import json

import pymupdf as fitz
import pytest

from engine import Block
from hybrid_engine import HybridEngine, merge
import recordings


def quad_of(box):
    x0, y0, x1, y1 = box
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def engines(truth):
    """Fake recognizers for scan_factory pages: one coarse block holding all
    lines, and line blocks whose second line is misread."""
    x0, y0 = min(b.box[0] for b in truth) - 0.01, min(b.box[1] for b in truth) - 0.01
    x1, y1 = max(b.box[2] for b in truth) + 0.01, max(b.box[3] for b in truth) + 0.01
    unlimited = [Block("\n".join(b.text for b in truth), (x0, y0, x1, y1))]
    misread = [b.text.replace("體", "体") for b in truth]
    rapid = [Block(text, b.box, quad_of(b.box), ((text[0], b.box),) if len(text) == 1 else ())
             for text, b in zip(misread, truth)]
    return (lambda png: unlimited), (lambda png: rapid)


def test_page_record_round_trips_through_json():
    png = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 40, 30), False).tobytes("png")
    blocks = [Block("區塊", (0.1, 0.2, 0.5, 0.4)),
              Block("AB", (0.1, 0.1, 0.3, 0.2), quad_of((0.1, 0.1, 0.3, 0.2)),
                    (("A", (0.1, 0.1, 0.2, 0.2)), ("B", (0.2, 0.1, 0.3, 0.2))))]
    record = json.loads(json.dumps(recordings.page_record(png, blocks)))
    assert (record["width"], record["height"]) == (40, 30)
    assert recordings.page_blocks(record) == blocks


@pytest.mark.parametrize("bad", [{"text": "x", "box": [0, 0, 1]}, {"text": "x", "box": [0, 0, 1, "nan"]},
                                 {"box": [0, 0, 1, 1]}, {"text": "x", "box": [0, 0, 1, 1], "quad": [[0, 0]] * 3}])
def test_malformed_recording_is_rejected(bad):
    with pytest.raises(ValueError):
        recordings.page_blocks({"blocks": [bad]})


def test_combine_is_merge_and_needs_the_same_page_image():
    unlimited = {"width": 100, "height": 50, "blocks": [{"text": "AB", "box": [0, 0, 1, 1]}]}
    rapid = {"width": 100, "height": 50, "blocks": [
        {"text": "AB", "box": [0.1, 0.1, 0.3, 0.2], "quad": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.2], [0.1, 0.2]],
         "chars": [["A", [0.1, 0.1, 0.2, 0.2]], ["B", [0.2, 0.1, 0.3, 0.2]]]}]}
    assert recordings.combine(unlimited, rapid) == merge(
        recordings.page_blocks(unlimited), recordings.page_blocks(rapid), 100, 50)
    with pytest.raises(ValueError, match="different page images"):
        recordings.combine(unlimited, {**rapid, "width": 101})


def test_recorded_engines_replay_to_the_live_hybrid_result(scan_factory, tmp_path):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    source, truth = scan_factory(native=True)
    source.rename(pdfs / "scan.pdf")
    (pdfs / "broken.pdf").write_bytes(b"%PDF-1.7 not really")
    unlimited, rapid = engines(truth)
    # Each engine recorded on its own, as if on different machines.
    recorded_u = json.loads(json.dumps(recordings.record(unlimited, pdfs)))
    recorded_r = json.loads(json.dumps(recordings.record(rapid, pdfs)))
    assert recorded_u["broken"] is None and len(recorded_u["scan"]) == 1  # native page skipped
    assert sorted(p.name for p in pdfs.iterdir()) == ["broken.pdf", "scan.pdf"]
    status = recordings.replay(recorded_u, recorded_r, pdfs, tmp_path / "out")
    assert status == {"broken": "not_recorded", "scan": "done"}

    class Engine:
        worker_address_space, cpu_threads = 0, 1

        def __init__(self, recognize):
            self.recognize = recognize
    live = tmp_path / "live.pdf"
    from pdf_layer import process_pdf
    process_pdf(pdfs / "scan.pdf", live, HybridEngine(Engine(unlimited), Engine(rapid)).recognize)
    with fitz.open(tmp_path / "out" / "scan.pdf") as replayed, fitz.open(live) as direct:
        assert replayed[0].get_text("rawdict") == direct[0].get_text("rawdict")
        assert "繁體中文文件測試" in replayed[0].get_text()


def test_replay_refuses_a_different_page_image(scan_factory, tmp_path):
    source, truth = scan_factory()
    unlimited, rapid = engines(truth)
    recorded = recordings.record(unlimited, source.parent)
    for pages in recorded.values():
        pages[0]["width"] += 1
    with pytest.raises(ValueError, match="differs from the recording"):
        recordings.replay(recorded, recorded, source.parent, tmp_path / "out")
