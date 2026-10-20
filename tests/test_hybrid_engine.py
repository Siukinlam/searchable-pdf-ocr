import random

import pymupdf as fitz
import pytest

from engine import Block, OcrError, create_engine
import hybrid_engine
from hybrid_engine import HybridEngine, merge
from pdf_layer import process_pdf


W = H = 1000  # page image pixels; boxes below are written in pixels


def block(text, x0, y0, x1, y1):
    """An Unlimited-OCR grounded block."""
    return Block(text, (x0 / W, y0 / H, x1 / W, y1 / H))


def line(text, x0, y0, x1, y1, chars=True, first=None, last=None):
    """A RapidOCR line; characters 80% of an even pitch between first and
    last (by default the line ends)."""
    glyphs = [c for c in text if not c.isspace()]
    start = x0 if first is None else first
    step = ((x1 if last is None else last) - start) / len(glyphs)
    boxes = tuple((c, ((start + k * step) / W, y0 / H, (start + (k + 0.8) * step) / W, y1 / H))
                  for k, c in enumerate(glyphs)) if chars else ()
    return Block(text, (x0 / W, y0 / H, x1 / W, y1 / H),
                 ((x0 / W, y0 / H), (x1 / W, y0 / H), (x1 / W, y1 / H), (x0 / W, y1 / H)), boxes)


def nonspace(blocks):
    return "".join("".join(b.text.split()) for b in blocks)


@pytest.fixture(params=["rapidfuzz", "difflib"])
def aligner(request, monkeypatch):
    if request.param == "rapidfuzz":
        pytest.importorskip("rapidfuzz")
        assert hybrid_engine.Levenshtein is not None
    else:
        monkeypatch.setattr(hybrid_engine, "Levenshtein", None)
    return request.param


def test_engine_selection_includes_hybrid(monkeypatch):
    monkeypatch.setenv("OCR_ENGINE", " Hybrid ")
    monkeypatch.setenv("OCR_RAPIDOCR_THREADS", "3")
    engine = create_engine()
    assert isinstance(engine, HybridEngine)
    # The worker limits follow the local CPU engine.
    assert engine.cpu_threads == 3 and engine.worker_address_space == 8 * 1024**3


def test_unlimited_characters_take_rapidocr_positions(aligner):
    rapid = line("虚擬内容一測試", 100, 100, 240, 120)
    merged = merge([block("虛擬內容—測試", 50, 80, 600, 140)], [rapid], W, H)
    # Unlimited's reading, including one-for-one misreads, on RapidOCR boxes.
    assert merged == [Block("虛擬內容—測試", rapid.box, rapid.quad,
                            tuple((c, box) for c, (_, box) in zip("虛擬內容—測試", rapid.chars)))]


def test_block_text_splits_onto_lines_and_keeps_word_gaps(aligner):
    lines = [line("Searchable docunent", 100, 100, 480, 120), line("繁体中文文件", 100, 130, 220, 150)]
    unlimited = block("Searchable document\n繁體中文\n文件", 90, 90, 500, 160)
    merged = merge([unlimited], lines, W, H)
    # A line break between CJK characters is not a word gap.
    assert [b.text for b in merged] == ["Searchable document", "繁體中文文件"]
    assert [b.quad for b in merged] == [l.quad for l in lines]
    assert [[box for _, box in b.chars] for b in merged] == [[box for _, box in l.chars] for l in lines]


def test_missed_characters_use_free_space_in_their_line(aligner):
    # RapidOCR's line box starts before its first recognized character.
    rapid = line("CR測試", 100, 100, 260, 120, first=130, last=230)
    merged = merge([block("OCR測試。", 50, 80, 600, 140)], [rapid], W, H)
    assert len(merged) == 1 and merged[0].text == "OCR測試。"
    o, c, *_, stop = merged[0].chars
    assert o == ("O", (0.1, 0.1, 0.13, 0.12)) and c == rapid.chars[0]
    # The missed full stop takes the room left at the end of the line.
    assert stop[0] == "。" and stop[1][0] == rapid.chars[-1][1][2] and stop[1][2] == 0.26


def test_missed_characters_without_room_share_a_neighbour(aligner):
    rapid = line("CR", 100, 100, 140, 120)
    merged = merge([block("OCR", 50, 80, 600, 140)], [rapid], W, H)
    (o, (ox0, _, ox1, _)), (c, (cx0, _, cx1, _)), r = merged[0].chars
    # O and C split C's span in reading order; R keeps its own box.
    assert (o, c) == ("O", "C") and ox0 == 0.1 and ox1 == pytest.approx(cx0) and cx1 == 0.116
    assert r == rapid.chars[1]


def test_unequal_misread_shares_the_rapidocr_span(aligner):
    rapid = line("modem", 100, 100, 200, 120)
    merged = merge([block("modern", 50, 80, 600, 140)], [rapid], W, H)
    assert merged[0].text == "modern" and len(merged[0].chars) == 6
    (_, (r0, _, r1, _)), (_, (n0, _, n1, _)) = merged[0].chars[4:]
    # r and n split the span RapidOCR gave m (aligners differ on the free
    # space after it).
    assert r0 == 0.18 and 0.196 <= n1 <= 0.2 and r1 == pytest.approx(n0)


def test_text_rapidocr_did_not_detect_keeps_block_placement(aligner):
    lines = [line("第一行文字", 100, 100, 200, 120), line("第三行文字", 100, 200, 200, 220)]
    unlimited = block("第一行文字\n第二行漏檢內容\n第三行文字", 50, 80, 600, 300)
    merged = merge([unlimited], lines, W, H)
    assert [b.text for b in merged] == ["第一行文字", "第二行漏檢內容", "第三行文字"]
    # The missed line keeps a block placement narrowed to the open band.
    assert merged[1] == Block("第二行漏檢內容", (0.05, 0.12, 0.6, 0.2))
    assert merged[0].quad == lines[0].quad and merged[2].quad == lines[1].quad


def test_rapidocr_only_text_is_never_added(aligner):
    title = block("標題", 50, 50, 300, 90)
    figure_note = block("圖一說明", 50, 400, 300, 440)
    lines = [line("標題", 100, 60, 140, 80), line("頁尾 99", 100, 900, 200, 920)]
    merged = merge([title, figure_note], lines, W, H)
    assert merged == [Block("標題", lines[0].box, lines[0].quad, lines[0].chars), figure_note]


def test_line_belongs_to_the_smallest_block_containing_its_center(aligner):
    table = block("合計\n金額", 0, 0, 1000, 1000)
    cell = block("金額", 400, 400, 600, 450)
    lines = [line("合計", 100, 100, 140, 120), line("金額", 450, 415, 490, 435)]
    merged = merge([table, cell], lines, W, H)
    # The table keeps its own line; its unmatched "金額" stays on the block.
    assert [(b.text, b.quad) for b in merged] == [("合計", lines[0].quad), ("金額", None),
                                                  ("金額", lines[1].quad)]


def test_table_cells_read_in_another_order_find_their_lines(aligner):
    # Unlimited reads cell by cell; RapidOCR reads row by row across cells.
    lines = [line("投標評審紀錄、涉", 100, 100, 260, 120), line("指定加密儲存區", 300, 100, 440, 120),
             line("及個人資料的個案", 100, 125, 260, 145), line("須經授權", 300, 125, 380, 145)]
    table = block("投標評審紀錄、涉及個人資料的個案 指定加密儲存區須經授權", 50, 50, 600, 200)
    merged = merge([table], lines, W, H)
    assert [(b.text, b.quad) for b in merged] == [
        ("投標評審紀錄、涉", lines[0].quad), ("及個人資料的個案", lines[2].quad),
        ("指定加密儲存區", lines[1].quad), ("須經授權", lines[3].quad)]
    assert all(b.chars == l.chars for b, l in zip(merged, [lines[i] for i in (0, 2, 1, 3)]))


def test_long_substitution_is_other_text_not_a_misread(aligner):
    # Same length, nothing in common: the cell is not where this text is.
    rapid = line("甲乙丙丁戊己", 100, 100, 220, 120)
    merged = merge([block("子丑寅卯辰巳", 50, 80, 600, 140)], [rapid], W, H)
    assert merged == [block("子丑寅卯辰巳", 50, 80, 600, 140)]


def test_line_without_character_boxes_uses_the_covered_part(aligner):
    rapid = line("ABCDEFGH", 100, 100, 260, 120, chars=False)
    merged = merge([block("EFGH", 150, 80, 300, 140)], [rapid], W, H)
    assert len(merged) == 1 and merged[0].text == "EFGH" and merged[0].chars == ()
    assert [v for point in merged[0].quad for v in point] == \
        pytest.approx([0.18, 0.1, 0.26, 0.1, 0.26, 0.12, 0.18, 0.12])


def test_text_is_never_lost_and_character_boxes_suit_the_layer(aligner):
    alphabet = "甲乙丙丁戊己庚辛壬癸ABCxyz0123—一。"
    generator = random.Random(20261020)
    for _ in range(300):
        rows = ["".join(generator.choice(alphabet) for _ in range(generator.randint(1, 12)))
                for _ in range(generator.randint(1, 4))]
        lines = []
        for n, row in enumerate(rows):
            # RapidOCR drops, adds, substitutes and reorders characters.
            noisy = [c if generator.random() > 0.2 else generator.choice(alphabet + "?")
                     for c in row if generator.random() > 0.15]
            noisy += [generator.choice(alphabet)] * (generator.random() < 0.2)
            if noisy and generator.random() > 0.15:
                lines.append(line("".join(noisy), 100, 100 + 30 * n, 100 + 25 * len(noisy),
                                  120 + 30 * n, chars=generator.random() > 0.2))
        generator.shuffle(lines)
        unlimited = block("\n".join(rows), 90, 90, 600, 230)
        merged = merge([unlimited], lines, W, H)
        assert nonspace(merged) == nonspace([unlimited])
        for piece in merged:
            if piece.chars:
                assert [c for c, _ in piece.chars] == [c for c in piece.text if not c.isspace()]
                starts = [box[0] for _, box in piece.chars]
                assert all(box[2] > box[0] for _, box in piece.chars) and starts == sorted(starts)


class FakeUnlimited:
    worker_address_space = 2 * 1024**3

    def __init__(self, blocks=(), error=None, ready=True):
        self.blocks, self.error, self.is_ready, self.images = list(blocks), error, ready, []

    def configured(self):
        return self.is_ready

    def ready(self, timeout=3):
        return self.is_ready

    def recognize(self, png):
        self.images.append(png)
        if self.error:
            raise OcrError(self.error)
        return self.blocks


class FakeRapid(FakeUnlimited):
    cpu_threads = 2
    worker_address_space = 8 * 1024**3


def test_hybrid_text_layer_passes_extraction_check(scan_factory, tmp_path, aligner):
    source, truth = scan_factory()
    # Unlimited reads all three lines correctly as one block; RapidOCR
    # misreads two and misses the third. No character boxes: line quads only.
    x0, y0 = min(b.box[0] for b in truth), min(b.box[1] for b in truth)
    x1, y1 = max(b.box[2] for b in truth), max(b.box[3] for b in truth)
    unlimited = FakeUnlimited([Block("\n".join(b.text for b in truth), (x0 - 0.01, y0 - 0.01, x1 + 0.01, y1 + 0.01))])
    misread = ["Searchable docunent 12345", "繁體中文文仵測試"]
    rapid = FakeRapid([Block(text, b.box, ((b.box[0], b.box[1]), (b.box[2], b.box[1]),
                                           (b.box[2], b.box[3]), (b.box[0], b.box[3])))
                       for text, b in zip(misread, truth)])
    output = tmp_path / "hybrid.pdf"
    report = process_pdf(source, output, HybridEngine(unlimited, rapid).recognize)
    assert report.pages_ocr_added == 1
    assert unlimited.images == rapid.images and len(unlimited.images) == 1
    with fitz.open(output) as doc:
        page = doc[0]
        assert page.get_text().split() == "Searchable document 12345 繁體中文文件測試 简体中文文档测试".split()
        for b in truth[:2]:
            found = page.search_for(b.text)
            assert len(found) == 1
            expected = fitz.Rect(b.box[0] * page.rect.width, b.box[1] * page.rect.height,
                                 b.box[2] * page.rect.width, b.box[3] * page.rect.height)
            assert all(abs(a - e) < 1.5 for a, e in zip(found[0], expected)), b.text
        assert page.search_for(truth[2].text)


def tiny_png():
    return fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 4, 3), False).tobytes("png")


@pytest.mark.parametrize("failing, code", [("unlimited", "model_unavailable"),
                                           ("rapid", "invalid_model_output")])
def test_either_recognizer_failure_fails_the_page_with_its_code(failing, code):
    unlimited = FakeUnlimited(error=code if failing == "unlimited" else None)
    rapid = FakeRapid(error=code if failing == "rapid" else None)
    with pytest.raises(OcrError, match=f"^{code}$"):
        HybridEngine(unlimited, rapid).recognize(tiny_png())
    # Both always see the page; neither is left running.
    assert len(unlimited.images) == len(rapid.images) == 1


def test_non_png_input_fails_closed():
    with pytest.raises(OcrError, match="^invalid_model_output$"):
        HybridEngine(FakeUnlimited(), FakeRapid()).recognize(b"\x89PNG not really")


@pytest.mark.parametrize("unlimited_ready, rapid_ready", [(True, False), (False, True)])
def test_ready_requires_both_recognizers(unlimited_ready, rapid_ready):
    engine = HybridEngine(FakeUnlimited(ready=unlimited_ready), FakeRapid(ready=rapid_ready))
    assert engine.ready() is False and engine.configured() is False
    assert HybridEngine(FakeUnlimited(), FakeRapid()).ready() is True
