"""Record each engine's per-page output as JSON, and replay recordings
through the hybrid combiner without running either engine.

    python recordings.py record ENGINE PDF_DIR OUT.json   # ENGINE: unlimited | rapidocr
    python recordings.py replay UNLIMITED.json RAPIDOCR.json PDF_DIR OUT_DIR

A recording maps each PDF's file stem to its recognized pages in order:
{"stem": [{"width": w, "height": h, "blocks": [{"text", "box", "quad", "chars"}]}]},
or null when the file failed. Pages are rendered by process_pdf exactly as in
a job, so a replay of the same PDFs sees the same pages and image sizes; the
two engines can be recorded on different machines.
"""
import json
import math
from pathlib import Path
import struct
import sys
import tempfile

from engine import Block, OcrError
from hybrid_engine import _png_size, merge
from pdf_layer import process_pdf


def page_record(png: bytes, blocks: list[Block]) -> dict:
    width, height = _png_size(png)
    return {"width": width, "height": height, "blocks": [
        {"text": b.text, "box": list(b.box),
         "quad": [list(p) for p in b.quad] if b.quad is not None else None,
         "chars": [[c, list(box)] for c, box in b.chars]} for b in blocks]}


def _numbers(values, count):
    values = tuple(float(v) for v in values)
    if len(values) != count or not all(math.isfinite(v) for v in values):
        raise ValueError("invalid recorded geometry")
    return values


def _quad(points):
    if points is None:
        return None
    if len(points) != 4:
        raise ValueError("invalid recorded geometry")
    return tuple(_numbers(p, 2) for p in points)


def page_blocks(record: dict) -> list[Block]:
    """The Blocks of one recorded page; malformed records raise ValueError."""
    try:
        return [Block(str(b["text"]), _numbers(b["box"], 4), _quad(b.get("quad")),
                      tuple((str(c), _numbers(box, 4)) for c, box in b.get("chars", ())))
                for b in record["blocks"]]
    except (KeyError, TypeError) as error:
        raise ValueError("invalid recorded page") from error


def combine(unlimited: dict, rapid: dict) -> list[Block]:
    """Hybrid output for one page from both engines' recordings of it."""
    if (unlimited["width"], unlimited["height"]) != (rapid["width"], rapid["height"]):
        raise ValueError("recordings are of different page images")
    return merge(page_blocks(unlimited), page_blocks(rapid), unlimited["width"], unlimited["height"])


def record(recognize, pdf_dir: Path) -> dict:
    recordings = {}
    with tempfile.TemporaryDirectory() as scratch:
        for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
            pages = []

            def recorder(png):
                blocks = recognize(png)
                pages.append(page_record(png, blocks))
                return blocks
            try:
                process_pdf(pdf, Path(scratch) / pdf.name, recorder)
                recordings[pdf.stem] = pages
            except OcrError as error:
                print(pdf.stem, error, file=sys.stderr)
                recordings[pdf.stem] = None
    return recordings


def replay(unlimited: dict, rapid: dict, pdf_dir: Path, out_dir: Path) -> dict:
    """Write hybrid PDFs for every file both recordings have; returns the
    status per file ("done", an error code, or "not_recorded")."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {}
    for pdf in sorted(Path(pdf_dir).glob("*.pdf")):
        pages_u, pages_r = unlimited.get(pdf.stem), rapid.get(pdf.stem)
        if not pages_u or not pages_r or len(pages_u) != len(pages_r):
            status[pdf.stem] = "not_recorded"
            continue
        pages = iter(zip(pages_u, pages_r))

        def recognize(png):
            page_u, page_r = next(pages)
            if [page_u["width"], page_u["height"]] != list(struct.unpack(">II", png[16:24])):
                raise ValueError("page image differs from the recording")
            return combine(page_u, page_r)
        try:
            process_pdf(pdf, out_dir / pdf.name, recognize)
            status[pdf.stem] = "done"
        except OcrError as error:
            status[pdf.stem] = str(error)
    return status


def main(argv):
    if len(argv) == 4 and argv[0] == "record" and argv[1] in {"unlimited", "rapidocr"}:
        if argv[1] == "rapidocr":
            from rapidocr_engine import RapidOCREngine
            engine = RapidOCREngine()
        else:
            from engine import UnlimitedOCR
            engine = UnlimitedOCR()
        Path(argv[3]).write_text(json.dumps(record(engine.recognize, Path(argv[2])),
                                            ensure_ascii=False), encoding="utf-8")
        return 0
    if len(argv) == 5 and argv[0] == "replay":
        load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
        status = replay(load(argv[1]), load(argv[2]), Path(argv[3]), Path(argv[4]))
        print(json.dumps(status, indent=1))
        return 0 if all(s == "done" for s in status.values()) else 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
