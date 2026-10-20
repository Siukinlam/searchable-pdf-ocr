"""Opt-in real model verification using only a generated synthetic scan.

Set OCR_MODEL_BASE_URL (ending /v1) and the optional upstream secret through the
environment, and OCR_ENGINE to choose the engine (default unlimited). This
script never starts/reconfigures the model or sends real files.
"""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "tests"))

from conftest import draw_scan  # noqa: E402  (synthetic scan drawn with the fixture font)
from engine import create_engine  # noqa: E402
from pdf_layer import process_pdf  # noqa: E402
import pdfgen  # noqa: E402
from pdfutil import find, open_pdf, pixels, text  # noqa: E402


def main(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "synthetic-scan.pdf"
    output = directory / "synthetic-searchable.pdf"
    lines = ["SEARCHABLE PDF TEST 12345", "繁體中文文件測試", "简体中文文档测试"]
    image, _ = draw_scan(lines, size=24, step=70, rect=False)
    image.save(directory / "before.png")
    pdfgen.write_pdf(source, [{"width": 600, "height": 800, "image": image}],
                     title="Synthetic OCR acceptance corpus")
    model = create_engine()
    if not model.ready():
        raise RuntimeError("configured OCR engine is not ready")
    started = time.monotonic()
    report = process_pdf(source, output, model.recognize)
    with open_pdf(source) as before, open_pdf(output) as after:
        extracted = text(after[0])
        pixel_equal = pixels(before[0], 2) == pixels(after[0], 2)
        after[0].render(scale=2).to_pil().save(directory / "after.png")
        terms = ["SEARCHABLE", "12345", "繁體中文", "简体中文"]
        checks = {term: bool(find(after[0], term)) for term in terms}
        receipt = {**report.public(), "engine": model.name, "real_model_inference": True,
                   "elapsed_seconds": round(time.monotonic() - started, 2),
                   "pixels_identical": pixel_equal,
                   "source_preserved": output.read_bytes().startswith(source.read_bytes()),
                   "search_checks": checks, "synthetic_extracted_text": extracted}
        (directory / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        if not pixel_equal or not all(checks.values()):
            raise RuntimeError("live OCR corpus acceptance failed; inspect synthetic receipt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
