"""Opt-in real model verification using only a generated synthetic scan.

Set OCR_MODEL_BASE_URL (ending /v1) and the optional upstream secret through the
environment. This script never starts/reconfigures the model or sends real files.
"""
import argparse
import json
from pathlib import Path
import time

import pymupdf as fitz

from engine import UnlimitedOCR
from pdf_layer import process_pdf


def main(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "synthetic-scan.pdf"
    output = directory / "synthetic-searchable.pdf"
    native = fitz.open()
    page = native.new_page(width=600, height=800)
    font = fitz.Font("cjk")
    page.insert_font(fontname="SyntheticFont", fontbuffer=font.buffer)
    for index, text in enumerate(["SEARCHABLE PDF TEST 12345", "繁體中文文件測試", "简体中文文档测试"]):
        page.insert_text((65, 100 + 70 * index), text, fontname="SyntheticFont", fontsize=24)
    image = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
    image.save(directory / "before.png")
    scanned = fitz.open()
    scan_page = scanned.new_page(width=600, height=800)
    scan_page.insert_image(scan_page.rect, stream=image.tobytes("png"))
    scanned.set_metadata({"title": "Synthetic OCR acceptance corpus"})
    scanned.save(source)
    scanned.close()
    native.close()
    model = UnlimitedOCR()
    if not model.ready():
        raise RuntimeError("configured OCR model is not ready")
    started = time.monotonic()
    report = process_pdf(source, output, model.recognize)
    with fitz.open(source) as before, fitz.open(output) as after:
        text = after[0].get_text()
        pixel_equal = before[0].get_pixmap(matrix=fitz.Matrix(2, 2)).samples == \
                      after[0].get_pixmap(matrix=fitz.Matrix(2, 2)).samples
        after[0].get_pixmap(matrix=fitz.Matrix(2, 2)).save(directory / "after.png")
        terms = ["SEARCHABLE", "12345", "繁體中文", "简体中文"]
        checks = {term: bool(after[0].search_for(term)) for term in terms}
        receipt = {**report.public(), "real_model_inference": True,
                   "elapsed_seconds": round(time.monotonic() - started, 2),
                   "pixels_identical": pixel_equal,
                   "source_preserved": output.read_bytes().startswith(source.read_bytes()),
                   "search_checks": checks, "synthetic_extracted_text": text,
                   "geometry": "model block bounds; line/word positions approximate"}
        (directory / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
        print(json.dumps(receipt, ensure_ascii=False, indent=2))
        if not pixel_equal or not all(checks.values()):
            raise RuntimeError("live OCR corpus acceptance failed; inspect synthetic receipt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)
