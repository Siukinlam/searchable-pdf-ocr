"""Run one engine over single-page PDFs, recording timing and raw blocks.

    python bench/run_engine.py ENGINE CORPUS_DIR OUT_DIR [--code DIR]

ENGINE is unlimited or rapidocr. Each page's recognized blocks are saved to
OUT_DIR/blocks/<page>.json so other combinations can be replayed offline.
"""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("engine", choices=["unlimited", "rapidocr"])
parser.add_argument("corpus", type=Path)
parser.add_argument("out", type=Path)
parser.add_argument("--code", type=Path, default=Path(__file__).resolve().parents[1])
args = parser.parse_args()
sys.path.insert(0, str(args.code))

from pdf_layer import process_pdf  # noqa: E402

if args.engine == "unlimited":
    from engine import UnlimitedOCR as Engine  # noqa: E402
else:
    from rapidocr_engine import RapidOCREngine as Engine  # noqa: E402

engine = Engine()
if args.engine == "rapidocr":
    engine._engine()  # load models once, outside the per-page timing
(args.out / "blocks").mkdir(parents=True, exist_ok=True)
results = {}
for pdf in sorted(args.corpus.glob("*.pdf")):
    record = {"recognize_s": 0.0}
    recorded = []

    def recognize(png):
        start = time.perf_counter()
        try:
            blocks = engine.recognize(png)
            recorded.extend(dataclasses.asdict(b) for b in blocks)
            return blocks
        finally:
            record["recognize_s"] += time.perf_counter() - start

    start = time.perf_counter()
    try:
        process_pdf(pdf, args.out / pdf.name, recognize)
        record["status"] = "done"
    except Exception as exc:
        record["status"] = "error:" + (str(exc) if type(exc).__name__ == "OcrError" else type(exc).__name__)
    record["total_s"] = round(time.perf_counter() - start, 3)
    record["recognize_s"] = round(record["recognize_s"], 3)
    (args.out / "blocks" / f"{pdf.stem}.json").write_text(json.dumps(recorded, ensure_ascii=False))
    results[pdf.stem] = record
    print(pdf.stem, record, flush=True)
(args.out / "timing.json").write_text(json.dumps(results, indent=1))
