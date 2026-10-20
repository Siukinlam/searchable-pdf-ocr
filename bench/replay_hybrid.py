"""Build hybrid results from two run_engine.py recordings, without either engine.

    python bench/replay_hybrid.py CORPUS_DIR UNLIMITED_DIR RAPIDOCR_DIR OUT_DIR

recognize_s is estimated as max(Unlimited, RapidOCR) + merge time, because the
hybrid engine runs RapidOCR while the model request is in flight.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import Block  # noqa: E402
from hybrid_engine import _png_size, merge  # noqa: E402
from pdf_layer import process_pdf  # noqa: E402


def load(path):
    blocks = []
    for b in json.loads(path.read_text()):
        quad = tuple(tuple(p) for p in b["quad"]) if b.get("quad") else None
        blocks.append(Block(b["text"], tuple(b["box"]), quad,
                            tuple((c, tuple(box)) for c, box in b.get("chars", ()))))
    return blocks


corpus, unlimited, rapid, out = (Path(a) for a in sys.argv[1:5])
out.mkdir(parents=True, exist_ok=True)
timing_u = json.loads((unlimited / "timing.json").read_text())
timing_r = json.loads((rapid / "timing.json").read_text())
results = {}
for pdf in sorted(corpus.glob("*.pdf")):
    if timing_u.get(pdf.stem, {}).get("status") != "done" or timing_r.get(pdf.stem, {}).get("status") != "done":
        results[pdf.stem] = {"status": "not_recorded"}
        continue
    a, b = load(unlimited / "blocks" / f"{pdf.stem}.json"), load(rapid / "blocks" / f"{pdf.stem}.json")
    spent = {"merge": 0.0}

    def recognize(png):
        start = time.perf_counter()
        try:
            return merge(a, b, *_png_size(png))
        finally:
            spent["merge"] += time.perf_counter() - start

    start = time.perf_counter()
    try:
        process_pdf(pdf, out / pdf.name, recognize)
        status = "done"
    except Exception as exc:
        status = "error:" + (str(exc) if type(exc).__name__ == "OcrError" else type(exc).__name__)
    layer = time.perf_counter() - start - spent["merge"]
    engines = max(timing_u[pdf.stem]["recognize_s"], timing_r[pdf.stem]["recognize_s"])
    results[pdf.stem] = {"status": status, "merge_s": round(spent["merge"], 3),
                         "recognize_s": round(engines + spent["merge"], 3),
                         "total_s": round(engines + spent["merge"] + layer, 3)}
    print(pdf.stem, results[pdf.stem], flush=True)
(out / "timing.json").write_text(json.dumps(results, indent=1))
