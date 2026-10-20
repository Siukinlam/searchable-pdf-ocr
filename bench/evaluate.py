"""Score engine outputs against make_corpus.py references.

    python bench/evaluate.py OUT_DIR name=RESULT_DIR [name=RESULT_DIR ...] [--overlays PAGE ...]

Metrics (whitespace and dot leaders removed, NFKC applied):
  cer       edit distance / reference characters (reading order counts)
  bag_f1    order-free character multiset F1
  center    distance between a reference glyph center and the text-layer
            character aligned to it (points); in_box = share of aligned
            text-layer centers inside the reference glyph box
"""
import argparse
import json
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw
from rapidfuzz.distance import Levenshtein

LEADERS = set(".…．·‧・")


def norm(c):
    return unicodedata.normalize("NFKC", c)


def clean(chars):
    out = []
    for c, box in chars:
        n = norm(c)
        if len(n) == 1 and not n.isspace() and n not in LEADERS:
            out.append((n, box))
    return out


def layer_chars(page):
    rm = page.rotation_matrix
    return [(c["c"], tuple(fitz.Rect(c["bbox"]) * rm)) for b in page.get_text("rawdict")["blocks"]
            for l in b.get("lines", []) for s in l["spans"] for c in s["chars"]]


def score(ref_chars, hyp_chars):
    r, h = clean(ref_chars), clean(hyp_chars)
    rs, hs = "".join(c for c, _ in r), "".join(c for c, _ in h)
    common = sum((Counter(rs) & Counter(hs)).values())
    p, q = (common / len(hs) if hs else 0), (common / len(rs) if rs else 0)
    errors, inside = [], 0
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(rs, hs):
        if tag != "equal":
            continue
        for k in range(i2 - i1):
            rb, hb = r[i1 + k][1], h[j1 + k][1]
            hx, hy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
            rx, ry = (rb[0] + rb[2]) / 2, (rb[1] + rb[3]) / 2
            errors.append(((rx - hx) ** 2 + (ry - hy) ** 2) ** 0.5)
            inside += rb[0] <= hx <= rb[2] and rb[1] <= hy <= rb[3]
    errors.sort()
    return {"ref_chars": len(rs), "cer": Levenshtein.distance(rs, hs) / max(1, len(rs)),
            "bag_f1": 2 * p * q / (p + q) if p + q else 0.0, "aligned": len(errors),
            "center_p50": statistics.median(errors) if errors else None,
            "center_p90": errors[int(0.9 * (len(errors) - 1))] if errors else None,
            "in_box": inside / len(errors) if errors else 0.0}


def overlay(src, layers, path):
    zoom = 2
    pix = src.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
    base = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    panels = []
    for label, chars, color in layers:
        im = base.copy()
        d = ImageDraw.Draw(im, "RGBA")
        for _, b in chars:
            d.rectangle([v * zoom for v in b], outline=color + (255,), fill=color + (40,))
        d.rectangle([0, 0, 12 * len(label) + 20, 30], fill=(255, 255, 255, 230))
        d.text((8, 8), label, fill=(0, 0, 0))
        panels.append(im)
    canvas = Image.new("RGB", (sum(p.width for p in panels) + 12 * (len(panels) - 1), base.height), "white")
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width + 12
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("engines", nargs="+")
    parser.add_argument("--overlays", nargs="*", default=[])
    args = parser.parse_args()
    refs = json.loads((args.out / "reference.json").read_text())
    engines = dict(e.split("=", 1) for e in args.engines)
    metrics = {}
    for name, folder in engines.items():
        folder = Path(folder)
        timing = json.loads((folder / "timing.json").read_text())
        pages = {}
        for page, ref in refs.items():
            t = timing.get(page, {})
            if t.get("status") != "done":
                pages[page] = {"status": t.get("status", "missing")}
                continue
            with fitz.open(folder / f"{page}.pdf") as doc:
                pages[page] = {"status": "done", "recognize_s": t["recognize_s"], "total_s": t["total_s"],
                               **score(ref["chars"], layer_chars(doc[0]))}
        metrics[name] = pages
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    groups = sorted({p.rsplit("-", 1)[0] for p in refs})
    lines = ["| engine | group | pages | recognize s/page | CER | bag F1 | center P50 (pt) | center P90 (pt) | in box |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, pages in metrics.items():
        for group in groups + ["all"]:
            rows = [v for k, v in pages.items() if v["status"] == "done" and (group == "all" or k.startswith(group + "-"))]
            if not rows:
                continue
            total = sum(r["ref_chars"] for r in rows)
            aligned = sum(r["aligned"] for r in rows)
            lines.append("| {} | {} | {} | {:.2f} | {:.1%} | {:.3f} | {:.2f} | {:.1f} | {:.1%} |".format(
                name, group, len(rows), statistics.fmean(r["recognize_s"] for r in rows),
                sum(r["cer"] * r["ref_chars"] for r in rows) / total,
                statistics.fmean(r["bag_f1"] for r in rows),
                statistics.median(r["center_p50"] for r in rows), statistics.median(r["center_p90"] for r in rows),
                sum(r["in_box"] * r["aligned"] for r in rows) / max(1, aligned)))
        failed = [k for k, v in pages.items() if v["status"] != "done"]
        if failed:
            lines.append(f"| {name} | failed pages | {len(failed)} | | | | | | |")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    colors = [(220, 30, 30), (30, 60, 220), (200, 120, 0), (120, 0, 160)]
    for page in args.overlays:
        corpus = args.out / "corpus" / f"{page}.pdf"
        with fitz.open(corpus) as src:
            layers = [("reference", [tuple(c) for c in refs[page]["chars"]], (0, 150, 0))]
            for (name, folder), color in zip(engines.items(), colors):
                f = Path(folder) / f"{page}.pdf"
                if f.exists():
                    with fitz.open(f) as doc:
                        layers.append((name, layer_chars(doc[0]), color))
            (args.out / "overlays").mkdir(exist_ok=True)
            overlay(src[0], layers, args.out / "overlays" / f"{page}.png")


if __name__ == "__main__":
    main()
