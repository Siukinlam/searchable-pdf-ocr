# Benchmark

All numbers below come from a corpus you can regenerate with the scripts in
`bench/`. The text is fictional and written for this benchmark.

## Corpus

`bench/make_corpus.py` draws 30 A4 pages with Pillow and records every
character's box as ground truth: its advance width by the font's ascent and
descent, as an extractor reports it. Each page is then wrapped as an
image-only PDF:

| Group | Pages | Content | Raster |
|---|---:|---|---|
| `para` | 10 | Traditional Chinese headings and numbered paragraphs | 150 dpi grayscale, JPEG q65 |
| `bilingual` | 6 | Alternating Traditional Chinese and English paragraphs | 150 dpi, q65 |
| `table` | 6 | A 15-row ruled table plus a paragraph | 150 dpi, q65 |
| `skew` | 4 | Paragraph pages rotated ±1.5° (references rotated with them) | 150 dpi, q65 |
| `lowres` | 4 | Paragraph pages | 120 dpi, JPEG q50 |

The published run used Songti TC (macOS). On Linux, Noto Serif CJK TC is picked
automatically. Every engine renders pages at 200 dpi through the same
`process_pdf`.

## Setup

- **`unlimited`:** Baidu Unlimited-OCR on vLLM, one GPU, on an aarch64 server
  shared with other workloads. Its recorded blocks are written by the v0.2.0
  text layer.
- **`rapidocr`:** PP-OCRv6 small (det/rec) plus the mobile direction
  classifier. ONNX Runtime CPU in a container with `--cpus 4` and
  `OCR_RAPIDOCR_THREADS=4` on the same aarch64 server, installed offline from a
  wheelhouse, with no PyMuPDF installed.
- **`hybrid`:** both recordings replayed through `hybrid_engine.merge` with
  `bench/replay_hybrid.py`. Its time is max(unlimited, rapidocr) + merge,
  because the hybrid engine runs RapidOCR while the model request is in flight.
  Merge itself averaged 4 ms per page.

## Metrics

Text and boxes are read back with PDFium. Whitespace and dot leaders are
removed and NFKC is applied before scoring.

- **CER:** edit distance divided by reference characters, on PDFium's
  extraction order.
- **bag F1:** order-free character multiset F1.
- **center P50 / P90:** distance in points between each reference glyph's
  center and the text-layer character aligned to it, using PDFium's loose
  (selection) boxes. P50 and P90 are per page; the table shows the median over
  pages.
- **in box:** share of aligned text-layer character centers inside the true
  glyph box. This is what you see when you search or select.

## Results

| engine | group | pages | recognize s/page | CER | bag F1 | center P50 (pt) | center P90 (pt) | in box |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| unlimited | bilingual | 6 | 5.23 | 0.0% | 1.000 | 38.32 | 210.6 | 13.5% |
| unlimited | lowres | 4 | 7.75 | 1.4% | 0.998 | 51.41 | 300.4 | 8.4% |
| unlimited | para | 10 | 7.45 | 2.3% | 1.000 | 54.10 | 311.3 | 8.5% |
| unlimited | skew | 4 | 7.57 | 7.9% | 1.000 | 33.86 | 165.5 | 12.2% |
| unlimited | table | 6 | 5.35 | 0.2% | 0.998 | 23.08 | 70.1 | 21.7% |
| unlimited | all | 30 | 6.64 | 2.1% | 0.999 | 42.74 | 243.2 | 11.8% |
| rapidocr | bilingual | 6 | 1.66 | 0.2% | 0.998 | 1.45 | 2.2 | 95.0% |
| rapidocr | lowres | 4 | 1.87 | 0.2% | 0.999 | 0.78 | 1.5 | 99.3% |
| rapidocr | para | 10 | 1.95 | 0.2% | 0.998 | 0.97 | 1.7 | 99.3% |
| rapidocr | skew | 4 | 1.83 | 0.2% | 0.999 | 0.70 | 1.5 | 99.2% |
| rapidocr | table | 6 | 1.65 | 0.9% | 0.993 | 1.24 | 2.2 | 92.3% |
| rapidocr | all | 30 | 1.80 | 0.3% | 0.997 | 1.09 | 1.9 | 97.4% |
| hybrid | bilingual | 6 | 5.24 | 0.0% | 1.000 | 1.45 | 2.2 | 95.1% |
| hybrid | lowres | 4 | 7.75 | 0.1% | 0.998 | 0.79 | 1.6 | 99.6% |
| hybrid | para | 10 | 7.45 | 0.0% | 1.000 | 0.99 | 1.8 | 99.7% |
| hybrid | skew | 4 | 7.58 | 0.1% | 1.000 | 0.70 | 1.5 | 99.8% |
| hybrid | table | 6 | 5.35 | 0.2% | 0.998 | 1.21 | 2.1 | 92.4% |
| hybrid | all | 30 | 6.65 | 0.1% | 0.999 | 1.08 | 1.9 | 97.7% |

![Text-layer boxes on a paragraph page](overlay.png)

- **Recognition:** all characters arrive (bag F1 ≥ 0.997). `unlimited`'s CER
  is mostly reading order: with block-level placement, PDFium sometimes
  extracts a paragraph number after its line. `hybrid` places the same text
  line by line and reads back in order (0.1%).
- **Placement:** block grounding (`unlimited`) puts about 1 in 8 characters on
  its glyph. Line and character grounding (`rapidocr`, `hybrid`) puts about 97%
  on the glyph, typically within 1 pt. Bilingual and table pages score lower
  (92–95%), mostly on Latin text and cell padding.
- **Output size:** each scanned page grew by about its own size, 4.9 MB of
  input became 10.1 MB, because PDFium's incremental save rewrites the objects
  of every modified page. The original bytes stay intact as the prefix.

## Text-layer writer: v0.2.0 (PDFium) vs v0.1.0 (PyMuPDF)

The same recorded engine output (v0.1.0's 30-page benchmark corpus, all three
engines) was written by both writers and measured with PDFium and poppler:

| Measure | v0.1.0 PyMuPDF | v0.2.0 PDFium |
|---|---|---|
| Every character extractable by PDFium | 90/90 pages | 90/90 |
| … by poppler `pdftotext` | 86/90 | 86/90 (same 4 block-placed pages) |
| `虛構` found by PDFium search and by `pdftotext` | on all pages whose text has it | same |
| Rendered pixels equal to the source (PDFium, `pdftoppm`) | 90/90 | 90/90 |
| In box: `hybrid` / `rapidocr` / `unlimited` | 98.63% / 98.44% / 11.33% | 98.53% / 98.35% / 11.44% |
| `pdftotext` order error: `hybrid` / `rapidocr` | 3.92% / 4.02% | 3.89% / 4.23% |
| Size added per page | +3.5 MB (embedded CJK font) | +0.17 MB (page objects rewritten, ~1 KB font) |

## Harder documents

On a separate private corpus of 34 dense, official-style pages (long
paragraphs, ruled tables, bilingual text, 1.5° skew), recognition differed much
more:

| Engine | CER (NFKC) | In box |
|---|---:|---:|
| unlimited | 4.3% | 22.1% |
| rapidocr, PP-OCRv6 small rec | 7.0% | 98.4% |
| rapidocr, PP-OCRv6 medium rec | 6.6% | 98.6% |
| rapidocr, PP-OCRv5 server rec (ch) | 7.6% | 95.9% |
| hybrid | 4.3% | 98.6% |

Typical RapidOCR errors there:

- Variant or Simplified forms of Traditional characters (虛→虚, 內→内).
- The em dash read as 一.
- Latin words inside Chinese sentences losing a letter.

The PP-OCRv5 server Chinese model outputs Simplified forms widely
(資→资, 試→试), so it is not recommended for Traditional text. PP-OCRv6 medium
removed the 虛→虚 errors at about 2.6× the small model's CPU time. Writing
those pages with the v0.2.0 text layer gave the same placement as v0.1.0
(98.7% in box for `hybrid`). **Validate on your own documents**; the
generated corpus above is optimistic.

## Reproduce

```sh
pip install -r requirements.txt -r requirements-rapidocr.txt -r bench/requirements.txt
python bench/make_corpus.py out
python bench/run_engine.py rapidocr out/corpus out/rapidocr
OCR_MODEL_BASE_URL=... python bench/run_engine.py unlimited out/corpus out/unlimited
python bench/replay_hybrid.py out/corpus out/unlimited out/rapidocr out/hybrid
python bench/evaluate.py out unlimited=out/unlimited rapidocr=out/rapidocr hybrid=out/hybrid --overlays para-02
```

Engines can be recorded on different machines; `run_engine.py` saves each
page's blocks as JSON. The service also includes `recordings.py`, which records
whole directories and replays them through the combiner.
