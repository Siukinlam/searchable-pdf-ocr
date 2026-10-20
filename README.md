# searchable-pdf-ocr

A self-hosted HTTP service that makes scanned PDFs searchable. It adds an
invisible Unicode text layer to a copy of the PDF, so viewers can search,
select and copy the recognized text. The original pages stay pixel-identical,
and the original file's bytes are kept as the prefix of the result.

Documents never leave your infrastructure. There are three recognition engines:

| `OCR_ENGINE` | Recognition | Text-layer geometry | Runs on |
|---|---|---|---|
| `unlimited` (default) | [Baidu Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR), a document VLM served by your own [vLLM](https://recipes.vllm.ai/baidu/Unlimited-OCR) endpoint | Block boxes; lines and characters inside a block are estimated | GPU (vLLM) |
| `rapidocr` | [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) PP-OCRv6 through [RapidOCR](https://github.com/RapidAI/RapidOCR) on ONNX Runtime | Line quads and per-character boxes | CPU, x86_64 or aarch64 |
| `hybrid` | Unlimited-OCR's text | RapidOCR's line and character boxes | both (RapidOCR runs while the model request is in flight) |

Why combine them? A document VLM reads text very accurately but only reports
where each **block** is. A classic PP-OCR pipeline reports where every **line
and character** is, but it misreads more. In `hybrid`, the VLM's text is
aligned by edit distance onto the PP-OCR line and character boxes. Searching
and selecting then hit the right glyphs without losing recognition accuracy.

![Text-layer boxes per engine on the same paragraph](docs/overlay.png)

## Benchmark

30 generated pages (paragraphs, bilingual, tables, ±1.5° skew, 120 dpi), all
engines on one aarch64 server. Full method, per-group numbers and a harder
private corpus are in [docs/BENCHMARK.md](docs/BENCHMARK.md).

| Engine | Recognize (s/page) | CER | Characters placed on their glyph | Median center error |
|---|---:|---:|---:|---:|
| `unlimited` (1 GPU, vLLM) | 7.33 | 0.1% | 11.0% | 48.8 pt |
| `rapidocr` (4 CPU cores) | 2.07 | 0.3% | 98.5% | 0.97 pt |
| `hybrid` | 7.34 | 0.1% | 98.6% | 0.96 pt |

On harder real-world documents the gap in recognition is larger. CER was 4.3%
for `unlimited` and `hybrid` vs 7.0% for `rapidocr`, mostly from Traditional
Chinese variant forms. Validate on your own documents.

## How it works

1. `POST /v1/ocr` accepts raw PDF bytes and returns an opaque job ID.
2. A bounded child process renders each page without a text layer at 200 dpi
   and sends it to the selected engine.
3. Recognized text is written with PDF render mode 3 (invisible) by an
   **incremental save**. Images, metadata, bookmarks and annotations are kept,
   and pages that already contain text are skipped.
4. The result is reopened and verified: page counts, extracted text equal to
   the recognized text, and source bytes as prefix. Then it can be downloaded.

### Text-layer placement

- **Block engines** (Unlimited-OCR): lines are spread inside the grounded
  block. Long paragraphs without line breaks are wrapped to fit.
- **Line engines** (RapidOCR, hybrid):
  - Each line is placed along its own baseline and follows skew, top-to-bottom
    columns and upside-down lines.
  - The line is split into runs wherever the font's proportional advances would
    miss a detected character start or word end by more than 20% of the line
    height. A run that ends inside a word is stretched to the next run, so
    extractors never see a false gap.
  - Character boxes that do not cover every character fall back to the line box.

Pages with no recognized text are counted in `pages_without_text`. A document
with no recognized text at all fails instead of returning an unchanged file.

## Quick start

```sh
python -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-rapidocr.txt
export OCR_ENGINE=rapidocr
export OCR_API_KEY_FILE=/run/secrets/ocr_api_key
export OCR_JOB_ROOT=/var/lib/ocr-jobs
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8092 --workers 1 --no-access-log
```

```sh
curl -s -H "Authorization: Bearer $KEY" -H "Content-Type: application/pdf" \
     --data-binary @scan.pdf http://127.0.0.1:8092/v1/ocr
```

```sh
curl -s -H "Authorization: Bearer $KEY" http://127.0.0.1:8092/v1/ocr/jobs/<job_id>
```

```sh
curl -s -H "Authorization: Bearer $KEY" -o scan_searchable.pdf \
     http://127.0.0.1:8092/v1/ocr/jobs/<job_id>/result
```

```sh
curl -s -X DELETE -H "Authorization: Bearer $KEY" http://127.0.0.1:8092/v1/ocr/jobs/<job_id>
```

Docker: `docker build --build-arg OCR_INSTALL_RAPIDOCR=1 -t searchable-pdf-ocr .`
The image runs as UID 10001 and exposes port 8092. Mount a private, writable job
directory owned by that UID, and mount secrets read-only. Run **one** ASGI
process; `OCR_WORKERS` bounds the document child processes.

## API

| Request | Result |
| --- | --- |
| `POST /v1/ocr` with a raw `application/pdf` body | `{"job_id": "<32 hex>", "status": "queued"}` |
| `GET /v1/ocr/jobs/{job_id}` | `status` (queued, running, done or error), progress 0–100, a safe error code and integer page counts |
| `GET /v1/ocr/jobs/{job_id}/result` | The PDF when the job is done, otherwise 409 |
| `DELETE /v1/ocr/jobs/{job_id}` | Kills the worker and removes the files |
| `GET /health` | 200 only when authentication is configured and the selected engine is ready |

The API accepts no filenames, client paths, callback URLs, per-request model
options or shell commands. Errors are constant codes such as
`encrypted_or_invalid_pdf`, `signed_pdf_not_supported`, `model_unavailable`,
`unsupported_ocr_glyph`, `ocr_timeout` and `canceled`. They never contain
document text, paths, upstream responses or exception messages.

Final counts satisfy `pages_with_text + pages_without_text = pages_total`.
Signed or certified PDFs, encrypted PDFs, PDFs that cannot be saved
incrementally, unsupported glyphs and invalid engine output are rejected.

## Configuration

### Engines

- `OCR_ENGINE`: `unlimited` (default), `rapidocr` or `hybrid`. Any other value
  fails closed: health returns 503 and jobs are rejected.
- **Unlimited-OCR**
  - `OCR_MODEL_BASE_URL`: an OpenAI-compatible URL ending in `/v1`.
  - `OCR_MODEL`: defaults to `baidu/Unlimited-OCR`.
  - `OCR_MODEL_API_KEY_FILE` or `OCR_MODEL_API_KEY`: optional upstream key.
  - Requests follow the official vLLM recipe: prompt `<image>document parsing.`,
    `temperature=0`, `skip_special_tokens=false`, and n-gram logits processor
    arguments.
  - Truncated, ungrounded or malformed output fails; it is never guessed.
  - URLs with credentials, query strings or fragments are rejected. Redirects
    and ambient proxies are disabled.
- **RapidOCR**
  - Install `requirements-rapidocr.txt`. The wheel bundles the PP-OCRv6 small
    detection, PP-OCRv6 small recognition and mobile direction-classifier ONNX
    models.
  - `OCR_RAPIDOCR_DET_MODEL`, `OCR_RAPIDOCR_REC_MODEL` and
    `OCR_RAPIDOCR_CLS_MODEL` select other **local** ONNX files of the same
    family. Models are never downloaded at runtime.
  - For Traditional Chinese, `PP-OCRv6_rec_medium.onnx` is recommended (see
    Benchmark).
  - `OCR_RAPIDOCR_THREADS`: default 4. `OCR_RAPIDOCR_MIN_SCORE`: default 0.5.
  - When the direction classifier marks a line as upside down, both
    orientations are recognized and the better-scoring reading is kept.

### Service

- **Authentication:** `OCR_API_KEY_FILE` (preferred) or `OCR_API_KEY`. Every job
  route requires `Authorization: Bearer …`. If the configured secret file is
  missing, the service fails closed. `OCR_ALLOW_UNAUTHENTICATED=1` is for
  development only.
- **Multi-tenant mode:** `OCR_REQUIRE_JOB_OWNER=1` puts the service behind a
  trusted gateway, which must send `X-OCR-Job-Owner`, an opaque 32–128
  character base64url ID derived from its authenticated session.
  - Jobs are visible only to their owner.
  - Other owners get the same 404 as for an unknown job.
  - The owner ID is never logged or returned.

### Limits

| Setting | Default / maximum |
| --- | --- |
| `OCR_MAX_PDF_BYTES` | 64 MiB / 64 MiB (upload and result) |
| `OCR_MAX_PAGES` | 200 / 500 |
| `OCR_WORKERS`, `OCR_QUEUE_SIZE` | 1 / 4 each |
| `OCR_MAX_RETAINED_JOBS` | 8 / 100 |
| `OCR_JOB_TIMEOUT_SECONDS` | 2100 / 2100 |
| `OCR_JOB_TTL_SECONDS` | 3600 / 86400 |
| `OCR_MAX_DISK_BYTES` | 1 GiB / 10 GiB, reserved at admission |
| `OCR_WORKER_MEMORY_BYTES` | Linux address-space limit: 2 GiB (Unlimited-OCR) or 8 GiB (RapidOCR, hybrid) |

**Why the address-space limit is higher for RapidOCR:** ONNX Runtime reserves
address space it never touches. Limits of 2–4 GiB made inference fail on an
aarch64 server although peak RSS was about 0.8 GiB. Bound resident memory with
the container's memory limit instead. The worker's CPU-time limit is the job
timeout multiplied by the engine's thread count.

## Air-gapped and aarch64 installs

- All RapidOCR dependencies publish manylinux aarch64 wheels. To install on a
  host without reliable PyPI access, build a wheelhouse on any machine:

  ```sh
  pip download --dest wheels --only-binary=:all: --python-version 3.11 \
      --platform manylinux_2_28_aarch64 --platform manylinux2014_aarch64 \
      -r requirements-rapidocr.txt
  ```

  ```sh
  pip install --no-index --find-links wheels -r requirements-rapidocr.txt
  ```

- PyPI's aarch64 onnxruntime is CPU-only. The `GPU device discovery failed`
  warning is harmless.

## Tests

```sh
pip install -r requirements-test.txt -r requirements-rapidocr.txt
python -m pytest -q tests
```

- HTTP tests use a loopback mock model. Synthetic English, Traditional Chinese
  and Simplified Chinese scans are rasterized, so they really have no text layer.
- The tests check reloaded text, search boxes, the source-bytes prefix, pixel
  equality, rotations 0/90/180/270, crop boxes, signatures and encryption,
  authentication, limits, cancellation, timeouts and TTL.
- With the RapidOCR packages installed, the same synthetic scans also run
  through the real local model.
- `python live_smoke.py --output DIR` runs one synthetic page against a real
  Unlimited-OCR endpoint.

## Limitations

- Recognition quality depends on the engine. RapidOCR PP-OCRv6 small sometimes
  outputs variant or Simplified forms of Traditional characters, and table text
  comes out row by row.
- Handwriting, faded print and real vertical documents have not been
  benchmarked.
- The text layer embeds the full Droid Sans Fallback CJK font, about 3.4 MB per
  document.
- Only searchability in PyMuPDF is verified. Check your PDF viewer.

## License

AGPL-3.0-only, because the service links [PyMuPDF](https://pymupdf.readthedocs.io/)
(AGPL-3.0 or Artifex commercial). Writing the text layer with pikepdf and
rendering pages with pypdfium2 would remove the AGPL dependency; that is
feasible but not implemented.

Third-party components:

| Component | License |
|---|---|
| RapidOCR | Apache-2.0 |
| PaddleOCR models | Apache-2.0 |
| ONNX Runtime | MIT |
| OpenCV (headless) | Apache-2.0 |
| Baidu Unlimited-OCR | MIT |
| FastAPI | MIT |
| httpx | BSD-3-Clause |
