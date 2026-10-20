"""One cancelable, resource-bounded child per document. No document logging."""
import json
import os
from pathlib import Path
import resource
import sys

from engine import OcrError, create_engine
from pdf_layer import process_pdf


SAFE_ERRORS = {
    "encrypted_or_invalid_pdf", "page_limit_exceeded", "signed_pdf_not_supported",
    "pdf_requires_repair", "invalid_pdf", "invalid_page_geometry", "no_text_recognized",
    "output_validation_failed", "invalid_model_output", "missing_grounding", "ungrounded_text",
    "invalid_grounding", "model_not_configured", "model_unavailable", "model_response_too_large",
    "model_output_incomplete", "output_too_large", "unsupported_ocr_glyph", "output_text_mismatch",
}


def main(work_dir: Path):
    os.umask(0o077)
    max_bytes = int(os.getenv("OCR_MAX_PDF_BYTES", str(64 * 1024 * 1024)))
    resource.setrlimit(resource.RLIMIT_FSIZE, (max_bytes, max_bytes))
    state = {"progress": 0, "pages_total": 0, "pages_with_text": 0, "pages_without_text": 0}

    def report(update):
        state.update(update)
        temporary = work_dir / "progress.tmp"
        temporary.write_text(json.dumps(state), encoding="ascii")
        temporary.replace(work_dir / "progress.json")

    try:
        engine = create_engine()
        # RLIMIT_CPU counts every thread. A local multithreaded engine must
        # not be killed before the wall-clock job deadline the manager enforces.
        cpu = int(os.getenv("OCR_JOB_TIMEOUT_SECONDS", "2100")) * max(1, engine.cpu_threads)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        if sys.platform == "linux":
            # Address space, not resident memory: ONNX Runtime reserves far
            # more than it touches, so each engine declares its own default.
            memory = int(os.getenv("OCR_WORKER_MEMORY_BYTES", str(engine.worker_address_space)))
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        result = process_pdf(work_dir / "source.pdf", work_dir / "result.pdf", engine.recognize,
                             report, max_pages=int(os.getenv("OCR_MAX_PAGES", "200")))
        if (work_dir / "result.pdf").stat().st_size > max_bytes:
            raise OcrError("output_too_large")
        report({**result.public(), "progress": 100, "status": "done"})
        return 0
    except Exception as exc:
        code = str(exc) if isinstance(exc, OcrError) and str(exc) in SAFE_ERRORS else "ocr_failed"
        (work_dir / "result.pdf").unlink(missing_ok=True)
        report({"status": "error", "error": code})
        return 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
