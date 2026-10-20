"""Authenticated PDF OCR job adapter for an existing Unlimited-OCR deployment."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
import json
import logging
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from engine import OcrError, create_engine, secret
from worker import SAFE_ERRORS


logger = logging.getLogger("ocr-service")
DIAGNOSTIC_ERRORS = frozenset(SAFE_ERRORS) | {"ocr_failed", "ocr_timeout", "canceled"}


def envint(name, default, low, high):
    try:
        return max(low, min(high, int(os.getenv(name, str(default)))))
    except ValueError:
        return default


@dataclass
class Settings:
    root: Path = field(default_factory=lambda: Path(os.getenv("OCR_JOB_ROOT", "/data/ocr-jobs")).resolve())
    max_bytes: int = field(default_factory=lambda: envint("OCR_MAX_PDF_BYTES", 64 * 1024**2, 1024, 64 * 1024**2))
    max_pages: int = field(default_factory=lambda: envint("OCR_MAX_PAGES", 200, 1, 500))
    workers: int = field(default_factory=lambda: envint("OCR_WORKERS", 1, 1, 4))
    queue_size: int = field(default_factory=lambda: envint("OCR_QUEUE_SIZE", 1, 0, 4))
    max_jobs: int = field(default_factory=lambda: envint("OCR_MAX_RETAINED_JOBS", 8, 1, 100))
    timeout: int = field(default_factory=lambda: envint("OCR_JOB_TIMEOUT_SECONDS", 2100, 1, 2100))
    ttl: int = field(default_factory=lambda: envint("OCR_JOB_TTL_SECONDS", 3600, 1, 86400))
    disk_bytes: int = field(default_factory=lambda: envint("OCR_MAX_DISK_BYTES", 1024**3, 1024, 10 * 1024**3))
    upload_timeout: int = field(default_factory=lambda: envint("OCR_UPLOAD_TIMEOUT_SECONDS", 120, 1, 300))


@dataclass
class Job:
    job_id: str
    work_dir: Path
    owner: str = ""
    status: str = "queued"
    progress: float = 0
    error: str = ""
    pages_total: int = 0
    pages_with_text: int = 0
    pages_without_text: int = 0
    pages_ocr_added: int = 0
    pages_native_text: int = 0
    updated: float = field(default_factory=time.monotonic)
    canceled: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    process_lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    process: subprocess.Popen | None = None
    submitted: bool = False
    slot_held: bool = True

    def public(self):
        return {key: getattr(self, key) for key in (
            "job_id", "status", "progress", "error", "pages_total", "pages_with_text",
            "pages_without_text", "pages_ocr_added", "pages_native_text")}


def kill_process(process):
    if process and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            # DELETE and the worker can observe cancellation together. A
            # concurrent reap may make killpg report EPERM on macOS; accept
            # that only after confirming our own child has already exited.
            if process.poll() is None:
                raise
        process.wait(timeout=5)


class Manager:
    def __init__(self, config: Settings):
        self.config = config
        self.jobs = {}
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=config.workers, thread_name_prefix="ocr")
        self.slots = threading.BoundedSemaphore(config.workers + config.queue_size)
        self.stop = threading.Event()
        self.config.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.config.root.chmod(0o700)
        # A fresh single-process service cannot resume child processes or jobs.
        # Its dedicated root contains only opaque generated job directories.
        for child in self.config.root.iterdir():
            if re.fullmatch(r"[0-9a-f]{32}", child.name) and child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
        self.janitor = threading.Thread(target=self._clean_loop, daemon=True)
        self.janitor.start()

    def used_bytes(self):
        return sum(p.stat().st_size for p in self.config.root.rglob("*") if p.is_file())

    def reserve(self, owner=""):
        with self.lock:
            active = sum(j.status in {"queued", "running"} for j in self.jobs.values())
            if active >= self.config.workers + self.config.queue_size or len(self.jobs) >= self.config.max_jobs:
                raise HTTPException(429, "OCR service is busy")
            # Reserve a full source+result allowance for every admitted job.
            # Output is capped by the worker's RLIMIT_FSIZE and final check.
            reservation = (active + 1) * (2 * self.config.max_bytes + 65536)
            if self.used_bytes() + reservation > self.config.disk_bytes or \
               shutil.disk_usage(self.config.root).free < reservation:
                raise HTTPException(507, "OCR storage capacity exceeded")
            # Deleted queued futures remain in ThreadPoolExecutor's internal
            # queue until a worker consumes them. Keep their admission slot
            # until then so submit/delete loops cannot grow that queue forever.
            if not self.slots.acquire(blocking=False):
                raise HTTPException(429, "OCR service is busy")
            job_id = uuid.uuid4().hex
            directory = self.config.root / job_id
            try:
                directory.mkdir(mode=0o700)
            except Exception:
                self.slots.release()
                raise
            job = Job(job_id, directory, owner=owner)
            self.jobs[job_id] = job
            return job

    def submit(self, job):
        with self.lock:
            job.submitted = True
            try:
                self.pool.submit(self._run, job)
            except Exception:
                job.submitted = False
                raise

    def _release_slot(self, job):
        with self.lock:
            if job.slot_held:
                job.slot_held = False
                self.slots.release()

    def _read_progress(self, job):
        try:
            path = job.work_dir / "progress.json"
            if path.stat().st_size > 65536:
                return
            data = json.loads(path.read_text())
            with self.lock:
                for key in ("pages_total", "pages_with_text", "pages_without_text",
                            "pages_ocr_added", "pages_native_text"):
                    value = data.get(key)
                    if type(value) is int and 0 <= value <= self.config.max_pages:
                        setattr(job, key, value)
                progress = data.get("progress")
                if type(progress) in {int, float} and math.isfinite(progress):
                    job.progress = max(0, min(99, progress))
            return data
        except (OSError, ValueError, AttributeError):
            return None

    def _run(self, job):
        try:
            # Popen can wait a long time for exec under host load. Serialize
            # startup and cancellation per job so unrelated status requests
            # and queued documents never wait on that process startup.
            with job.process_lock:
                with self.lock:
                    if job.canceled.is_set():
                        return
                    job.status = "running"
                env = os.environ.copy()
                env.update(OCR_MAX_PDF_BYTES=str(self.config.max_bytes),
                           OCR_MAX_PAGES=str(self.config.max_pages),
                           OCR_JOB_TIMEOUT_SECONDS=str(self.config.timeout),
                           PYTHONDONTWRITEBYTECODE="1")
                job.process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).with_name("worker.py")), str(job.work_dir)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True, env=env)
                if job.canceled.is_set() or self.stop.is_set():
                    kill_process(job.process)
            deadline = time.monotonic() + self.config.timeout
            while job.process.poll() is None:
                if job.canceled.is_set() or self.stop.is_set() or time.monotonic() >= deadline:
                    with job.process_lock:
                        kill_process(job.process)
                    job.error = "canceled" if job.canceled.is_set() else "ocr_timeout"
                    break
                self._read_progress(job)
                job.canceled.wait(0.1)
            result = self._read_progress(job) or {}
            with self.lock:
                valid = (result.get("status") == "done" and job.process.returncode == 0 and
                         0 < job.pages_with_text <= job.pages_total and
                         job.pages_with_text + job.pages_without_text == job.pages_total and
                         (job.work_dir / "result.pdf").is_file() and
                         (job.work_dir / "result.pdf").stat().st_size <= self.config.max_bytes)
                if valid and not job.canceled.is_set() and not job.error:
                    job.status, job.progress = "done", 100
                else:
                    job.status = "error"
                    if job.canceled.is_set():
                        job.error = "canceled"
                    else:
                        job.error = job.error or (result.get("error") if result.get("error") in SAFE_ERRORS else "ocr_failed")
                    (job.work_dir / "result.pdf").unlink(missing_ok=True)
                job.updated = time.monotonic()
        except Exception:
            with job.process_lock:
                kill_process(job.process)
            with self.lock:
                code = "canceled" if job.canceled.is_set() else "ocr_failed"
                job.status, job.error, job.updated = "error", code, time.monotonic()
        finally:
            if job.status == "error":
                # Clients delete terminal jobs, so retain only the bounded
                # cause needed for diagnosis. Never log documents, job IDs,
                # paths, upstream responses, or exception details.
                code = job.error if type(job.error) is str and job.error in DIAGNOSTIC_ERRORS else "ocr_failed"
                logger.warning("OCR job failed: code=%s", code)
            self._release_slot(job)
            job.finished.set()

    def delete(self, job):
        job.canceled.set()
        # Wait for this job's Popen to publish its child before deleting files;
        # never hold the manager lock while waiting for startup or termination.
        with job.process_lock:
            kill_process(job.process)
        # A queued task is released by the bounded executor once preceding work
        # leaves. It checks canceled before touching its removed directory.
        if job.process:
            job.finished.wait(6)
        with self.lock:
            shutil.rmtree(job.work_dir, ignore_errors=True)
            self.jobs.pop(job.job_id, None)
            if not job.submitted:
                self._release_slot(job)

    def _clean_loop(self):
        while not self.stop.wait(min(30, self.config.ttl)):
            with self.lock:
                expired = [job for job in self.jobs.values() if job.status in {"done", "error"}
                           and time.monotonic() - job.updated >= self.config.ttl]
            for job in expired:
                self.delete(job)

    def close(self):
        self.stop.set()
        with self.lock:
            all_jobs = list(self.jobs.values())
        for job in all_jobs:
            self.delete(job)
        self.pool.shutdown(wait=True, cancel_futures=True)
        self.janitor.join(timeout=1)


def require_job_owner():
    return os.getenv("OCR_REQUIRE_JOB_OWNER", "0") == "1"


def request_owner(request: Request):
    if not require_job_owner():
        return ""
    owner = request.headers.get("x-ocr-job-owner", "")
    if not re.fullmatch(r"[A-Za-z0-9_-]{32,128}", owner):
        raise HTTPException(403, "Managed job owner is required")
    return owner


def authenticated(request: Request):
    expected = secret("OCR_API_KEY")
    if not expected:
        if not require_job_owner() and os.getenv("OCR_ALLOW_UNAUTHENTICATED", "0") == "1":
            return
        raise HTTPException(503, "OCR service authentication is not configured")
    supplied = request.headers.get("authorization", "")
    if not supplied.lower().startswith("bearer ") or \
       not secrets.compare_digest(supplied[7:].strip(), expected):
        raise HTTPException(401, "Invalid API key")
    request_owner(request)


def engine_ready():
    try:
        return create_engine().ready()
    except OcrError:
        return False


def create_app(config=None):
    config = config or Settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.manager = Manager(config)
        yield
        await run_in_threadpool(app.state.manager.close)

    app = FastAPI(title="Searchable PDF OCR", lifespan=lifespan, docs_url=None, redoc_url=None)

    def find_job(job_id, request):
        manager = app.state.manager
        with manager.lock:
            if not re.fullmatch(r"[0-9a-f]{32}", job_id) or job_id not in manager.jobs:
                raise HTTPException(404, "OCR job not found")
            job = manager.jobs[job_id]
            if not secrets.compare_digest(job.owner, request_owner(request)):
                raise HTTPException(404, "OCR job not found")
            return job

    @app.get("/health")
    async def health():
        try:
            engine = create_engine()
            name = engine.name
        except OcrError:
            engine, name = None, "unsupported"
        upstream = bool(engine) and await run_in_threadpool(engine.ready)
        auth = bool(secret("OCR_API_KEY")) or (not require_job_owner() and os.getenv("OCR_ALLOW_UNAUTHENTICATED", "0") == "1")
        ready = upstream and auth
        return JSONResponse({"ready": ready, "engine": name, "model_available": upstream,
                             "text_layer": "unicode-invisible", "local_language_packs": "not_applicable"},
                            status_code=200 if ready else 503)

    @app.post("/v1/ocr", dependencies=[Depends(authenticated)])
    async def submit(request: Request):
        if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/pdf":
            raise HTTPException(415, "Expected application/pdf")
        # Only raw bytes are accepted; there are no paths, filenames, model
        # options, callback URLs, query switches, or client-selected commands.
        if request.query_params:
            raise HTTPException(400, "OCR options are not accepted")
        try:
            length = int(request.headers.get("content-length", "0"))
        except ValueError:
            raise HTTPException(400, "Invalid content length") from None
        if length < 0 or length > config.max_bytes:
            raise HTTPException(413, "PDF exceeds size limit")
        if not await run_in_threadpool(engine_ready):
            raise HTTPException(503, "OCR model is unavailable")
        manager = app.state.manager
        job = manager.reserve(request_owner(request))
        try:
            count = 0
            async with asyncio.timeout(config.upload_timeout):
                with (job.work_dir / "source.pdf").open("xb") as stream:
                    async for chunk in request.stream():
                        count += len(chunk)
                        if count > config.max_bytes:
                            raise HTTPException(413, "PDF exceeds size limit")
                        stream.write(chunk)
            with (job.work_dir / "source.pdf").open("rb") as stream:
                if not count or not stream.read(1024).lstrip().startswith(b"%PDF-"):
                    raise HTTPException(400, "Invalid PDF")
            manager.submit(job)
            return {"job_id": job.job_id, "status": "queued"}
        except BaseException as exc:
            await run_in_threadpool(manager.delete, job)
            if isinstance(exc, TimeoutError):
                raise HTTPException(408, "PDF upload timed out") from None
            raise

    @app.get("/v1/ocr/jobs/{job_id}", dependencies=[Depends(authenticated)])
    def status(job_id: str, request: Request):
        with app.state.manager.lock:
            return find_job(job_id, request).public()

    @app.get("/v1/ocr/jobs/{job_id}/result", dependencies=[Depends(authenticated)])
    def result(job_id: str, request: Request):
        with app.state.manager.lock:
            job = find_job(job_id, request)
            if job.status != "done":
                raise HTTPException(409, "OCR result is not ready")
            # Open under the lock; a concurrent DELETE can unlink the path but
            # cannot replace the authenticated download's already-open handle.
            stream = (job.work_dir / "result.pdf").open("rb")
            size = os.fstat(stream.fileno()).st_size
        def chunks():
            try:
                while data := stream.read(65536):
                    yield data
            finally:
                stream.close()
        return StreamingResponse(chunks(), media_type="application/pdf",
                                 headers={"Content-Length": str(size), "Cache-Control": "no-store"},
                                 background=BackgroundTask(stream.close))

    @app.delete("/v1/ocr/jobs/{job_id}", dependencies=[Depends(authenticated)])
    def delete(job_id: str, request: Request):
        job = find_job(job_id, request)
        app.state.manager.delete(job)
        return {"job_id": job_id, "status": "deleted"}

    return app


app = create_app()
