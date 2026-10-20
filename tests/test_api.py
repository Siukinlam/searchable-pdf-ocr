import json
import logging
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time

from fastapi.testclient import TestClient
import pymupdf as fitz
import pytest

from app import Manager, Settings, create_app, kill_process
from engine import UnlimitedOCR


@pytest.fixture
def upstream(monkeypatch):
    state = {"raw": "", "calls": [], "wait": threading.Event(), "delay": False,
             "model": "baidu/Unlimited-OCR"}
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, data):
            raw = json.dumps(data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            assert self.path == "/v1/models"
            self.reply({"data": [{"id": state["model"]}]})

        def do_POST(self):
            assert self.path == "/v1/chat/completions"
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["calls"].append(request)
            if state["delay"]:
                state["wait"].wait(10)
            self.reply({"choices": [{"finish_reason": "stop", "message": {"content": state["raw"]}}]})
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("OCR_MODEL_BASE_URL", f"http://127.0.0.1:{server.server_port}/v1")
    monkeypatch.setenv("OCR_MODEL", "baidu/Unlimited-OCR")
    monkeypatch.delenv("OCR_MODEL_API_KEY_FILE", raising=False)
    monkeypatch.delenv("OCR_MODEL_API_KEY", raising=False)
    yield state
    state["wait"].set()
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setenv("OCR_API_KEY", "api-test-key")
    monkeypatch.delenv("OCR_API_KEY_FILE", raising=False)
    monkeypatch.delenv("OCR_ALLOW_UNAUTHENTICATED", raising=False)
    return Settings(root=tmp_path / "jobs", ttl=60)


AUTH = {"Authorization": "Bearer api-test-key"}
PDF_AUTH = {**AUTH, "Content-Type": "application/pdf"}


def wait_status(client, job_id, target={"done", "error"}):
    for _ in range(200):
        status = client.get("/v1/ocr/jobs/" + job_id, headers=AUTH).json()
        if status["status"] in target:
            return status
        time.sleep(0.05)
    pytest.fail("job did not reach expected status")


def grounding(blocks):
    return "\n".join('<|det|>text ' + json.dumps([round(x * 999, 5) for x in b.box]) +
                     '<|/det|>' + b.text for b in blocks)


def test_real_subprocess_upload_poll_download_delete(config, upstream, scan_factory):
    source, blocks = scan_factory(native=True)
    upstream["raw"] = grounding(blocks)
    with TestClient(create_app(config)) as client:
        assert client.get("/health").json()["ready"] is True
        submitted = client.post("/v1/ocr", content=source.read_bytes(), headers=PDF_AUTH)
        assert submitted.status_code == 200
        job_id = submitted.json()["job_id"]
        assert len(job_id) == 32
        result = wait_status(client, job_id)
        assert result["status"] == "done", result
        assert result["progress"] == 100
        assert result["pages_total"] == result["pages_with_text"] == 2
        assert result["pages_without_text"] == 0
        pdf = client.get(f"/v1/ocr/jobs/{job_id}/result", headers=AUTH)
        assert pdf.status_code == 200
        assert pdf.headers["content-type"] == "application/pdf"
        assert pdf.content.startswith(source.read_bytes())
        with fitz.open(stream=pdf.content, filetype="pdf") as doc:
            assert "繁體中文文件測試" in doc[0].get_text()
            assert doc[0].search_for("Searchable")
        assert len(upstream["calls"]) == 1
        assert client.delete(f"/v1/ocr/jobs/{job_id}", headers=AUTH).status_code == 200
        assert not list(config.root.iterdir())
        assert client.get(f"/v1/ocr/jobs/{job_id}", headers=AUTH).status_code == 404


def test_authentication_fail_closed_on_every_route(config, monkeypatch, tmp_path):
    with TestClient(create_app(config)) as client:
        for method, url in [("POST", "/v1/ocr"), ("GET", "/v1/ocr/jobs/" + "a" * 32),
                            ("GET", "/v1/ocr/jobs/" + "a" * 32 + "/result"),
                            ("DELETE", "/v1/ocr/jobs/" + "a" * 32)]:
            assert client.request(method, url).status_code == 401
        monkeypatch.delenv("OCR_API_KEY")
        assert client.post("/v1/ocr").status_code == 503
        monkeypatch.setenv("OCR_API_KEY", "api-test-key")
        monkeypatch.setenv("OCR_API_KEY_FILE", str(tmp_path / "missing-secret"))
        assert client.post("/v1/ocr", headers=PDF_AUTH).status_code == 503


def test_wrong_model_health_is_not_ready(config, upstream):
    upstream["model"] = "unrelated-ASR-model"
    with TestClient(create_app(config)) as client:
        assert client.get("/health").status_code == 503
        result = client.post("/v1/ocr", content=b"%PDF-1.7", headers=PDF_AUTH)
        assert result.status_code == 503
        assert not list(config.root.iterdir())


def test_validation_and_stream_limit(config, upstream):
    config.max_bytes = 1024
    with TestClient(create_app(config)) as client:
        assert client.post("/v1/ocr", content=b"%PDF-1.7", headers=AUTH).status_code == 415
        assert client.post("/v1/ocr?filename=private.pdf", content=b"%PDF-1.7", headers=PDF_AUTH).status_code == 400
        assert client.post("/v1/ocr", content=b"not a PDF", headers=PDF_AUTH).status_code == 400
        assert client.post("/v1/ocr", content=b"%PDF-1.7" + b"x" * 2000,
                           headers={**PDF_AUTH, "Content-Length": "10"}).status_code == 413
        assert not list(config.root.iterdir())


def test_cancel_kills_child_and_cleans_before_return(config, upstream, scan_factory):
    source, blocks = scan_factory()
    upstream.update(raw=grounding(blocks), delay=True)
    app = create_app(config)
    with TestClient(app) as client:
        job_id = client.post("/v1/ocr", content=source.read_bytes(), headers=PDF_AUTH).json()["job_id"]
        wait_status(client, job_id, {"running"})
        job = app.state.manager.jobs[job_id]
        assert client.get(f"/v1/ocr/jobs/{job_id}/result", headers=AUTH).status_code == 409
        start = time.monotonic()
        assert client.delete(f"/v1/ocr/jobs/{job_id}", headers=AUTH).status_code == 200
        assert time.monotonic() - start < 3
        assert job.process.poll() is not None
        assert not job.work_dir.exists()


def test_capacity_timeout_and_ttl(config, upstream, scan_factory):
    source, blocks = scan_factory()
    upstream.update(raw=grounding(blocks), delay=True)
    config.queue_size, config.timeout, config.ttl = 0, 1, 1
    with TestClient(create_app(config)) as client:
        job_id = client.post("/v1/ocr", content=source.read_bytes(), headers=PDF_AUTH).json()["job_id"]
        assert client.post("/v1/ocr", content=source.read_bytes(), headers=PDF_AUTH).status_code == 429
        result = wait_status(client, job_id)
        assert result["status"] == "error"
        assert result["error"] == "ocr_timeout"
        for _ in range(80):
            if not list(config.root.iterdir()):
                break
            time.sleep(0.05)
        assert not list(config.root.iterdir())


def test_disk_reservation_prevents_upload(config, upstream):
    config.disk_bytes = 1024
    with TestClient(create_app(config)) as client:
        assert client.post("/v1/ocr", content=b"%PDF-1.7", headers=PDF_AUTH).status_code == 507


def test_errors_never_return_model_content(config, upstream, scan_factory, caplog):
    source, _ = scan_factory()
    upstream["raw"] = "PRIVATE MODEL RESPONSE MUST NOT ESCAPE"
    caplog.set_level(logging.WARNING, logger="ocr-service")
    with TestClient(create_app(config)) as client:
        job_id = client.post("/v1/ocr", content=source.read_bytes(), headers=PDF_AUTH).json()["job_id"]
        result = wait_status(client, job_id)
        assert result["status"] == "error"
        assert result["error"] == "missing_grounding"
        assert "PRIVATE" not in json.dumps(result)
        assert client.get(f"/v1/ocr/jobs/{job_id}/result", headers=AUTH).status_code == 409
        assert client.app.state.manager.jobs[job_id].finished.wait(2)
        messages = [r.getMessage() for r in caplog.records if r.name == "ocr-service"]
        assert messages == ["OCR job failed: code=missing_grounding"]
        assert "PRIVATE" not in caplog.text


@pytest.mark.parametrize("failure", ["spawn_exception", "worker_error"])
def test_failure_diagnostics_discard_untrusted_details(config, monkeypatch, caplog, failure):
    private = "PRIVATE EXCEPTION /customer/document.pdf upstream reply must not escape"
    caplog.set_level(logging.WARNING, logger="ocr-service")
    manager = Manager(config)
    try:
        job = manager.reserve()
        if failure == "spawn_exception":
            def fail(*args, **kwargs):
                raise RuntimeError(private)
            monkeypatch.setattr("app.subprocess.Popen", fail)
        else:
            class FailedProcess:
                returncode = 1
                def poll(self):
                    return self.returncode
            monkeypatch.setattr("app.subprocess.Popen", lambda *a, **kw: FailedProcess())
            monkeypatch.setattr(manager, "_read_progress", lambda _: {"status": "error", "error": private})
        manager.submit(job)
        assert job.finished.wait(2)
        assert job.status == "error"
        assert job.error == "ocr_failed"
        records = [r for r in caplog.records if r.name == "ocr-service"]
        assert [r.getMessage() for r in records] == ["OCR job failed: code=ocr_failed"]
        assert all(r.exc_info is None and r.stack_info is None for r in records)
        assert private not in caplog.text
        assert job.job_id not in caplog.text
        assert str(job.work_dir) not in caplog.text
    finally:
        manager.close()


@pytest.mark.parametrize("already_exited", [True, False])
def test_duplicate_cancel_ignores_signal_error_only_after_child_exit(monkeypatch, already_exited):
    class Process:
        pid = 123
        waited = False
        polls = iter([None, -9 if already_exited else None])
        def poll(self):
            return next(self.polls)
        def wait(self, timeout):
            self.waited = True
            return -9
    process = Process()
    def denied(*args):
        raise PermissionError("synthetic concurrent process-group reap")
    monkeypatch.setattr("app.os.killpg", denied)
    if already_exited:
        kill_process(process)
        assert process.waited
    else:
        with pytest.raises(PermissionError):
            kill_process(process)
        assert not process.waited


def test_blocked_startup_keeps_status_responsive_and_cancel_owns_child(config, monkeypatch, caplog):
    started, release_startup = threading.Event(), threading.Event()
    spawned, killed = [], []
    caplog.set_level(logging.WARNING, logger="ocr-service")
    monkeypatch.setattr(UnlimitedOCR, "ready", lambda _: True)

    class Process:
        returncode = None
        def poll(self):
            return self.returncode

    def blocked_startup(*args, **kwargs):
        started.set()
        assert release_startup.wait(10)
        process = Process()
        spawned.append(process)
        return process

    def stop_child(process):
        if process and process.poll() is None:
            killed.append(process)
            process.returncode = -9

    monkeypatch.setattr("app.subprocess.Popen", blocked_startup)
    monkeypatch.setattr("app.kill_process", stop_child)
    app = create_app(config)
    with TestClient(app) as client:
        def request_async(method, url, **kwargs):
            future = Future()
            def run():
                try:
                    future.set_result(client.request(method, url, **kwargs))
                except BaseException as exc:
                    future.set_exception(exc)
            threading.Thread(target=run, daemon=True).start()
            return future

        cancel = None
        try:
            first = client.post("/v1/ocr", content=b"%PDF-1.7", headers=PDF_AUTH).json()["job_id"]
            assert started.wait(2)
            job = app.state.manager.jobs[first]
            status = request_async("GET", f"/v1/ocr/jobs/{first}", headers=AUTH).result(timeout=2)
            assert status.json()["status"] == "running"
            queued = request_async("POST", "/v1/ocr", content=b"%PDF-1.7", headers=PDF_AUTH).result(timeout=2)
            assert queued.status_code == 200
            second = queued.json()["job_id"]
            status = request_async("GET", f"/v1/ocr/jobs/{second}", headers=AUTH).result(timeout=2)
            assert status.json()["status"] == "queued"

            cancel = request_async("DELETE", f"/v1/ocr/jobs/{first}", headers=AUTH)
            assert job.canceled.wait(2)
            assert not cancel.done()
            assert job.work_dir.exists()
            # Cancellation waits only for A's unpublished child. The status
            # route and deleting queued B still complete before A can start.
            status = request_async("GET", f"/v1/ocr/jobs/{first}", headers=AUTH).result(timeout=2)
            assert status.status_code == 200
            deleted = request_async("DELETE", f"/v1/ocr/jobs/{second}", headers=AUTH).result(timeout=2)
            assert deleted.status_code == 200
        finally:
            release_startup.set()
            if cancel:
                assert cancel.result(timeout=5).status_code == 200

        assert job.finished.wait(2)
        assert len(spawned) == len(killed) == 1
        assert spawned[0] is killed[0]
        assert spawned[0].poll() == -9
        assert job.status == "error" and job.error == "canceled"
        assert not job.work_dir.exists()
        assert not list(config.root.iterdir())
        assert client.get(f"/v1/ocr/jobs/{first}", headers=AUTH).status_code == 404
        assert [r.getMessage() for r in caplog.records if r.name == "ocr-service"] == [
            "OCR job failed: code=canceled"]


def wait_upstream_calls(upstream, count):
    for _ in range(100):
        if len(upstream["calls"]) >= count:
            return
        time.sleep(0.05)
    pytest.fail("expected inference request did not start")


def test_two_jobs_serialize_and_keep_their_own_results(config, upstream, scan_factory):
    first_source, blocks = scan_factory(name="first")
    second_source, _ = scan_factory(name="second", native=True)
    upstream.update(raw=grounding(blocks), delay=True)
    with TestClient(create_app(config)) as client:
        first = client.post("/v1/ocr", content=first_source.read_bytes(), headers=PDF_AUTH).json()["job_id"]
        wait_upstream_calls(upstream, 1)
        second = client.post("/v1/ocr", content=second_source.read_bytes(), headers=PDF_AUTH).json()["job_id"]
        assert first != second
        assert client.get(f"/v1/ocr/jobs/{second}", headers=AUTH).json()["status"] == "queued"
        assert len(upstream["calls"]) == 1
        upstream["wait"].set()
        for job_id, source, expected_pages in [(first, first_source, 1), (second, second_source, 2)]:
            status = wait_status(client, job_id)
            assert status["status"] == "done", status
            assert status["pages_total"] == status["pages_with_text"] == expected_pages
            response = client.get(f"/v1/ocr/jobs/{job_id}/result", headers=AUTH)
            assert response.status_code == 200
            assert response.content.startswith(source.read_bytes())
            with fitz.open(stream=response.content, filetype="pdf") as doc:
                assert len(doc) == expected_pages
                assert "繁體中文文件測試" in doc[0].get_text()
            assert client.delete(f"/v1/ocr/jobs/{job_id}", headers=AUTH).status_code == 200
        assert len(upstream["calls"]) == 2
        assert not list(config.root.iterdir())


def test_cancel_running_job_does_not_cancel_queued_document(config, upstream, scan_factory):
    first_source, blocks = scan_factory(name="cancel-first")
    second_source, _ = scan_factory(name="keep-second", native=True)
    upstream.update(raw=grounding(blocks), delay=True)
    with TestClient(create_app(config)) as client:
        first = client.post("/v1/ocr", content=first_source.read_bytes(), headers=PDF_AUTH).json()["job_id"]
        wait_upstream_calls(upstream, 1)
        second = client.post("/v1/ocr", content=second_source.read_bytes(), headers=PDF_AUTH).json()["job_id"]
        assert client.get(f"/v1/ocr/jobs/{second}", headers=AUTH).json()["status"] == "queued"
        # Keep A's mock response blocked while B can answer, so B's completion
        # proves canceling A releases its worker without killing the next job.
        upstream["delay"] = False
        assert client.delete(f"/v1/ocr/jobs/{first}", headers=AUTH).status_code == 200
        assert client.get(f"/v1/ocr/jobs/{first}", headers=AUTH).status_code == 404
        status = wait_status(client, second)
        assert status["status"] == "done", status
        assert status["pages_total"] == status["pages_with_text"] == 2
        result = client.get(f"/v1/ocr/jobs/{second}/result", headers=AUTH)
        assert result.status_code == 200
        assert result.content.startswith(second_source.read_bytes())
        assert len(upstream["calls"]) == 2
        assert client.delete(f"/v1/ocr/jobs/{second}", headers=AUTH).status_code == 200
        assert not list(config.root.iterdir())


def test_canceling_queued_jobs_cannot_grow_executor_queue(config, upstream, scan_factory):
    source, blocks = scan_factory()
    upstream.update(raw=grounding(blocks), delay=True)
    with TestClient(create_app(config)) as client:
        running = client.post("/v1/ocr", content=source.read_bytes(), headers=PDF_AUTH).json()["job_id"]
        wait_status(client, running, {"running"})
        queued = client.post("/v1/ocr", content=source.read_bytes(), headers=PDF_AUTH).json()["job_id"]
        assert client.delete(f"/v1/ocr/jobs/{queued}", headers=AUTH).status_code == 200
        assert client.post("/v1/ocr", content=source.read_bytes(), headers=PDF_AUTH).status_code == 429
        assert client.delete(f"/v1/ocr/jobs/{running}", headers=AUTH).status_code == 200


def test_managed_owner_boundaries_and_standalone_compatibility(config, monkeypatch):
    monkeypatch.setenv("OCR_REQUIRE_JOB_OWNER", "1")
    monkeypatch.setattr(UnlimitedOCR, "ready", lambda self: True)
    monkeypatch.setattr(Manager, "submit", lambda self, job: None)
    owner = "a" * 43
    owned = {**AUTH, "X-OCR-Job-Owner": owner}
    other = {**AUTH, "X-OCR-Job-Owner": "b" * 43}
    app = create_app(config)
    with TestClient(app) as client:
        assert client.post("/v1/ocr", headers=PDF_AUTH, content=b"%PDF-1.7").status_code == 403
        created = client.post("/v1/ocr", headers={**owned, "Content-Type": "application/pdf"}, content=b"%PDF-1.7")
        assert created.status_code == 200
        path = "/v1/ocr/jobs/" + created.json()["job_id"]
        job = app.state.manager.jobs[created.json()["job_id"]]
        job.status = "done"
        (job.work_dir / "result.pdf").write_bytes(b"%PDF-1.7\n%%EOF\n")
        for method, url in [("GET", path), ("GET", path + "/result"), ("DELETE", path)]:
            assert client.request(method, url, headers=other).status_code == 404
            assert client.request(method, url, headers=AUTH).status_code == 403
        assert client.get(path, headers=owned).status_code == 200
        assert owner not in client.get(path, headers=owned).text
        assert client.get(path + "/result", headers=owned).status_code == 200
        assert client.delete(path, headers=owned).status_code == 200
        assert not job.work_dir.exists()
        monkeypatch.delenv("OCR_API_KEY")
        monkeypatch.setenv("OCR_ALLOW_UNAUTHENTICATED", "1")
        assert client.post("/v1/ocr", headers={**owned, "Content-Type": "application/pdf"}, content=b"%PDF-1.7").status_code == 503
    # Standalone deployment ignores untrusted owner headers and retains its
    # original shared-key semantics; it is never an alternate managed route.
    monkeypatch.setenv("OCR_API_KEY", "api-test-key")
    monkeypatch.setenv("OCR_REQUIRE_JOB_OWNER", "0")
    with TestClient(create_app(config)) as client:
        created = client.post("/v1/ocr", headers=PDF_AUTH, content=b"%PDF-1.7")
        path = "/v1/ocr/jobs/" + created.json()["job_id"]
        assert client.get(path, headers=other).status_code == 200
        assert client.delete(path, headers=AUTH).status_code == 200


def test_unknown_engine_is_not_ready_and_rejects_jobs(config, monkeypatch):
    monkeypatch.setenv("OCR_ENGINE", "not-an-engine")
    with TestClient(create_app(config)) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json()["engine"] == "unsupported"
        assert client.post("/v1/ocr", content=b"%PDF-1.7", headers=PDF_AUTH).status_code == 503
        assert not list(config.root.iterdir())


def test_local_rapidocr_engine_runs_same_job_protocol(config, monkeypatch, scan_factory):
    pytest.importorskip("rapidocr")
    pytest.importorskip("onnxruntime")
    monkeypatch.setenv("OCR_ENGINE", "rapidocr")
    monkeypatch.setenv("OCR_RAPIDOCR_THREADS", "2")
    # No upstream model endpoint is configured or contacted.
    monkeypatch.delenv("OCR_MODEL_BASE_URL", raising=False)
    source, _ = scan_factory(native=True)
    with TestClient(create_app(config)) as client:
        health = client.get("/health").json()
        assert health["ready"] is True and health["engine"] == "RapidOCR PP-OCRv6"
        job_id = client.post("/v1/ocr", content=source.read_bytes(), headers=PDF_AUTH).json()["job_id"]
        result = wait_status(client, job_id)
        assert result["status"] == "done", result
        assert result["pages_total"] == result["pages_with_text"] == 2
        pdf = client.get(f"/v1/ocr/jobs/{job_id}/result", headers=AUTH)
        assert pdf.content.startswith(source.read_bytes())
        with fitz.open(stream=pdf.content, filetype="pdf") as doc:
            text = doc[0].get_text()
            assert "Searchable document 12345" in text and "简体中文文档测试" in text
            assert doc[0].search_for("繁體中文")
        assert client.delete(f"/v1/ocr/jobs/{job_id}", headers=AUTH).status_code == 200
        assert not list(config.root.iterdir())


def test_hybrid_engine_runs_same_job_protocol(config, upstream, monkeypatch, scan_factory):
    pytest.importorskip("rapidocr")
    pytest.importorskip("onnxruntime")
    monkeypatch.setenv("OCR_ENGINE", "hybrid")
    monkeypatch.setenv("OCR_RAPIDOCR_THREADS", "2")
    source, blocks = scan_factory(native=True)
    # One coarse Unlimited block for all three lines; RapidOCR places them.
    box = (min(b.box[0] for b in blocks) - 0.02, min(b.box[1] for b in blocks) - 0.02,
           max(b.box[2] for b in blocks) + 0.02, max(b.box[3] for b in blocks) + 0.02)
    upstream["raw"] = '<|det|>text ' + json.dumps([round(x * 999, 5) for x in box]) + '<|/det|>' + \
        "<br>".join(b.text for b in blocks)
    with TestClient(create_app(config)) as client:
        health = client.get("/health").json()
        assert health["ready"] is True and health["engine"] == "Unlimited-OCR + RapidOCR PP-OCRv6"
        job_id = client.post("/v1/ocr", content=source.read_bytes(), headers=PDF_AUTH).json()["job_id"]
        result = wait_status(client, job_id)
        assert result["status"] == "done", result
        assert len(upstream["calls"]) == 1
        pdf = client.get(f"/v1/ocr/jobs/{job_id}/result", headers=AUTH)
        with fitz.open(stream=pdf.content, filetype="pdf") as doc:
            page = doc[0]
            assert "".join(page.get_text().split()) == "".join("".join(b.text.split()) for b in blocks)
            for b in blocks:
                found = page.search_for(b.text)
                assert found, b.text
                expected = fitz.Rect(b.box[0] * page.rect.width, b.box[1] * page.rect.height,
                                     b.box[2] * page.rect.width, b.box[3] * page.rect.height)
                # Line placement, not an even split of the coarse block.
                assert abs(found[0].x0 - expected.x0) < 5 and abs(found[-1].x1 - expected.x1) < 5
                assert abs((found[0].y0 + found[0].y1) - (expected.y0 + expected.y1)) / 2 < 4
        assert client.delete(f"/v1/ocr/jobs/{job_id}", headers=AUTH).status_code == 200


def test_hybrid_engine_is_not_ready_without_the_model(config, monkeypatch):
    monkeypatch.setenv("OCR_ENGINE", "hybrid")
    monkeypatch.delenv("OCR_MODEL_BASE_URL", raising=False)
    with TestClient(create_app(config)) as client:
        health = client.get("/health")
        assert health.status_code == 503 and health.json()["model_available"] is False
        assert client.post("/v1/ocr", content=b"%PDF-1.7", headers=PDF_AUTH).status_code == 503
