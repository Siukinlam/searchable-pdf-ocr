from types import SimpleNamespace

import httpx
import pytest

import engine
from engine import OcrError, UnlimitedOCR, parse_grounding


def test_both_official_grounding_formats_and_999_coordinates():
    old = '<|ref|>text<|/ref|><|det|>[[0, 100, 999, 200]]<|/det|>繁體中文'
    new = '<|det|>text [0, 100, 999, 200]<|/det|>繁體中文'
    assert parse_grounding(old) == parse_grounding(new)
    assert parse_grounding(old)[0].box == (0, 100 / 999, 1, 200 / 999)
    mixed = new + '\n<|det|>image [100, 200, 900, 900]<|/det|>![](unused.png)'
    assert len(parse_grounding(mixed)) == 1


@pytest.mark.parametrize("raw", [
    "plain prose without boxes", '<|det|>text [0, 1, 1000, 2]<|/det|>text',
    '<|det|>text [0, 1, 0, 2]<|/det|>text',
    '<|det|>text [0, 1, NaN, 2]<|/det|>text',
    '<|det|>text [[0, 1, 2, 3], [4, 5, 6, 7]]<|/det|>ambiguous text',
    'ungrounded prefix <|det|>text [0, 1, 2, 3]<|/det|>text',
    '<|det|>text [0, 1, 2, 3]<|/det|>text <|det|>BROKEN',
    '<|det|>text [__import__("os").system("false")]<|/det|>text',
])
def test_no_guessing_or_eval_of_model_output(raw):
    with pytest.raises(OcrError):
        parse_grounding(raw)


def test_request_recipe_and_truncation(monkeypatch):
    model = UnlimitedOCR()
    seen = []
    def request(method, suffix, payload):
        seen.append(payload)
        return {"choices": [{"finish_reason": "stop", "message": {
            "content": '<|det|>text [0, 1, 999, 100]<|/det|>Recognized text'}}]}
    monkeypatch.setattr(model, "_request", request)
    assert model.recognize(b"image")[0].text == "Recognized text"
    payload = seen[0]
    assert payload["messages"][0]["content"][0]["text"] == "<image>document parsing."
    assert payload["skip_special_tokens"] is False
    assert payload["vllm_xargs"] == {"ngram_size": 35, "window_size": 128}
    monkeypatch.setattr(model, "_request", lambda *args: {
        "choices": [{"finish_reason": "length", "message": {"content": "partial"}}]})
    with pytest.raises(OcrError, match="incomplete"):
        model.recognize(b"image")


def test_markup_is_not_copied_into_text_layer():
    raw = '<|det|>title [0, 1, 999, 100]<|/det|># **Heading**\n[Visible link](https://example.test)'
    assert parse_grounding(raw)[0].text == "Heading\nVisible link"


def test_missing_explicit_upstream_secret_file_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setenv("OCR_MODEL_BASE_URL", "https://ocr.example/v1")
    monkeypatch.setenv("OCR_MODEL_API_KEY", "must-not-fallback")
    monkeypatch.setenv("OCR_MODEL_API_KEY_FILE", str(tmp_path / "missing-secret"))
    assert UnlimitedOCR().configured() is False


def test_special_tokens_and_literal_identifiers():
    raw = '<｜begin▁of▁sentence｜><|det|>text [0, 1, 999, 100]<|/det|>`__init__` a**b<｜end▁of▁sentence｜>'
    assert parse_grounding(raw)[0].text == "__init__ a**b"


@pytest.fixture
def recovery_clock(monkeypatch):
    clock = SimpleNamespace(now=0.0, run_thread=True, threads=[])

    class Event:
        def __init__(self):
            self.value = False

        def is_set(self):
            return self.value

        def set(self):
            self.value = True

        def wait(self, timeout):
            if not self.value:
                clock.now += timeout
            return self.value

    class Thread:
        def __init__(self, *, target, daemon):
            self.target = target
            assert daemon is True
            clock.threads.append(self)

        def start(self):
            if clock.run_thread:
                self.target()

    monkeypatch.setattr(engine, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(engine, "threading", SimpleNamespace(Event=Event, Thread=Thread))
    return clock


def _recognized_response(content='<|det|>text [0, 1, 999, 100]<|/det|>Synthetic text',
                         finish_reason="stop"):
    return {"choices": [{"finish_reason": finish_reason, "message": {"content": content}}]}


def test_unavailable_recovers_same_model_then_retries_once(monkeypatch, recovery_clock):
    model = UnlimitedOCR()
    posts, probes = [], []

    def request(method, suffix, payload=None, timeout=None):
        if method == "POST":
            assert suffix == "/chat/completions"
            posts.append(payload)
            if len(posts) == 1:
                raise OcrError("model_unavailable")
            return _recognized_response()
        assert suffix == "/models"
        assert payload is None
        probes.append((recovery_clock.now, timeout))
        if len(probes) == 1:
            return {"data": [{"id": "another-model"}]}
        if len(probes) == 2:
            raise OcrError("model_unavailable")
        return {"data": [{"id": model.model}]}

    monkeypatch.setattr(model, "_request", request)
    assert model.recognize(b"synthetic image")[0].text == "Synthetic text"
    assert len(posts) == 2 and posts[0] is posts[1]
    assert probes == [(0, 3), (2, 3), (4, 3)]
    assert len(recovery_clock.threads) == 1


def test_never_ready_stops_at_60_seconds_without_retry(monkeypatch, recovery_clock):
    model = UnlimitedOCR()
    posts, probes = [], []

    def request(method, suffix, payload=None, timeout=None):
        if method == "POST":
            posts.append(payload)
            raise OcrError("model_unavailable")
        probes.append((recovery_clock.now, timeout))
        return {"data": []}

    monkeypatch.setattr(model, "_request", request)
    with pytest.raises(OcrError, match="^model_unavailable$"):
        model.recognize(b"synthetic image")
    assert recovery_clock.now == 60
    assert len(posts) == 1
    assert len(probes) == 30 and probes[-1] == (58, 2)


def test_stalled_readiness_thread_cannot_extend_caller_deadline(monkeypatch, recovery_clock):
    model = UnlimitedOCR()
    recovery_clock.run_thread = False
    calls = []

    def request(method, suffix, payload=None, timeout=None):
        calls.append(method)
        raise OcrError("model_unavailable")

    monkeypatch.setattr(model, "_request", request)
    with pytest.raises(OcrError, match="^model_unavailable$"):
        model.recognize(b"synthetic image")
    assert recovery_clock.now == 60
    assert calls == ["POST"]
    # A delayed metadata thread sees the stop signal and does no further I/O.
    recovery_clock.threads[0].target()
    assert calls == ["POST"]


def test_second_inference_failure_is_terminal(monkeypatch, recovery_clock):
    model = UnlimitedOCR()
    calls = []

    def request(method, suffix, payload=None, timeout=None):
        calls.append(method)
        if method == "GET":
            return {"data": [{"id": model.model}]}
        raise OcrError("model_unavailable")

    monkeypatch.setattr(model, "_request", request)
    with pytest.raises(OcrError, match="^model_unavailable$"):
        model.recognize(b"synthetic image")
    assert calls == ["POST", "GET", "POST"]
    assert len(recovery_clock.threads) == 1


@pytest.mark.parametrize("response,code", [
    ({}, "invalid_model_output"),
    (_recognized_response("No grounding"), "missing_grounding"),
    (_recognized_response('<|det|>text [0, 1, 0, 2]<|/det|>Text'), "invalid_grounding"),
    (_recognized_response('Prefix <|det|>text [0, 1, 999, 100]<|/det|>Text'), "ungrounded_text"),
    (_recognized_response("Partial", "length"), "model_output_incomplete"),
])
def test_invalid_output_never_enters_recovery(monkeypatch, recovery_clock, response, code):
    model = UnlimitedOCR()
    calls = []

    def request(method, suffix, payload=None, timeout=None):
        calls.append(method)
        return response

    monkeypatch.setattr(model, "_request", request)
    with pytest.raises(OcrError, match=f"^{code}$"):
        model.recognize(b"synthetic image")
    assert calls == ["POST"] and recovery_clock.threads == []


@pytest.mark.parametrize("code", ["model_response_too_large", "model_not_configured",
                                 "unsupported_ocr_glyph", "invalid_model_output"])
def test_other_errors_never_enter_recovery(monkeypatch, recovery_clock, code):
    model = UnlimitedOCR()

    def request(*args, **kwargs):
        raise OcrError(code)

    monkeypatch.setattr(model, "_request", request)
    with pytest.raises(OcrError, match=f"^{code}$"):
        model.recognize(b"synthetic image")
    assert recovery_clock.threads == []


def test_malformed_response_json_is_not_retried(monkeypatch, recovery_clock):
    monkeypatch.setenv("OCR_MODEL_BASE_URL", "http://synthetic.invalid/v1")
    monkeypatch.delenv("OCR_MODEL_API_KEY_FILE", raising=False)
    model = UnlimitedOCR()
    calls = []

    def handle(request):
        calls.append(request.method)
        return httpx.Response(200, content=b"not json")

    client_type = httpx.Client
    monkeypatch.setattr(engine.httpx, "Client", lambda **kwargs: client_type(
        transport=httpx.MockTransport(handle), **kwargs))
    with pytest.raises(OcrError, match="^invalid_model_output$"):
        model.recognize(b"synthetic image")
    assert calls == ["POST"] and recovery_clock.threads == []


@pytest.mark.parametrize("failure", ["http", "transport"])
def test_http_and_transport_failures_recover(monkeypatch, recovery_clock, failure):
    monkeypatch.setenv("OCR_MODEL_BASE_URL", "http://synthetic.invalid/v1")
    monkeypatch.delenv("OCR_MODEL_API_KEY_FILE", raising=False)
    model = UnlimitedOCR()
    calls = []

    def handle(request):
        calls.append(request.method)
        if len(calls) == 1:
            if failure == "transport":
                raise httpx.ConnectError("Synthetic connection failure", request=request)
            return httpx.Response(503)
        if request.method == "GET":
            assert request.url.path == "/v1/models"
            assert request.extensions["timeout"]["read"] == 3
            return httpx.Response(200, json={"data": [{"id": model.model}]})
        return httpx.Response(200, json=_recognized_response())

    client_type = httpx.Client
    monkeypatch.setattr(engine.httpx, "Client", lambda **kwargs: client_type(
        transport=httpx.MockTransport(handle), **kwargs))
    assert model.recognize(b"synthetic image")[0].text == "Synthetic text"
    assert calls == ["POST", "GET", "POST"]
