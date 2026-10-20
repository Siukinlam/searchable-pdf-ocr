"""Unlimited-OCR's grounded output and vLLM request recipe.

No model code executes here: images are sent only to an operator-configured
endpoint. Grounding coordinates follow Baidu's 0..999 convention.
"""
from dataclasses import dataclass
import base64
from html import unescape
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from urllib.parse import urlsplit

import httpx


class OcrError(Exception):
    """Public errors must be constant codes, never upstream text or filenames."""


_MODEL_RECOVERY_SECONDS = 60
_MODEL_RECOVERY_POLL_SECONDS = 2
_MODEL_READY_TIMEOUT_SECONDS = 3


def secret(name: str) -> str:
    # An explicitly configured file wins; missing files fail closed.
    if path := os.getenv(name + "_FILE", "").strip():
        try:
            return Path(path).read_text(encoding="utf-8").strip()
        except OSError:
            return ""
    return os.getenv(name, "").strip()


@dataclass(frozen=True)
class Block:
    """Recognized text in displayed-page coordinates normalized to 0..1.

    Unlimited-OCR grounds whole blocks: only ``text`` and ``box`` are set and
    line/word placement inside the box is estimated. Line-level detectors also
    set ``quad`` (top-left, top-right, bottom-right, bottom-left in reading
    order) and, when available, ``chars``: one (character, box) pair per
    non-whitespace character of ``text``.
    """
    text: str
    box: tuple[float, float, float, float]
    quad: tuple[tuple[float, float], ...] | None = None
    chars: tuple[tuple[str, tuple[float, float, float, float]], ...] = ()


class _PlainHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"br", "p", "div", "tr", "li"}:
            self.parts.append("\n")
        elif tag in {"td", "th"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in {"p", "div", "tr", "li"}:
            self.parts.append("\n")

    def handle_data(self, data):
        self.parts.append(data)


def plain_text(text: str) -> str:
    text = re.sub(r"<\|[^<>]+\|>", "", text)
    for token in ("<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜▁pad▁｜>"):
        text = text.replace(token, "")
    if re.search(r"</?(?:table|tr|td|th|div|p|br)\b", text, re.I):
        parser = _PlainHTML()
        parser.feed(text)
        text = "".join(parser.parts)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s+", "", text)
    # Strip presentation only when unambiguous. In particular __init__ and
    # a**b are literal identifiers/operators, not expendable decoration.
    text = re.sub(r"(?<!\w)\*\*(?=\S)(.+?)(?<=\S)\*\*(?!\w)", r"\1", text)
    text = re.sub(r"(?m)^\s*```[^\n]*$", "", text)
    text = re.sub(r"`([^`\n]+)`", r"\1", text)
    text = "\n".join(" ".join(line.split()) for line in unescape(text).splitlines())
    return text.strip()


# The official implementation accepts these two generations of grounding tags:
# <|ref|>text<|/ref|><|det|>[[x0,y0,x1,y1]]<|/det|>actual text
# <|det|>text [x0,y0,x1,y1]<|/det|>actual text
_MARKER = re.compile(
    r"(?:<\|ref\|>(?P<ref>[^<>]{1,80})<\|/ref\|>\s*)?"
    r"<\|det\|>\s*(?:(?P<label>[A-Za-z_][\w-]*)\s*)?"
    r"(?P<box>\[[^<>]{1,4096}\])\s*<\|/det\|>", re.DOTALL)


def parse_grounding(raw: str) -> list[Block]:
    if not isinstance(raw, str) or len(raw) > 2_000_000:
        raise OcrError("invalid_model_output")
    matches = list(_MARKER.finditer(raw))
    if not matches:
        if plain_text(raw):
            raise OcrError("missing_grounding")
        return []
    if plain_text(raw[:matches[0].start()]):
        raise OcrError("ungrounded_text")
    blocks = []
    for index, match in enumerate(matches):
        label = (match.group("ref") or match.group("label") or "").strip().lower()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        body = raw[match.end():end]
        # Reject malformed or nested grounding rather than silently losing text.
        if "<|det|>" in body or "<|ref|>" in body:
            raise OcrError("invalid_grounding")
        if label in {"image", "figure"}:
            continue
        text = plain_text(body)
        if not text:
            continue
        try:
            coords = json.loads(match.group("box"))
            if isinstance(coords, list) and len(coords) == 1 and isinstance(coords[0], list):
                coords = coords[0]
            if not isinstance(coords, list) or len(coords) != 4:
                raise ValueError()
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                   not math.isfinite(v) or not 0 <= v <= 999 for v in coords):
                raise ValueError()
            x0, y0, x1, y1 = coords
            if x1 <= x0 or y1 <= y0:
                raise ValueError()
        except (ValueError, TypeError):
            raise OcrError("invalid_grounding") from None
        blocks.append(Block(text, tuple(float(v) / 999 for v in coords)))
    return blocks


class UnlimitedOCR:
    name = "Unlimited-OCR"
    cpu_threads = 1
    worker_address_space = 2 * 1024**3

    def __init__(self):
        self.base = os.getenv("OCR_MODEL_BASE_URL", "").strip().rstrip("/")
        self.model = os.getenv("OCR_MODEL", "baidu/Unlimited-OCR").strip()
        self.key = secret("OCR_MODEL_API_KEY")
        self.timeout = max(1, min(1200, int(os.getenv("OCR_MODEL_TIMEOUT_SECONDS", "300"))))
        self.max_tokens = max(256, min(16384, int(os.getenv("OCR_MODEL_MAX_TOKENS", "4096"))))
        self.max_response = 4 * 1024 * 1024

    def configured(self) -> bool:
        p = urlsplit(self.base)
        return bool(p.scheme in {"http", "https"} and p.hostname and not p.username
                    and not p.password and not p.query and not p.fragment and self.model
                    and (not os.getenv("OCR_MODEL_API_KEY_FILE", "").strip() or self.key))

    def _request(self, method, suffix, payload=None, timeout=None):
        if not self.configured():
            raise OcrError("model_not_configured")
        headers = {"Authorization": "Bearer " + self.key} if self.key else {}
        try:
            # Disable ambient proxies and redirects: document bytes stay on the
            # endpoint explicitly selected by the service operator.
            with httpx.Client(trust_env=False, follow_redirects=False,
                              timeout=timeout or self.timeout) as client:
                with client.stream(method, self.base + suffix, headers=headers, json=payload) as r:
                    if r.status_code != 200:
                        raise OcrError("model_unavailable")
                    raw = bytearray()
                    for chunk in r.iter_bytes():
                        raw.extend(chunk)
                        if len(raw) > self.max_response:
                            raise OcrError("model_response_too_large")
            return json.loads(raw)
        except (httpx.HTTPError, OSError):
            raise OcrError("model_unavailable") from None
        except (ValueError, TypeError):
            raise OcrError("invalid_model_output") from None

    def ready(self, timeout=_MODEL_READY_TIMEOUT_SECONDS) -> bool:
        try:
            result = self._request("GET", "/models",
                                   timeout=min(_MODEL_READY_TIMEOUT_SECONDS, timeout))
            return any(item.get("id") == self.model for item in result.get("data", []))
        except (OcrError, AttributeError, TypeError):
            return False

    def _wait_for_model_recovery(self) -> bool:
        deadline = time.monotonic() + _MODEL_RECOVERY_SECONDS
        finished = threading.Event()
        stopped = threading.Event()
        recovered = False

        def poll():
            nonlocal recovered
            try:
                while not stopped.is_set():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return
                    if self.ready(timeout=min(_MODEL_READY_TIMEOUT_SECONDS, remaining)):
                        recovered = time.monotonic() < deadline and not stopped.is_set()
                        return
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or stopped.wait(min(_MODEL_RECOVERY_POLL_SECONDS, remaining)):
                        return
            except Exception:
                # Readiness failures have one public outcome; never print an
                # upstream exception from the background thread.
                return
            finally:
                finished.set()

        # httpx timeouts apply per I/O operation, so a slow response is not a
        # total request deadline. Bound the caller's wait independently. This
        # daemon only requests model metadata, never document bytes; worker
        # termination still cancels the complete job including an in-flight GET.
        threading.Thread(target=poll, daemon=True).start()
        finished.wait(max(0, deadline - time.monotonic()))
        stopped.set()
        return recovered and time.monotonic() < deadline

    def recognize(self, png: bytes) -> list[Block]:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "<image>document parsing."},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64," + base64.b64encode(png).decode("ascii")}},
            ]}],
            "temperature": 0.0, "max_tokens": self.max_tokens, "stream": False,
            "stop": ["<｜end▁of▁sentence｜>"],
            "skip_special_tokens": False,
            "vllm_xargs": {"ngram_size": 35, "window_size": 128},
        }
        try:
            response = self._request("POST", "/chat/completions", payload)
        except OcrError as error:
            if str(error) != "model_unavailable" or not self._wait_for_model_recovery():
                raise
            # Exactly one retry of the captured request after the same model
            # returns. Validation errors below must never enter recovery.
            response = self._request("POST", "/chat/completions", payload)
        try:
            choice = response["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise OcrError("model_output_incomplete")
            return parse_grounding(choice["message"]["content"])
        except (KeyError, IndexError, TypeError):
            raise OcrError("invalid_model_output") from None


ENGINES = ("unlimited", "rapidocr", "hybrid")


def engine_name() -> str:
    name = os.getenv("OCR_ENGINE", "unlimited").strip().lower() or "unlimited"
    if name not in ENGINES:
        # Unknown names fail closed instead of silently using another engine.
        raise OcrError("model_not_configured")
    return name


def create_engine():
    """The operator-selected backend: OCR_ENGINE=unlimited (default), rapidocr,
    or hybrid (Unlimited-OCR text on RapidOCR line and character geometry)."""
    name = engine_name()
    if name == "rapidocr":
        from rapidocr_engine import RapidOCREngine
        return RapidOCREngine()
    if name == "hybrid":
        from hybrid_engine import HybridEngine
        return HybridEngine()
    return UnlimitedOCR()
