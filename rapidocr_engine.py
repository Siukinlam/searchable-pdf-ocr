"""Local RapidOCR backend: PaddleOCR PP-OCRv6 ONNX models on ONNX Runtime CPU.

Page images never leave the worker process. Model files load only from explicit
local paths (the wheel's bundled PP-OCRv6 small models by default), so the
library's download fallback is never reached. Unlike Unlimited-OCR, every
result is one detected text line with a four-point box, plus per-character
boxes when the recognizer's CTC alignment covers every character.
"""
import math
import os
from pathlib import Path

from engine import Block, OcrError


DEFAULT_MODELS = {
    "det": "PP-OCRv6_det_small.onnx",
    "rec": "PP-OCRv6_rec_small.onnx",
    "cls": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
}


def _bundled_models():
    try:
        import rapidocr
    except ImportError:
        return None
    return Path(rapidocr.__file__).with_name("models")


def _envnum(name, default, low, high, kind=int):
    try:
        value = kind(os.getenv(name, str(default)))
    except ValueError:
        return default
    return max(low, min(high, value)) if math.isfinite(value) else default


def _point(value, width, height):
    x, y = (float(v) for v in value)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError()
    # Detection boxes are unclipped outward and may extend past the image.
    return (min(max(x / width, 0.0), 1.0), min(max(y / height, 0.0), 1.0))


def _bounds(points):
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _reading_frame(quad, flipped):
    """Order quad corners as top-left, top-right, bottom-right, bottom-left
    of the text as it was read. RapidOCR crops along p0->p1, turns crops at
    least 1.5 times taller than wide by 90 degrees counterclockwise, and turns
    lines its direction classifier marks as 180 degrees upside down."""
    p0, p1, p2, p3 = quad
    width = max(math.dist(p0, p1), math.dist(p3, p2))
    height = max(math.dist(p0, p3), math.dist(p1, p2))
    frame = (p1, p2, p3, p0) if width > 0 and height / width >= 1.5 else quad
    return frame[2:] + frame[:2] if flipped else frame


def _mirror(points, frame):
    # Character columns of a flipped crop are reported as if it were not
    # flipped. Reflect them along the reading axis about the line center.
    (ax, ay), (bx, by) = frame[0], frame[1]
    length = math.hypot(bx - ax, by - ay)
    ux, uy = (bx - ax) / length, (by - ay) / length
    cx, cy = sum(p[0] for p in frame) / 4, sum(p[1] for p in frame) / 4
    mirrored = []
    for x, y in points:
        d = (x - cx) * ux + (y - cy) * uy
        mirrored.append((x - 2 * d * ux, y - 2 * d * uy))
    return mirrored


def to_blocks(boxes, txts, word_results, width, height, flipped=None) -> list[Block]:
    """Convert RapidOCR pixel output into normalized line blocks.

    flipped holds one direction-classifier result per line, or None when it
    is unknown; then character boxes are not trusted for upside-down lines.
    """
    if width <= 0 or height <= 0:
        raise OcrError("invalid_model_output")
    blocks = []
    word_results = word_results or ()
    for index, (box, raw_text) in enumerate(zip(boxes, txts)):
        text = " ".join(str(raw_text).split())
        if not text:
            continue
        try:
            if len(box) != 4:
                raise ValueError()
            quad = tuple(_point(p, width, height) for p in box)
        except (TypeError, ValueError):
            raise OcrError("invalid_model_output") from None
        x0, y0, x1, y1 = _bounds(quad)
        if x1 <= x0 or y1 <= y0:
            continue
        upside_down = bool(flipped[index]) if flipped is not None else False
        frame = _reading_frame(quad, upside_down)
        if math.dist(frame[0], frame[1]) <= 0:
            continue
        chars = ()
        words = word_results[index] if index < len(word_results) else None
        if words and flipped is not None:
            try:
                pairs = []
                for char, _score, char_box in words:
                    points = [_point(p, width, height) for p in char_box]
                    pairs.append((str(char), _bounds(_mirror(points, frame) if upside_down else points)))
            except (TypeError, ValueError):
                pairs = []
            # Character boxes are only trusted when they cover every
            # non-whitespace character in order; otherwise use the line box.
            if "".join(c for c, _ in pairs) == "".join(text.split()) and \
               all(len(c) == 1 for c, _ in pairs):
                chars = tuple(pairs)
        blocks.append(Block(text, (x0, y0, x1, y1), frame, chars))
    return blocks


class RapidOCREngine:
    name = "RapidOCR PP-OCRv6"
    # Measured on an aarch64 Linux server: 2-4 GiB RLIMIT_AS makes inference fail while
    # peak RSS stays near 0.75 GiB. Bound resident memory with the container.
    worker_address_space = 8 * 1024**3

    def __init__(self):
        self.cpu_threads = _envnum("OCR_RAPIDOCR_THREADS", 4, 1, 32)
        self.min_score = _envnum("OCR_RAPIDOCR_MIN_SCORE", 0.5, 0.0, 1.0, float)
        bundled = _bundled_models()
        self.models = {}
        for task, filename in DEFAULT_MODELS.items():
            explicit = os.getenv(f"OCR_RAPIDOCR_{task.upper()}_MODEL", "").strip()
            self.models[task] = Path(explicit) if explicit else (bundled / filename if bundled else None)
        self._ocr = None

    def configured(self) -> bool:
        return _bundled_models() is not None and all(
            path is not None and path.is_file() for path in self.models.values())

    def ready(self, timeout=None) -> bool:
        # Local files only; loading the models belongs to the job worker.
        return self.configured()

    def _engine(self):
        if self._ocr is None:
            if not self.configured():
                raise OcrError("model_not_configured")
            try:
                from rapidocr import RapidOCR
                self._ocr = RapidOCR(params={
                    "Global.log_level": "critical",
                    "Global.return_word_box": True,
                    "Global.return_single_char_box": True,
                    "Global.text_score": self.min_score,
                    "Det.model_path": str(self.models["det"]),
                    "Rec.model_path": str(self.models["rec"]),
                    "Cls.model_path": str(self.models["cls"]),
                    "EngineConfig.onnxruntime.intra_op_num_threads": self.cpu_threads,
                    "EngineConfig.onnxruntime.inter_op_num_threads": 1,
                })
            except Exception:
                raise OcrError("model_unavailable") from None
        return self._ocr

    def _steps(self, ocr, prepared, record):
        """RapidOCR.run_ocr_steps with a second opinion on upside-down lines.

        The mobile direction classifier flags some upright lines as 180
        degrees with confidence above its threshold; recognizing the turned
        crop then yields unrelated characters. A flagged line keeps whichever
        orientation the recognizer reads with the higher score.
        """
        from rapidocr.main import RapidOCRError
        try:
            crops, det = ocr.detect_and_crop(prepared, record)
        except RapidOCRError:
            return None
        images, cls = ocr.cls_and_rotate(crops)
        labels = list(cls.cls_res)
        threshold = ocr.cfg.Cls.cls_thresh
        flagged = [i for i, (label, score) in enumerate(labels) if "180" in label and score > threshold]
        if flagged:
            turned = ocr.recognize_txt([images[i] for i in flagged]).scores
            upright = ocr.recognize_txt([crops[i] for i in flagged]).scores
            for i, turned_score, upright_score in zip(flagged, turned, upright):
                if upright_score > turned_score:
                    images[i], labels[i] = crops[i], ("0", labels[i][1])
        cls.img_list, cls.cls_res = images, labels
        rec = ocr.recognize_txt(images)
        return det, cls, rec, crops

    def recognize(self, png: bytes) -> list[Block]:
        ocr = self._engine()
        try:
            # RapidOCR.__call__ split into its public steps so each line's
            # direction result survives into the line geometry.
            image = ocr.load_img(png)
            prepared, record = ocr.preprocess_img(image)
            steps = self._steps(ocr, prepared, record)
            if steps is None:
                return []
            det, cls, rec, crops = steps
            texts, scores = tuple(rec.txts or ()), tuple(rec.scores or ())
            labels = list(cls.cls_res or ())
            result = ocr.build_final_output(image, det, cls, rec, crops, record)
        except Exception:
            # Never surface library messages; they may echo image content.
            raise OcrError("invalid_model_output") from None
        if getattr(result, "boxes", None) is None or getattr(result, "txts", None) is None:
            return []
        # Reproduce build_final_output's empty-text and score filters to align
        # direction results with the surviving lines; drop them on mismatch.
        kept = [i for i, text in enumerate(texts)
                if text.strip() and i < len(scores) and scores[i] >= ocr.text_score]
        flipped = None
        if len(labels) == len(texts) and [texts[i] for i in kept] == list(result.txts):
            threshold = ocr.cfg.Cls.cls_thresh
            flipped = [("180" in labels[i][0] and labels[i][1] > threshold) for i in kept]
        height, width = image.shape[:2]
        return to_blocks(result.boxes, result.txts, result.word_results, width, height, flipped)
