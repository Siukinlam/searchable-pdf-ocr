import pymupdf as fitz
import pytest

from engine import OcrError, UnlimitedOCR, create_engine
from pdf_layer import process_pdf
import rapidocr_engine
from rapidocr_engine import RapidOCREngine, to_blocks


def test_engine_selection_defaults_to_unlimited_and_rejects_unknown(monkeypatch):
    monkeypatch.delenv("OCR_ENGINE", raising=False)
    assert isinstance(create_engine(), UnlimitedOCR)
    monkeypatch.setenv("OCR_ENGINE", " RapidOCR ")
    assert isinstance(create_engine(), RapidOCREngine)
    # A typo must not silently fall back to another engine or route.
    monkeypatch.setenv("OCR_ENGINE", "tesseract")
    with pytest.raises(OcrError, match="model_not_configured"):
        create_engine()


def test_line_quads_and_character_boxes_are_normalized():
    boxes = [[[100, 50], [500, 52], [500, 90], [100, 88]], [[-5, 200], [80, 200], [80, 240], [-5, 240]]]
    words = [(("A", 0.9, [[110, 50], [130, 50], [130, 90], [110, 90]]),
              ("B", 0.9, [[140, 50], [160, 50], [160, 90], [140, 90]])),
             (("文", 0.9, [[0, 200], [40, 200], [40, 240], [0, 240]]),)]
    blocks = to_blocks(boxes, ["  A   B ", "文"], words, 1000, 500, [False, False])
    assert [b.text for b in blocks] == ["A B", "文"]
    assert blocks[0].quad == ((0.1, 0.1), (0.5, 0.104), (0.5, 0.18), (0.1, 0.176))
    assert blocks[0].box == (0.1, 0.1, 0.5, 0.18)
    assert blocks[0].chars == (("A", (0.11, 0.1, 0.13, 0.18)), ("B", (0.14, 0.1, 0.16, 0.18)))
    # Detection boxes are unclipped outward; coordinates stay on the page.
    assert blocks[1].quad[0] == (0.0, 0.4)


@pytest.mark.parametrize("words, flipped", [
    ((("A", 0.9, [[1, 1], [2, 1], [2, 2], [1, 2]]),), [False]),   # misses B
    ((("AB", 0.9, [[1, 1], [2, 1], [2, 2], [1, 2]]),), [False]),  # not per character
    ((("A", 0.9, [[1, 1], [2, 1], [2, 2], [1, 2]]),
      ("B", 0.9, [[3, 1], [4, 1], [4, 2], [3, 2]])), None),        # direction unknown
])
def test_untrusted_character_boxes_fall_back_to_line_box(words, flipped):
    blocks = to_blocks([[[0, 0], [10, 0], [10, 3], [0, 3]]], ["AB"], [words], 10, 10, flipped)
    assert blocks[0].chars == ()
    assert blocks[0].quad == ((0, 0), (1, 0), (1, 0.3), (0, 0.3))


@pytest.mark.parametrize("box", [[[0, 0], [1, 0], [1, 1]], [[0, 0], [1, "x"], [1, 1], [0, 1]],
                                 [[0, 0], [float("nan"), 0], [1, 1], [0, 1]]])
def test_malformed_engine_geometry_fails_closed(box):
    with pytest.raises(OcrError, match="invalid_model_output"):
        to_blocks([box], ["text"], None, 10, 10)


def test_reading_frame_follows_crop_rotation_and_direction_classifier():
    tall = [[50, 0], [60, 0], [60, 100], [50, 100]]
    words = [(("上", 0.9, [[50, 0], [60, 0], [60, 40], [50, 40]]),
              ("下", 0.9, [[50, 60], [60, 60], [60, 100], [50, 100]]))]
    column = to_blocks([tall], ["上下"], words, 100, 100, [False])[0]
    # Tall crops are read top-down with glyph tops toward the right.
    assert column.quad == ((0.6, 0.0), (0.6, 1.0), (0.5, 1.0), (0.5, 0.0))
    assert [c for c, _ in column.chars] == ["上", "下"]
    wide = [[0, 0], [100, 0], [100, 10], [0, 10]]
    words = [(("A", 0.9, [[0, 0], [30, 0], [30, 10], [0, 10]]),
              ("B", 0.9, [[60, 0], [100, 0], [100, 10], [60, 10]]))]
    upside_down = to_blocks([wide], ["AB"], words, 100, 100, [True])[0]
    assert upside_down.quad == ((1.0, 0.1), (0.0, 0.1), (0.0, 0.0), (1.0, 0.0))
    # Columns of a flipped crop are mirrored back about the line center.
    assert upside_down.chars == (("A", (0.7, 0.0, 1.0, 0.1)), ("B", (0.0, 0.0, 0.4, 0.1)))


def test_missing_local_model_is_not_ready_and_never_downloads(monkeypatch, tmp_path):
    monkeypatch.setenv("OCR_RAPIDOCR_DET_MODEL", str(tmp_path / "missing.onnx"))
    model = RapidOCREngine()
    assert model.ready() is False
    with pytest.raises(OcrError, match="model_not_configured"):
        model.recognize(b"\x89PNG")
    monkeypatch.setattr(rapidocr_engine, "_bundled_models", lambda: None)
    monkeypatch.delenv("OCR_RAPIDOCR_DET_MODEL")
    assert RapidOCREngine().configured() is False


def test_library_errors_do_not_escape(monkeypatch):
    class Broken:
        def load_img(self, image):
            raise RuntimeError("PRIVATE document text must not escape")
    model = RapidOCREngine()
    monkeypatch.setattr(model, "_engine", lambda: Broken())
    with pytest.raises(OcrError, match="^invalid_model_output$"):
        model.recognize(b"image")


def test_thread_setting_is_bounded(monkeypatch):
    monkeypatch.setenv("OCR_RAPIDOCR_THREADS", "999")
    assert RapidOCREngine().cpu_threads == 32
    monkeypatch.setenv("OCR_RAPIDOCR_THREADS", "many")
    assert RapidOCREngine().cpu_threads == 4


@pytest.fixture
def real_engine():
    pytest.importorskip("rapidocr")
    pytest.importorskip("onnxruntime")
    model = RapidOCREngine()
    if not model.configured():
        pytest.skip("bundled RapidOCR models are not installed")
    return model


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("crop", [False, True])
def test_real_rapidocr_line_geometry_on_synthetic_scan(scan_factory, tmp_path, real_engine,
                                                      rotation, crop):
    source, truth = scan_factory(rotation=rotation, crop=crop, native=True)
    output = tmp_path / "rapid.pdf"
    report = process_pdf(source, output, real_engine.recognize)
    assert report.pages_ocr_added == report.pages_native_text == 1
    assert output.read_bytes().startswith(source.read_bytes())
    with fitz.open(source) as before, fitz.open(output) as after:
        page = after[0]
        assert page.get_pixmap().samples == before[0].get_pixmap().samples
        for block in truth:
            assert block.text in page.get_text()
            matches = page.search_for(block.text)
            assert matches
            actual = fitz.Rect()
            for rect in matches:
                actual |= rect * page.rotation_matrix
            expected = fitz.Rect(block.box[0] * page.rect.width, block.box[1] * page.rect.height,
                                 block.box[2] * page.rect.width, block.box[3] * page.rect.height)
            # Line-grounded text lands on the scanned glyphs, not just
            # somewhere inside a model block.
            assert abs(actual.x0 - expected.x0) < 5 and abs(actual.x1 - expected.x1) < 5
            assert abs((actual.y0 + actual.y1) - (expected.y0 + expected.y1)) / 2 < 4


def test_direction_classifier_needs_a_better_reading_to_turn_a_line(monkeypatch):
    pytest.importorskip("rapidocr")
    from types import SimpleNamespace

    class Fake:
        cfg = SimpleNamespace(Cls=SimpleNamespace(cls_thresh=0.9))

        def detect_and_crop(self, image, record):
            return ["upright", "misjudged", "inverted"], SimpleNamespace(boxes=[1, 2, 3])

        def cls_and_rotate(self, crops):
            # The classifier is confident about all three turns, wrongly once.
            return (["upright", "misjudged-turned", "inverted-turned"],
                    SimpleNamespace(cls_res=[("0", 1.0), ("180", 0.95), ("180", 0.97)]))

        def recognize_txt(self, images):
            score = {"upright": 0.99, "misjudged": 0.99, "misjudged-turned": 0.52,
                     "inverted": 0.31, "inverted-turned": 0.98}
            return SimpleNamespace(txts=list(images), scores=[score[i] for i in images])

    det, cls, rec, crops = RapidOCREngine()._steps(Fake(), None, {})
    assert rec.txts == ["upright", "misjudged", "inverted-turned"]
    assert [label for label, _ in cls.cls_res] == ["0", "0", "180"]
