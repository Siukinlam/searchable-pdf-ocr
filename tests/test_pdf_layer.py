import hashlib
import unicodedata

import pymupdf as fitz
import pytest

from engine import Block, OcrError
from pdf_layer import _fit_lines, process_pdf


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("crop", [False, True])
def test_unicode_search_pixels_source_metadata_and_geometry(scan_factory, tmp_path, rotation, crop):
    source, blocks = scan_factory(rotation=rotation, crop=crop, native=True)
    source_bytes = source.read_bytes()
    output = tmp_path / "result.pdf"
    called = []
    def recognize(image):
        assert image.startswith(b"\x89PNG\r\n\x1a\n")
        called.append(True)
        return blocks
    report = process_pdf(source, output, recognize)
    assert len(called) == 1  # Native second page was not submitted to the model.
    assert report.pages_total == report.pages_with_text == 2
    assert report.pages_without_text == 0
    assert report.pages_ocr_added == report.pages_native_text == 1
    assert source.read_bytes() == source_bytes
    assert output.read_bytes().startswith(source_bytes)
    with fitz.open(source) as before, fitz.open(output) as after:
        assert not before[0].get_text().strip()
        assert after.metadata == before.metadata
        assert after.get_toc() == before.get_toc()
        assert after[1].get_text() == before[1].get_text()
        for i in range(2):
            assert after[i].rect == before[i].rect
            assert after[i].cropbox == before[i].cropbox
            assert after[i].rotation == before[i].rotation
            assert after[i].get_pixmap().samples == before[i].get_pixmap().samples
        page = after[0]
        for block in blocks:
            assert block.text in page.get_text()
            matches = page.search_for(block.text)
            assert matches
            actual = matches[0] * page.rotation_matrix
            expected = fitz.Rect(block.box[0] * page.rect.width, block.box[1] * page.rect.height,
                                 block.box[2] * page.rect.width, block.box[3] * page.rect.height)
            assert all(abs(a - b) < 1 for a, b in zip(actual, expected))


def test_partial_pages_reported_and_empty_result_rejected(scan_factory, tmp_path):
    source, blocks = scan_factory()
    with fitz.open(source) as doc:
        doc.new_page()
        doc.saveIncr()
    outputs = iter([blocks, []])
    report = process_pdf(source, tmp_path / "partial.pdf", lambda _: next(outputs))
    assert (report.pages_total, report.pages_with_text, report.pages_without_text) == (2, 1, 1)
    empty, _ = scan_factory("empty", blank=True)
    target = tmp_path / "rejected.pdf"
    with pytest.raises(OcrError, match="no_text_recognized"):
        process_pdf(empty, target, lambda _: [])
    assert not target.exists()


def test_encrypted_and_signed_sources_rejected_before_inference(scan_factory, tmp_path):
    source, _ = scan_factory()
    protected = tmp_path / "encrypted.pdf"
    with fitz.open(source) as doc:
        doc.save(protected, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="reader")
    signed = tmp_path / "signed.pdf"
    with fitz.open(source) as doc:
        sig = doc.get_new_xref()
        doc.update_object(sig, "<< /Type /Sig /ByteRange [0 1 2 3] /Contents <00> >>")
        doc.xref_set_key(doc.pdf_catalog(), "Perms", "<< /DocMDP %d 0 R >>" % sig)
        doc.save(signed)
    def forbidden(_):
        pytest.fail("protected document must not reach model")
    for path, error in [(protected, "encrypted"), (signed, "signed_pdf")]:
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        with pytest.raises(OcrError, match=error):
            process_pdf(path, tmp_path / "forbidden.pdf", forbidden)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_page_limit_prevents_inference(scan_factory, tmp_path):
    source, _ = scan_factory(native=True)
    with pytest.raises(OcrError, match="page_limit"):
        process_pdf(source, tmp_path / "limited.pdf", lambda _: pytest.fail("not called"), max_pages=1)


def test_unsupported_glyph_is_not_silently_replaced(scan_factory, tmp_path):
    source, _ = scan_factory()
    with pytest.raises(OcrError, match="unsupported_ocr_glyph"):
        process_pdf(source, tmp_path / "unsupported.pdf",
                    lambda _: [Block("\U0001f9fe", (0.1, 0.1, 0.9, 0.2))])


def test_existing_font_resource_does_not_replace_ocr_unicode(scan_factory, tmp_path):
    source, blocks = scan_factory()
    with fitz.open(source) as doc:
        doc[0].insert_font(fontname="OCRUnicode", fontbuffer=fitz.Font("helv").buffer)
        doc[0].insert_font(fontname="OCRUnicode1", fontbuffer=fitz.Font("cour").buffer)
        doc.saveIncr()
    output = tmp_path / "font-collision.pdf"
    process_pdf(source, output, lambda _: blocks)
    with fitz.open(output) as doc:
        assert all(block.text in doc[0].get_text() for block in blocks)
        assert "OCRUnicode2" in {f[4] for f in doc[0].get_fonts(full=True)}


def test_opaque_annotation_cover_is_respected_in_model_image(scan_factory, tmp_path):
    source, _ = scan_factory()
    with fitz.open(source) as doc:
        cover = doc[0].add_rect_annot(fitz.Rect(40, 60, 560, 270))
        cover.set_colors(stroke=(0, 0, 0), fill=(0, 0, 0))
        cover.set_opacity(1)
        cover.update()
        doc.saveIncr()
    with fitz.open(source) as doc:
        expected = doc[0].get_pixmap(matrix=fitz.Matrix(200 / 72, 200 / 72),
                                      alpha=False, colorspace=fitz.csRGB, annots=True).samples
    called = []
    def recognize(png):
        assert fitz.Pixmap(png).samples == expected
        called.append(True)
        return []
    with pytest.raises(OcrError, match="no_text_recognized"):
        process_pdf(source, tmp_path / "covered.pdf", recognize)
    assert called


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("crop", [False, True])
def test_dense_paragraph_preserves_repeated_characters_and_search(scan_factory, tmp_path,
                                                               rotation, crop):
    source, _ = scan_factory(rotation=rotation, crop=crop)
    original = source.read_bytes()
    # A model paragraph has one box and no line breaks. The previous extreme
    # horizontal compression made extraction discard the second l/s/c/0/f.
    text = "文件 all assessment access 100 offline " * 16
    block = Block(text.strip(), (0.1, 0.1, 0.9, 0.35))
    output = tmp_path / "dense.pdf"
    report = process_pdf(source, output, lambda _: [block])
    assert report.pages_ocr_added == 1
    assert source.read_bytes() == original
    assert output.read_bytes().startswith(original)
    with fitz.open(source) as before, fitz.open(output) as after:
        page = after[0]
        assert "".join(page.get_text().split()) == "".join(text.split())
        assert page.get_pixmap().samples == before[0].get_pixmap().samples
        for word in ("all", "assessment", "access", "100", "offline"):
            assert len(page.search_for(word)) == 16


def test_long_unbroken_cjk_keeps_combining_marks_with_their_base():
    text = "か\u3099文件測試" * 30
    lines, line_height = _fit_lines([text], 30, 100, fitz.Font("cjk"))
    assert len(lines) > 1 and len(lines) * line_height <= 100
    assert "".join(lines) == text
    assert all(not unicodedata.category(line[0]).startswith("M") for line in lines)


def line_block(text, rect, page_rect, chars=(), quad=None):
    """A line-grounded block in normalized displayed-page coordinates."""
    w, h = page_rect.width, page_rect.height
    if quad is None:
        quad = ((rect.x0, rect.y0), (rect.x1, rect.y0), (rect.x1, rect.y1), (rect.x0, rect.y1))
    return Block(text, (rect.x0 / w, rect.y0 / h, rect.x1 / w, rect.y1 / h),
                 tuple((x / w, y / h) for x, y in quad),
                 tuple((c, (r.x0 / w, r.y0 / h, r.x1 / w, r.y1 / h)) for c, r in chars))


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_character_boxes_place_each_word_where_it_was_scanned(scan_factory, tmp_path, rotation):
    source, _ = scan_factory(rotation=rotation, blank=True)
    with fitz.open(source) as doc:
        page_rect = doc[0].rect
    # Wide, uneven gaps that the embedded font's proportional advances cannot
    # reproduce from a single line box.
    words = [("Total", 60, 110), ("amount", 200, 270), ("HK$", 300, 330), ("1,280", 440, 490)]
    chars = []
    for word, x0, x1 in words:
        step = (x1 - x0) / len(word)
        # Recognizer boxes leave small gaps between characters of a word.
        chars += [(c, fitz.Rect(x0 + i * step, 100, x0 + (i + 0.8) * step, 120))
                  for i, c in enumerate(word)]
    block = line_block("Total amount HK$ 1,280", fitz.Rect(60, 100, 490, 120), page_rect, chars)
    output = tmp_path / "chars.pdf"
    process_pdf(source, output, lambda _: [block])
    with fitz.open(output) as doc:
        page = doc[0]
        assert page.get_text().split() == ["Total", "amount", "HK$", "1,280"]
        for word, x0, x1 in words:
            found = page.search_for(word)
            assert len(found) == 1, word
            box = found[0] * page.rotation_matrix
            assert abs(box.x0 - x0) < 2 and abs(box.x1 - (x1 - 0.2 * (x1 - x0) / len(word))) < 2, (word, box)
            assert 98 < (box.y0 + box.y1) / 2 < 122
        assert page.search_for("amount HK$")


def test_line_box_without_characters_spans_detected_line(scan_factory, tmp_path):
    source, _ = scan_factory(blank=True)
    with fitz.open(source) as doc:
        page_rect = doc[0].rect
    block = line_block("繁體中文 Searchable", fitz.Rect(80, 300, 420, 330), page_rect)
    output = tmp_path / "line.pdf"
    process_pdf(source, output, lambda _: [block])
    with fitz.open(output) as doc:
        found = doc[0].search_for("繁體中文 Searchable")
        assert len(found) == 1
        assert all(abs(a - b) < 1.5 for a, b in zip(found[0], (80, 300, 420, 330)))


def test_vertical_and_skewed_lines_follow_their_quads(scan_factory, tmp_path):
    source, _ = scan_factory(blank=True)
    with fitz.open(source) as doc:
        page_rect = doc[0].rect
    # Top-to-bottom column: reading frame starts at the top-right corner.
    column = line_block("直排文字", fitz.Rect(500, 300, 530, 420), page_rect,
                        quad=((530, 300), (530, 420), (500, 420), (500, 300)))
    # A line scanned 3 degrees clockwise.
    skew = line_block("skewed scan line", fitz.Rect(60, 500, 360, 540), page_rect,
                      quad=((60, 500), (359.6, 515.7), (358.5, 535.7), (59, 520)))
    output = tmp_path / "geometry.pdf"
    process_pdf(source, output, lambda _: [column, skew])
    with fitz.open(output) as doc:
        page = doc[0]
        chars = [c for b in page.get_text("rawdict")["blocks"] for line in b.get("lines", [])
                 for span in line["spans"] for c in span["chars"]]
        vertical = [c for c in chars if c["c"] in "直排文字"]
        assert [c["c"] for c in vertical] == list("直排文字")
        ys = [c["origin"][1] for c in vertical]
        assert ys == sorted(ys) and ys[0] >= 299 and ys[-1] < 420
        assert all(500 <= c["origin"][0] <= 530 for c in vertical)
        found = page.search_for("skewed")
        assert found and found[0].y1 < page.search_for("line")[0].y1
