import hashlib
import io
import re
import shutil
import subprocess
import unicodedata

from PIL import Image
import pytest

from engine import Block, OcrError
import pdf_layer
from pdf_layer import GlyphLessFont, PageSpace, _fit_lines, process_pdf, render_page
import pdfgen
from pdfutil import chars, display_size, find, metadata, open_pdf, outline, pixels, text


def close(a, b, tolerance):
    return all(abs(x - y) < tolerance for x, y in zip(a, b))


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
    with open_pdf(source) as before, open_pdf(output) as after:
        assert not text(before[0]).strip()
        assert metadata(after) == metadata(before) and metadata(after)["Title"] == "Synthetic OCR corpus"
        assert outline(after) == outline(before) == [(0, "Synthetic bookmark", 0)]
        assert text(after[1]) == text(before[1])
        for i in range(2):
            assert after[i].get_cropbox() == before[i].get_cropbox()
            assert after[i].get_rotation() == before[i].get_rotation()
            assert pixels(after[i]) == pixels(before[i])
        page = after[0]
        width, height = display_size(page)
        for block in blocks:
            assert block.text in text(page)
            matches = find(page, block.text)
            assert len(matches) == 1
            expected = (block.box[0] * width, block.box[1] * height,
                        block.box[2] * width, block.box[3] * height)
            # PDFium derives loose character boxes from the glyph outline, about
            # 1.5% above the font ascent. Block placement on 90/270-degree pages
            # fills a tall box with one line, so allow 2% of the box extent.
            extent = max(expected[2] - expected[0], expected[3] - expected[1])
            assert close(matches[0], expected, max(1, 0.02 * extent)), (matches[0], expected)


def test_text_layer_is_extractable_by_poppler(scan_factory, tmp_path):
    if not shutil.which("pdftotext"):
        pytest.skip("poppler pdftotext is not installed")
    source, blocks = scan_factory()
    output = tmp_path / "result.pdf"
    process_pdf(source, output, lambda _: blocks)
    extracted = subprocess.run(["pdftotext", "-enc", "UTF-8", str(output), "-"],
                               capture_output=True, text=True, check=True).stdout
    for block in blocks:
        assert block.text in extracted


def test_partial_pages_reported_and_empty_result_rejected(scan_factory, tmp_path):
    source, blocks = scan_factory()
    with open_pdf(source) as doc:
        scan = Image.open(io.BytesIO(render_page(doc[0], dpi=72)))
    two_pages = tmp_path / "two.pdf"
    pdfgen.write_pdf(two_pages, [{"width": 600, "height": 800, "image": scan},
                                 {"width": 600, "height": 800}])
    outputs = iter([blocks, []])
    report = process_pdf(two_pages, tmp_path / "partial.pdf", lambda _: next(outputs))
    assert (report.pages_total, report.pages_with_text, report.pages_without_text) == (2, 1, 1)
    empty, _ = scan_factory("empty", blank=True)
    target = tmp_path / "rejected.pdf"
    with pytest.raises(OcrError, match="no_text_recognized"):
        process_pdf(empty, target, lambda _: [])
    assert not target.exists()


@pytest.mark.parametrize("kind, error", [("encrypted", "encrypted"), ("field", "signed_pdf"),
                                         ("certified", "signed_pdf")])
def test_encrypted_and_signed_sources_rejected_before_inference(tmp_path, kind, error):
    path = tmp_path / f"{kind}.pdf"
    pdfgen.write_pdf(path, [{"width": 600, "height": 800, "image": Image.new("RGB", (60, 80), "white")}],
                     signed=None if kind == "encrypted" else kind, encrypted=kind == "encrypted")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(OcrError, match=error):
        process_pdf(path, tmp_path / "forbidden.pdf",
                    lambda _: pytest.fail("protected document must not reach model"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert not (tmp_path / "forbidden.pdf").exists()


def test_signature_inside_compressed_object_stream_is_found(tmp_path):
    import zlib
    body = b"<< /Type /Sig /ByteRange [0 1 2 3] >>"
    data = (b"%PDF-1.7\n1 0 obj\n<< /Type /ObjStm /N 1 /First 4 /Filter /FlateDecode >>\nstream\n" +
            zlib.compress(b"2 0 " + body) + b"\nendstream\nendobj\n")
    assert pdf_layer._has_signature_marker(data)
    assert not pdf_layer._has_signature_marker(data.replace(b"/ObjStm", b"/XObject"))


def test_broken_cross_reference_requires_repair(scan_factory, tmp_path):
    source, blocks = scan_factory()
    broken = tmp_path / "broken.pdf"
    data = source.read_bytes()
    tail = data.rindex(b"startxref")
    broken.write_bytes(data[:tail] + b"startxref\n12\n%%EOF\n")
    with pytest.raises(OcrError, match="pdf_requires_repair"):
        process_pdf(broken, tmp_path / "out.pdf", lambda _: pytest.fail("not called"))


def test_page_limit_prevents_inference(scan_factory, tmp_path):
    source, _ = scan_factory(native=True)
    with pytest.raises(OcrError, match="page_limit"):
        process_pdf(source, tmp_path / "limited.pdf", lambda _: pytest.fail("not called"), max_pages=1)


@pytest.mark.parametrize("bad", ["\U0001f9fe", "bell\x07"])
def test_unsupported_glyph_is_not_silently_replaced(scan_factory, tmp_path, bad):
    source, _ = scan_factory()
    with pytest.raises(OcrError, match="unsupported_ocr_glyph"):
        process_pdf(source, tmp_path / "unsupported.pdf",
                    lambda _: [Block(bad, (0.1, 0.1, 0.9, 0.2))])


def test_combining_marks_round_trip(scan_factory, tmp_path):
    source, _ = scan_factory()
    output = tmp_path / "marks.pdf"
    process_pdf(source, output, lambda _: [Block("café résumé", (0.1, 0.1, 0.9, 0.15))])
    with open_pdf(output) as doc:
        assert unicodedata.normalize("NFC", text(doc[0])).split() == ["café", "résumé"]


def test_ocr_font_never_reuses_an_existing_resource(tmp_path):
    # PDFium names its fonts /FXF<n>. A source already defining /FXF1 for a
    # different font must not have the OCR text drawn in that font.
    path = tmp_path / "fonts.pdf"
    pdfgen.write_pdf(path, [{"width": 600, "height": 800, "image": Image.new("RGB", (60, 80), "white")}])
    data = path.read_bytes().replace(b"/XObject << /Im0", b"/Font << /FXF1 3 0 R >> /XObject << /Im0")
    path.write_bytes(_fix_xref(data))
    output = tmp_path / "out.pdf"
    process_pdf(path, output, lambda _: [Block("繁體中文 Searchable", (0.1, 0.1, 0.9, 0.15))])
    with open_pdf(output) as doc:
        assert text(doc[0]).split() == ["繁體中文", "Searchable"]
    tail = output.read_bytes()[len(path.read_bytes()):]
    fonts = re.findall(rb"/Font\s*<<([^>]*)>>", tail)
    assert fonts and all(b" 3 0 R" not in entry for entry in fonts)


def test_resources_used_by_original_content_are_kept(tmp_path):
    # PDFium rebuilds /Resources of a modified page; optional content,
    # shadings and graphics states the original content uses must survive.
    w = pdfgen.Writer()
    catalog, pages = w.reserve(), w.reserve()
    group = w.add("<< /Type /OCG /Name (Layer) >>")
    shading = w.add("<< /ShadingType 2 /ColorSpace /DeviceRGB /Coords [0 0 600 0] /Function "
                    "<< /FunctionType 2 /Domain [0 1] /C0 [1 0 0] /C1 [0 0 1] /N 1 >> >>")
    state = w.add("<< /Type /ExtGState /ca 0.5 >>")
    content = w.stream(b"/OC /oc1 BDC q /GS1 gs 0 0 600 300 re W n /Sh1 sh Q EMC")
    page = w.add(f"<< /Type /Page /Parent {pages} 0 R /MediaBox [0 0 600 800] /Contents {content} 0 R "
                 f"/Resources << /Properties << /oc1 {group} 0 R >> /Shading << /Sh1 {shading} 0 R >> "
                 f"/ExtGState << /GS1 {state} 0 R >> >> >>")
    w.set(pages, f"<< /Type /Pages /Kids [{page} 0 R] /Count 1 >>")
    w.set(catalog, f"<< /Type /Catalog /Pages {pages} 0 R /OCProperties << /OCGs [{group} 0 R] "
                   f"/D << /ON [{group} 0 R] >> >> >>")
    source, output = tmp_path / "layers.pdf", tmp_path / "out.pdf"
    w.save(source, catalog)
    process_pdf(source, output, lambda _: [Block("Layer test", (0.1, 0.6, 0.5, 0.65))])
    tail = output.read_bytes()[len(source.read_bytes()):]
    for entry in (b"/oc1 %d 0 R" % group, b"/Sh1 %d 0 R" % shading, b"/GS1 %d 0 R" % state):
        assert entry in tail
    with open_pdf(source) as before, open_pdf(output) as after:
        assert pixels(after[0]) == pixels(before[0])


def _fix_xref(data):
    """Recompute xref offsets after editing an object in place."""
    offsets = {int(m.group(1)): m.start() for m in re.finditer(rb"(?m)^(\d+) 0 obj", data)}
    head = data[:data.index(b"xref\n")]
    xref = len(head)
    rows = b"".join(b"%010d 00000 n \n" % offsets[n] for n in sorted(offsets))
    trailer = data[data.index(b"trailer"):data.index(b"startxref")]
    return (head + b"xref\n0 %d\n0000000000 65535 f \n" % (len(offsets) + 1) + rows + trailer +
            b"startxref\n%d\n%%%%EOF\n" % xref)


def test_opaque_annotation_cover_is_respected_in_model_image(tmp_path):
    scan = Image.new("RGB", (600, 800), "white")
    path = tmp_path / "covered.pdf"
    pdfgen.write_pdf(path, [{"width": 600, "height": 800, "image": scan, "cover": (40, 60, 560, 270)}])
    seen = []

    def recognize(png):
        image = Image.open(io.BytesIO(png)).convert("RGB")
        scale = image.width / 600
        seen.append((image.getpixel((int(300 * scale), int(150 * scale))),
                     image.getpixel((int(300 * scale), int(500 * scale)))))
        return []
    with pytest.raises(OcrError, match="no_text_recognized"):
        process_pdf(path, tmp_path / "out.pdf", recognize)
    assert seen == [((0, 0, 0), (255, 255, 255))]


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("crop", [False, True])
def test_dense_paragraph_preserves_repeated_characters_and_search(scan_factory, tmp_path,
                                                               rotation, crop):
    source, _ = scan_factory(rotation=rotation, crop=crop)
    original = source.read_bytes()
    # A model paragraph has one box and no line breaks. Extreme horizontal
    # compression makes extractors discard adjacent repeated glyphs.
    paragraph = "文件 all assessment access 100 offline " * 16
    block = Block(paragraph.strip(), (0.1, 0.1, 0.9, 0.35))
    output = tmp_path / "dense.pdf"
    report = process_pdf(source, output, lambda _: [block])
    assert report.pages_ocr_added == 1
    assert source.read_bytes() == original
    assert output.read_bytes().startswith(original)
    with open_pdf(source) as before, open_pdf(output) as after:
        page = after[0]
        assert sorted("".join(text(page).split())) == sorted("".join(paragraph.split()))
        assert pixels(page) == pixels(before[0])
        for word in ("all", "assessment", "access", "100", "offline"):
            assert len(find(page, word)) == 16, word


def test_long_unbroken_cjk_keeps_combining_marks_with_their_base():
    paragraph = "が文件測試" * 30
    lines, line_height = _fit_lines([paragraph], 30, 100, GlyphLessFont())
    assert len(lines) > 1 and len(lines) * line_height <= 100
    assert "".join(lines) == paragraph
    assert all(not unicodedata.category(line[0]).startswith("M") for line in lines)


def test_generated_font_is_small_and_maps_every_character():
    program = GlyphLessFont.program("繁體中文文件測試 Searchable 12345")
    assert len(program) < 4000
    from fontTools.ttLib import TTFont
    cmap = TTFont(io.BytesIO(program)).getBestCmap()
    assert {chr(code) for code in cmap} >= set("繁體中文文件測試Searchable12345 ")
    assert len(set(cmap.values())) == len(cmap)  # one glyph per character


def line_block(text_, rect, size, chars_=(), quad=None):
    """A line-grounded block in normalized displayed-page coordinates."""
    w, h = size
    x0, y0, x1, y1 = rect
    if quad is None:
        quad = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    return Block(text_, (x0 / w, y0 / h, x1 / w, y1 / h),
                 tuple((x / w, y / h) for x, y in quad),
                 tuple((c, (r[0] / w, r[1] / h, r[2] / w, r[3] / h)) for c, r in chars_))


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_character_boxes_place_each_word_where_it_was_scanned(scan_factory, tmp_path, rotation):
    source, _ = scan_factory(rotation=rotation, blank=True)
    with open_pdf(source) as doc:
        size = display_size(doc[0])
    # Wide, uneven gaps that proportional advances cannot reproduce from a
    # single line box.
    words = [("Total", 60, 110), ("amount", 200, 270), ("HK$", 300, 330), ("1,280", 440, 490)]
    boxes = []
    for word, x0, x1 in words:
        step = (x1 - x0) / len(word)
        # Recognizer boxes leave small gaps between characters of a word.
        boxes += [(c, (x0 + i * step, 100, x0 + (i + 0.8) * step, 120)) for i, c in enumerate(word)]
    block = line_block("Total amount HK$ 1,280", (60, 100, 490, 120), size, boxes)
    output = tmp_path / "chars.pdf"
    process_pdf(source, output, lambda _: [block])
    with open_pdf(output) as doc:
        page = doc[0]
        assert text(page).split() == ["Total", "amount", "HK$", "1,280"]
        for word, x0, x1 in words:
            found = find(page, word)
            assert len(found) == 1, word
            box = found[0]
            assert abs(box[0] - x0) < 2 and abs(box[2] - (x1 - 0.2 * (x1 - x0) / len(word))) < 2, (word, box)
            assert 98 < (box[1] + box[3]) / 2 < 122
        assert find(page, "amount HK$")


def test_line_box_without_characters_spans_detected_line(scan_factory, tmp_path):
    source, _ = scan_factory(blank=True)
    with open_pdf(source) as doc:
        size = display_size(doc[0])
    block = line_block("繁體中文 Searchable", (80, 300, 420, 330), size)
    output = tmp_path / "line.pdf"
    process_pdf(source, output, lambda _: [block])
    with open_pdf(output) as doc:
        found = find(doc[0], "繁體中文 Searchable")
        assert len(found) == 1
        assert close(found[0], (80, 300, 420, 330), 1.5), found


def test_vertical_and_skewed_lines_follow_their_quads(scan_factory, tmp_path):
    source, _ = scan_factory(blank=True)
    with open_pdf(source) as doc:
        size = display_size(doc[0])
    # Top-to-bottom column: reading frame starts at the top-right corner.
    column = line_block("直排文字", (500, 300, 530, 420), size,
                        quad=((530, 300), (530, 420), (500, 420), (500, 300)))
    # A line scanned 3 degrees clockwise.
    skew = line_block("skewed scan line", (60, 500, 360, 540), size,
                      quad=((60, 500), (359.6, 515.7), (358.5, 535.7), (59, 520)))
    output = tmp_path / "geometry.pdf"
    process_pdf(source, output, lambda _: [column, skew])
    with open_pdf(output) as doc:
        page = doc[0]
        vertical = [(c, box) for c, box in chars(page) if c in "直排文字"]
        assert [c for c, _ in vertical] == list("直排文字")
        centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for _, b in vertical]
        assert [y for _, y in centers] == sorted(y for _, y in centers)
        assert all(299 <= y <= 421 and 499 <= x <= 531 for x, y in centers)
        assert find(page, "skewed")[0][3] < find(page, "line")[0][3]


def test_display_space_round_trips_every_rotation(tmp_path):
    for rotation in (0, 90, 180, 270):
        path = tmp_path / f"r{rotation}.pdf"
        pdfgen.write_pdf(path, [{"width": 600, "height": 800, "rotate": rotation, "crop": (25, 20, 575, 780)}])
        with open_pdf(path) as doc:
            space = PageSpace(doc[0])
            assert (space.width, space.height) == ((760, 550) if rotation in (90, 270) else (550, 760))
            for point in [(0, 0), (12.5, 40), (space.width, space.height)]:
                assert close(space.to_display(*space.to_user(*point)), point, 1e-9)
