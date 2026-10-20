"""Preserve the PDF and append an invisible Unicode text layer (render mode 3).

Pages are rendered and the text layer is written with PDFium (pypdfium2). The
text uses a font generated per document: one empty glyph per character that
occurs, with advance widths and a Unicode cmap, so extraction, search and
selection work while nothing is drawn and no font program is embedded beyond a
few kilobytes. The update is saved incrementally after the original bytes.
"""
from collections import Counter
import ctypes
from dataclasses import dataclass, asdict
import hashlib
import io
import math
from pathlib import Path
import re
import shutil
import unicodedata
import zlib

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c

from engine import Block, OcrError


@dataclass
class Report:
    pages_total: int = 0
    pages_with_text: int = 0
    pages_without_text: int = 0
    pages_ocr_added: int = 0
    pages_native_text: int = 0

    def public(self):
        return asdict(self)


class GlyphLessFont:
    """Advance widths and font program for invisible OCR text.

    Every character gets its own glyph so PDFium's generated ToUnicode map
    (CID = glyph ID) maps it back to exactly that character. Each glyph is a
    rectangle filling its cell: render mode 3 never paints it, and PDFium
    skips text whose glyphs have empty bounds (a one-character run would
    vanish from extraction) while it derives character boxes from them.
    Widths only approximate a proportional font; line placement stretches
    each run to the detected geometry.
    """
    units = 1000
    ascender = 0.88
    descender = -0.12

    @staticmethod
    def width(char):
        if unicodedata.east_asian_width(char) in {"W", "F"}:
            return 1000
        if unicodedata.category(char).startswith("M"):
            return 0
        if char == " ":
            return 280
        if char in "ijlI|.,:;'!`":
            return 280
        if char in "frt()[]{}-/\\\"":
            return 350
        if char in "mwMW@%":
            return 830
        if char.isupper() or char.isdigit():
            return 620
        return 520

    def text_length(self, text, fontsize=1.0):
        return sum(self.width(char) for char in text) * fontsize / self.units

    @classmethod
    def program(cls, chars) -> bytes:
        codes = sorted({ord(c) for c in chars} | {0x20})
        names = {code: f"u{code:04X}" for code in codes}
        builder = FontBuilder(cls.units, isTTF=True)
        builder.setupGlyphOrder([".notdef"] + [names[c] for c in codes])
        builder.setupCharacterMap(names)
        ascent, descent = round(cls.ascender * cls.units), round(cls.descender * cls.units)

        def cell(advance):
            pen = TTGlyphPen(None)
            right = max(advance, 1)
            pen.moveTo((0, descent))
            pen.lineTo((0, ascent))
            pen.lineTo((right, ascent))
            pen.lineTo((right, descent))
            pen.closePath()
            return pen.glyph()
        cells = {}
        glyphs = {".notdef": TTGlyphPen(None).glyph()}
        for code in codes:
            advance = cls.width(chr(code))
            if advance not in cells:
                cells[advance] = cell(advance)
            glyphs[names[code]] = cells[advance]
        builder.setupGlyf(glyphs)
        builder.setupHorizontalMetrics({".notdef": (500, 0),
                                        **{names[c]: (cls.width(chr(c)), 0) for c in codes}})
        builder.setupHorizontalHeader(ascent=ascent, descent=descent)
        builder.setupNameTable({"familyName": "GlyphLessOCR", "styleName": "Regular"})
        builder.setupOS2(sTypoAscender=ascent, sTypoDescender=descent,
                         usWinAscent=ascent, usWinDescent=-descent)
        builder.setupPost()
        buffer = io.BytesIO()
        builder.save(buffer)
        return buffer.getvalue()


def _supported(char):
    # Control characters, surrogates and unassigned code points cannot
    # round-trip through a PDF text string and ToUnicode map, and PDFium drops
    # characters outside the Basic Multilingual Plane from text objects.
    category = unicodedata.category(char)
    return ord(char) <= 0xFFFF and not (category in {"Cc", "Cs", "Cn"} and not char.isspace())


def _characters(text):
    # PDFium may extract a base letter and combining mark as one composed
    # character; compare canonical compositions.
    return Counter(unicodedata.normalize("NFC", "".join(text.split())))


def page_text(page) -> str:
    textpage = page.get_textpage()
    try:
        return textpage.get_text_range()
    finally:
        textpage.close()


def has_text(page):
    return bool(page_text(page).strip())


class PageSpace:
    """Displayed page coordinates (points, origin top-left, y down, after
    /Rotate and the crop box) to and from unrotated PDF user space."""

    def __init__(self, page):
        self.left, self.bottom, self.right, self.top = page.get_cropbox()
        self.rotation = page.get_rotation() % 360
        w, h = self.right - self.left, self.top - self.bottom
        self.width, self.height = (h, w) if self.rotation in {90, 270} else (w, h)

    def to_user(self, x, y):
        r = self.rotation
        if r == 90:
            return self.left + y, self.bottom + x
        if r == 180:
            return self.right - x, self.bottom + y
        if r == 270:
            return self.right - y, self.top - x
        return self.left + x, self.top - y

    def vector(self, dx, dy):
        x0, y0 = self.to_user(0, 0)
        x1, y1 = self.to_user(dx, dy)
        return x1 - x0, y1 - y0

    def to_display(self, x, y):
        r = self.rotation
        if r == 90:
            return y - self.bottom, x - self.left
        if r == 180:
            return self.right - x, y - self.bottom
        if r == 270:
            return self.top - y, self.right - x
        return x - self.left, self.top - y

    def rect_to_display(self, left, bottom, right, top):
        xs, ys = zip(*(self.to_display(x, y) for x in (left, right) for y in (bottom, top)))
        return min(xs), min(ys), max(xs), max(ys)


_SIGNATURE_MARKERS = (b"/ByteRange", b"/DocMDP", b"/UR3", b"/Type/Sig", b"/Type /Sig",
                      b"/FT/Sig", b"/FT /Sig")


def _has_signature_marker(data: bytes) -> bool:
    # PDFium only counts signature fields. Certification (/Perms /DocMDP) and
    # orphaned signature dictionaries can sit in plain objects or in
    # compressed object streams, so scan both. A false positive only rejects.
    if any(marker in data for marker in _SIGNATURE_MARKERS):
        return True
    for match in re.finditer(rb"stream\r?\n", data):
        end = data.find(b"endstream", match.end())
        if end < 0:
            break
        try:
            chunk = zlib.decompressobj().decompress(data[match.end():end], 4 * 1024 * 1024)
        except zlib.error:
            continue
        if b"/ObjStm" in data[max(0, match.start() - 512):match.start()] and \
           any(marker in chunk for marker in _SIGNATURE_MARKERS):
            return True
    return False


def _well_formed_tail(data: bytes) -> bool:
    # An incremental update appends after the last cross-reference section;
    # a file whose own startxref is wrong only opens after repair, and an
    # update chained to it would not be readable without repair either.
    match = re.search(rb"startxref\s+(\d+)\s+%%EOF\s*$", data[-2048:])
    if not match:
        return False
    offset = int(match.group(1))
    head = data[offset:offset + 64]
    return head.startswith(b"xref") or re.match(rb"\d+\s+\d+\s+obj", head) is not None


def validate(source: Path, max_pages=200):
    data = Path(source).read_bytes()
    try:
        doc = pdfium.PdfDocument(data)
    except pdfium.PdfiumError:
        raise OcrError("encrypted_or_invalid_pdf") from None
    try:
        if pdfium_c.FPDF_GetSecurityHandlerRevision(doc.raw) != -1:
            raise OcrError("encrypted_or_invalid_pdf")
        if not 0 < len(doc) <= max_pages:
            raise OcrError("page_limit_exceeded")
        # Never silently invalidate a signature or certification.
        if pdfium_c.FPDF_GetSignatureCount(doc.raw) > 0 or _has_signature_marker(data):
            raise OcrError("signed_pdf_not_supported")
        if not _well_formed_tail(data):
            raise OcrError("pdf_requires_repair")
    finally:
        doc.close()


def _fit_lines(lines, width, height, font):
    # Model grounding can describe a whole paragraph with no line breaks.
    # Crushing that paragraph into one tall, narrow line makes PDF extractors
    # mistake adjacent repeated letters for duplicate overprinted glyphs.
    # Wrap at measured word boundaries (or characters for an oversized word)
    # and fit the font before applying the existing page rotation transform.
    font_height = font.ascender - font.descender
    original_line_height = height / len(lines)
    original_size = original_line_height / font_height
    if not any(len("".join(line.split())) > 64 and
               font.text_length(line, fontsize=original_size) > width * 2
               for line in lines):
        # Preserve short grounded lines, including tall boxes on rotated
        # pages. Only severely compressed long paragraphs need reflow.
        return lines, original_line_height
    advances = {char: font.text_length(char, fontsize=1)
                for char in set("".join(lines) + " ")}
    word_lines = []
    for line in lines:
        words = []
        for word in line.split():
            units = []
            for char in word:
                if units and unicodedata.category(char).startswith("M"):
                    units[-1] += char
                else:
                    units.append(char)
            units = [(unit, sum(advances[char] for char in unit)) for unit in units]
            words.append((word, sum(advance for _, advance in units), units))
        word_lines.append(words)

    def wrap(max_width):
        result = []
        for words in word_lines:
            current, current_width = "", 0.0
            for word, word_width, units in words:
                if current and current_width + advances[" "] + word_width <= max_width:
                    current += " " + word
                    current_width += advances[" "] + word_width
                    continue
                if current:
                    result.append(current)
                    current, current_width = "", 0.0
                for unit, advance in units:
                    if current and current_width + advance > max_width:
                        result.append(current)
                        current, current_width = "", 0.0
                    current += unit
                    current_width += advance
            if current:
                result.append(current)
        return result

    widest_unit = max(advance for words in word_lines for _, _, units in words
                      for _, advance in units)
    upper = min(height / (len(lines) * font_height), width / widest_unit)
    if upper <= 0 or not math.isfinite(upper):
        raise OcrError("output_text_mismatch")
    fitted = wrap(width / upper)
    if len(fitted) * upper * font_height <= height:
        return fitted, upper * font_height
    lower = 0.0
    for _ in range(24):
        size = (lower + upper) / 2
        fitted = wrap(width / size)
        if len(fitted) * size * font_height <= height:
            lower = size
        else:
            upper = size
    return wrap(width / lower), lower * font_height


def _runs(text, starts, ends, advance, tolerance):
    """Split a line into (text, start, scale) runs wherever the font's
    proportional layout misses a detected character start, or a word end, by
    more than tolerance. Whitespace has no detected box; it stays with the
    preceding run so extraction keeps the word gap. A run ending inside a
    word stretches to the next run: recognizer character boxes leave small
    gaps inside words, and an uncovered gap makes extractors insert a space."""
    glyphs, gaps = [], []
    for char in text:
        if not char.isspace():
            glyphs.append(char)
            gaps.append("")
        elif glyphs:
            gaps[-1] += char
    widths = [advance(g) for g in glyphs]
    spaces = [advance(g) if g else 0.0 for g in gaps]

    def target(k):
        return starts[k + 1] if not gaps[k] and k + 1 < len(glyphs) else ends[k]

    def layout(first, last):
        total = sum(widths[first:last + 1]) + sum(spaces[first:last])
        span = target(last) - starts[first]
        return span / total if span > 0 and total > 0 else 0.0

    def fits(first, last, scale):
        position = starts[first]
        for k in range(first, last + 1):
            end = position + widths[k] * scale
            if abs(position - starts[k]) > tolerance or \
               ((gaps[k] or k == last) and abs(end - target(k)) > tolerance):
                return False
            position += (widths[k] + spaces[k]) * scale
        return True

    runs, first = [], 0
    while first < len(glyphs):
        last, scale = first, layout(first, first)
        for candidate in range(first + 1, len(glyphs)):
            trial = layout(first, candidate)
            if trial <= 0 or not fits(first, candidate, trial):
                break
            last, scale = candidate, trial
        # A lone zero-advance glyph keeps its natural width rather than
        # vanishing and failing the extracted-text check.
        run = "".join(g + s for g, s in zip(glyphs[first:last + 1], gaps[first:last + 1]))
        runs.append((run, starts[first], scale if scale > 0 else 1.0))
        first = last + 1
    return runs


class TextLayer:
    """Invisible text objects on one document, all in one generated font."""

    def __init__(self, doc, chars):
        self.doc = doc
        self.metrics = GlyphLessFont()
        program = GlyphLessFont.program(chars)
        self._buffer = (ctypes.c_uint8 * len(program)).from_buffer_copy(program)
        self.font = pdfium_c.FPDFText_LoadFont(doc.raw, self._buffer, len(program),
                                               pdfium_c.FPDF_FONT_TRUETYPE, True)
        if not self.font:
            raise OcrError("output_validation_failed")

    def close(self):
        pdfium_c.FPDFFont_Close(self.font)

    def place(self, page, space, text, origin, direction, scale, size):
        """One run: origin is the baseline start and direction the unit text
        direction, both in displayed points; scale stretches the advances."""
        if not text or size <= 0 or scale <= 0:
            return
        obj = pdfium_c.FPDFPageObj_CreateTextObj(self.doc.raw, self.font, ctypes.c_float(size))
        if not obj:
            raise OcrError("output_validation_failed")
        encoded = (text + "\0").encode("utf-16-le")
        buffer = ctypes.create_string_buffer(encoded, len(encoded))
        if not pdfium_c.FPDFText_SetText(obj, ctypes.cast(buffer, ctypes.POINTER(pdfium_c.FPDF_WCHAR))):
            pdfium_c.FPDFPageObj_Destroy(obj)
            raise OcrError("unsupported_ocr_glyph")
        pdfium_c.FPDFTextObj_SetTextRenderMode(obj, pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE)
        # Text space x follows the reading direction, y the glyph's up side;
        # both are mapped from displayed (y-down) into user space.
        ux, uy = space.vector(*direction)
        vx, vy = space.vector(direction[1], -direction[0])
        ox, oy = space.to_user(*origin)
        pdfium_c.FPDFPageObj_Transform(obj, ux * scale, uy * scale, vx, vy, ox, oy)
        pdfium_c.FPDFPage_InsertObject(page.raw, obj)


def _add_line(layer, page, space, text, box):
    x0, y0, x1, y1 = box
    font = layer.metrics
    size = (y1 - y0) / (font.ascender - font.descender)
    width = font.text_length(text, fontsize=size)
    if size <= 0 or width <= 0:
        return
    layer.place(page, space, text, (x0, y0 + size * font.ascender), (1.0, 0.0),
                (x1 - x0) / width, size)


def _add_grounded_line(layer, page, space, block):
    w, h = space.width, space.height
    font = layer.metrics
    (ax, ay), (bx, by), _, (dx, dy) = ((x * w, y * h) for x, y in block.quad)
    length, thickness = math.hypot(bx - ax, by - ay), math.hypot(dx - ax, dy - ay)
    if length <= 0 or thickness <= 0:
        return
    u = ((bx - ax) / length, (by - ay) / length)
    n = ((dx - ax) / thickness, (dy - ay) / thickness)
    size = thickness / (font.ascender - font.descender)
    base = (ax + n[0] * size * font.ascender, ay + n[1] * size * font.ascender)
    natural = font.text_length(block.text, fontsize=size)
    runs = [(block.text, 0.0, length / natural)] if natural > 0 else []
    # PDFium orders the text objects of a line by their displayed position.
    # On a line shown upside down that reverses split runs ("t 12345
    # Searchable documen"), so such a line stays one run on its line box.
    if block.chars and u[0] > -0.5:
        # Project each character box onto the reading direction.
        starts, ends = [], []
        for _, (x0, y0, x1, y1) in block.chars:
            along = [(x * w - ax) * u[0] + (y * h - ay) * u[1]
                     for x in (x0, x1) for y in (y0, y1)]
            starts.append(max(0.0, min(along)))
            ends.append(min(length, max(along)))
        if all(b > a for a, b in zip(starts, ends)) and \
           all(b >= a for a, b in zip(starts, starts[1:])):
            runs = _runs(block.text, starts, ends,
                         lambda unit: font.text_length(unit, fontsize=size), 0.2 * thickness)
    for run, start, scale in runs:
        layer.place(page, space, run, (base[0] + u[0] * start, base[1] + u[1] * start), u,
                    scale, size)


def add_blocks(layer, page, blocks: list[Block]):
    space = PageSpace(page)
    for block in blocks:
        if any(not _supported(char) for char in block.text):
            raise OcrError("unsupported_ocr_glyph")
        if block.quad is not None:
            _add_grounded_line(layer, page, space, block)
            continue
        lines = [line for line in block.text.splitlines() if line.strip()]
        if not lines:
            continue
        x0, y0, x1, y1 = block.box
        x0, x1 = x0 * space.width, x1 * space.width
        y0, y1 = y0 * space.height, y1 * space.height
        lines, line_height = _fit_lines(lines, x1 - x0, y1 - y0, layer.metrics)
        for n, line in enumerate(lines):
            _add_line(layer, page, space, line, (x0, y0 + n * line_height,
                                                 x1, y0 + (n + 1) * line_height))
    if not pdfium_c.FPDFPage_GenerateContent(page.raw):
        raise OcrError("output_validation_failed")
    # A nonempty ASCII fragment is not evidence that every recognized glyph
    # survived. Every recognized character must be extractable exactly once.
    # PDFium orders extracted text by its own line analysis (a paragraph
    # number may follow its line), so the order is not compared.
    if _characters("".join(block.text for block in blocks)) != _characters(page_text(page)):
        raise OcrError("output_text_mismatch")


def render_page(page, dpi=200, max_pixels=20_000_000) -> bytes:
    """The displayed page (rotation, crop box, annotations) as an RGB PNG."""
    space = PageSpace(page)
    scale = min(dpi / 72, math.sqrt(max_pixels / max(1.0, space.width * space.height)))
    if scale <= 0 or not math.isfinite(scale):
        raise OcrError("invalid_page_geometry")
    image = page.render(scale=scale, draw_annots=True, rev_byteorder=True).to_pil().convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _digests(doc):
    return [hashlib.sha256(page_text(doc[i]).encode("utf-8")).digest() for i in range(len(doc))]


def process_pdf(source: Path, result: Path, recognize, progress=lambda _: None,
                max_pages=200, dpi=200, max_pixels=20_000_000) -> Report:
    source, result = Path(source), Path(result)
    # Preflight occurs before copy or inference. Keep source bytes untouched.
    try:
        validate(source, max_pages)
    except OcrError:
        raise
    except Exception:
        raise OcrError("invalid_pdf") from None
    report = Report()
    try:
        recognized = {}
        with pdfium.PdfDocument(source) as doc:
            report.pages_total = len(doc)
            progress(report.public())
            for index in range(len(doc)):
                page = doc[index]
                if has_text(page):
                    report.pages_native_text += 1
                else:
                    recognized[index] = recognize(render_page(page, dpi, max_pixels))
                progress({**report.public(),
                          "progress": round((index + 1) * 90 / report.pages_total, 1)})
        chars = {char for blocks in recognized.values() for block in blocks for char in block.text}
        written = {index for index, blocks in recognized.items()
                   if "".join("".join(b.text.split()) for b in blocks)}
        report.pages_with_text = report.pages_native_text + len(written)
        report.pages_ocr_added = len(written)
        report.pages_without_text = report.pages_total - report.pages_with_text
        if report.pages_with_text == 0:
            raise OcrError("no_text_recognized")
        if not written:
            shutil.copyfile(source, result)
        else:
            with pdfium.PdfDocument(source) as doc:
                layer = TextLayer(doc, chars)
                try:
                    for index in sorted(written):
                        add_blocks(layer, doc[index], recognized[index])
                    expected = _digests(doc)
                    # Incremental save keeps the original bytes as the prefix,
                    # with images, metadata, bookmarks and annotations.
                    doc.save(result, flags=pdfium_c.FPDF_INCREMENTAL)
                finally:
                    layer.close()
            if not result.read_bytes().startswith(source.read_bytes()):
                raise OcrError("output_validation_failed")
            with pdfium.PdfDocument(result) as reloaded:
                if len(reloaded) != report.pages_total or _digests(reloaded) != expected:
                    raise OcrError("output_validation_failed")
        progress({**report.public(), "progress": 95})
        return report
    except Exception:
        result.unlink(missing_ok=True)
        raise
