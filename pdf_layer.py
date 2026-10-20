"""Preserve the PDF and append Unicode text with PDF render mode 3."""
from dataclasses import dataclass, asdict
import hashlib
import math
from pathlib import Path
import shutil
import unicodedata

import pymupdf as fitz

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


def has_text(page):
    return bool(page.get_text().strip())


def validate(doc, max_pages=200):
    if not doc.is_pdf or doc.needs_pass or doc.is_encrypted:
        raise OcrError("encrypted_or_invalid_pdf")
    if not 0 < doc.page_count <= max_pages:
        raise OcrError("page_limit_exceeded")
    # Reject signed/certified forms as well as signature dictionaries not
    # reachable via widgets. Never silently invalidate a signature.
    if doc.get_sigflags() > 0:
        raise OcrError("signed_pdf_not_supported")
    for xref in range(1, doc.xref_length()):
        if doc.xref_get_key(xref, "Type")[1] == "/Sig" or \
           doc.xref_get_key(xref, "FT")[1] == "/Sig" or \
           doc.xref_get_key(xref, "ByteRange")[0] != "null":
            raise OcrError("signed_pdf_not_supported")
    if not doc.can_save_incrementally():
        raise OcrError("pdf_requires_repair")


def _add_line(page, text, box, font, fontname):
    # Work in the displayed-page space used by the model, then apply the
    # inverse /Rotate transform to the insertion origin and scaling axes.
    rect = fitz.Rect(box)
    size = rect.height / (font.ascender - font.descender)
    width = font.text_length(text, fontsize=size)
    if size <= 0 or width <= 0:
        return
    origin = fitz.Point(rect.x0, rect.y0 + size * font.ascender)
    origin = origin * page.derotation_matrix
    scale = rect.width / width
    # Scaling is in unrotated axes; horizontal displayed text maps to the
    # PDF's Y axis on 90/270-degree pages.
    morph = fitz.Matrix(1, scale) if page.rotation in {90, 270} else fitz.Matrix(scale, 1)
    page.insert_text(origin, text, fontname=fontname, fontsize=size,
                     render_mode=3, rotate=page.rotation, morph=(origin, morph), overlay=True)


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


def _place_run(page, text, origin, direction, scale, size, fontname):
    # origin is the baseline start and direction the unit text direction, both
    # in the displayed-page space used by the OCR image. Stretch the run along
    # its own axis (conjugated by /Rotate, as in _add_line), then turn it to
    # follow skew or a vertical column. Morph angles run opposite to
    # displayed y-down angles.
    if size <= 0 or scale <= 0:
        return
    angle = math.degrees(math.atan2(direction[1], direction[0]))
    rotation = fitz.Matrix(page.rotation_matrix)
    rotation.e = rotation.f = 0
    morph = rotation * fitz.Matrix(scale, 1) * ~rotation * fitz.Matrix(-angle)
    origin = fitz.Point(origin) * page.derotation_matrix
    page.insert_text(origin, text, fontname=fontname, fontsize=size,
                     render_mode=3, rotate=page.rotation, morph=(origin, morph), overlay=True)


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


def _add_grounded_line(page, block, font, fontname):
    w, h = page.rect.width, page.rect.height
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
    if block.chars:
        # Project each character box onto the reading direction.
        starts, ends = [], []
        for _, (x0, y0, x1, y1) in block.chars:
            along = [(x * w - ax) * u[0] + (y * h - ay) * u[1]
                     for x in (x0, x1) for y in (y0, y1)]
            starts.append(max(0.0, min(along)))
            ends.append(min(length, max(along)))
        if all(b > a for a, b in zip(starts, ends)) and \
           all(b >= a for a, b in zip(starts, starts[1:])):
            cache = {}
            def advance(unit):
                if unit not in cache:
                    cache[unit] = font.text_length(unit, fontsize=size)
                return cache[unit]
            runs = _runs(block.text, starts, ends, advance, 0.2 * thickness)
    for run, start, scale in runs:
        _place_run(page, run, (base[0] + u[0] * start, base[1] + u[1] * start), u,
                   scale, size, fontname)


def add_blocks(page, blocks: list[Block]):
    font = fitz.Font("cjk")
    # Resource names belong to the source PDF. Reusing an existing name makes
    # PyMuPDF silently keep that font, even if its glyph mapping differs.
    existing = {item[4] for item in page.get_fonts(full=True)}
    fontname = "OCRUnicode"
    suffix = 0
    while fontname in existing:
        suffix += 1
        fontname = f"OCRUnicode{suffix}"
    if blocks:
        page.insert_font(fontname=fontname, fontbuffer=font.buffer)
    for block in blocks:
        if any(not font.has_glyph(ord(char)) for char in block.text if not char.isspace()):
            raise OcrError("unsupported_ocr_glyph")
        if block.quad is not None:
            _add_grounded_line(page, block, font, fontname)
            continue
        lines = [line for line in block.text.splitlines() if line.strip()]
        if not lines:
            continue
        x0, y0, x1, y1 = block.box
        rect = page.rect
        x0, x1 = x0 * rect.width, x1 * rect.width
        y0, y1 = y0 * rect.height, y1 * rect.height
        lines, line_height = _fit_lines(lines, x1 - x0, y1 - y0, font)
        for n, line in enumerate(lines):
            _add_line(page, line, (x0, y0 + n * line_height,
                                  x1, y0 + (n + 1) * line_height), font, fontname)
    # A nonempty ASCII fragment is not evidence that every recognized glyph
    # survived. Check the inserted Unicode content before accepting the page.
    expected = "".join("".join(block.text.split()) for block in blocks)
    if expected != "".join(page.get_text().split()):
        raise OcrError("output_text_mismatch")


def process_pdf(source: Path, result: Path, recognize, progress=lambda _: None,
                max_pages=200, dpi=200, max_pixels=20_000_000) -> Report:
    # Preflight occurs before copy or inference. Keep source bytes untouched.
    try:
        with fitz.open(source) as doc:
            validate(doc, max_pages)
    except OcrError:
        raise
    except Exception:
        raise OcrError("invalid_pdf") from None
    shutil.copyfile(source, result)
    report = Report()
    text_digests = []
    try:
        with fitz.open(result) as doc:
            report.pages_total = doc.page_count
            progress(report.public())
            for page in doc:
                if has_text(page):
                    report.pages_native_text += 1
                    report.pages_with_text += 1
                else:
                    scale = min(dpi / 72, math.sqrt(max_pixels / max(1, page.rect.get_area())))
                    if scale <= 0 or not math.isfinite(scale):
                        raise OcrError("invalid_page_geometry")
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False,
                                             colorspace=fitz.csRGB, annots=True)
                    blocks = recognize(pixmap.tobytes("png"))
                    add_blocks(page, blocks)
                    if has_text(page):
                        report.pages_with_text += 1
                        report.pages_ocr_added += 1
                    else:
                        report.pages_without_text += 1
                text_digests.append(hashlib.sha256(page.get_text().encode("utf-8")).digest())
                progress({**report.public(), "progress": round((page.number + 1) * 95 / doc.page_count, 1)})
            if report.pages_with_text == 0:
                raise OcrError("no_text_recognized")
            if report.pages_ocr_added:
                # Incremental save retains original source bytes, streams,
                # images, metadata, bookmarks and annotations.
                doc.saveIncr()
        with fitz.open(result) as reloaded:
            if reloaded.page_count != report.pages_total or \
               sum(has_text(page) for page in reloaded) != report.pages_with_text or \
               [hashlib.sha256(page.get_text().encode("utf-8")).digest() for page in reloaded] != text_digests:
                raise OcrError("output_validation_failed")
        return report
    except Exception:
        result.unlink(missing_ok=True)
        raise
