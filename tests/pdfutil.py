"""PDFium-based assertions for tests: text, search boxes, pixels, metadata."""
from pathlib import Path

import pypdfium2 as pdfium

from pdf_layer import PageSpace, page_text


def open_pdf(source):
    """A PDFium document from a path or from bytes."""
    return pdfium.PdfDocument(source if isinstance(source, (bytes, bytearray)) else Path(source))


def text(page) -> str:
    return page_text(page)


def find(page, term):
    """Each match of term as one (x0, y0, x1, y1) box in displayed points."""
    space = PageSpace(page)
    textpage = page.get_textpage()
    try:
        searcher = textpage.search(term, match_case=True)
        boxes = []
        while (hit := searcher.get_next()) is not None:
            index, count = hit
            rects = [space.rect_to_display(*textpage.get_charbox(i, loose=True))
                     for i in range(index, index + count)
                     if not textpage.get_text_range(i, 1).isspace()]
            boxes.append((min(r[0] for r in rects), min(r[1] for r in rects),
                          max(r[2] for r in rects), max(r[3] for r in rects)))
        searcher.close()
        return boxes
    finally:
        textpage.close()


def chars(page):
    """(character, display box) for every extracted non-space character."""
    space = PageSpace(page)
    textpage = page.get_textpage()
    try:
        out = []
        for i in range(textpage.count_chars()):
            char = textpage.get_text_range(i, 1)
            if char and not char.isspace():
                out.append((char, space.rect_to_display(*textpage.get_charbox(i, loose=True))))
        return out
    finally:
        textpage.close()


def pixels(page, scale=1.0) -> bytes:
    bitmap = page.render(scale=scale, draw_annots=True)
    return bytes(bitmap.buffer)


def display_size(page):
    space = PageSpace(page)
    return space.width, space.height


def metadata(doc):
    return {k: v for k, v in doc.get_metadata_dict().items() if v}


def outline(doc):
    items = []
    for item in doc.get_toc():
        dest = item.get_dest()
        items.append((item.level, item.get_title(), dest.get_index() if dest else None))
    return items
