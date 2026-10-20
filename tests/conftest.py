from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PIL import Image, ImageDraw, ImageFont
import pytest

from engine import Block
from pdf_layer import PageSpace
import pdfgen
from pdfutil import open_pdf

FONT = Path(__file__).with_name("fixtures") / "DroidSansFallback-subset.ttf"
LINES = ["Searchable document 12345", "繁體中文文件測試", "简体中文文档测试"]
CROP = (25, 20, 575, 780)  # x0, y0, x1, y1 from the top-left of a 600x800 page


def draw_scan(lines, width=600, height=800, zoom=2, size=20, origin=(65, 100), step=55, rect=True):
    """A rasterized page and each line's box in points (top-left origin): the
    advance width by the font's ascent and descent, as a text extractor
    reports a line, not the tighter ink bounds."""
    image = Image.new("RGB", (width * zoom, height * zoom), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(FONT), size * zoom)
    ascent, descent = font.getmetrics()
    boxes = []
    for index, line in enumerate(lines):
        x, y = origin[0] * zoom, (origin[1] + index * step) * zoom
        draw.text((x, y), line, font=font, fill="black", anchor="ls")
        boxes.append((x / zoom, (y - ascent) / zoom, (x + font.getlength(line)) / zoom, (y + descent) / zoom))
    if rect:
        draw.rectangle((40 * zoom, 40 * zoom, 560 * zoom, 270 * zoom), outline=(51, 102, 178), width=2)
    return image, boxes


@pytest.fixture
def scan_factory(tmp_path):
    def create(name="scan", rotation=0, native=False, blank=False, crop=False):
        image, boxes = draw_scan([] if blank else LINES)
        page = {"width": 600, "height": 800, "image": image, "rotate": rotation}
        if crop:
            page["crop"] = CROP
        pages = [page]
        if native:
            pages.append({"width": 600, "height": 800, "text": [(80, 100, "Existing native text stays exact.")]})
        path = tmp_path / (name + ".pdf")
        pdfgen.write_pdf(path, pages, title="Synthetic OCR corpus", author="Synthetic QA",
                         outline="Synthetic bookmark")
        # Ground truth in normalized displayed coordinates of the scanned page.
        with open_pdf(path) as doc:
            space = PageSpace(doc[0])
            blocks = []
            for line, (x0, y0, x1, y1) in zip(LINES, boxes):
                corners = [space.to_display(x, 800 - y) for x in (x0, x1) for y in (y0, y1)]
                xs, ys = [c[0] for c in corners], [c[1] for c in corners]
                blocks.append(Block(line, (min(xs) / space.width, min(ys) / space.height,
                                           max(xs) / space.width, max(ys) / space.height)))
        return path, blocks
    return create
