from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pymupdf as fitz
import pytest

from engine import Block


@pytest.fixture
def scan_factory(tmp_path):
    def create(name="scan", rotation=0, native=False, blank=False, crop=False):
        lines = ["Searchable document 12345", "繁體中文文件測試", "简体中文文档测试"]
        original = fitz.open()
        page = original.new_page(width=600, height=800)
        font = fitz.Font("cjk")
        page.insert_font(fontname="CorpusFont", fontbuffer=font.buffer)
        if not blank:
            for index, line in enumerate(lines):
                page.insert_text((65, 100 + index * 55), line,
                                 fontsize=20, fontname="CorpusFont")
        page.draw_rect((40, 40, 560, 270), color=(0.2, 0.4, 0.7), width=1)
        # Rasterize content so the new test input genuinely has no text layer.
        image = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).tobytes("png")
        boxes = [fitz.Rect(line["bbox"]) for block in page.get_text("dict")["blocks"]
                 if block["type"] == 0 for line in block["lines"]]
        doc = fitz.open()
        scanned = doc.new_page(width=600, height=800)
        scanned.insert_image(scanned.rect, stream=image)
        if crop:
            scanned.set_cropbox(fitz.Rect(25, 20, 575, 780))
            boxes = [box + (-25, -20, -25, -20) for box in boxes]
        scanned.set_rotation(rotation)
        blocks = []
        for line, box in zip(lines, boxes):
            rect = box * scanned.rotation_matrix
            blocks.append(Block(line, (rect.x0 / scanned.rect.width, rect.y0 / scanned.rect.height,
                                       rect.x1 / scanned.rect.width, rect.y1 / scanned.rect.height)))
        if native:
            p = doc.new_page(width=600, height=800)
            p.insert_text((80, 100), "Existing native text stays exact.")
        doc.set_metadata({"title": "Synthetic OCR corpus", "author": "Synthetic QA"})
        doc.set_toc([[1, "Synthetic bookmark", 1]])
        path = tmp_path / (name + ".pdf")
        doc.save(path)
        doc.close()
        original.close()
        return path, blocks
    return create
