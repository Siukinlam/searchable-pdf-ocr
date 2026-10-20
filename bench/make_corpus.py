"""Generate a reproducible synthetic scan corpus with exact ground truth.

Each page is first laid out as a native PDF text page (the reference text and
character boxes), then rasterized to an image-only single-page PDF the OCR
engines see. All text is fictional and written for this benchmark.

    python bench/make_corpus.py OUT_DIR [--cjk-font PATH] [--cjk-index N]

The CJK font needs Traditional Chinese glyphs; without --cjk-font a common
system font is tried (Noto Serif/Sans CJK on Linux, Songti TC on macOS).
"""
import argparse
import io
import json
import math
import random
from pathlib import Path

import pymupdf as fitz
from PIL import Image

CJK_CANDIDATES = [
    ("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc", 2),   # Noto Serif CJK TC
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 2),    # Noto Sans CJK TC
    ("/System/Library/Fonts/Supplemental/Songti.ttc", 7),             # Songti TC Regular
]

TC = [
    "河濱社區圖書館於本年度完成館藏數碼化計劃第一期，共掃描舊報紙及地方文獻一萬二千三百頁。",
    "所有掃描檔案須經人手覆核，確認內容完整後方可上載至公開檢索平台。",
    "借閱服務全年開放三百一十二日，總借閱量為八萬六千四百五十冊次，較上年度增加百分之七點二。",
    "兒童閱讀角於週末舉辦說故事活動，參加人數累計二千零四十人次。",
    "本館資料庫採用全文檢索技術，讀者可按標題、作者或內文關鍵詞查詢。",
    "為保障個人資料私隱，讀者記錄只保存二十四個月，期滿後即予刪除。",
    "義工團隊共有六十八名成員，負責整理書架、修補舊書及協助長者使用電子資源。",
    "虛擬導覽系統於三月正式啟用，訪客可在網上瀏覽各樓層的設施與館藏分佈。",
    "館內設有自助還書機兩部，還書後系統會即時發出電子收據。",
    "年度預算支出總額為港幣四百二十五萬元，其中資訊科技項目佔百分之十八。",
    "如發現資料錯誤或掃描影像不清晰，請透過意見表格通知館方，我們會於十個工作天內回覆。",
    "特藏室收藏地方族譜、碑帖拓本及早期地圖，須預約方可入內閱覽。",
    "本年度共舉辦講座及工作坊四十五場，主題包括電子書製作、數碼攝影及家族歷史研究。",
    "館方與鄰近三所中學合作推行閱讀大使計劃，學生每月推介一本好書。",
    "影印及列印服務收費維持不變，黑白每頁港幣一元，彩色每頁港幣五元。",
    "新設的安靜研習區提供四十個座位，每個座位均配備電源插座及檯燈。",
    "OCR 處理後的文件可供全文搜尋，但手寫筆跡及褪色字跡的辨識率仍有待提升。",
    "下年度計劃擴建多媒體室，並引入可借用的平板電腦及電子閱讀器。",
    "所有公開活動均須事先登記，名額有限，先到先得。",
    "館務委員會每季開會一次，會議紀錄於會後兩星期內在網站公佈。",
]
EN = [
    "The library completed phase one of its digitization program this year, scanning 12,300 pages of local newspapers.",
    "Every scanned file is reviewed by staff before it is published to the public search portal.",
    "Lending services were available on 312 days, with 86,450 loans recorded, up 7.2% on the previous year.",
    "Reader records are kept for 24 months and are deleted when that period expires.",
    "A virtual tour went live in March, letting visitors browse each floor online.",
    "Total expenditure was HK$4.25 million, of which IT projects accounted for 18 percent.",
    "Text recognized by OCR is searchable, although handwriting and faded print remain difficult.",
    "Next year the multimedia room will be expanded and tablets will be available for loan.",
    "All public events require registration in advance; places are limited.",
    "Meeting minutes are published on the website within two weeks of each quarterly meeting.",
]
HEADINGS = ["一、年度概況", "二、館藏發展", "三、讀者服務", "四、社區活動", "五、資訊科技", "六、財務摘要", "七、未來計劃"]


def find_font(path, index):
    if path:
        return path, index
    for candidate, idx in CJK_CANDIDATES:
        if Path(candidate).is_file():
            return candidate, idx
    raise SystemExit("no Traditional Chinese font found; pass --cjk-font PATH --cjk-index N")


def load_face(path, index):
    """Font bytes; PyMuPDF cannot pick a face inside a .ttc collection."""
    if not path.lower().endswith((".ttc", ".otc")):
        return Path(path).read_bytes()
    from fontTools.ttLib import TTCollection
    buffer = io.BytesIO()
    TTCollection(path)[index].save(buffer)
    return buffer.getvalue()


def wrap(text, font, size, width):
    lines, current = [], ""
    for ch in text:
        if current and font.text_length(current + ch, fontsize=size) > width:
            # Keep Latin words whole where possible.
            if ch != " " and " " in current and current[-1] != " " and ord(ch) < 0x3000:
                head, _, tail = current.rpartition(" ")
                lines.append(head)
                current = tail + ch
                continue
            lines.append(current.rstrip())
            current = "" if ch == " " else ch
        else:
            current += ch
    if current.strip():
        lines.append(current.rstrip())
    return lines


class Page:
    def __init__(self, doc, fonts):
        self.page = doc.new_page(width=595.28, height=841.89)
        self.fonts = fonts
        for name, font in fonts.items():
            self.page.insert_font(fontname=name, fontbuffer=font.buffer)
        self.y = 72

    def text(self, x, y, s, name="cjk", size=11):
        self.page.insert_text((x, y), s, fontname=name, fontsize=size)

    def paragraph(self, s, name="cjk", size=11, indent=0, leading=1.6):
        font = self.fonts[name]
        for line in wrap(s, font, size, 595.28 - 144 - indent):
            self.y += size * leading
            self.text(72 + indent, self.y, line, name, size)
        self.y += size * 0.6


def build(kind, number, rng, fonts):
    doc = fitz.open()
    p = Page(doc, fonts)
    p.text(72, 48, f"河濱社區圖書館年報 — Riverside Community Library Annual Report", "cjk", 8)
    if kind in ("para", "skew", "lowres"):
        for section in rng.sample(range(len(HEADINGS)), 3):
            p.y += 10
            p.text(72, p.y + 14, HEADINGS[section], "cjk", 14)
            p.y += 20
            for k in range(rng.randint(2, 3)):
                p.paragraph(f"{section + 1}.{k + 1}　" +
                            "".join(rng.sample(TC, rng.randint(2, 4))), size=rng.choice([10.5, 11, 12]))
    elif kind == "bilingual":
        for _ in range(4):
            p.paragraph("".join(rng.sample(TC, 2)), size=11)
            p.paragraph(" ".join(rng.sample(EN, 2)), name="tiro", size=11)
    elif kind == "table":
        p.text(72, 90, "表一：分館服務統計 Table 1: Branch statistics", "cjk", 13)
        cols = [72, 200, 290, 380, 470, 523]
        head = ["分館 Branch", "借閱量", "訪客", "活動", "備註"]
        rows = [head] + [[f"第{n}分館", f"{rng.randint(1000, 99999):,}", f"{rng.randint(100, 9999):,}",
                          str(rng.randint(1, 60)), rng.choice(["正常", "裝修中", "延長開放", "—"])]
                         for n in range(1, 15)]
        y = 110
        for r in rows:
            p.page.draw_rect(fitz.Rect(cols[0], y, cols[-1], y + 26), color=(0.3, 0.3, 0.3), width=0.6)
            for c in cols[1:-1]:
                p.page.draw_line((c, y), (c, y + 26), color=(0.3, 0.3, 0.3), width=0.6)
            for x, cell in zip(cols, r):
                p.text(x + 5, y + 17, cell, "cjk", 10)
            y += 26
        p.y = y + 10
        p.paragraph("".join(rng.sample(TC, 2)), size=11)
    p.text(72, 815, f"本文件為基準測試用虛構樣本　第 {number} 頁", "cjk", 8)
    return doc


def rasterize(doc, kind, rng):
    page = doc[0]
    chars = [(c["c"], tuple(c["bbox"])) for b in page.get_text("rawdict")["blocks"]
             for l in b.get("lines", []) for s in l["spans"] for c in s["chars"] if not c["c"].isspace()]
    dpi, quality = (120, 50) if kind == "lowres" else (150, 65)
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), colorspace=fitz.csGRAY, alpha=False)
    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    if kind == "skew":
        deg = rng.choice([-1.5, 1.5])
        img = img.rotate(-deg, resample=Image.BICUBIC, fillcolor=255)
        cx, cy = page.rect.width / 2, page.rect.height / 2
        a = math.radians(deg)
        def rot(x, y):
            dx, dy = x - cx, y - cy
            return cx + dx * math.cos(a) - dy * math.sin(a), cy + dx * math.sin(a) + dy * math.cos(a)
        chars = [(c, (lambda pts: (min(q[0] for q in pts), min(q[1] for q in pts),
                                   max(q[0] for q in pts), max(q[1] for q in pts)))(
                     [rot(x, y) for x in (b[0], b[2]) for y in (b[1], b[3])])) for c, b in chars]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    out = fitz.open()
    target = out.new_page(width=page.rect.width, height=page.rect.height)
    target.insert_image(target.rect, stream=buf.getvalue())
    return out, chars


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--cjk-font")
    parser.add_argument("--cjk-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    path, index = find_font(args.cjk_font, args.cjk_index)
    fonts = {"cjk": fitz.Font(fontbuffer=load_face(path, index)), "tiro": fitz.Font("tiro")}
    corpus = args.out / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    plan = [("para", 10), ("bilingual", 6), ("table", 6), ("skew", 4), ("lowres", 4)]
    refs, n = {}, 0
    for kind, count in plan:
        for i in range(count):
            n += 1
            doc = build(kind, n, rng, fonts)
            scan, chars = rasterize(doc, kind, rng)
            name = f"{kind}-{i + 1:02d}"
            scan.save(corpus / f"{name}.pdf")
            refs[name] = {"text": "".join(c for c, _ in chars), "chars": chars}
    (args.out / "reference.json").write_text(json.dumps(refs, ensure_ascii=False))
    print(len(refs), "pages from", path, "face", index)


if __name__ == "__main__":
    main()
