"""A tiny PDF writer for test fixtures (no third-party PDF library).

Writes scanned pages (one JPEG each), optional native Helvetica text, /Rotate,
/CropBox, annotations, an Info dictionary, one outline item, and signature or
encryption markers, with a classic cross-reference table.
"""
import io

from PIL import Image


def _pdf_string(text):
    return "(" + text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)") + ")"


class Writer:
    def __init__(self):
        self.objects = []  # bytes per object, numbered from 1

    def add(self, body) -> int:
        self.objects.append(body if isinstance(body, bytes) else body.encode("latin-1"))
        return len(self.objects)

    def reserve(self) -> int:
        return self.add(b"null")

    def set(self, number, body):
        self.objects[number - 1] = body if isinstance(body, bytes) else body.encode("latin-1")

    def stream(self, data: bytes, entries="") -> int:
        return self.add(f"<< {entries} /Length {len(data)} >>\nstream\n".encode("latin-1") + data +
                        b"\nendstream")

    def save(self, path, root, info=None, extra_trailer=""):
        out = io.BytesIO()
        out.write(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
        offsets = []
        for number, body in enumerate(self.objects, 1):
            offsets.append(out.tell())
            out.write(f"{number} 0 obj\n".encode() + body + b"\nendobj\n")
        xref = out.tell()
        out.write(f"xref\n0 {len(self.objects) + 1}\n0000000000 65535 f \n".encode())
        for offset in offsets:
            out.write(f"{offset:010d} 00000 n \n".encode())
        trailer = f"/Size {len(self.objects) + 1} /Root {root} 0 R"
        if info:
            trailer += f" /Info {info} 0 R"
        out.write(f"trailer\n<< {trailer} {extra_trailer} >>\nstartxref\n{xref}\n%%EOF\n".encode())
        with open(path, "wb") as handle:
            handle.write(out.getvalue())


def jpeg(image: Image.Image, quality=90) -> bytes:
    buffer = io.BytesIO()
    image.convert("L" if image.mode == "L" else "RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def write_pdf(path, pages, title=None, author=None, outline=None, signed=None, encrypted=False):
    """pages: dicts with width, height and any of image (PIL; quality), text
    [(x, y_from_top, string)], rotate, crop (x0, y0, x1, y1 from the top-left),
    cover (x0, y0, x1, y1 from the top-left, an opaque black square annotation).
    signed: None, "field" (signature form field) or "certified" (/Perms only).
    """
    w = Writer()
    catalog, pages_id = w.reserve(), w.reserve()
    helvetica = w.add("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    kids, first_page = [], None
    for spec in pages:
        width, height = spec["width"], spec["height"]
        resources, content = [], []
        if spec.get("image") is not None:
            image = spec["image"]
            data = jpeg(image, spec.get("quality", 90))
            gray = image.mode == "L"
            img = w.stream(data, f"/Type /XObject /Subtype /Image /Width {image.width} /Height {image.height} "
                                 f"/ColorSpace /Device{'Gray' if gray else 'RGB'} /BitsPerComponent 8 "
                                 f"/Filter /DCTDecode")
            resources.append(f"/XObject << /Im0 {img} 0 R >>")
            content.append(f"q {width} 0 0 {height} 0 0 cm /Im0 Do Q")
        if spec.get("text"):
            resources.append(f"/Font << /F1 {helvetica} 0 R >>")
            for x, y, string in spec["text"]:
                content.append(f"BT /F1 12 Tf {x} {height - y} Td {_pdf_string(string)} Tj ET")
        stream = w.stream("\n".join(content).encode("latin-1"))
        entries = (f"/Type /Page /Parent {pages_id} 0 R /MediaBox [0 0 {width} {height}] "
                   f"/Resources << {' '.join(resources)} >> /Contents {stream} 0 R")
        if spec.get("rotate"):
            entries += f" /Rotate {spec['rotate']}"
        if spec.get("crop"):
            x0, y0, x1, y1 = spec["crop"]
            entries += f" /CropBox [{x0} {height - y1} {x1} {height - y0}]"
        if spec.get("cover"):
            x0, y0, x1, y1 = spec["cover"]
            rect = f"{x0} {height - y1} {x1} {height - y0}"
            ap = w.stream(f"0 g 0 0 {x1 - x0} {y1 - y0} re f".encode(),
                          f"/Type /XObject /Subtype /Form /BBox [0 0 {x1 - x0} {y1 - y0}]")
            annot = w.add(f"<< /Type /Annot /Subtype /Square /Rect [{rect}] /C [0 0 0] /IC [0 0 0] "
                          f"/F 4 /AP << /N {ap} 0 R >> >>")
            entries += f" /Annots [{annot} 0 R]"
        page = w.add(f"<< {entries} >>")
        kids.append(page)
        first_page = first_page or page
    w.set(pages_id, f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>")
    root_entries = f"/Type /Catalog /Pages {pages_id} 0 R"
    if outline:
        outlines = w.reserve()
        item = w.add(f"<< /Title {_pdf_string(outline)} /Parent {outlines} 0 R /Dest [{first_page} 0 R /Fit] >>")
        w.set(outlines, f"<< /Type /Outlines /First {item} 0 R /Last {item} 0 R /Count 1 >>")
        root_entries += f" /Outlines {outlines} 0 R"
    if signed:
        sig = w.add("<< /Type /Sig /Filter /Adobe.PPKLite /SubFilter /adbe.pkcs7.detached "
                    "/ByteRange [0 1 2 3] /Contents <00> >>")
        if signed == "field":
            field = w.add(f"<< /FT /Sig /T (Signature1) /V {sig} 0 R /Type /Annot /Subtype /Widget "
                          f"/Rect [0 0 0 0] /P {first_page} 0 R >>")
            root_entries += f" /AcroForm << /Fields [{field} 0 R] /SigFlags 3 >>"
        else:
            root_entries += f" /Perms << /DocMDP {sig} 0 R >>"
    w.set(catalog, f"<< {root_entries} >>")
    info = None
    if title or author:
        info = w.add("<< " + (f"/Title {_pdf_string(title)} " if title else "") +
                     (f"/Author {_pdf_string(author)}" if author else "") + " >>")
    extra = ""
    if encrypted:
        # A Standard security handler whose password checks cannot pass.
        enc = w.add("<< /Filter /Standard /V 2 /R 3 /Length 128 /P -44 "
                    "/O <" + "11" * 32 + "> /U <" + "22" * 32 + "> >>")
        extra = f"/Encrypt {enc} 0 R /ID [<{'33' * 16}> <{'33' * 16}>]"
    w.save(path, catalog, info, extra)
