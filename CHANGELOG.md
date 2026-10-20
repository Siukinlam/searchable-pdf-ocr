# Changelog

## 0.2.0 (unreleased)

### Changed
- **License:** Apache-2.0. Version 0.1.0 remains available under AGPL-3.0.
- **PDF backend:** PyMuPDF replaced by [pypdfium2](https://github.com/pypdfium2-team/pypdfium2)
  (PDFium) for rendering pages, detecting existing text and writing the
  invisible text layer. No AGPL or GPL dependency remains.
- **Text-layer font:** the embedded 3.4 MB Droid Sans Fallback is replaced by a
  font generated per document with fontTools. It has one rectangular glyph per
  character used, never painted in render mode 3, and a Unicode cmap so every
  character round-trips through extraction. A typical document now grows by
  a few kilobytes for the font instead of 3.4 MB.
- **Text check:** every recognized character must be extractable exactly once.
  The order is no longer compared, because PDFium orders extracted text by its
  own line analysis.
- **Signature detection:** PDFium's signature-field count plus a scan for
  signature and certification dictionaries, including inside compressed
  object streams.
- **Damaged files:** a source whose final `startxref` does not point at a
  cross-reference section is rejected as `pdf_requires_repair`.
- **Tests:** fixtures are built with a small built-in PDF writer and a subset
  of Droid Sans Fallback; there is no test-time dependency on a PDF library
  other than pypdfium2.

### Known differences from 0.1.0
- **Output size:** PDFium's incremental save re-serializes every object it
  parsed on a modified page, including the page image. A scanned page
  therefore adds roughly its own size again, and the original bytes are still
  kept as the prefix. Documents larger than half of `OCR_MAX_PDF_BYTES` may
  hit `output_too_large`.
- **Unused page resources:** PDFium rebuilds `/Resources` of modified pages
  and drops resources the page does not use. Resources the content uses,
  including optional content, shadings and graphics states, are kept.
- **Upside-down lines:** lines displayed upside down are written as one run on
  the line box, without per-character placement. PDFium would otherwise
  reverse the order of split runs.
- **Non-BMP characters:** characters outside the Basic Multilingual Plane are
  rejected with `unsupported_ocr_glyph`. 0.1.0 also rejected them, because
  its font lacked them.

## 0.1.0

First release (AGPL-3.0): `unlimited`, `rapidocr` and `hybrid` engines with a
PyMuPDF text layer.
