"""Build minimal, valid text PDFs for tests, so no binary fixtures live in the repo."""


def _escape(line: str) -> str:
    return line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def build_pdf(pages: list[list[str]]) -> bytes:
    """Return PDF bytes with one page per entry; each entry is that page's text lines.

    A page given as an empty list has no text layer, like a scanned image page.
    """
    page_count = len(pages)
    font_id = 3
    first_page_id = 4
    # objects: 1 catalog, 2 pages, 3 font, then (page, content) pairs
    kids = " ".join(f"{first_page_id + 2 * i} 0 R" for i in range(page_count))
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{kids}] /Count {page_count} >>".encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, lines in enumerate(pages):
        content_id = first_page_id + 2 * i + 1
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode()
        )
        if lines:
            shown = " T* ".join(f"({_escape(line)}) Tj" for line in lines)
            stream = f"BT /F1 12 Tf 14 TL 72 720 Td {shown} ET".encode()
        else:
            stream = b""
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n%s\nendobj\n" % (number, body)
    xref_offset = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % offset for offset in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref_offset,
    )
    return bytes(out)
