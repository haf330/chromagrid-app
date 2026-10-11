"""Read the assignment PDF/TXT sources; no Markdown runtime dependency."""

from pathlib import Path
from pypdf import PdfReader

from .catalogue import Catalogue, CatalogueError


def load_business_reference(root):
    root = Path(root)
    try:
        summary = (root / "me" / "business_summary.txt").read_text(encoding="utf-8")
        pdf_text = "\n".join(page.extract_text() or "" for page in PdfReader(root / "me" / "about_business.pdf").pages)
    except Exception as exc:
        raise CatalogueError("The required business PDF and TXT could not be read.") from exc
    if not summary.strip() or not pdf_text.strip():
        raise CatalogueError("The required business PDF and TXT must contain text.")
    reference = "Business summary (TXT):\n" + summary + "\nFull approved business profile (PDF):\n" + pdf_text
    return reference, Catalogue.from_pdf_text(pdf_text)
