from __future__ import annotations
import fitz

def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from a PDF using PyMuPDF."""
    with fitz.open(pdf_path) as doc:
        return "\n".join(p.get_text("text") for p in doc)