"""Shared text extraction helpers for PDF, DOCX, HTML, and plain text."""

import io
import logging

import pdfplumber
from bs4 import BeautifulSoup
from docx import Document as DocxDocument

logger = logging.getLogger(__name__)


def extract_text(data: bytes, content_type: str) -> str:
    """Extract text from binary data based on content type.

    Args:
        data: Raw file bytes
        content_type: MIME type string (e.g. "application/pdf")

    Returns:
        Extracted text, or empty string on failure
    """
    ct = (content_type or "").lower()

    if "pdf" in ct:
        return pdf_to_text(data)

    if (
        "word" in ct
        or ct.endswith("/msword")
        or ct.endswith("/vnd.openxmlformats-officedocument.wordprocessingml.document")
    ):
        return docx_to_text(data)

    if "html" in ct or "text/html" in ct:
        return html_to_text(data)

    # Default: try UTF-8 decoding
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def pdf_to_text(data: bytes) -> str:
    """Extract text from PDF bytes using pdfplumber."""
    txt = []
    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                try:
                    extracted = page.extract_text()
                    if extracted:
                        txt.append(extracted)
                except Exception:
                    continue
    except Exception as e:
        logger.warning(f"PDF extraction failed: {e}")

    return "\n".join(t for t in txt if t)


def docx_to_text(data: bytes) -> str:
    """Extract text from DOCX bytes using python-docx."""
    try:
        bio = io.BytesIO(data)
        doc = DocxDocument(bio)
        return "\n".join(p.text for p in doc.paragraphs if p.text)
    except Exception as e:
        logger.warning(f"DOCX extraction failed: {e}")
        return ""


def html_to_text(data: bytes) -> str:
    """Extract text from HTML bytes using BeautifulSoup."""
    try:
        soup = BeautifulSoup(data, "html.parser")

        # Remove script and style tags
        for tag in soup(["script", "style"]):
            tag.decompose()

        # Get text and normalize whitespace
        text = soup.get_text(" ")
        return " ".join(text.split())
    except Exception as e:
        logger.warning(f"HTML extraction failed: {e}")
        return ""
