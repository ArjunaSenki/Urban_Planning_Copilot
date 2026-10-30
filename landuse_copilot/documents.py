"""Offline planning-document extraction and passage retrieval."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

from docx.opc.exceptions import PackageNotFoundError
from lxml.etree import XMLSyntaxError
from pypdf.errors import PyPdfError
SUPPORTED_DOCUMENT_SUFFIXES = {".pdf", ".txt", ".docx"}
MAX_DOCUMENT_BYTES = 20 * 1024 * 1024
MAX_PASSAGE_CHARS = 700


class DocumentValidationError(ValueError):
    """A supplied planning document cannot be read in a supported format."""


@dataclass(frozen=True)
class Passage:
    document: str
    location: str
    text: str
    score: float


def extract_passages(filename: str, file_bytes: bytes) -> list[tuple[str, str]]:
    """Extract PDF pages or text/document paragraphs with traceable locations."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
        raise DocumentValidationError("Supported planning documents are PDF, TXT, and DOCX.")
    if not file_bytes:
        raise DocumentValidationError(f"{filename} is empty.")
    if len(file_bytes) > MAX_DOCUMENT_BYTES:
        raise DocumentValidationError(f"{filename} exceeds the 20 MB per-document limit.")
    try:
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(__import__("io").BytesIO(file_bytes), strict=True)
            if reader.is_encrypted:
                raise DocumentValidationError(f"{filename} is encrypted and cannot be searched.")
            passages = [
                (f"page {number}", text.strip())
                for number, page in enumerate(reader.pages, start=1)
                if (text := page.extract_text())
                and text.strip()
            ]
        elif suffix == ".docx":
            from docx import Document

            document = Document(__import__("io").BytesIO(file_bytes))
            passages = [
                (f"paragraph {number}", paragraph.text.strip())
                for number, paragraph in enumerate(document.paragraphs, start=1)
                if paragraph.text.strip()
            ]
        else:
            text = file_bytes.decode("utf-8-sig")
            paragraphs = re.split(r"\n\s*\n", text.strip())
            passages = [
                (f"section {number}", paragraph.strip())
                for number, paragraph in enumerate(paragraphs, start=1)
                if paragraph.strip()
            ]
    except DocumentValidationError:
        raise
    except (
        OSError,
        ValueError,
        UnicodeDecodeError,
        BadZipFile,
        PackageNotFoundError,
        PyPdfError,
        XMLSyntaxError,
    ) as error:
        raise DocumentValidationError(
            f"{filename} could not be read. Check that the file is a valid, unprotected {suffix.upper()} document."
        ) from error
    if not passages:
        raise DocumentValidationError(
            f"No searchable text was found in {filename}. Scanned PDFs require OCR, which is not included."
        )
    return passages


def retrieve_passages(
    documents: list[tuple[str, list[tuple[str, str]]]],
    query: str,
    limit: int = 5,
) -> list[Passage]:
    """Rank exact source passages using local TF-IDF-style term weighting."""
    tokens = re.findall(r"[a-z0-9][a-z0-9'-]{1,}", query.lower())
    query_terms = set(tokens)
    if not query_terms:
        return []
    candidates: list[tuple[str, str, str, set[str], Counter[str]]] = []
    for filename, passages in documents:
        for location, text in passages:
            words = re.findall(r"[a-z0-9][a-z0-9'-]{1,}", text.lower())
            term_counts = Counter(words)
            candidates.append((filename, location, text, set(words), term_counts))
    if not candidates:
        return []
    document_frequency = Counter(
        term for term in query_terms if any(term in candidate[3] for candidate in candidates)
    )
    ranked: list[Passage] = []
    total = len(candidates)
    for filename, location, text, word_set, term_counts in candidates:
        score = 0.0
        for term in query_terms & word_set:
            inverse_frequency = 1.0 + (total / (1 + document_frequency[term])) ** 0.5
            score += inverse_frequency * (1.0 + min(term_counts[term], 3) / 3)
        if score:
            excerpt = re.sub(r"\s+", " ", text).strip()
            if len(excerpt) > MAX_PASSAGE_CHARS:
                excerpt = excerpt[: MAX_PASSAGE_CHARS - 1].rsplit(" ", 1)[0] + "…"
            ranked.append(Passage(filename, location, excerpt, score))
    ranked.sort(key=lambda passage: (-passage.score, passage.document, passage.location))
    return ranked[: max(limit, 0)]
