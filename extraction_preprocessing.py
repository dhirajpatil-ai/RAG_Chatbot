# extraction_preprocessing.py

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any

from unstructured.partition.pdf import partition_pdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# CONFIGURATION
# ============================================================

CHUNK_SIZE = 800
CHUNK_OVERLAP = 160

# Tesseract path.
# If tesseract is already in PATH, keep this as None.
TESSERACT_CMD = None

# For Windows, if needed:
# TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    """
    Basic deterministic text normalization.
    """

    if not text:
        return ""

    # Normalize line endings
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Remove excessive spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove strange control characters
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    return text.strip()


def normalize_header_footer(text: str) -> str:
    """
    Lightweight header/footer normalization.

    This removes common repeating page-number patterns.
    """

    lines = text.splitlines()

    cleaned = []

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # Page 1 / Page 2 / Page 25
        if re.fullmatch(
            r"(page\s*)?\d+(\s*(of|/)\s*\d+)?",
            line,
            flags=re.IGNORECASE
        ):
            continue

        # "Page 1 of 100"
        if re.fullmatch(
            r"page\s+\d+\s+of\s+\d+",
            line,
            flags=re.IGNORECASE
        ):
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


# ============================================================
# BOUNDING BOX
# ============================================================

def get_bbox(metadata: Any):
    """
    Extract bounding box from Unstructured metadata if available.
    """

    try:
        coordinates = metadata.coordinates

        if coordinates is None:
            return None

        points = getattr(coordinates, "points", None)

        if not points:
            return None

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]

        return {
            "x1": min(xs),
            "y1": min(ys),
            "x2": max(xs),
            "y2": max(ys),
        }

    except Exception:
        return None


# ============================================================
# PDF EXTRACTION
# ============================================================

def extract_pdf(pdf_path: str | Path) -> List[Document]:
    """
    Extract PDF content using Unstructured.

    strategy="hi_res" is important because the PDF may contain:
        - native text
        - scanned pages
        - images
        - tables

    OCR is used where text extraction is unavailable.
    """

    pdf_path = Path(pdf_path)

    print(f"\nProcessing: {pdf_path.name}")

    elements = partition_pdf(
        filename=str(pdf_path),

        # High-resolution strategy handles scanned/image-heavy PDFs.
        strategy="hi_res",

        # Extract table structure when possible.
        infer_table_structure=True,

        # OCR language.
        languages=["eng"],

        # Keep coordinates when available.
        include_page_breaks=False,
    )

    documents = []

    for element in elements:

        text = str(element).strip()

        if not text:
            continue

        text = clean_text(text)
        text = normalize_header_footer(text)

        if not text:
            continue

        metadata = element.metadata

        page_number = getattr(
            metadata,
            "page_number",
            None
        )

        category = getattr(
            element,
            "category",
            None
        )

        bbox = get_bbox(metadata)

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "pdf_id": hashlib.sha256(
                        pdf_path.name.encode()
                    ).hexdigest()[:16],

                    "filename": pdf_path.name,

                    "page_number": page_number
                    if page_number is not None
                    else -1,

                    "element_type": category
                    if category
                    else "unknown",

                    "bbox": bbox
                    if bbox
                    else {},
                }
            )
        )

    print(
        f"Extracted {len(documents)} elements "
        f"from {pdf_path.name}"
    )

    return documents


# ============================================================
# CHUNKING
# ============================================================

def chunk_documents(
    documents: List[Document]
) -> List[Document]:

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",

        chunk_size=CHUNK_SIZE,

        chunk_overlap=CHUNK_OVERLAP,

        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ],
    )

    chunks = splitter.split_documents(documents)

    output = []

    for index, chunk in enumerate(chunks):

        content = clean_text(chunk.page_content)

        if len(content) < 30:
            continue

        metadata = dict(chunk.metadata)

        metadata["chunk_index"] = index

        # Deterministic chunk ID
        raw_id = (
            f"{metadata.get('filename', '')}|"
            f"{metadata.get('page_number', '')}|"
            f"{index}|"
            f"{content}"
        )

        chunk_id = hashlib.sha256(
            raw_id.encode("utf-8")
        ).hexdigest()

        metadata["chunk_id"] = chunk_id

        output.append(
            Document(
                page_content=content,
                metadata=metadata
            )
        )

    print(f"Created {len(output)} chunks")

    return output


# ============================================================
# COMPLETE INGESTION
# ============================================================

def process_pdf(pdf_path: str | Path) -> List[Document]:

    documents = extract_pdf(pdf_path)

    chunks = chunk_documents(documents)

    return chunks


def process_directory(directory: str | Path) -> List[Document]:

    directory = Path(directory)

    pdf_files = sorted(
        directory.glob("*.pdf")
    )

    if not pdf_files:
        raise ValueError(
            f"No PDF files found in {directory}"
        )

    print(
        f"\nFound {len(pdf_files)} PDF files."
    )

    all_chunks = []

    for pdf_file in pdf_files:

        chunks = process_pdf(pdf_file)

        all_chunks.extend(chunks)

    print(
        f"\nTotal chunks created: "
        f"{len(all_chunks)}"
    )

    return all_chunks