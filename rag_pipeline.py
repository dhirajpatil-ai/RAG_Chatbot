# rag_pipeline.py

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
import os
from dotenv import load_dotenv


from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from extraction_preprocessing import process_directory

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError(
        "HF_TOKEN not found. Please add HF_TOKEN to .env"
    )


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PDF_DIRECTORY = BASE_DIR / "data" / "pdfs"

CHROMA_DIRECTORY = BASE_DIR / "chroma_db"

COLLECTION_NAME = "private_pdf_knowledge_base"


# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


# ============================================================
# LLM
# ============================================================

LLM_MODEL = "meta-llama/Llama-3.2-1B-Instruct"


# ============================================================
# RETRIEVAL CONFIG
# ============================================================

TOP_K = 8

FINAL_CONTEXT_CHUNKS = 5


# ============================================================
# EMBEDDING MODEL
# ============================================================

print("\nLoading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,

    model_kwargs={
        "device": "cuda"
        if torch.cuda.is_available()
        else "cpu"
    },

    encode_kwargs={
        "normalize_embeddings": True
    }
)


# ============================================================
# CHROMA VECTOR DATABASE
# ============================================================

print("\nLoading ChromaDB...")

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,

    embedding_function=embeddings,

    persist_directory=str(
        CHROMA_DIRECTORY
    ),

    collection_metadata={
        "hnsw:space": "cosine"
    }
)


# ============================================================
# LOAD LLAMA 3. 8B 4-BIT
# ============================================================

print("\nLoading LLM_MODEL = Llama-3.2-1B...")

quantization_config = BitsAndBytesConfig(

    load_in_4bit=True,

    bnb_4bit_quant_type="nf4",

    bnb_4bit_compute_dtype=(
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    ),

    bnb_4bit_use_double_quant=True
)


tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL,
    token=HF_TOKEN
)


model = AutoModelForCausalLM.from_pretrained(

    LLM_MODEL,
    token=HF_TOKEN,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=(
        torch.float16
        if torch.cuda.is_available()
        else torch.float32
    ),

    low_cpu_mem_usage=True
)


# ============================================================
# TEXT GENERATION PIPELINE
# ============================================================

generator = pipeline(

    "text-generation",

    model=model,

    tokenizer=tokenizer,

    max_new_tokens=400,

    do_sample=False,

    temperature=None,

    top_p=None,

    return_full_text=False
)


# ============================================================
# INGESTION
# ============================================================

def ingest_documents() -> Dict[str, Any]:

    start_time = time.perf_counter()

    pdf_files = sorted(
        PDF_DIRECTORY.glob("*.pdf")
    )

    # --------------------------------------------------------
    # Validate minimum PDF requirement
    # --------------------------------------------------------

    if len(pdf_files) < 10:

        raise ValueError(
            f"At least 10 PDFs are required. "
            f"Found {len(pdf_files)}."
        )

    # --------------------------------------------------------
    # Process PDFs
    # --------------------------------------------------------

    chunks = process_directory(
        PDF_DIRECTORY
    )

    if not chunks:

        raise ValueError(
            "No chunks were generated."
        )

    # --------------------------------------------------------
    # Generate deterministic IDs
    # --------------------------------------------------------

    ids = [
        chunk.metadata["chunk_id"]
        for chunk in chunks
    ]

    print(
        f"\nAdding {len(chunks)} chunks "
        f"to ChromaDB..."
    )

    vectorstore.add_documents(
        documents=chunks,
        ids=ids
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    return {

        "status": "success",

        "pdf_count": len(pdf_files),

        "chunks_added": len(chunks),

        "ingestion_time_seconds": round(
            elapsed,
            2
        )
    }


# ============================================================
# LEXICAL SCORE
# ============================================================

def lexical_score(
    query: str,
    text: str
) -> float:

    query_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            query.lower()
        )
    )

    text_words = set(
        re.findall(
            r"\b[a-zA-Z0-9]+\b",
            text.lower()
        )
    )

    if not query_words:

        return 0.0

    overlap = (
        query_words.intersection(
            text_words
        )
    )

    return (
        len(overlap)
        /
        len(query_words)
    )


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    query: str,
    k: int = TOP_K
):

    start = time.perf_counter()

    results = (
        vectorstore
        .similarity_search_with_score(
            query,
            k=k
        )
    )

    retrieved = []

    for document, score in results:

        lexical = lexical_score(
            query,
            document.page_content
        )

        combined_score = (

            (
                1.0
                /
                (1.0 + float(score))
            )
            * 0.85

            +

            lexical * 0.15
        )

        retrieved.append({

            "document": document,

            "vector_score":
                float(score),

            "lexical_score":
                lexical,

            "combined_score":
                combined_score
        })

    retrieved.sort(
        key=lambda x:
            x["combined_score"],
        reverse=True
    )

    retrieval_time = (
        time.perf_counter()
        - start
    )

    return (
        retrieved,
        retrieval_time
    )


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(
    retrieved
):

    selected = retrieved[
        :FINAL_CONTEXT_CHUNKS
    ]

    context_parts = []

    for index, item in enumerate(
        selected
    ):

        document = item["document"]

        filename = (
            document.metadata
            .get(
                "filename",
                "Unknown"
            )
        )

        page = (
            document.metadata
            .get(
                "page_number",
                "Unknown"
            )
        )

        context_parts.append(

            f"""
SOURCE [{index + 1}]
File: {filename}
Page: {page}

Content:
{document.page_content}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    query: str,
    retrieved
):

    context = build_context(
        retrieved
    )

    messages = [

        {
            "role": "system",

            "content": """
You are a private-document
Retrieval-Augmented Generation assistant.

Your job is to answer questions
using ONLY the supplied document context.

Rules:

1. Do not use outside knowledge.

2. Do not invent facts.

3. If the answer cannot be found
   in the supplied context, say:

   "I could not find this information
   in the provided documents."

4. Give a clear and concise answer.

5. Cite factual statements using
   the supplied source numbers.

6. Citation format must be:

   [Source 1, Page 25]

7. Never invent a PDF filename
   or page number.

8. Only cite a source if it actually
   supports the statement.
"""
        },

        {
            "role": "user",

            "content": f"""
DOCUMENT CONTEXT:

{context}


USER QUESTION:

{query}


Answer using only the document context.
"""
        }
    ]


    # Llama 3.1 chat template
    prompt = tokenizer.apply_chat_template(

        messages,

        tokenize=False,

        add_generation_prompt=True
    )


    output = generator(
        prompt
    )


    answer = output[0]["generated_text"]

    return answer.strip()


# ============================================================
# ASK QUESTION
# ============================================================

def ask_question(
    query: str
) -> Dict[str, Any]:

    total_start = (
        time.perf_counter()
    )


    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    retrieved, retrieval_time = (
        retrieve(query)
    )


    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    generation_start = (
        time.perf_counter()
    )

    answer = generate_answer(
        query,
        retrieved
    )

    generation_time = (
        time.perf_counter()
        -
        generation_start
    )


    total_time = (
        time.perf_counter()
        -
        total_start
    )


    # --------------------------------------------------------
    # Source metadata
    # --------------------------------------------------------

    sources = []


    for rank, item in enumerate(

        retrieved[
            :FINAL_CONTEXT_CHUNKS
        ],

        start=1
    ):

        document = (
            item["document"]
        )


        sources.append({

            "rank": rank,

            "filename":
                document.metadata.get(
                    "filename"
                ),

            "page":
                document.metadata.get(
                    "page_number"
                ),

            "score":
                round(
                    item[
                        "combined_score"
                    ],
                    4
                ),

            "vector_score":
                round(
                    item[
                        "vector_score"
                    ],
                    4
                ),

            "preview":
                document.page_content[
                    :500
                ]
        })


    return {

        "answer": answer,

        "sources": sources,

        "retrieval_time_seconds":
            round(
                retrieval_time,
                3
            ),

        "generation_time_seconds":
            round(
                generation_time,
                3
            ),

        "total_latency_seconds":
            round(
                total_time,
                3
            )
    }


# ============================================================
# DATABASE STATS
# ============================================================

def get_collection_stats():

    try:

        collection = (
            vectorstore._collection
        )

        count = collection.count()

        return {

            "collection":
                COLLECTION_NAME,

            "embedding_model":
                EMBEDDING_MODEL,

            "llm_model":
                LLM_MODEL,

            "quantization":
                "4-bit NF4",

            "chunks":
                count,

            "pdf_directory":
                str(
                    PDF_DIRECTORY
                )
        }

    except Exception as e:

        return {
            "error": str(e)
        }