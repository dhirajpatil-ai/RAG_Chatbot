# app.py

from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException
)

from fastapi.responses import HTMLResponse

from pydantic import BaseModel

from rag_pipeline import (
    ingest_documents,
    ask_question,
    get_collection_stats,
    PDF_DIRECTORY
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Open Source PDF RAG Chatbot",
    version="1.0.0"
)


PDF_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    question: str


# ============================================================
# UPLOAD PDFs
# ============================================================

@app.post("/upload")
async def upload_pdfs(
    files: List[UploadFile] = File(...)
):

    uploaded = []

    for file in files:

        if not file.filename.lower().endswith(
            ".pdf"
        ):
            continue

        destination = (
            PDF_DIRECTORY /
            Path(file.filename).name
        )

        with open(
            destination,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

        uploaded.append(
            file.filename
        )

    if not uploaded:

        raise HTTPException(
            status_code=400,
            detail="No PDF files were uploaded."
        )

    return {
        "status": "success",
        "uploaded_files": uploaded,
        "count": len(uploaded)
    }


# ============================================================
# INGEST
# ============================================================

@app.post("/ingest")
def ingest():

    try:

        result = ingest_documents()

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# CHAT
# ============================================================

@app.post("/chat")
def chat(
    request: ChatRequest
):

    if not request.question.strip():

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    try:

        return ask_question(
            request.question
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# ============================================================
# STATS
# ============================================================

@app.get("/stats")
def stats():

    return get_collection_stats()


# ============================================================
# SIMPLE UI
# ============================================================

HTML = """
<!DOCTYPE html>

<html>

<head>

<title>PDF RAG Chatbot</title>

<style>

body {
    font-family: Arial, sans-serif;
    background: #f4f6f8;
    margin: 0;
    padding: 30px;
}

.container {
    max-width: 1100px;
    margin: auto;
}

h1 {
    margin-bottom: 5px;
}

.subtitle {
    color: #666;
    margin-bottom: 25px;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}

button {
    padding: 10px 18px;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    background: #222;
    color: white;
}

button:hover {
    opacity: 0.85;
}

textarea {
    width: 100%;
    min-height: 90px;
    padding: 12px;
    box-sizing: border-box;
    border: 1px solid #ccc;
    border-radius: 6px;
    resize: vertical;
}

.answer {
    white-space: pre-wrap;
    line-height: 1.6;
    background: #fafafa;
    padding: 15px;
    border-radius: 6px;
}

.source {
    border-left: 4px solid #333;
    padding: 10px 15px;
    margin-top: 10px;
    background: #fafafa;
}

.source-title {
    font-weight: bold;
}

.preview {
    color: #555;
    margin-top: 8px;
}

.metrics {
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
}

.metric {
    padding: 10px;
    background: #f1f1f1;
    border-radius: 6px;
}

#status {
    margin-top: 10px;
    color: #555;
}

</style>

</head>

<body>

<div class="container">

<h1>📚 Open Source PDF RAG Chatbot</h1>

<div class="subtitle">
Private PDF Knowledge Base • Semantic Search • Local LLM
</div>


<div class="card">

<h2>1. Upload PDFs</h2>

<input
    type="file"
    id="pdfFiles"
    multiple
    accept=".pdf"
/>

<br><br>

<button onclick="uploadPDFs()">
    Upload PDFs
</button>

<div id="uploadStatus"></div>

</div>


<div class="card">

<h2>2. Build Vector Database</h2>

<p>
PDF → Unstructured/OCR → Chunking →
Embeddings → ChromaDB
</p>

<button onclick="ingest()">
    Start Ingestion
</button>

<div id="ingestStatus"></div>

</div>


<div class="card">

<h2>3. Ask Question</h2>

<textarea
    id="question"
    placeholder="Ask something about your PDF documents..."
></textarea>

<br><br>

<button onclick="askQuestion()">
    Ask
</button>

</div>


<div class="card">

<h2>Answer</h2>

<div id="answer" class="answer">
No question asked yet.
</div>

<div id="metrics" class="metrics"></div>

</div>


<div class="card">

<h2>Retrieved Chunks</h2>

<div id="sources">
No retrieval performed yet.
</div>

</div>

</div>


<script>

async function uploadPDFs() {

    const files =
        document.getElementById(
            "pdfFiles"
        ).files;

    if (!files.length) {

        alert("Select PDF files first.");

        return;
    }

    const formData =
        new FormData();

    for (const file of files) {

        formData.append(
            "files",
            file
        );
    }

    document.getElementById(
        "uploadStatus"
    ).innerText =
        "Uploading...";

    const response =
        await fetch(
            "/upload",
            {
                method: "POST",
                body: formData
            }
        );

    const data =
        await response.json();

    document.getElementById(
        "uploadStatus"
    ).innerText =
        JSON.stringify(
            data,
            null,
            2
        );
}


async function ingest() {

    document.getElementById(
        "ingestStatus"
    ).innerText =
        "Ingestion started. Large PDFs may take time...";

    const response =
        await fetch(
            "/ingest",
            {
                method: "POST"
            }
        );

    const data =
        await response.json();

    document.getElementById(
        "ingestStatus"
    ).innerText =
        JSON.stringify(
            data,
            null,
            2
        );
}


async function askQuestion() {

    const question =
        document.getElementById(
            "question"
        ).value.trim();

    if (!question) {

        alert("Enter a question.");

        return;
    }

    document.getElementById(
        "answer"
    ).innerText =
        "Thinking...";

    document.getElementById(
        "sources"
    ).innerHTML =
        "Retrieving...";

    const start =
        performance.now();

    const response =
        await fetch(
            "/chat",
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({
                    question: question
                })
            }
        );

    const data =
        await response.json();

    const end =
        performance.now();

    if (!response.ok) {

        document.getElementById(
            "answer"
        ).innerText =
            data.detail ||
            "Something went wrong.";

        return;
    }

    document.getElementById(
        "answer"
    ).innerText =
        data.answer;


    document.getElementById(
        "metrics"
    ).innerHTML = `

        <div class="metric">
            Retrieval:
            ${data.retrieval_time_seconds}s
        </div>

        <div class="metric">
            Generation:
            ${data.generation_time_seconds}s
        </div>

        <div class="metric">
            Total:
            ${data.total_latency_seconds}s
        </div>

        <div class="metric">
            Browser:
            ${((end-start)/1000).toFixed(3)}s
        </div>
    `;


    let html = "";

    data.sources.forEach(
        function(source) {

            html += `

            <div class="source">

                <div class="source-title">

                    #${source.rank}
                    ${source.filename}

                    — Page
                    ${source.page}

                </div>

                <div>
                    Combined Score:
                    ${source.score}
                </div>

                <div>
                    Vector Score:
                    ${source.vector_score}
                </div>

                <div class="preview">
                    ${escapeHtml(
                        source.preview
                    )}
                </div>

            </div>
            `;
        }
    );

    document.getElementById(
        "sources"
    ).innerHTML = html;
}


function escapeHtml(text) {

    const div =
        document.createElement(
            "div"
        );

    div.textContent = text;

    return div.innerHTML;
}

</script>

</body>

</html>
"""


# ============================================================
# HOME
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
def home():

    return HTML