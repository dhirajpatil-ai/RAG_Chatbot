# Private PDF RAG Chatbot

A production-oriented **Retrieval-Augmented Generation (RAG) chatbot** for querying a large private PDF knowledge base containing native text, scanned documents, tables, and images.

The system extracts and preprocesses PDF content, performs OCR when required, creates semantic chunks, generates embeddings using an open-source embedding model, stores vectors in ChromaDB, retrieves the most relevant information, and generates grounded answers using **Llama-3.2-1B-Instruct** with source/page citations.

---

## 🚀 Features

* Supports large private PDF collections
* Designed for 10+ PDFs with 200+ pages per PDF
* Native PDF text extraction
* OCR support for scanned PDF pages
* Structured PDF processing using Unstructured
* Text cleaning and normalization
* Header/footer normalization
* Recursive semantic chunking
* Configurable chunk size and overlap
* Page-level metadata
* Bounding-box metadata where available
* Deterministic chunk IDs using SHA-256
* Open embedding model: `BAAI/bge-small-en-v1.5`
* Persistent ChromaDB vector database
* HNSW approximate nearest-neighbor search
* Cosine similarity
* Top-K semantic retrieval
* Lightweight lexical + semantic reranking
* Local Llama 3.2 1B Instruct generation
* 4-bit model quantization using bitsandbytes
* Grounded RAG prompting
* Source/page citations
* Hallucination reduction through context-only generation
* FastAPI backend
* Simple browser-based UI
* PDF upload and ingestion
* Vector database statistics
* Fully local inference after required models/dependencies are installed

---

## 🏗️ Architecture

```text
                         ┌───────────────────────┐
                         │      Browser UI       │
                         │   HTML/CSS/JavaScript │
                         └───────────┬───────────┘
                                     │
                                     ▼
                         ┌───────────────────────┐
                         │        FastAPI        │
                         │        app.py         │
                         └───────────┬───────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
                    ▼                                 ▼
          ┌───────────────────┐             ┌───────────────────┐
          │ PDF Ingestion     │             │ Question Answering│
          └─────────┬─────────┘             └─────────┬─────────┘
                    │                                 │
                    ▼                                 ▼
          ┌───────────────────┐             ┌───────────────────┐
          │   Unstructured    │             │ Query Embedding   │
          │ PDF Processing    │             │ BGE-small         │
          └─────────┬─────────┘             └─────────┬─────────┘
                    │                                 │
                    ▼                                 ▼
          ┌───────────────────┐             ┌───────────────────┐
          │ OCR / Extraction  │             │    ChromaDB       │
          │ Cleaning          │             │ HNSW + Cosine     │
          │ Chunking          │             └─────────┬─────────┘
          └─────────┬─────────┘                       │
                    │                                 ▼
                    ▼                       ┌───────────────────┐
          ┌───────────────────┐             │ Top-K Retrieval   │
          │ BGE Embeddings    │             │ + Reranking       │
          └─────────┬─────────┘             └─────────┬─────────┘
                    │                                 │
                    ▼                                 ▼
          ┌───────────────────┐             ┌───────────────────┐
          │     ChromaDB      │             │ Llama 3.2 1B      │
          │ Persistent Vector │             │ Instruct 4-bit    │
          │      Store        │             └─────────┬─────────┘
          └───────────────────┘                       │
                                                     ▼
                                          ┌─────────────────────┐
                                          │ Grounded Answer +   │
                                          │ Source/Page Citation│
                                          └─────────────────────┘
```

---

## 🔄 RAG Pipeline

### Document Ingestion

```text
PDF
 │
 ▼
Unstructured PDF Processing
 │
 ├── Native Text Extraction
 │
 ├── OCR for Scanned Content
 │
 └── Table/Layout Processing
 │
 ▼
Text Cleaning
 │
 ▼
Header/Footer Normalization
 │
 ▼
Recursive Chunking
 │
 ▼
Metadata Generation
 │
 ▼
BGE Embeddings
 │
 ▼
ChromaDB
 │
 ▼
HNSW Index
```

### Question Answering

```text
User Question
 │
 ▼
BGE Embedding
 │
 ▼
ChromaDB Similarity Search
 │
 ▼
Top-K Candidates
 │
 ▼
Semantic + Lexical Ranking
 │
 ▼
Relevant Context
 │
 ▼
Llama 3.2 1B Instruct
 │
 ▼
Grounded Answer
 │
 ▼
Source/Page Citations
```

---

## 🧰 Technology Stack

| Component              | Technology                               |
| ---------------------- | ---------------------------------------- |
| Programming Language   | Python                                   |
| API Framework          | FastAPI                                  |
| PDF Processing         | Unstructured                             |
| PDF Extraction         | PyMuPDF / PDF processing dependencies    |
| OCR                    | Tesseract / Unstructured OCR pipeline    |
| Text Splitting         | LangChain RecursiveCharacterTextSplitter |
| Embedding Model        | `BAAI/bge-small-en-v1.5`                 |
| Vector Database        | ChromaDB                                 |
| ANN Index              | HNSW                                     |
| Similarity             | Cosine                                   |
| LLM                    | `meta-llama/Llama-3.2-1B-Instruct`       |
| Quantization           | bitsandbytes 4-bit                       |
| Model Framework        | Hugging Face Transformers                |
| Environment Management | python-dotenv                            |
| Containerization       | Docker-ready architecture                |
| Frontend               | Simple HTML/CSS/JavaScript               |

---

## 📁 Project Structure

```text
RAG_Chatbot/
│
├── app.py
├── rag_pipeline.py
├── extraction_preprocessing.py
├── requirements.txt
├── .env
├── .gitignore
│
├── data/
│   └── pdfs/
│       ├── document1.pdf
│       ├── document2.pdf
│       └── ...
│
└── chroma_db/
```

### File Responsibilities

#### `extraction_preprocessing.py`

Responsible for:

* PDF extraction
* OCR-oriented processing
* Text cleaning
* Header/footer normalization
* Chunking
* Metadata generation
* Deterministic chunk IDs

#### `rag_pipeline.py`

Responsible for:

* Embedding model
* ChromaDB
* Vector indexing
* Retrieval
* Semantic/lexical ranking
* Llama model loading
* Prompt construction
* Answer generation
* Source citation handling
* Document ingestion orchestration

#### `app.py`

Responsible for:

* FastAPI application
* PDF upload endpoint
* Ingestion endpoint
* Chat endpoint
* Statistics endpoint
* Simple web UI

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd RAG_Chatbot
```

### 2. Create a virtual environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

If required by your Unstructured PDF configuration:

```bash
pip install "unstructured[pdf]"
pip install unstructured-inference
```

---

## 🔐 Hugging Face Authentication

The project uses:

```text
meta-llama/Llama-3.2-1B-Instruct
```

Create a Hugging Face access token with access to the model.

Create a `.env` file in the project root:

```text
HF_TOKEN=hf_your_token_here
```

The application reads the token using `python-dotenv`.

### ⚠️ Security

Never commit your `.env` file.

Your `.gitignore` should contain:

```text
.env
.venv/
__pycache__/
*.pyc
chroma_db/
```

---

## 📚 Add PDF Documents

Place your private PDF corpus inside:

```text
data/pdfs/
```

Example:

```text
data/pdfs/
├── document1.pdf
├── document2.pdf
├── document3.pdf
├── document4.pdf
├── document5.pdf
├── document6.pdf
├── document7.pdf
├── document8.pdf
├── document9.pdf
└── document10.pdf
```

The intended evaluation corpus contains at least:

```text
10 PDFs
200+ pages per PDF
```

---

## ▶️ Run the Application

Start FastAPI:

```bash
uvicorn app:app
```

For development with automatic reload:

```bash
uvicorn app:app --reload
```

Open the application:

```text
http://localhost:8000
```

or:

```text
http://127.0.0.1:8000
```

---

## 🧠 Models

### Embedding Model

```text
BAAI/bge-small-en-v1.5
```

The embedding model converts both documents and user questions into vector representations.

```text
Document Chunk
      ↓
BGE-small
      ↓
Embedding Vector
```

The same process is applied to the user query.

```text
User Question
      ↓
BGE-small
      ↓
Query Vector
```

The query vector is then compared against vectors stored in ChromaDB.

---

### Generation Model

```text
meta-llama/Llama-3.2-1B-Instruct
```

The model generates the final response from the retrieved context.

```text
Question
   +
Retrieved Context
   ↓
Llama 3.2 1B
   ↓
Answer
```

The model is loaded using 4-bit quantization:

```text
4-bit NF4
```

This reduces model memory requirements compared with loading the full-precision weights.

---

## 🗄️ Vector Database

The project uses:

```text
ChromaDB
```

with persistent storage.

The vectors contain both:

```text
Embedding
+
Metadata
```

Example metadata:

```json
{
  "filename": "document1.pdf",
  "page_number": 42,
  "pdf_id": "..."
}
```

This metadata allows the application to provide document provenance.

---

## 🔎 Retrieval

The retrieval process initially retrieves the top candidates using vector similarity.

Example:

```text
User Query
    ↓
Embedding
    ↓
ChromaDB
    ↓
Top 8 candidates
    ↓
Semantic + lexical scoring
    ↓
Top relevant context
```

The final context is then provided to the LLM.

---

## 📌 Chunking Strategy

The default chunking configuration is approximately:

```text
Chunk size:     800 tokens
Overlap:        160 tokens
Overlap ratio:  20%
```

The project uses LangChain's:

```text
RecursiveCharacterTextSplitter
```

The splitter attempts to preserve natural document boundaries such as:

```text
Paragraph
   ↓
Line
   ↓
Sentence
   ↓
Word
```

This helps maintain semantic coherence.

---

## 🧾 Provenance and Citations

Each chunk contains document metadata.

Example:

```text
Filename: document1.pdf
Page: 42
```

The LLM is instructed to cite retrieved information using:

```text
[Source 1, Page 42]
```

Example response:

```text
RAG combines information retrieval with language generation.

[Source 1, Page 42]
```

The model is also instructed not to invent page numbers or filenames.

---

## 🛡️ Hallucination Control

The RAG prompt instructs the LLM to:

1. Answer using only retrieved context.
2. Avoid unsupported information.
3. Avoid inventing citations.
4. State when the requested information is not available in the retrieved documents.
5. Include source/page citations.

Conceptually:

```text
             Retrieved Evidence
                     │
                     ▼
User Question ──► Llama 3.2
                     │
                     ▼
             Grounded Response
                     │
                     ▼
              Source Citation
```

---

## 📊 Evaluation Metrics

The project can be evaluated using the following metrics.

### Retrieval Recall @ K

Measures whether the relevant document/chunk appears within the top K retrieved results.

```text
R@K
```

---

### Mean Reciprocal Rank

Measures the position of the first relevant result.

```text
MRR
```

Higher values indicate that relevant results appear earlier in the ranking.

---

### Citation Accuracy

Measures whether generated citations correctly point to the document/page containing the supporting information.

---

### Hallucination Rate

Measures how frequently the generated response contains unsupported information.

---

### Latency

Measure:

```text
Retrieval latency
Generation latency
End-to-end latency
```

The target end-to-end latency is approximately:

```text
2–5 seconds
```

depending on hardware, document size, retrieval configuration, and local LLM performance.

---

### p95 Latency

The 95th percentile latency is useful for understanding worst-case user experience.

For example:

```text
Average latency = 2.5 sec
p95 latency     = 4.2 sec
```

means approximately 95% of requests complete within 4.2 seconds.

---

## ⚡ Performance Considerations

Performance depends heavily on:

* GPU model
* GPU VRAM
* CPU
* RAM
* PDF complexity
* OCR requirements
* Number of chunks
* Retrieval K
* LLM generation length

### Ingestion

OCR-heavy PDFs can take considerably longer to process than native-text PDFs.

### Querying

Querying generally follows:

```text
Question
 ↓
Embedding
 ↓
Vector Search
 ↓
Context Construction
 ↓
LLM Generation
```

The LLM generation stage can dominate total latency when running locally.

---

## 🔒 Privacy

The architecture is designed for private document collections.

After the required model files and dependencies are available locally, the RAG pipeline can perform:

```text
PDF Processing
      ↓
Embeddings
      ↓
Vector Search
      ↓
LLM Generation
```

locally without sending document content to a hosted LLM API.

This makes the architecture suitable for scenarios where documents should remain within the local environment.

---

## 🧪 Example Query

After ingesting the documents, ask:

```text
What are the main components of the architecture described in the documents?
```

The pipeline performs:

```text
Question
   ↓
BGE embedding
   ↓
ChromaDB search
   ↓
Retrieve relevant chunks
   ↓
Rank candidates
   ↓
Construct context
   ↓
Llama 3.2 1B
   ↓
Answer with citations
```

---

## 🔌 API Endpoints

### Upload PDFs

```text
POST /upload
```

Uploads PDF files to the document directory.

---

### Ingest Documents

```text
POST /ingest
```

Processes PDFs and stores their embeddings in ChromaDB.

---

### Ask a Question

```text
POST /chat
```

Example:

```json
{
  "question": "What is retrieval augmented generation?"
}
```

---

### Database Statistics

```text
GET /stats
```

Returns information about the current vector database.

---

## 🧩 Design Principles

The project follows several RAG design principles:

### Separation of Concerns

```text
PDF Processing
      ↓
RAG Pipeline
      ↓
API/UI
```

Each major responsibility is isolated into a separate file.

### Deterministic Processing

The same PDF and chunk configuration should produce reproducible chunk IDs.

### Precomputed Embeddings

Document embeddings are generated during ingestion rather than every time a user asks a question.

### Metadata Preservation

Page-level metadata is preserved for provenance and citation.

### Local Generation

The LLM runs locally using Hugging Face Transformers.

---

## 🔮 Future Improvements

Potential improvements include:

* Multilingual OCR
* Better table extraction
* Image understanding
* Multimodal embeddings
* Hybrid BM25 + vector retrieval
* Cross-encoder reranking
* Query expansion
* Parent-child document retrieval
* Context compression
* Streaming LLM responses
* Redis caching
* Batch embedding
* Async ingestion
* Background PDF processing
* Evaluation dashboard
* Retrieval visualization
* Page-level source highlighting
* Docker deployment
* Kubernetes deployment
* Prometheus/Grafana monitoring
* Automated RAG evaluation
* RAGAS-based evaluation

---

## ⚠️ Notes

### Llama Model

`meta-llama/Llama-3.2-1B-Instruct` requires appropriate access on Hugging Face.

You must accept the applicable model terms and provide a valid Hugging Face token.

### Hardware

Running an 8B parameter model locally can require significant RAM/VRAM even with 4-bit quantization.

Actual memory usage depends on:

* Quantization configuration
* GPU architecture
* Transformers/PyTorch version
* KV cache
* Context length
* Generation settings

### First Startup

The first application startup can take significantly longer because Hugging Face may need to download the model files.

Subsequent startups normally use the locally cached model files, although the model still needs to be loaded into memory.

---

## 📜 License

Add the appropriate license for your own source code and verify the licenses/terms of all third-party models and libraries used by this project.

---

## 👨‍💻 Project Summary

This project demonstrates an end-to-end local RAG architecture capable of processing a large private PDF corpus and answering natural-language questions using semantic retrieval and a locally hosted LLM.

The main technologies are:

```text
Python
FastAPI
LangChain
Unstructured
OCR
BGE Embeddings
ChromaDB
HNSW
Hugging Face Transformers
Llama 3.2 1B Instruct
bitsandbytes
```

The core objective is to provide **grounded, traceable answers from private PDF documents while keeping the retrieval and generation pipeline local**.
