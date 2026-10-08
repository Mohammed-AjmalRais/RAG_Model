# 🧠 Enterprise Hybrid RAG (Retrieval-Augmented Generation)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39%2B-FF4B4B.svg)](https://streamlit.io/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorStore-orange.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced, production-grade **Retrieval-Augmented Generation (RAG)** platform designed for querying multi-format knowledge bases with high precision, verified citations, and zero hallucinations.

---

## 🌟 What Was Improved? (Before vs. After)

| Feature | Simple Toy RAG (Previous) | Enterprise Hybrid RAG (Current) |
| :--- | :--- | :--- |
| **Document Support** | PDF only (crashed if PyMuPDF was missing) | **Universal**: PDF (`pypdf` + `fitz` fallback), TXT, Markdown (`.md`), CSV |
| **Retrieval Strategy** | Naive Vector-only Cosine Similarity | **Hybrid Search**: Dense Vector Embeddings + Sparse BM25 / TF-IDF Lexical Index |
| **Rank Fusion** | None (pure single-vector distances) | **Reciprocal Rank Fusion (RRF)** with rank constant $k=60$ for superior recall |
| **LLM Support** | Hardcoded to Anthropic Claude only | **Multi-Provider**: Google Gemini, Groq (Llama 3.3), Claude, OpenAI, Local Ollama |
| **Zero-Key Execution** | Crashed immediately without `ANTHROPIC_API_KEY` | **Offline Extractive Fallback Mode**: Generates cited answers out-of-the-box |
| **Streaming UI** | Frozen spinner waiting for full generation | **Real-Time Token Streaming**: SSE streaming with live typewriter typing effect |
| **Knowledge Base Scope** | Single document only, isolated in memory | **Global vs Document Target**: Search across entire knowledge library or single file |
| **Persistence** | In-memory python dict (lost on server restart) | **Persistent Disk Catalog** (`data/metadata.json` + ChromaDB persistent vector store) |
| **Hallucination Guard** | Basic prompt | **Strict Grounding Guardrails**: Page citations, confidence scores, source previews |
| **Verification Suite** | None | Automated end-to-end test suite (`python test.py`) |

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Multi-Format Ingestion Pipeline"]
        A["Document Upload (PDF, TXT, MD, CSV)"] --> B["Text Extraction & Normalization"]
        B --> C["Recursive Character Splitting<br/>(800 chars, 150 overlap)"]
        C --> D["Dense Vector Embeddings"]
        C --> E["Sparse Lexical Indexing (BM25 / TF-IDF)"]
        D --> F[("Persistent ChromaDB<br/>Vector Store")]
        E --> G[("In-Memory & Disk<br/>Sparse Index")]
        B --> H[("Metadata Catalog<br/>(metadata.json)")]
    end

    subgraph Retrieval ["2. Hybrid Retrieval Engine"]
        Q["User Query"] --> QD["Dense Vector Query"]
        Q --> QS["Sparse Lexical Query"]
        QD --> F
        QS --> G
        F --> R1["Top Dense Candidates"]
        G --> R2["Top Lexical Candidates"]
        R1 & R2 --> RRF["Reciprocal Rank Fusion (RRF)<br/>Score = 1/(60 + r_dense) + 1/(60 + r_sparse)"]
        RRF --> TopK["Re-Ranked Top-K Context Excerpts"]
    end

    subgraph Generation ["3. Multi-Provider Generation & Streaming"]
        TopK --> Prompt["Grounded Prompt Builder<br/>(Strict Anti-Hallucination & Citations)"]
        Prompt --> LLM{"LLM Provider Router"}
        LLM -->|"GEMINI_API_KEY"| Gem["Google Gemini 2.5 Flash"]
        LLM -->|"GROQ_API_KEY"| Groq["Groq Llama 3.3 70B"]
        LLM -->|"ANTHROPIC_API_KEY"| Claude["Claude 3.5 Sonnet"]
        LLM -->|"OPENAI_API_KEY / Local"| OAI["OpenAI / Ollama"]
        LLM -->|"No Key Provided"| Offline["Offline Extractive Engine"]
        Gem & Groq & Claude & OAI & Offline --> Stream["Server-Sent Events (SSE) Stream"]
        Stream --> UI["Streamlit Interactive Chat & Citation Drawer"]
    end
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python **3.10+**

### 2. Installation
```bash
python -m pip install -r requirements.txt
```

### 3. Environment Configuration
Copy the `.env.example` file to `.env`:
```bash
cp .env.example .env
```

Add your preferred API key (Gemini, Groq, Claude, or OpenAI), or run in offline extractive mode with zero keys.

### 4. Running the Stack
- **Terminal 1 (Backend API):**
  ```bash
  python main.py
  ```
- **Terminal 2 (Streamlit UI):**
  ```bash
  streamlit run app.py
  ```
- **Automated Verification:**
  ```bash
  python ../test.py
  ```

---

## 📁 Directory Layout

```
files/
├── rag_pipeline.py       # Ingestion, chunking, ChromaDB, BM25, RRF, LLM routers
├── main.py               # FastAPI backend: REST API, SSE streaming, document CRUD
├── app.py                # Streamlit UI: Chat interface, Knowledge Base explorer, citations
├── requirements.txt      # Clean project dependencies
├── .env.example          # Backend environment template
├── README.md             # Complete documentation
└── data/                 # Auto-generated persistent storage
    ├── uploads/          # Raw document files
    ├── chroma_db/        # Persistent ChromaDB vector collections
    └── metadata.json     # Persistent document catalog
```
