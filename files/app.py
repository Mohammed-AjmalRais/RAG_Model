import os
import json
import time
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Enterprise Hybrid RAG Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished modern aesthetic
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 9999px;
        background-color: #EEF2FF;
        color: #4F46E5;
        margin-right: 0.5rem;
    }
    .badge-success {
        background-color: #ECFDF5;
        color: #059669;
    }
    .source-card {
        border-left: 3px solid #3B82F6;
        padding: 0.6rem 0.8rem;
        background-color: #F8FAFC;
        border-radius: 0 6px 6px 0;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    .stChatMessage {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_doc_id" not in st.session_state:
    st.session_state.active_doc_id = "all"
if "documents" not in st.session_state:
    st.session_state.documents = []


def fetch_documents():
    try:
        resp = requests.get(f"{API_BASE}/documents", timeout=5)
        if resp.status_code == 200:
            st.session_state.documents = resp.json()
        else:
            st.session_state.documents = []
    except Exception:
        st.session_state.documents = []


fetch_documents()


# ---------------------------------------------------------------------------
# Sidebar: Knowledge Base & Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📚 Knowledge Base")
    st.caption("Upload documents to build your vector and lexical index.")

    uploaded_files = st.file_uploader(
        "Upload files (PDF, TXT, MD, CSV)",
        type=["pdf", "txt", "md", "csv"],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("⚡ Index Documents", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file_obj in enumerate(uploaded_files):
            status_text.caption(f"Indexing {file_obj.name}...")
            files = {"file": (file_obj.name, file_obj.getvalue(), file_obj.type or "application/octet-stream")}
            try:
                resp = requests.post(f"{API_BASE}/upload", files=files, timeout=60)
                if resp.status_code == 200:
                    st.toast(f"✅ Ingested: {file_obj.name}")
                else:
                    st.error(f"Failed to ingest {file_obj.name}: {resp.json().get('detail')}")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.empty()
        progress_bar.empty()
        fetch_documents()
        st.rerun()

    st.divider()

    # Document Library
    st.markdown("#### 📂 Document Library")
    if st.session_state.documents:
        doc_options = {"🌐 Entire Knowledge Base (All Documents)": "all"}
        for d in st.session_state.documents:
            label = f"📄 {d['filename']} ({d['num_chunks']} chunks, {d['num_pages']} p.)"
            doc_options[label] = d["doc_id"]

        selected_label = st.selectbox(
            "Query Target:",
            options=list(doc_options.keys()),
            index=0
        )
        st.session_state.active_doc_id = doc_options[selected_label]

        # Document deletion expander
        with st.expander("Manage Stored Documents"):
            for d in st.session_state.documents:
                col1, col2 = st.columns([3, 1])
                col1.caption(f"**{d['filename']}** ({d['num_chunks']} chunks)")
                if col2.button("🗑️", key=f"del_{d['doc_id']}", help="Delete from vector store"):
                    requests.delete(f"{API_BASE}/documents/{d['doc_id']}")
                    fetch_documents()
                    st.rerun()
    else:
        st.info("No documents indexed yet. Upload a PDF or TXT above.")
        st.session_state.active_doc_id = "all"

    st.divider()

    # Retrieval & LLM Tuning
    st.markdown("#### ⚙️ Retrieval & Model Settings")
    search_mode = st.selectbox(
        "Search Strategy:",
        options=["hybrid", "dense", "sparse"],
        format_func=lambda x: {
            "hybrid": "Hybrid RRF (Dense + BM25)",
            "dense": "Dense Semantic Only",
            "sparse": "Sparse Keyword (BM25) Only"
        }[x],
        index=0
    )

    top_k = st.slider("Top Chunks (Top-K):", min_value=1, max_value=15, value=5)

    provider = st.selectbox(
        "LLM Engine:",
        options=["auto", "gemini", "groq", "claude", "openai", "offline"],
        format_func=lambda x: {
            "auto": "Auto-Detect from .env",
            "gemini": "Google Gemini (Fast)",
            "groq": "Groq Llama 3.3 (Ultra-Fast)",
            "claude": "Anthropic Claude (High Quality)",
            "openai": "OpenAI / Local Ollama",
            "offline": "Offline Extractive (Zero API Key)"
        }[x],
        index=0
    )

    with st.expander("🔑 Quick API Key Override"):
        custom_key = st.text_input("Enter API Key (Optional):", type="password")
        key_provider = st.selectbox("Key for:", ["GEMINI_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"])
        if custom_key and st.button("Apply Key to Session"):
            os.environ[key_provider] = custom_key
            st.success(f"{key_provider} applied!")

    if st.button("🧹 Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main Chat Area
# ---------------------------------------------------------------------------
col_title, col_stats = st.columns([3, 1])
with col_title:
    st.markdown('<div class="main-header">🧠 Enterprise Hybrid RAG Assistant</div>', unsafe_allow_html=True)
    target_name = "Entire Knowledge Base" if st.session_state.active_doc_id == "all" else "Single Document"
    st.markdown(
        f'<span class="badge badge-success">Target: {target_name}</span>'
        f'<span class="badge">Strategy: {search_mode.upper()}</span>'
        f'<span class="badge">Docs Indexed: {len(st.session_state.documents)}</span>',
        unsafe_allow_html=True
    )

with col_stats:
    if st.session_state.messages and st.button("📥 Export Chat (Markdown)"):
        chat_md = "# RAG Chat Export\n\n"
        for m in st.session_state.messages:
            chat_md += f"### {m['role'].capitalize()}:\n{m['content']}\n\n"
        st.download_button(
            label="Download Markdown",
            data=chat_md,
            file_name=f"rag_conversation_{int(time.time())}.md",
            mime="text/markdown"
        )

st.divider()

# Empty State Helper
if not st.session_state.messages:
    st.markdown("""
    ### Welcome to the Enhanced RAG System! 👋
    **Getting Started:**
    1. Upload your documents (PDF, TXT, MD, CSV) on the left sidebar.
    2. Select your retrieval mode (Hybrid Dense+BM25 is recommended for optimal accuracy).
    3. Type your question below to receive grounded answers with citations.
    """)
    if st.session_state.documents:
        st.caption("💡 Quick Prompts to try:")
        col1, col2, col3 = st.columns(3)
        if col1.button("📌 Summarize the main topics"):
            st.session_state.prompt_input = "Summarize the key findings and main topics in the documents."
        if col2.button("🔍 What are the key facts & figures?"):
            st.session_state.prompt_input = "Extract all key facts, numbers, dates, and figures mentioned."
        if col3.button("❓ What questions can this doc answer?"):
            st.session_state.prompt_input = "What are the 3 most important questions these documents answer?"

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📑 Sources & Citations ({len(msg['sources'])} excerpts | Confidence: {msg.get('confidence_score', 'N/A')}%)"):
                for s in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-card">
                        <strong>Source:</strong> {s.get('source')} | <strong>Page:</strong> {s.get('page', 1)} | <strong>Score:</strong> {s.get('score', 'N/A')}
                        <br>
                        <em>"{s.get('preview')}"</em>
                    </div>
                    """, unsafe_allow_html=True)

# Question Input
user_question = st.chat_input("Ask a question about your documents...") or getattr(st.session_state, "prompt_input", None)
if hasattr(st.session_state, "prompt_input"):
    del st.session_state.prompt_input

if user_question:
    if not st.session_state.documents:
        st.warning("⚠️ Please upload and index at least one document before asking questions.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # Assistant response container
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            sources = []
            confidence = 0.0

            payload = {
                "question": user_question,
                "doc_id": st.session_state.active_doc_id,
                "top_k": top_k,
                "search_mode": search_mode,
                "provider": provider if provider != "auto" else None,
            }

            try:
                # Use SSE stream endpoint for real-time typing effect
                with requests.post(f"{API_BASE}/ask/stream", json=payload, stream=True, timeout=60) as resp:
                    if resp.status_code == 200:
                        for line in resp.iter_lines():
                            if line:
                                line_str = line.decode("utf-8")
                                if line_str.startswith("data: "):
                                    event_data = json.loads(line_str[6:])
                                    if event_data.get("event") == "meta":
                                        sources = event_data.get("sources", [])
                                    elif event_data.get("event") == "token":
                                        full_response += event_data.get("token", "")
                                        response_container.markdown(full_response + "▌")
                                    elif event_data.get("event") == "done":
                                        break
                        response_container.markdown(full_response)
                    else:
                        # Fallback to standard /ask if streaming encounters issue
                        resp_fallback = requests.post(f"{API_BASE}/ask", json=payload, timeout=30)
                        if resp_fallback.status_code == 200:
                            data = resp_fallback.json()
                            full_response = data["answer"]
                            sources = data.get("sources", [])
                            confidence = data.get("confidence_score", 0.0)
                            response_container.markdown(full_response)
                        else:
                            full_response = f"Error: {resp_fallback.json().get('detail', 'Failed to retrieve answer')}"
                            response_container.error(full_response)
            except Exception as e:
                full_response = f"Connection error: {str(e)}. Make sure the FastAPI backend (`python main.py`) is running on port 8000."
                response_container.error(full_response)

            # Display citations expander
            if sources:
                with st.expander(f"📑 Sources & Citations ({len(sources)} excerpts)"):
                    for s in sources:
                        st.markdown(f"""
                        <div class="source-card">
                            <strong>Source:</strong> {s.get('source')} | <strong>Page:</strong> {s.get('page', 1)} | <strong>Score:</strong> {s.get('score', 'N/A')}
                            <br>
                            <em>"{s.get('preview')}"</em>
                        </div>
                        """, unsafe_allow_html=True)

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
                "confidence_score": confidence,
            })
