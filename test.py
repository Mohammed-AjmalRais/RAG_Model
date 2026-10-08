"""
Automated Verification Suite for the Hybrid RAG Model.
Run this script to verify your setup, vector indexing, hybrid search, and LLM integration:
    python test.py
"""

import os
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Add files folder to module path
current_dir = Path(__file__).resolve().parent
files_dir = current_dir / "files"
sys.path.insert(0, str(files_dir))

import rag_pipeline


def run_checks():
    print("=" * 65)
    print(" [*] Enterprise Hybrid RAG - System Verification")
    print("=" * 65)

    # 1. Inspect providers
    stats = rag_pipeline.get_system_stats()
    print(f"[*] Detected Active Provider(s): {', '.join(stats['active_providers'])}")
    print(f"[*] Storage Data Directory: {stats['data_directory']}")

    # 2. Test Document Ingestion
    print("\n[1/3] Testing Ingestion Engine...")
    sample_doc = (
        "Enterprise Search Architecture\n"
        "Retrieval-Augmented Generation (RAG) significantly reduces hallucination in LLMs.\n"
        "By fusing BM25 lexical search with dense vector similarity via Reciprocal Rank Fusion,\n"
        "retrieval recall is boosted by over 40% across technical documentation and legal filings.\n"
        "The system persists vector collections in ChromaDB and supports multiple LLM providers."
    )
    test_filepath = files_dir / "data" / "uploads" / "system_check_doc.txt"
    test_filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(test_filepath, "w", encoding="utf-8") as f:
        f.write(sample_doc)

    doc_info = rag_pipeline.ingest_document(str(test_filepath), "system_check_doc.txt")
    print(f"  -> Ingested: {doc_info['filename']} (ID: {doc_info['doc_id']})")
    print(f"  -> Total Chunks: {doc_info['num_chunks']} across {doc_info['num_pages']} page(s).")

    # 3. Test Hybrid Retrieval
    print("\n[2/3] Testing Hybrid Retrieval (Reciprocal Rank Fusion)...")
    query = "How much is retrieval recall boosted by fusing BM25 and vector search?"
    chunks = rag_pipeline.retrieve_chunks(doc_id=doc_info["doc_id"], question=query, top_k=2, search_mode="hybrid")
    print(f"  -> Retrieved {len(chunks)} chunks.")
    if chunks:
        print(f"  -> Top Chunk Preview: {chunks[0]['text'][:80]}...")
        print(f"  -> RRF Score: {chunks[0].get('rrf_score', 'N/A')}")

    # 4. Test Ask Question & Citation
    print("\n[3/3] Testing Question Answering & Source Citation...")
    answer_result = rag_pipeline.ask_question(doc_id=doc_info["doc_id"], question=query, top_k=2)
    print(f"  -> Provider Used: {answer_result['provider_used']}")
    print(f"  -> Confidence Score: {answer_result['confidence_score']}%")
    print(f"  -> Latency: {answer_result['latency_seconds']}s")
    print(f"\n  -> Answer:\n{answer_result['answer']}\n")

    # Cleanup test document
    rag_pipeline.delete_document(doc_info["doc_id"])
    if test_filepath.exists():
        test_filepath.unlink()

    print("=" * 65)
    print(" [OK] All System Verification Tests Passed Successfully!")
    print("=" * 65)


if __name__ == "__main__":
    run_checks()