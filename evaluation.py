"""
Evaluation script for the RAG pipeline.
Uploads sample docs, runs queries, computes Hit@5 and MRR.

Usage:
    python evaluation.py --base-url http://localhost:8000
"""
import argparse
import tempfile
import os
import time
import requests

SAMPLE_DOCS = [
    {
        "filename": "ai_overview.txt",
        "content": (
            "Artificial intelligence (AI) is the simulation of human intelligence by machines. "
            "Machine learning is a subset of AI that enables systems to learn from data. "
            "Deep learning uses neural networks with multiple layers. "
            "Natural language processing (NLP) allows machines to understand human language. "
            "Computer vision enables machines to interpret visual information from images and videos."
        ),
    },
    {
        "filename": "rag_systems.txt",
        "content": (
            "Retrieval-Augmented Generation (RAG) combines information retrieval with text generation. "
            "RAG systems first retrieve relevant documents from a knowledge base using vector similarity. "
            "The retrieved context is then passed to a language model to generate accurate responses. "
            "ChromaDB is a popular vector database for storing embeddings. "
            "CLIP embeddings can encode both text and images into the same vector space."
        ),
    },
]

QUERIES = [
    {"query": "What is artificial intelligence?", "expected_doc": "ai_overview.txt"},
    {"query": "How does RAG work?", "expected_doc": "rag_systems.txt"},
    {"query": "What is deep learning?", "expected_doc": "ai_overview.txt"},
    {"query": "What vector database does RAG use?", "expected_doc": "rag_systems.txt"},
    {"query": "What is NLP?", "expected_doc": "ai_overview.txt"},
]


def upload_docs(base_url):
    print("Uploading sample documents...")
    doc_ids = []

    for doc in SAMPLE_DOCS:
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, 'w') as f:
            f.write(doc["content"])

        with open(path, 'rb') as f:
            resp = requests.post(
                f"{base_url}/api/upload",
                files={"file": (doc["filename"], f, "text/plain")},
            )

        os.unlink(path)

        if resp.status_code == 200:
            doc_id = resp.json()["document_id"]
            doc_ids.append(doc_id)
            print(f"  ✓ {doc['filename']} -> {doc_id}")
        else:
            print(f"  ✗ {doc['filename']} failed: {resp.text}")

    # wait for processing
    print("Waiting for ingestion...")
    time.sleep(10)
    return doc_ids


def run_queries(base_url):
    print("\nRunning evaluation queries...")
    results = []

    for q in QUERIES:
        resp = requests.post(
            f"{base_url}/api/chat",
            json={"message": q["query"]},
        )

        if resp.status_code != 200:
            print(f"  ✗ Query failed: {q['query']}")
            results.append({"hit": False, "rank": None})
            continue

        data = resp.json()
        sources = data.get("sources", [])

        # check if expected doc appears in top 5
        hit = False
        rank = None
        for i, src in enumerate(sources[:5]):
            if src.get("filename") == q["expected_doc"]:
                hit = True
                rank = i + 1
                break

        results.append({"hit": hit, "rank": rank})
        status = f"Hit@rank={rank}" if hit else "Miss"
        print(f"  {'✓' if hit else '✗'} \"{q['query']}\" -> {status}")

    return results


def compute_metrics(results):
    hits = sum(1 for r in results if r["hit"])
    hit_at_5 = hits / len(results) if results else 0

    mrr = 0
    for r in results:
        if r["rank"]:
            mrr += 1.0 / r["rank"]
    mrr = mrr / len(results) if results else 0

    return hit_at_5, mrr


def main():
    parser = argparse.ArgumentParser(description="RAG evaluation")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--skip-upload", action="store_true", help="Skip doc upload if already done")
    args = parser.parse_args()

    if not args.skip_upload:
        upload_docs(args.base_url)

    results = run_queries(args.base_url)
    hit5, mrr = compute_metrics(results)

    print("\n" + "=" * 40)
    print(f"  Hit@5:  {hit5:.2%}")
    print(f"  MRR:    {mrr:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
