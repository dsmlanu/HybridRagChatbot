from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from ingest import run_ingestion
import numpy as np


# -----------------------------
# 1. LOAD CHROMA (SEMANTIC)
# -----------------------------
def load_chroma(persist_dir="chroma_db"):
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding
    )


# -----------------------------
# 2. BM25 KEYWORD SEARCH
# -----------------------------
def bm25_search(chunks, query, k=5):
    tokenized_corpus = [chunk.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    scores = bm25.get_scores(query.lower().split())
    top_k_idx = np.argsort(scores)[-k:][::-1]

    return [chunks[i] for i in top_k_idx]


# -----------------------------
# 3. HYBRID SEARCH (FUSION)
# -----------------------------
def hybrid_search(query, chunks, k=5):
    chroma = load_chroma()
    if query.lower().startswith(("what", "why")):
        k=8
    else:
        k=5
            

    # Semantic Search
    semantic_docs = chroma.similarity_search(query, k=k)
    semantic_texts = [doc.page_content for doc in semantic_docs]

    # Keyword Search
    keyword_texts = bm25_search(chunks, query, k=k)

    # Fusion (remove duplicates, preserve order)
    combined = list(dict.fromkeys(semantic_texts + keyword_texts))

    return combined[:k]


# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    pdf_path = "app/data/documents/attensionisallyouneed.pdf"

    # Get chunks (same as Step-1)
    chunks = run_ingestion(pdf_path)

    query = "Explain the main purpose of this document"

    results = hybrid_search(query, chunks)

    print("\n🔍 Hybrid Search Results:\n")
    for i, res in enumerate(results):
        print(f"Result {i+1}:\n{res}\n")
