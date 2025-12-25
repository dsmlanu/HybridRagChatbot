import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from ingest import run_ingestion


# -----------------------------
# 1. EMBEDDING MODEL
# -----------------------------
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# -----------------------------
# 2. CREATE / LOAD CHROMA DB
# -----------------------------
def create_chroma_db(chunks, persist_dir="chroma_db"):
    embedding = load_embedding_model()

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embedding,
        persist_directory=persist_dir
    )

    vectordb.persist()
    print("✅ ChromaDB created and persisted")

    return vectordb


# -----------------------------
# 3. SEMANTIC SEARCH TEST
# -----------------------------
def search_query(query, persist_dir="chroma_db", k=3):
    embedding = load_embedding_model()

    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding
    )

    results = vectordb.similarity_search(query, k=k)
    return results


