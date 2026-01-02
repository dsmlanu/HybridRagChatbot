import streamlit as st
import os
from ingest import run_ingestion
from vector_store import create_chroma_db
from rag_pipeline import rag_pipeline

# -----------------------------
# APP CONFIG
# -----------------------------
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
st.set_page_config(page_title="Hybrid RAG Chatbot", layout="wide")
st.title("📄 Hybrid RAG PDF Chatbot")
st.write("Upload a PDF and ask questions using Hybrid Search + LLaMA")

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# -----------------------------
# SESSION STATE
# -----------------------------
if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False

# -----------------------------
# PDF UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    pdf_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ PDF uploaded successfully")

    if st.button("🔍 Process Document"):
        with st.spinner("Processing PDF..."):
            chunks = run_ingestion(pdf_path)
            create_chroma_db(chunks)

            st.session_state.chunks = chunks
            st.session_state.pdf_processed = True

        st.success("✅ Document indexed successfully")

# -----------------------------
# QUESTION ANSWERING
# -----------------------------
if st.session_state.pdf_processed:
    st.subheader("💬 Ask a Question")

    query = st.text_input("Enter your question")

    if st.button("Ask"):
        with st.spinner("Thinking..."):
            answer = rag_pipeline(query, st.session_state.chunks)

        st.markdown("### 🧠 Answer")
        st.write(answer)
