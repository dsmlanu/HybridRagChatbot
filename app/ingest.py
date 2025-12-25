import re
import os
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter


# -----------------------------
# 1. PDF TEXT EXTRACTION
# -----------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


# -----------------------------
# 2. CLEANING & PREPROCESSING
# -----------------------------
def clean_text(text: str) -> str:
    # Remove multiple newlines
    text = re.sub(r"\n{2,}", "\n", text)

    # Remove page numbers (simple heuristic)
    text = re.sub(r"\n\d+\n", "\n", text)

    # Remove headers/footers (common patterns)
    text = re.sub(r"Page \d+ of \d+", "", text, flags=re.IGNORECASE)

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# -----------------------------
# 3. CHUNKING STRATEGY
# -----------------------------
def chunk_text(text: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,        # experiment between 200–500
        chunk_overlap=50,      # 30–50 words overlap
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks


# -----------------------------
# 4. PIPELINE EXECUTION
# -----------------------------
def run_ingestion(pdf_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"File not found: {pdf_path}")

    print("📄 Extracting text...")
    raw_text = extract_text_from_pdf(pdf_path)

    print("🧹 Cleaning text...")
    cleaned_text = clean_text(raw_text)

    print("✂️ Chunking text...")
    chunks = chunk_text(cleaned_text)

    #print(f"✅ Total Chunks Created: {len(chunks)}")
   # print("\n🔹 Sample Chunk:\n")
    #print(chunks[0])

    return chunks



