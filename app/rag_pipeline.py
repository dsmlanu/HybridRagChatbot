from hybrid_search import hybrid_search
from ingest import run_ingestion
import subprocess

import sys
sys.stdout.reconfigure(encoding="utf-8")
# -----------------------------
# 1. BUILD CONTEXT
# -----------------------------
def build_context(docs, max_chars=3000):
    context = ""
    for doc in docs:
        if len(context) + len(doc) <= max_chars:
            context += doc + "\n\n"
    return context.strip()


# -----------------------------
# 2. PROMPT TEMPLATE (RAG SAFE)
# -----------------------------
def create_prompt(context, query):
    prompt = f"""
You are a helpful AI assistant.
Answer the question ONLY using the context below.
If the answer is not in the context, say "This question is not covered in the uploaded document."

Context:
{context}

Question:
{query}

Answer:
"""
    return prompt
# -----------------------------
# 4. for handling UnicodeEncodeError: 'charmap' codec can't encode character '\u2217' in position 1255:
#character maps to <undefined>
# -----------------------------

def safe_text(text):
    return text.encode("utf-8", errors="ignore").decode("utf-8")
# -----------------------------
# 3. CALL LLM (OLLAMA)
# -----------------------------
def call_llm(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        capture_output=True,
        text=True,          # ✅ VERY IMPORTANT
        encoding="utf-8"    # ✅ avoids Unicode errors
    )

    return result.stdout.strip()



# -----------------------------
# 4. FULL RAG PIPELINE
# -----------------------------
def rag_pipeline(query, chunks):
    retrieved_docs = hybrid_search(query, chunks, k=5)

    context = build_context(retrieved_docs)

    prompt = create_prompt(context, query)

    answer = call_llm(prompt)
   
    return answer


