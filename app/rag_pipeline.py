from hybrid_search import hybrid_search
from ingest import run_ingestion
import subprocess
import requests

BACKEND_URL = "https://hybrid-rag-fastapi-backend.onrender.comvenv/generate"

# ----------------------------------------------------------------sw2   1-------
# 1. CALL LLM (OLLAMA)  This method we can use when we  use fast api
# -----------------------------------------------------------------------
import os
import requests

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def call_llm(prompt: str):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    r = requests.post(url, json=data, headers=headers)
    r.raise_for_status()

    return r.json()["choices"][0]["message"]["content"]


'''def call_llm(prompt):
    payload = {"prompt": prompt}
    r = requests.post(BACKEND_URL, json=payload)
    return r.json().get("response", "")'''

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
# -----------------------------------------------------------------------
# 3. CALL LLM (OLLAMA)  This method we can use when we don't use fast api
# -----------------------------------------------------------------------
"""def call_llm(prompt: str) -> str:
    result = subprocess.run(
        ["ollama", "run", "llama3"],
        input=prompt,
        capture_output=True,
        text=True,          # ✅ VERY IMPORTANT
        encoding="utf-8"    # ✅ avoids Unicode errors
    )

    return result.stdout.strip()"""



# -----------------------------
# 4. FULL RAG PIPELINE
# -----------------------------
def rag_pipeline(query, chunks):
    retrieved_docs = hybrid_search(query, chunks, k=5)

    context = build_context(retrieved_docs)

    prompt = create_prompt(context, query)

    answer = call_llm(prompt)
   
    return answer


