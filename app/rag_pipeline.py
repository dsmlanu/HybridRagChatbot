from hybrid_search import hybrid_search
from ingest import run_ingestion
import subprocess
import requests
import os

os.environ["GROQ_API_KEY"] = "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"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#-----------------------------------------------------------
# 1. LLM method which use groq_api_key
#--------------------------------------------------------

def call_llm(prompt: str):
    
    url = "https://api.groq.com/openai/v1/chat/completions"  # ✅ corrected URL

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.3-70b-versatile",  # valid Groq model
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    r = requests.post(url, json=data, headers=headers)
    r.raise_for_status()

    return r.json()["choices"][0]["message"]["content"]
# -----------------------------
# 2. BUILD CONTEXT
# -----------------------------
def build_context(docs, max_chars=3000):
    context = ""
    for doc in docs:
        if len(context) + len(doc) <= max_chars:
            context += doc + "\n\n"
    return context.strip()


# -----------------------------
# 3. PROMPT TEMPLATE (RAG SAFE)
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
# 4. FULL RAG PIPELINE
# -----------------------------
def rag_pipeline(query, chunks):
    retrieved_docs = hybrid_search(query, chunks, k=5)

    context = build_context(retrieved_docs)

    prompt = create_prompt(context, query)

    answer = call_llm(prompt)
   
    return answer


