# HybridRagChatbot
🧠 Hybrid RAG Chatbot (Retrieval-Augmented Generation)

This project is a Hybrid RAG Chatbot that improves answer accuracy by combining:

✔ Keyword-based search (BM25 / traditional IR)
✔ Semantic search using embeddings
✔ LLM reasoning on top of retrieved context

Instead of relying only on the model, the bot retrieves relevant knowledge from documents and then uses an LLM to generate reliable, grounded answers.

🚀 Key Features

🔍 Hybrid Search

BM25 / keyword search for exact matches

Vector search for meaning-based results

Merges and re-ranks both results for the best answer

📄 Multi-document support

PDFs, text files, knowledge base docs

🏗️ Chunking + Metadata

Smart chunking to preserve context
Document-aware retrieval

🤖 LLM Powered

Uses retrieved content as context

Reduces hallucinations

Generates clear, grounded responses

⚡ Streaming responses (optional)

🗂️ Easy to extend with custom datasets
