# ⚡ RAG Chatbot

A production-ready chatbot built with Groq's Llama 3.1, supporting both general conversation and document-based Q&A (PDF, TXT, DOCX).

## Features

- 💬 Normal chat powered by Groq (Llama 3.1 8B)
- 📂 Upload documents and ask questions about them (RAG)
- 🧠 FAISS vector search with HuggingFace embeddings
- 💾 Persistent chat history stored in PostgreSQL (Supabase)
- ☁️ Uploaded files stored securely in Supabase Storage
- 🔒 File validation, filename sanitization, and rate limiting
- 🚫 Strict prompt grounding — answers only from uploaded documents, no hallucination

## Tech Stack

- **LLM**: Groq (Llama 3.1 8B Instant)
- **Framework**: Streamlit
- **Vector Store**: FAISS
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2)
- **Database**: PostgreSQL (via Supabase)
- **File Storage**: Supabase Storage

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # add your keys
streamlit run app.py
```

## Environment Variables
