⚡ GROQ-powered RAG Chatbot
A conversational AI chatbot built with Streamlit and powered by Groq's LLaMA 3.1 model. It supports two modes:

💬 Normal Chat — a general-purpose AI assistant for everyday conversations.
📂 Chat with Documents — upload PDF, TXT, or DOCX files and ask questions about their content using Retrieval-Augmented Generation (RAG). Documents are chunked, embedded using HuggingFace's MiniLM, stored in a FAISS vector store, and relevant chunks are retrieved to provide context-aware answers.

Tech Stack:

Frontend: Streamlit
LLM: Groq (LLaMA 3.1 8B)
Embeddings: HuggingFace all-MiniLM-L6-v2
Vector Store: FAISS
Document Loaders: LangChain (PDF, TXT, DOCX)
