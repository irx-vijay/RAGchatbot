import os
import re
import uuid
import time
import hashlib
import logging
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional

import bleach
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

from db import init_db, save_message, load_history, delete_session
from storage import upload_file, delete_session_files

# ─── Startup ────────────────────────────────────────────────────────────────
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag")

# ─── Groq Client ────────────────────────────────────────────────────────────
@st.cache_resource
def get_groq_client():
    key = os.getenv("GROQ_API_KEY")
    if not key:
        st.error("❌ GROQ_API_KEY missing.")
        st.stop()
    return Groq(api_key=key)

client = get_groq_client()

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Chatbot", page_icon="⚡", layout="wide")
st.title("⚡ GROQ-powered RAG Chatbot")

# ─── Config ─────────────────────────────────────────────────────────────────
APP_CONFIG = {
    "chunk_size": 600,
    "chunk_overlap": 120,
    "retrieval_k": 4,
    "max_tokens": 700,
    "temperature_chat": 0.7,
    "temperature_rag": 0.2,
    "max_file_mb": 20,
    "supported_types": ["pdf", "txt", "docx"],
    "rate_limit": 20,
    "rate_window_seconds": 60,
}

# ─── System Prompts ──────────────────────────────────────────────────────────
SYSTEM_PROMPT_CHAT = (
    "You are a knowledgeable, concise AI assistant. "
    "Answer clearly and honestly. If you don't know, say so."
)

SYSTEM_PROMPT_RAG = """You are a document analysis assistant with ONE strict rule:
ONLY answer using information explicitly present in the context.
- If the answer is NOT in the context, say exactly:
  "I couldn't find that in the uploaded documents."
- Always mention the source filename.
- Never use outside knowledge.
"""

# ─── Security ────────────────────────────────────────────────────────────────
def sanitize_filename(name: str) -> str:
    name = bleach.clean(name)
    name = re.sub(r"[^\w\.\-]", "_", name)
    name = re.sub(r"\.{2,}", ".", name)
    return name[:200]

def validate_file(f) -> Optional[str]:
    if len(f.getvalue()) == 0:
        return "File is empty."
    size_mb = len(f.getvalue()) / (1024 * 1024)
    if size_mb > APP_CONFIG["max_file_mb"]:
        return f"Exceeds {APP_CONFIG['max_file_mb']} MB limit."
    ext = Path(f.name).suffix.lower().lstrip(".")
    if ext not in APP_CONFIG["supported_types"]:
        return f"Unsupported type: .{ext}"
    return None

_rate_store: dict = defaultdict(list)

def check_rate_limit(session_id: str) -> bool:
    now = datetime.utcnow()
    window = timedelta(seconds=APP_CONFIG["rate_window_seconds"])
    _rate_store[session_id] = [t for t in _rate_store[session_id] if now - t < window]
    if len(_rate_store[session_id]) >= APP_CONFIG["rate_limit"]:
        return False
    _rate_store[session_id].append(now)
    return True

# ─── Session State ───────────────────────────────────────────────────────────
def init_state():
    if "session_id" not in st.session_state:
        params = st.query_params
        if "sid" in params:
            st.session_state["session_id"] = params["sid"]
        else:
            new_sid = str(uuid.uuid4())
            st.session_state["session_id"] = new_sid
            st.query_params["sid"] = new_sid

    if "messages" not in st.session_state:
        st.session_state["messages"] = load_history(st.session_state["session_id"])

    if "file_hash" not in st.session_state:
        st.session_state["file_hash"] = None

    if "vectorstore" not in st.session_state:
        st.session_state["vectorstore"] = None

init_db()      # ← first, creates table
init_state()   # ← second, loads history safely

if len(st.session_state["messages"]) == 0:
    st.session_state["messages"] = load_history(st.session_state["session_id"])
# ─── Embeddings ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ─── Vector Store ────────────────────────────────────────────────────────────
def build_vector_store(uploaded_files):
    docs = []
    for uf in uploaded_files:
        safe_name = sanitize_filename(uf.name)
        upload_file(uf.getvalue(), safe_name, st.session_state["session_id"])
        suffix = Path(safe_name).suffix

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uf.getvalue())
            tmp_path = tmp.name

        try:
            ext = suffix.lower().lstrip(".")
            if ext == "pdf":
                loader = PyPDFLoader(tmp_path)
            elif ext == "txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
            elif ext == "docx":
                loader = Docx2txtLoader(tmp_path)
            else:
                continue
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source"] = safe_name
            docs.extend(loaded)
        except Exception as e:
            st.warning(f"⚠️ Could not read {safe_name}: {e}")
        finally:
            os.unlink(tmp_path)

    if not docs:
        st.error("No readable content found.")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=APP_CONFIG["chunk_size"],
        chunk_overlap=APP_CONFIG["chunk_overlap"],
    )
    splits = splitter.split_documents(docs)
    return FAISS.from_documents(splits, get_embeddings())

# ─── Groq with Retry ─────────────────────────────────────────────────────────
def call_groq(messages: list, temperature: float, max_tokens: int) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content
        except Exception as e:
            if "rate_limit" in str(e).lower() and attempt < 2:
                time.sleep(2 ** attempt)
            else:
                st.error("⚠️ AI service error. Please try again.")
                st.stop()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚡ RAG Chatbot")
    mode = st.radio("Mode", ["💬 Normal Chat", "📂 Chat with Documents"])

    if mode == "📂 Chat with Documents":
        uploaded_files = st.file_uploader(
            "PDF · TXT · DOCX",
            type=APP_CONFIG["supported_types"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )
        if uploaded_files:
            errors = [validate_file(f) for f in uploaded_files]
            for err in [e for e in errors if e]:
                st.warning(f"⚠️ {err}")
            valid_files = [f for f in uploaded_files if not validate_file(f)]
            if valid_files:
                new_hash = hashlib.md5(b"".join(f.getvalue() for f in valid_files)).hexdigest()
                if new_hash != st.session_state["file_hash"]:
                    with st.spinner("Indexing…"):
                        st.session_state["vectorstore"] = build_vector_store(valid_files)
                    st.session_state["file_hash"] = new_hash
                    st.success(f"✅ {len(valid_files)} file(s) indexed")
    else:
        uploaded_files = []

    if st.button("🗑️ Clear", use_container_width=True):
        sid = st.session_state["session_id"]
        delete_session(sid)
        delete_session_files(sid)
        st.session_state.clear()
        st.rerun()

# ─── Main Chat ────────────────────────────────────────────────────────────────
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ─── Normal Chat ──────────────────────────────────────────────────────────────
if mode == "💬 Normal Chat":
    user_input = st.chat_input("Type your message…")
    if user_input:
        sid = st.session_state["session_id"]
        if not check_rate_limit(sid):
            st.warning("⚠️ Too many requests. Wait a moment.")
            st.stop()

        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        api_messages = [{"role": "system", "content": SYSTEM_PROMPT_CHAT}] + [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state["messages"][-20:]
        ]
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = call_groq(api_messages, APP_CONFIG["temperature_chat"], APP_CONFIG["max_tokens"])
            st.markdown(reply)
        st.session_state["messages"].append({"role": "assistant", "content": reply})
        save_message(sid, "user", user_input)
        save_message(sid, "assistant", reply)

# ─── Document Chat ────────────────────────────────────────────────────────────
elif mode == "📂 Chat with Documents":
    vs = st.session_state.get("vectorstore")
    if not vs:
        st.info("📂 Upload documents in the sidebar.")
    else:
        user_input = st.chat_input("Ask about your documents…")
        if user_input:
            sid = st.session_state["session_id"]
            if not check_rate_limit(sid):
                st.warning("⚠️ Too many requests. Wait a moment.")
                st.stop()

            st.session_state["messages"].append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            retriever = vs.as_retriever(search_kwargs={"k": APP_CONFIG["retrieval_k"]})
            retrieved_docs = retriever.invoke(user_input)
            context = "\n\n---\n\n".join(
                f"[Source: {d.metadata.get('source','?')}]\n{d.page_content}"
                for d in retrieved_docs
            )
            sources = list({d.metadata.get("source", "?") for d in retrieved_docs})

            api_messages = [
                {"role": "system", "content": SYSTEM_PROMPT_RAG},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_input}"}
            ]
            with st.chat_message("assistant"):
                with st.spinner("Searching…"):
                    reply = call_groq(api_messages, APP_CONFIG["temperature_rag"], APP_CONFIG["max_tokens"])
                st.markdown(reply)
                if sources:
                    st.caption(f"📄 Sources: {', '.join(sources)}")

            st.session_state["messages"].append({"role": "assistant", "content": reply})
            save_message(sid, "user", user_input)
            save_message(sid, "assistant", reply, sources=sources)
