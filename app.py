import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found.")
    st.stop()

client = Groq(api_key=groq_api_key)

st.set_page_config(page_title="RAG Chatbot", page_icon="⚡", layout="wide")
st.title("⚡ GROQ-powered RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Your AI assistant"}]

SYSTEM_PROMPT = "You are a helpful AI assistant."

@st.cache_resource
def build_vector_store(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            st.warning(f"⚠️ Unsupported file type: {uploaded_file.name}")
            continue
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(splits, embeddings)

mode = st.radio("Choose Mode:", ["💬 Normal Chat", "📂 Chat with Documents"])

# display messages
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if mode == "💬 Normal Chat":
    user_input = st.chat_input("Type your message...")
    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state["messages"],
                temperature=0.7,
                max_tokens=500
            )
            bot_reply = response.choices[0].message.content
            st.write(bot_reply)
        st.session_state["messages"].append({"role": "assistant", "content": bot_reply})

elif mode == "📂 Chat with Documents":
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True
    )

    if uploaded_files:
        vectorstore = build_vector_store(uploaded_files)
        user_input = st.chat_input("Ask something about your documents...")
        if user_input:
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.chat_message("user").write(user_input)

            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            retrieved_docs = retriever.invoke(user_input)
            context = "\n\n".join([d.page_content for d in retrieved_docs])

            with st.chat_message("assistant"):
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {"role": "system", "content": "Answer using the provided context."},
                        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{user_input}"}
                    ],
                    temperature=0.3,
                    max_tokens=500
                )
                bot_reply = response.choices[0].message.content
                st.write(bot_reply)
            st.session_state["messages"].append({"role": "assistant", "content": bot_reply})
    else:
        st.info("📂 Please upload at least one document to start chatting.")