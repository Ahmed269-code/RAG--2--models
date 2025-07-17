import streamlit as st
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import bs4
import os

# ------------------------ Load FLAN-T5 Model ------------------------
@st.cache_resource
def load_flan_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# ------------------------ Gemini API Function ------------------------
def ask_gemini_api(context, question, api_key):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": f"Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}"
                    }
                ]
            }
        ]
    }
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key
    }

    response = requests.post(url, headers=headers, json=payload)
    try:
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Error: {e}\nResponse text: {response.text}"

# ------------------------ FLAN Answering ------------------------
def generate_flan_answer(context, question, tokenizer, model, device="cpu"):
    prompt = f"""You are an intelligent assistant helping answer questions based on provided documents.

Only use the information from the context to answer the question. Do not make up any information.

If the answer is not found in the context, say: "The answer is not available in the provided context."

---

Context:
{context}

---

Question: {question}

Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# ------------------------ Vectorstore Function ------------------------
@st.cache_resource
def load_vectorstore():
    loader = WebBaseLoader(
        web_paths=[
            "https://en.wikipedia.org/wiki/Artificial_intelligence",
            "https://en.wikipedia.org/wiki/Machine_learning",
            "https://en.wikipedia.org/wiki/Deep_learning",
        ],
        bs_kwargs=dict(parse_only=bs4.SoupStrainer("p"))
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
    splits = splitter.split_documents(docs)
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_documents(documents=splits, embedding=embedding_model)

# ------------------------ Streamlit UI ------------------------
st.set_page_config(page_title="🔍 RAG QA App", layout="wide")
st.title("RAGChat")

model_choice = st.selectbox("Choose Model:", ["Gemini API", "FLAN-T5"])

question = st.text_input("💬 Ask your question:")

if st.button("Search"):
    if not question:
        st.warning("Please enter a question.")
        st.stop()

    st.write("🔍 Searching relevant context...")
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever()
    relevant_docs = retriever.get_relevant_documents(question)
    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    st.write("⏳ Generating Answer...")

    if model_choice == "Gemini API":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("GEMINI_API_KEY not found. Please set it in your Streamlit secrets.")
            st.stop()
        answer = ask_gemini_api(context, question, api_key)
    else:
        tokenizer, model = load_flan_model()
        answer = generate_flan_answer(context, question, tokenizer, model, device="cuda" if torch.cuda.is_available() else "cpu")

    st.subheader("📌 Answer:")
    st.write(answer)

    with st.expander("📚 Retrieved Context"):
        st.write(context)
