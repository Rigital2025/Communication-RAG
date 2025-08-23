import os
from pathlib import Path
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

DATA_DIR = Path("data")
DB_DIR = Path(".chroma")
EMBED_MODEL = "all-MiniLM-L6-v2"  # small, fast, no API key

def load_docs(data_dir: Path):
    docs = []
    for pdf in sorted(data_dir.glob("*.pdf")):
        loader = PyPDFLoader(str(pdf))
        pages = loader.load()
        for p in pages:
            p.metadata["source"] = pdf.name
        docs.extend(pages)
    return docs

def build_or_load_db(force_rebuild: bool = False):
    embeddings = SentenceTransformerEmbeddings(model_name=EMBED_MODEL)

    if force_rebuild and DB_DIR.exists():
        for p in DB_DIR.glob("**/*"):
            try:
                p.unlink()
            except IsADirectoryError:
                pass
        try:
            DB_DIR.rmdir()
        except Exception:
            pass

    if DB_DIR.exists() and not force_rebuild:
        return Chroma(persist_directory=str(DB_DIR), embedding_function=embeddings)

    raw_docs = load_docs(DATA_DIR)
    if not raw_docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(raw_docs)
    db = Chroma.from_documents(chunks, embeddings, persist_directory=str(DB_DIR))
    db.persist()
    return db

st.set_page_config(page_title="Communication RAG", page_icon="🧭", layout="wide")
st.title("Communication‑RAG 🧭📜")
st.caption("Ask questions about your Communication Game & All The Smoke PDFs. Returns grounded passages with citations.")

with st.sidebar:
    st.subheader("Index")
    if st.button("🔄 Rebuild index"):
        db = build_or_load_db(force_rebuild=True)
        if db is None:
            st.error("No PDFs found in ./data. Add files and try again.")
        else:
            st.success("Index rebuilt.")
    st.markdown("---")
    st.info("Put PDFs in `./data/` then click **Rebuild index**.")

db = build_or_load_db(force_rebuild=False)

q = st.text_input("Your question", placeholder="e.g., What is LUV‑FFO and how is it used?")
k = st.slider("Results to show", 1, 5, 3)

if st.button("Search") or (q and st.session_state.get("auto_run")):
    if db is None:
        st.error("I can't find any PDFs. Add them to `./data/` and rebuild the index.")
    else:
        retriever = db.as_retriever(search_kwargs={"k": k})
        results = retriever.get_relevant_documents(q)
        if not results:
            st.warning("No matches found. Try rephrasing.")
        else:
            st.subheader("Top matches")
            for i, doc in enumerate(results, 1):
                meta = doc.metadata or {}
                src = meta.get("source", "unknown.pdf")
                page = meta.get("page", "?")
                with st.expander(f"{i}. {src} — page {page}"):
                    st.write(doc.page_content)
