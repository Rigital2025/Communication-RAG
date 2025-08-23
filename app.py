# app.py
import os, glob
import streamlit as st
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

DB_DIR = Path("chroma_db")
DATA_DIR = Path("data")
COLLECTION = "docs"

@st.cache_resource
def get_client():
    DB_DIR.mkdir(exist_ok=True)
    return chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=str(DB_DIR)))

@st.cache_resource
def get_model():
    # small, fast, good quality
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def load_pdfs():
    docs = []
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf_path))
            for i, page in enumerate(reader.pages):
                text = (page.extract_text() or "").strip()
                if text:
                    docs.append({
                        "id": f"{pdf_path.name}-p{i+1}",
                        "text": text,
                        "meta": {"file": pdf_path.name, "page": i+1}
                    })
        except Exception as e:
            st.warning(f"Skipping {pdf_path.name}: {e}")
    return docs

def rebuild_index():
    cl = get_client()
    try:
        cl.delete_collection(COLLECTION)
    except Exception:
        pass
    col = cl.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    model = get_model()
    docs = load_pdfs()
    if not docs:
        return 0
    # batch insert
    B = 64
    for i in range(0, len(docs), B):
        batch = docs[i:i+B]
        embeddings = model.encode([d["text"] for d in batch], show_progress_bar=False).tolist()
        col.add(
            ids=[d["id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[d["meta"] for d in batch],
            embeddings=embeddings,
        )
    # persist
    cl.persist()
    return len(docs)

def query_index(q, k=3):
    cl = get_client()
    try:
        col = cl.get_collection(COLLECTION)
    except Exception:
        return []
    model = get_model()
    q_emb = model.encode([q]).tolist()
    res = col.query(query_embeddings=q_emb, n_results=k)
    out = []
    for doc, meta, dist in zip(res.get("documents",[[]])[0], res.get("metadatas",[[]])[0], res.get("distances",[[]])[0]):
        out.append((doc, meta, dist))
    return out

# -------- UI --------
st.set_page_config(page_title="Communication‑RAG", page_icon="🚀", layout="wide")
st.title("Communication‑RAG 🚀")

with st.sidebar:
    st.subheader("Index")
    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    st.caption(f"Found {len(pdfs)} PDF(s) in ./data")
    if st.button("🔁 Rebuild index", use_container_width=True):
        n = rebuild_index()
        st.success(f"Indexed {n} page chunks.")

st.write("Ask about your PDFs and get grounded snippets with file + page.")

q = st.text_input("Your question")
top_k = st.slider("Results", 1, 10, 3)
if q:
    hits = query_index(q, k=top_k)
    if not hits:
        st.info("No index yet. Click **Rebuild index** in the sidebar.")
    else:
        for i, (doc, meta, dist) in enumerate(hits, 1):
            st.markdown(f"**{i}. {meta['file']} · p.{meta['page']}** — _(distance: {dist:.3f})_")
            st.write(doc[:1200] + ("…" if len(doc) > 1200 else ""))
            st.divider()
