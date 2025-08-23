# app.py
import os, glob
import streamlit as st
from pathlib import Path
from pypdf import PdfReader

# Chroma 0.5+ API
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# ---------- paths & constants ----------
DB_DIR   = Path("chroma_db")     # persisted vector DB
DATA_DIR = Path("data")          # where your PDFs live
COLLECTION = "docs"

# ---------- cached singletons ----------
@st.cache_resource
def get_client() -> PersistentClient:
    DB_DIR.mkdir(exist_ok=True)
    return PersistentClient(path=str(DB_DIR))

@st.cache_resource
def get_collection():
    client = get_client()
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(COLLECTION, embedding_function=ef)

# ---------- helpers ----------
def load_pdfs():
    docs = []
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        try:
            txt = []
            with open(pdf_path, "rb") as f:
                reader = PdfReader(f)
                for p in reader.pages:
                    t = p.extract_text() or ""
                    if t.strip():
                        txt.append(t)
            if txt:
                docs.append({"file": pdf_path.name, "text": "\n".join(txt)})
        except Exception as e:
            st.warning(f"Failed to read {pdf_path.name}: {e}")
    return docs

def rebuild_index():
    cl = get_client()
    try:
        cl.delete_collection(COLLECTION)
    except Exception:
        pass
    col = get_collection()
    docs = load_pdfs()

    ids, texts, metadatas = [], [], []
    for i, d in enumerate(docs):
        chunk = d["text"]
        # coarse chunking – simple but fine for now
        for j, s in enumerate(chunk.split("\n\n")):
            s = s.strip()
            if not s:
                continue
            ids.append(f"{d['file']}::{i}:{j}")
            texts.append(s)
            metadatas.append({"file": d["file"], "chunk": j})

    if texts:
        col.add(ids=ids, documents=texts, metadatas=metadatas)
    return len(texts)

def query_index(q, k=3):
    col = get_collection()
    try:
        res = col.query(query_texts=[q], n_results=k)
        hits = list(zip(res.get("documents", [[]])[0],
                        res.get("metadatas", [[]])[0]))
        return hits
    except Exception as e:
        st.error(f"Query failed: {e}")
        return []

# ---------- UI ----------
st.sidebar.header("Index")
DATA_DIR.mkdir(exist_ok=True)
st.sidebar.write(f"Found {len(list(DATA_DIR.glob('*.pdf')))} PDF(s) in ./data")

if st.sidebar.button("🔄 Rebuild index"):
    with st.spinner("Building embeddings…"):
        n = rebuild_index()
    st.sidebar.success(f"Indexed {n} chunks.")

st.title("Communication‑RAG 🚀")
st.caption("Ask about your PDFs and get grounded snippets with file + page.")

q = st.text_input("Your question", placeholder="Ask something about the PDFs…")
k = st.slider("Results", 1, 10, 3)

if q:
    hits = query_index(q, k=k)
    if not hits:
        st.info("No results yet. Try **Rebuild index** in the sidebar.")
    else:
        for i, (doc, meta) in enumerate(hits, 1):
            with st.expander(f"{i}. {meta.get('file','?')} • chunk {meta.get('chunk','?')}"):
                st.write(doc or "")
