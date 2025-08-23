# app.py
import os, glob
import streamlit as st
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# --- Paths & constants ---
DB_DIR = Path("chroma_db")
DATA_DIR = Path("data")
COLLECTION = "docs"

# ------------------ CACHED RESOURCES ------------------

@st.cache_resource
def get_client() -> PersistentClient:
    DB_DIR.mkdir(exist_ok=True)
    return PersistentClient(path=str(DB_DIR))

@st.cache_resource
def get_collection():
    # build the embedding function once and keep it with the collection
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    client = get_client()
    return client.get_or_create_collection(COLLECTION, embedding_function=ef)

@st.cache_resource
def get_model():
    # optional: if you ever need to embed locally outside Chroma (not used here)
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ------------------ PDF LOADING ------------------

def load_pdfs():
    docs = []
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf_path))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = text.strip()
                if not text:
                    continue
                docs.append(
                    {
                        "id": f"{pdf_path.name}-p{i+1}",
                        "text": text,
                        "metadata": {"file": pdf_path.name, "page": i + 1},
                    }
                )
        except Exception as e:
            st.warning(f"Could not read {pdf_path.name}: {e}")
    return docs

# ------------------ INDEX OPS ------------------

def rebuild_index():
    client = get_client()
    # drop & recreate to ensure a clean slate
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    col = get_collection()
    docs = load_pdfs()
    if not docs:
        return 0

    col.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[d["metadata"] for d in docs],
    )
    return len(docs)

def query_index(q: str, k: int = 3):
    col = get_collection()
    try:
        res = col.query(query_texts=[q], n_results=k)
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return list(zip(ids, docs, metas))
    except Exception as e:
        st.error(f"Query failed: {e}")
        return []

# ------------------ UI ------------------

st.title("Communication‑RAG 🚀")
st.caption("Ask about your PDFs and get grounded snippets with file + page.")

with st.sidebar:
    st.subheader("Index")
    st.write(f"Found {len(list(DATA_DIR.glob('*.pdf')))} PDF(s) in ./data")
    if st.button("🔁 Rebuild index", use_container_width=True):
        n = rebuild_index()
        st.success(f"Rebuilt index with {n} page chunks.")

q = st.text_input("Your question", "")
k = st.slider("Results", 1, 10, 3)

if q:
    hits = query_index(q, k)
    for _id, text, meta in hits:
        with st.expander(f"{meta.get('file','?')} – page {meta.get('page','?')}"):
            st.write(text)
