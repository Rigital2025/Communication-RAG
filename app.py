# app.py
from __future__ import annotations

import os
import glob
from pathlib import Path
from typing import List, Dict, Any, Tuple

import streamlit as st
from pypdf import PdfReader

from sentence_transformers import SentenceTransformer  # (pulled for env pre-loads)
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# ---------------- Paths & constants ----------------
DB_DIR = Path("chroma_db")
DATA_DIR = Path("data")
COLLECTION = "docs"

# --------------- Helpers ----------------
def _pdf_paths() -> List[Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return [Path(p) for p in glob.glob(str(DATA_DIR / "*.pdf"))]

def _read_pdf(p: Path) -> List[Dict[str, Any]]:
    """Return a list of page docs: {id, text, metadata} for a single PDF."""
    docs: List[Dict[str, Any]] = []
    try:
        reader = PdfReader(str(p))
        for i, page in enumerate(reader.pages):
            text = (page.extract_text() or "").strip()
            if not text:
                continue
            docs.append(
                {
                    "id": f"{p.name}-p{i+1}",
                    "text": text,
                    "metadata": {"source": p.name, "page": i + 1},
                }
            )
    except Exception as e:
        st.warning(f"Failed to read {p.name}: {e}")
    return docs

def load_pdfs() -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for p in _pdf_paths():
        docs.extend(_read_pdf(p))
    return docs

# --------------- Chroma clients & collections ----------------
@st.cache_resource
def get_client() -> PersistentClient:
    """Create (and cache) a Chroma PersistentClient at DB_DIR."""
    DB_DIR.mkdir(parents=True, exist_ok=True)
    return PersistentClient(path=str(DB_DIR))

@st.cache_resource
def get_collection(_client: PersistentClient):
    """
    IMPORTANT: the leading underscore in `_client` tells Streamlit not to hash it.
    That avoids UnhashableParamError on chromadb Client objects.
    """
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return _client.get_or_create_collection(COLLECTION, embedding_function=ef)

# --------------- Index ops ----------------
def rebuild_index() -> int:
    client = get_client()
    # Drop collection if it exists (clean slate)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    col = get_collection(client)  # pass the client in

    docs = load_pdfs()
    if not docs:
        return 0

    col.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[d["metadata"] for d in docs],
    )
    return len(docs)

def query_index(q: str, k: int = 3) -> List[Tuple[str, str, Dict[str, Any]]]:
    client = get_client()
    col = get_collection(client)  # pass the client in

    try:
        res = col.query(query_texts=[q], n_results=k)
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return list(zip(ids, docs, metas))
    except Exception as e:
        st.error(f"Query failed: {e}")
        return []

# --------------- UI ----------------
st.set_page_config(page_title="Communication‑RAG", page_icon="🚀", layout="wide")
st.title("Communication‑RAG 🚀")

with st.sidebar:
    pdfs = _pdf_paths()
    st.subheader("Index")
    st.caption(f"Found {len(pdfs)} PDF(s) in ./data")
    if st.button("🔁 Rebuild index"):
        with st.spinner("Rebuilding…"):
            n = rebuild_index()
        st.success(f"Indexed {n} page snippet(s).")

st.divider()

col1, col2 = st.columns([3, 1])
with col1:
    q = st.text_input("Ask about your PDFs", placeholder="e.g., What is LUV‑FFO?")

with col2:
    top_k = st.number_input("Results", 1, 10, 3, step=1)

if q:
    with st.spinner("Searching..."):
        hits = query_index(q, k=top_k)

    if not hits:
        st.info("No results yet. Try rebuilding the index or asking a different question.")
    else:
        for i, (hit_id, text, meta) in enumerate(hits, start=1):
            src = meta.get("source", "unknown")
            page = meta.get("page", "?")
            with st.expander(f"{i}. {src} — p.{page}  ·  id={hit_id}"):
                st.write(text)
                st.caption(f"Source: {src}, page {page}")
