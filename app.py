# app.py
import os, glob
import streamlit as st
from pathlib import Path
from pypdf import PdfReader

# --- Chroma (new API) ---
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# -------------------------
# Paths & constants
# -------------------------
DB_DIR = Path("chroma_db")         # where Chroma stores the vector index
DATA_DIR = Path("data")            # where your PDFs live
COLLECTION = "docs"                # collection name inside Chroma

# -------------------------
# Cache helpers
# -------------------------
@st.cache_resource
def get_client() -> PersistentClient:
    DB_DIR.mkdir(exist_ok=True)
    return PersistentClient(path=str(DB_DIR))

@st.cache_resource
def get_collection(client: PersistentClient):
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    # Create or fetch the collection with our embedding function
    return client.get_or_create_collection(
        COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

# -------------------------
# PDF loading & chunking
# -------------------------
def load_pdfs():
    """Yield (text, metadata) chunks from every PDF in DATA_DIR."""
    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        for page_num, page in enumerate(reader.pages, start=1):
            raw = page.extract_text() or ""
            text = " ".join(raw.split())
            if not text:
                continue
            # simple chunking by characters (keeps it robust for long pages)
            chunk_size, overlap = 900, 150
            start = 0
            while start < len(text):
                end = min(len(text), start + chunk_size)
                chunk = text[start:end]
                meta = {
                    "file": pdf_path.name,
                    "path": str(pdf_path),
                    "page": page_num,
                }
                yield chunk, meta
                start = end - overlap

# -------------------------
# Indexing
# -------------------------
def rebuild_index():
    client = get_client()
    # nuke & recreate the collection to keep it simple
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = get_collection(client)

    texts, metadatas, ids = [], [], []
    i = 0
    for chunk, meta in load_pdfs():
        texts.append(chunk)
        metadatas.append(meta)
        ids.append(f"{meta['file']}|p{meta['page']}|{i}")
        i += 1

    if texts:
        col.add(documents=texts, metadatas=metadatas, ids=ids)
    return len(texts)

def query_index(q: str, k: int = 3):
    client = get_client()
    col = get_collection(client)
    res = col.query(query_texts=[q], n_results=max(1, k), include=["documents", "metadatas", "distances", "ids"])
    # Normalize output
    hits = []
    if res and res.get("documents"):
        docs = res["documents"][0]
        metas = res["metadatas"][0]
        dists = res.get("distances", [[None]])[0]
        ids = res.get("ids", [[None]])[0]
        for doc, meta, dist, _id in zip(docs, metas, dists, ids):
            hits.append({"text": doc, "meta": meta, "score": dist, "id": _id})
    return hits

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Communication-RAG", page_icon="🚀", layout="wide")
st.sidebar.header("Index")
pdf_count = len(list(DATA_DIR.glob("*.pdf")))
st.sidebar.write(f"Found {pdf_count} PDF(s) in ./data")

if st.sidebar.button("🔁 Rebuild index"):
    with st.spinner("Rebuilding index…"):
        n = rebuild_index()
    st.sidebar.success(f"Indexed {n} chunk(s).")

st.title("Communication-RAG 🚀")
st.caption("Ask about your PDFs and get grounded snippets with file + page.")

q = st.text_input("Your question", value="")
k = st.slider("Results", min_value=1, max_value=10, value=3)

if q.strip():
    with st.spinner("Searching…"):
        hits = query_index(q.strip(), k=k)

    if not hits:
        st.warning("No results yet. Try **Rebuild index** in the sidebar first, then ask again.")
    else:
        for i, h in enumerate(hits, start=1):
            meta = h["meta"]
            file = meta.get("file", "unknown")
            page = meta.get("page", "?")
            score = h.get("score", None)
            with st.container(border=True):
                st.markdown(f"**{i}. {file} — page {page}**")
                if score is not None:
                    st.write(f"distance: {score:.4f}")
                st.write(h["text"])

