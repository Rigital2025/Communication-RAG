# app.py
import os, glob
import streamlit as st
from pathlib import Path
from pypdf import PdfReader

# ✅ New Chroma API (no Settings)
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# ---- Paths / constants -------------------------------------------------------
DB_DIR = Path("chroma_db")          # where Chroma stores the index
DATA_DIR = Path("data")             # put your PDFs here
COLLECTION = "docs"

# ---- Caching helpers ---------------------------------------------------------
@st.cache_resource
def get_client():
    DB_DIR.mkdir(exist_ok=True)
    return PersistentClient(path=str(DB_DIR))

@st.cache_resource
def get_collection(client):
    # Use a local Sentence-Transformers embedding function (no API key needed)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

# ---- PDF loading -------------------------------------------------------------
def load_pdfs():
    docs = []
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf_path))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = " ".join(text.split())
                if not text:
                    continue
                docs.append(
                    {
                        "id": f"{pdf_path.stem}-{i+1}",
                        "document": text,
                        "metadata": {"source": pdf_path.name, "page": i + 1},
                    }
                )
        except Exception as e:
            st.warning(f"Could not read {pdf_path.name}: {e}")
    return docs

# ---- Index build / query -----------------------------------------------------
def rebuild_index():
    client = get_client()
    # drop any old collection (safe even if it's not there)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass

    col = get_collection(client)
    docs = load_pdfs()
    if not docs:
        return 0

    # Chroma add() expects parallel lists
    col.add(
        ids=[d["id"] for d in docs],
        documents=[d["document"] for d in docs],
        metadatas=[d["metadata"] for d in docs],
    )
    return len(docs)

def query_index(q, k=3):
    client = get_client()
    col = get_collection(client)
    try:
        res = col.query(query_texts=[q], n_results=k)
    except Exception:
        return []

    hits = []
    for doc, meta in zip(res.get("documents", [[]])[0], res.get("metadatas", [[]])[0]):
        hits.append({"text": doc, "source": meta.get("source"), "page": meta.get("page")})
    return hits

# ---- UI ----------------------------------------------------------------------
st.set_page_config(page_title="Communication‑RAG", page_icon="🚀", layout="wide")

with st.sidebar:
    pdf_count = len(list(DATA_DIR.glob("*.pdf")))
    st.caption(f"Found {pdf_count} PDF(s) in ./data")
    if st.button("🔄 Rebuild index", use_container_width=True):
        with st.spinner("Indexing…"):
            n = rebuild_index()
        st.success(f"Indexed {n} chunks", icon="✅")

st.title("Communication‑RAG 🚀")
st.write("Ask about your PDFs and get grounded snippets with file + page.")

q = st.text_input("Your question", placeholder="e.g., What is the Communication Game about?")
k = st.slider("Results", min_value=1, max_value=10, value=3)

if q:
    with st.spinner("Searching…"):
        hits = query_index(q, k)
    if not hits:
        st.info("No results yet. Try **Rebuild index** on the left, then ask again.")
    else:
        for i, h in enumerate(hits, 1):
            st.markdown(f"**{i}. {h['source']} — p.{h['page']}**")
            st.write(h["text"])
            st.divider()

