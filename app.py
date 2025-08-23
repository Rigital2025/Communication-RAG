# app.py
import os, glob
import streamlit as st
from pathlib import Path
from pypdf import PdfReader

# NEW Chroma imports (no Settings)
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

DB_DIR = Path("chroma_db")
DATA_DIR = Path("data")
COLLECTION = "docs"

@st.cache_resource
def get_client():
    DB_DIR.mkdir(exist_ok=True)
    return PersistentClient(path=str(DB_DIR))

@st.cache_resource
def get_collection(client):
    # use small, fast sentence-transformers
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(COLLECTION, embedding_function=ef)

def load_pdfs():
    docs = []
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf_path))
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = " ".join(text.split())
                if text:
                    docs.append(
                        {
                            "id": f"{pdf_path.name}-{i+1}",
                            "text": text,
                            "meta": {"file": pdf_path.name, "page": i + 1},
                        }
                    )
        except Exception as e:
            st.warning(f"Failed to read {pdf_path.name}: {e}")
    return docs

def rebuild_index():
    client = get_client()
    # drop and recreate collection each time
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = get_collection(client)

    docs = load_pdfs()
    if not docs:
        return 0

    # add in chunks to avoid large batch issues
    B = 100
    for i in range(0, len(docs), B):
        batch = docs[i : i + B]
        col.add(
            ids=[d["id"] for d in batch],
            documents=[d["text"] for d in batch],
            metadatas=[d["meta"] for d in batch],
        )
    return len(docs)

def query_index(q, k=3):
    client = get_client()
    try:
        col = get_collection(client)
    except Exception:
        return []
    r = col.query(query_texts=[q], n_results=k)
    hits = []
    if r and r.get("documents"):
        for doc, meta in zip(r["documents"][0], r["metadatas"][0]):
            hits.append({"text": doc, "file": meta.get("file"), "page": meta.get("page")})
    return hits

# ---------------- UI ----------------
st.set_page_config(page_title="Communication-RAG", layout="wide")
st.sidebar.header("Index")
pdfs_found = list(DATA_DIR.glob("*.pdf"))
st.sidebar.write(f"Found {len(pdfs_found)} PDF(s) in ./data")
if st.sidebar.button("🔁 Rebuild index"):
    with st.spinner("Rebuilding..."):
        n = rebuild_index()
    st.success(f"Indexed {n} pages.")

st.title("Communication-RAG 🚀")
st.caption("Ask about your PDFs and get grounded snippets with file + page.")

q = st.text_input("Your question", placeholder="What is the Communication Game about?")
k = st.slider("Results", 1, 10, 3)

if q:
    with st.spinner("Searching..."):
        results = query_index(q, k=k)
    if not results:
        st.info("No results (try rebuilding the index or checking your PDFs).")
    for i, r in enumerate(results, 1):
        st.markdown(f"**{i}. {r['file']} — p.{r['page']}**")
        st.write(r["text"])

