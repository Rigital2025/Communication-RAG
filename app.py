# app.py
import os, glob
import streamlit as st
from pathlib import Path
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
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
def get_collection(_client):
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return client.get_or_create_collection(COLLECTION, embedding_function=ef)

def load_pdfs():
    docs = []
    for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
        try:
            r = PdfReader(str(pdf_path))
            for i, page in enumerate(r.pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append({
                        "id": f"{pdf_path.name}-{i}",
                        "text": text,
                        "metadata": {"file": pdf_path.name, "page": i+1}
                    })
        except Exception as e:
            st.warning(f"Could not read {pdf_path.name}: {e}")
    return docs

def rebuild_index():
    cl = get_client()
    # drop & recreate collection
    try:
        cl.delete_collection(COLLECTION)
    except Exception:
        pass
    col = get_collection(cl)
    # add docs
    docs = load_pdfs()
    if not docs:
        return 0
    col.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[d["metadata"] for d in docs],
    )
    return len(docs)

def query_index(q, k=3):
    cl = get_client()
    try:
        col = cl.get_collection(COLLECTION)
    except Exception:
        return []
    res = col.query(query_texts=[q], n_results=k)
    hits = []
    for i in range(len(res["ids"][0])):
        hits.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "meta": res["metadatas"][0][i]
        })
    return hits

# ---------------- UI ----------------
st.set_page_config(page_title="Communication-RAG", page_icon="🚀", layout="wide")
st.sidebar.header("Index")
data_count = len(list(DATA_DIR.glob("*.pdf")))
st.sidebar.write(f"Found {data_count} PDF(s) in ./data")

if st.sidebar.button("🔄 Rebuild index"):
    with st.spinner("Rebuilding…"):
        n = rebuild_index()
    st.success(f"Indexed {n} chunks.")

st.title("Communication-RAG 🚀")
st.caption("Ask about your PDFs and get grounded snippets with file + page.")

q = st.text_input("Your question")
k = st.slider("Results", 1, 10, 3)

if q:
    with st.spinner("Searching…"):
        hits = query_index(q, k)
    if not hits:
        st.error("No results (try rebuilding the index).")
    else:
        for h in hits:
            with st.expander(f'{h["meta"]["file"]} — p.{h["meta"]["page"]}  •  {h["id"]}'):
                st.write(h["text"])

