# Communication‑RAG 📜⚙️

**Retrieval‑Augmented Generation** over two originals by Rodrick Chambers:
- *Refine: The Communication Game – Breakdown to Breakthrough*
- *All The Smoke – Communication Course Coaching*

This app lets you **ask questions** about the materials and get **grounded answers with citations** to page snippets. Built to showcase practical IR + LLM orchestration, basic safety, and evaluation.

---

## ✨ Features (MVP → near‑term)
- 🔎 **Semantic search** over PDFs (FAISS/Chroma)
- 🧠 **LLM answerer** constrained to context (Streamlit UI)
- 📚 **Citations** with file + page numbers
- 🛡️ **Safety checks** (PII & prompt‑injection heuristics)
- 📏 **Early evaluation** with RAGAS (precision/faithfulness)

**Planned**
- 🧪 Eval dashboard (RAGAS) + small golden set
- 🐳 Docker image for one‑command run
- 🚀 Hugging Face Space for click‑to‑try demo

---

## 🧰 Tech Stack
- Python, Streamlit
- LangChain / (or LlamaIndex), FAISS/Chroma
- OpenAI or HF LLM + embeddings
- PyPDF for document loading
- RAGAS for evaluation

---

## 📦 Project Structure (target)

