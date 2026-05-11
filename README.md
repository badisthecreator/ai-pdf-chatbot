# 1. Install dependencies
pip install -r requirements.txt

# 2. Get a FREE Groq key at console.groq.com
#    then create your .env file:
cp .env.example .env
# paste your key inside .env

# 3. Launch
python frontend/app.py
# → open http://localhost:7860

# 📄 AI PDF Chatbot — Advanced RAG

An intelligent chatbot that lets you upload any PDF and have a full conversation about its content.  
Built with **RAG (Retrieval-Augmented Generation)** — no hallucinations, every answer is grounded in your document.

---

## 🏗️ Architecture

```
ai_pdf_chatbot/
├── backend/
│   ├── config.py        # all configuration constants
│   ├── ingestion.py     # PDF → chunks → embeddings → ChromaDB
│   ├── retriever.py     # hybrid search (vector + BM25)
│   └── rag_chain.py     # RAG pipeline + conversational memory + citations
├── frontend/
│   └── app.py           # Gradio UI
├── data/                # put test PDFs here(optional)
├── vectorstore/         # ChromaDB persisted data (auto-created)
├── .env.example         # copy to .env and add your key
└── requirements.txt
```

---

## ⚙️ RAG Avance Features Implemented

| Feature | Implementation |
|---|---|
| ✅ Advanced chunking with overlap |
| ✅ Hybrid search | ChromaDB (vector) + BM25 (`rank_bm25`) merged & deduplicated |
| ✅ Conversational memory | Last 4 turns injected into every prompt |
| ✅ Source citations | Page number + filename shown under every answer |
| ✅ Streaming responses | Groq stream=True, tokens displayed progressively |
| ✅ File upload UI | Gradio file component |

---

## 🚀 Setup & Run

### 1. Clone / download the project
```bash
cd ai_pdf_chatbot
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt # if i missed something there don't panic just search it out
```

### 4. Get a FREE Groq API key
Go to https://console.groq.com → sign up → copy your API key.

### 5. Configure environment
```bash
cp .env.example .env
# Open .env and paste your GROQ_API_KEY
```

### 6. Run the app
```bash
python frontend/app.py
```

Open your browser at **http://localhost:7860**

---

## 🔄 How It Works (step by step)

```
User uploads PDF
      │
      ▼
[ingestion.py]
  PyMuPDF loads pages → RecursiveCharacterTextSplitter chunks with overlap
  → HuggingFace embeds chunks → ChromaDB stores chunks + vectors
      │
      ▼
User asks a question
      │
      ▼
[retriever.py]
  ChromaDB vector search (top 5)  +  BM25 keyword search (top 5)
  → Merge & deduplicate → top 5 best chunks
      │
      ▼
[rag_chain.py]
  Format context (with page numbers) + inject chat history
  → Build prompt → call Groq LLaMA 3 (stream)
      │
      ▼
[app.py]
  Stream tokens to Gradio chat → append citations below answer
```

---

## 🧠 Technologies

| Tool | Role |
|---|---|
| **Groq + LLaMA 3** | Free, ultra-fast LLM API |
| **LangChain** | RAG pipeline orchestration |
| **ChromaDB** | Local vector database |
| **sentence-transformers** | Local embeddings (all-MiniLM-L6-v2) |
| **rank-bm25** | Keyword search |
| **PyMuPDF** | PDF loading with page metadata |
| **Gradio** | Web UI |

---

## ⚠️ Known Limitations & Future Improvements

- **Single PDF at a time**
- **Local embeddings**: `all-MiniLM-L6-v2` is fast but not the most powerful. Could swap for OpenAI embeddings.
- **No OCR**: scanned PDFs (image-only) won't be read.
- **Memory length**: only last 4 turns kept. Long conversations lose early context.

---

## 👤 Author

Project built with love for the *l'IA Generative* course.  
Option chosen: **RAG Avance (/18)**
