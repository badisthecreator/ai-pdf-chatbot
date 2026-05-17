# 📄 DocuMind AI — Advanced PDF Assistant

A chatbot that lets you upload a PDF and talk to it. Ask questions, get summaries, find specific info — all without reading the whole document yourself.

Built by **Badis Kefi** for the *l'IA Generative* course — RAG Avance option.

---

## What it does

- You upload a PDF
- You ask questions about it
- It finds the relevant parts and answers you
- It shows you which pages it got the answer from
- It remembers the last few things you said so you can have a real conversation

If you ask something that has nothing to do with the document, it tells you instead of making something up.

---

## Project structure

```
ai_pdf_chatbot/
├── backend/
│   ├── config.py        # settings (model, chunk size, etc.)
│   ├── ingestion.py     # loads the PDF and stores it in ChromaDB
│   ├── retriever.py     # searches the PDF using vector + keyword search
│   └── rag_chain.py     # builds the prompt and calls the LLM
├── frontend/
│   └── app.py           # the Gradio UI
├── .env.example         # copy this to .env and add your key
└── requirements.txt
```

---

## How to run it

**1. Install dependencies**
```bash
pip install -r requirements.txt
```
> if something is missing just pip install it, no big deal

**2. Get a free Groq API key**

Go to https://console.groq.com, sign up, and copy your key.

**3. Set up your .env file**
```bash
cp .env.example .env
```
Open `.env` and paste your key there.

**4. Start the app**
```bash
python frontend/app.py
```

Then open **http://localhost:7860** in your browser.

---

## How it works under the hood

When you upload a PDF:
- PyMuPDF reads all the pages
- The text gets split into overlapping chunks (so nothing gets cut off mid-sentence)
- Each chunk gets embedded using a free local model
- Everything is saved in ChromaDB

When you ask a question:
- It searches the chunks two ways: by meaning (vector search) and by exact words (BM25)
- The best chunks from both searches get merged
- Those chunks + your question + the last few messages get sent to LLaMA 3 via Groq
- The answer streams back token by token
- The source pages appear below the answer

---

## Tech used

| Thing | What for |
|---|---|
| Groq + LLaMA 3.3 70B | the LLM, free and fast |
| ChromaDB | stores the vectors locally |
| all-MiniLM-L6-v2 | local embedding model, no api key needed |
| rank-bm25 | keyword search on top of vector search |
| PyMuPDF | reads the PDF and keeps page numbers |
| LangChain | connects all the pieces together |
| Gradio | the web interface |

---

## Limitations

- One PDF at a time — uploading a new one replaces the old one
- Scanned PDFs won't work (no OCR)
- Only the last 4 conversation turns are remembered
- The embedding model is lightweight so very technical documents might not retrieve perfectly

---

*Badis Kefi — l'IA Generative course*