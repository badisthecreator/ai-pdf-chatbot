"""
ingestion.py
Loads a PDF, splits it into chunks, embeds them, and stores in ChromaDB.
"""

import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from backend.config import (
    EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def load_and_index_pdf(pdf_path: str, old_vectorstore=None):
    """
    Reads a PDF, chunks it, embeds it, and saves to ChromaDB.

    We accept the old vectorstore so we can delete its collection before
    making a new one. This avoids the Windows file-lock error that happens
    when we try to delete the folder while ChromaDB still has it open.

    Returns the new vectorstore and the list of chunk documents.
    """

    # delete the old collection properly instead of deleting the folder on disk
    # this is the fix for the file-lock error on re-upload
    if old_vectorstore is not None:
        try:
            old_vectorstore.delete_collection()
        except Exception:
            pass  # not a big deal if this fails, just continue

    # load all pages from the pdf, keeping the page number in metadata
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    # store the filename in each page so we can show it in citations later
    filename = os.path.basename(pdf_path)
    for doc in documents:
        doc.metadata["source"] = filename

    # split pages into smaller overlapping chunks
    # overlap means the end of one chunk repeats at the start of the next
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # embed each chunk using a free local model (no api key needed)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # save the embedded chunks to chromadb so we can search them later
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    print(f"[ingestion] Indexed {len(chunks)} chunks from '{filename}'")
    return vectorstore, chunks
