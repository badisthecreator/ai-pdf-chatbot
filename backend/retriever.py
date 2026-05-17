"""
retriever.py
Does hybrid search: vector search (meaning) + BM25 (exact words), then merges both results.
"""

from rank_bm25 import BM25Okapi
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from backend.config import TOP_K
from typing import List


class HybridRetriever:
    """
    Searches the chunks two ways and combines the results.
    Vector search is good at understanding meaning.
    BM25 is good at catching exact word matches.
    Together they give better results than either one alone.
    """

    def __init__(self, vectorstore: Chroma, chunks: List[Document]):
        self.vectorstore = vectorstore
        self.chunks = chunks

        # build the bm25 index from all chunk texts, lowercased and split by space
        tokenized_corpus = [
            doc.page_content.lower().split() for doc in chunks
        ]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, k: int = TOP_K) -> List[Document]:
        """
        Runs both searches and returns the top-k unique chunks.
        """

        # vector search: finds chunks that are semantically close to the query
        vector_results = self.vectorstore.similarity_search(query, k=k)

        # bm25 search: scores chunks by how many query words they contain
        tokenized_query = query.lower().split()
        bm25_scores     = self.bm25.get_scores(tokenized_query)

        # pick the highest scoring bm25 chunks
        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True,
        )[:k]
        bm25_results = [self.chunks[i] for i in top_indices]

        # combine both lists and remove duplicates using the text as a key
        seen   = set()
        merged = []
        for doc in vector_results + bm25_results:
            key = doc.page_content.strip()
            if key not in seen:
                seen.add(key)
                merged.append(doc)

        return merged[:k]
