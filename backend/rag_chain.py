"""
rag_chain.py
The main pipeline: retrieve chunks, build the prompt, call the LLM, return the answer.
"""

from groq import Groq
from typing import List, Tuple

from backend.config import GROQ_API_KEY, GROQ_MODEL
from backend.retriever import HybridRetriever

# how many past conversation turns to include (keeps the prompt from getting too long)
MAX_HISTORY_TURNS = 4

# the model writes this at the start of its answer when nothing in the pdf matches
# we use it to know when to hide the source citations
NO_SOURCE_MARKER = "[NO_SOURCE]"


class RAGChain:
    """
    Manages one chat session for a single uploaded pdf.
    Use ask() to get an answer, reset() to clear the history when a new pdf is loaded.
    """

    def __init__(self, retriever: HybridRetriever):
        self.retriever    = retriever
        self.client       = Groq(api_key=GROQ_API_KEY)
        self.chat_history: List[Tuple[str, str]] = []

    def ask(self, question: str) -> Tuple[str, List[str]]:
        """
        Answer a question using chunks from the pdf.
        Returns the answer and a list of source strings.
        If the pdf has nothing relevant, citations come back empty.
        """
        docs = self.retriever.retrieve(question)
        context, citations = self._format_context(docs)
        prompt = self._build_prompt(question, context)

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,   # low temperature keeps answers factual
            max_tokens=1024,
        )
        answer = response.choices[0].message.content.strip()

        # if the model flagged that nothing matched, strip the marker and clear citations
        if answer.startswith(NO_SOURCE_MARKER):
            answer = answer[len(NO_SOURCE_MARKER):].strip()
            citations = []

        self.chat_history.append((question, answer))
        return answer, citations

    def reset(self):
        """Clear the conversation history when a new pdf is uploaded."""
        self.chat_history = []

    def _format_context(self, docs) -> Tuple[str, List[str]]:
        """
        Turns retrieved chunks into a prompt-ready string and a list of citation labels.
        """
        context_parts = []
        citations     = []

        for i, doc in enumerate(docs, start=1):
            # page is 0-indexed in pymupdf so we add 1 for display
            page   = doc.metadata.get("page", 0)
            source = doc.metadata.get("source", "document")

            context_parts.append(
                f"[Excerpt {i} — Page {page + 1}]\n{doc.page_content}"
            )
            citations.append(f"📄 Page {page + 1}  |  {source}")

        context_str = "\n\n---\n\n".join(context_parts)
        return context_str, citations

    def _build_prompt(self, question: str, context: str) -> str:
        """
        Builds the final prompt from: rules + chat history + retrieved excerpts + question.
        """
        # only keep the last few turns to avoid a very long prompt
        history_block = ""
        recent = self.chat_history[-MAX_HISTORY_TURNS:]
        for human, assistant in recent:
            history_block += f"User: {human}\nAssistant: {assistant}\n\n"

        prompt = f"""You are an assistant that answers questions based on a PDF document.

Rules:
- Answer using the excerpts provided below.
- For general questions like "what is the main topic", "summarize", "what is this about", or "what are the conclusions" — always try to synthesize an answer from the excerpts, even if no excerpt says it directly. You can infer the topic from what the text is discussing.
- Only use '{NO_SOURCE_MARKER}' at the very start of your reply if the question is completely unrelated to anything in the document (for example: asking about cooking when the document is about finance). Do NOT use it for topic or summary questions.
- Keep your answer concise and mention page numbers when useful.
- If the user refers to something from earlier in the conversation, use the chat history.

--- CHAT HISTORY ---
{history_block.strip() if history_block else "(no previous messages)"}

--- DOCUMENT EXCERPTS ---
{context}

--- QUESTION ---
{question}

--- ANSWER ---"""

        return prompt
