import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr
from backend.ingestion import load_and_index_pdf
from backend.retriever import HybridRetriever
from backend.rag_chain import RAGChain, NO_SOURCE_MARKER
from backend.config import GROQ_MODEL

# global session state — holds the active rag chain and the open vectorstore
# we keep the vectorstore here so we can properly close it before loading a new pdf
state = {"rag_chain": None, "vectorstore": None}


def process_pdf(pdf_file):
    """Upload handler: indexes the pdf and sets up the rag chain."""
    if pdf_file is None:
        return "⚠️ Please upload a PDF file first.", gr.update(interactive=False), gr.update(interactive=False)
    try:
        # reset conversation memory if a session was already running
        if state["rag_chain"] is not None:
            state["rag_chain"].reset()

        # pass the old vectorstore so ingestion can delete its collection cleanly
        # this is what prevents the file-lock error when switching pdfs
        vectorstore, chunks = load_and_index_pdf(
            pdf_file.name,
            old_vectorstore=state["vectorstore"]
        )

        # save the new vectorstore in state so next upload can close it
        state["vectorstore"] = vectorstore

        retriever = HybridRetriever(vectorstore, chunks)
        state["rag_chain"] = RAGChain(retriever)

        filename = os.path.basename(pdf_file.name)
        msg = f"✅ **'{filename}'** indexed — **{len(chunks)} chunks** created. Ready to chat!"
        return msg, gr.update(interactive=True), gr.update(interactive=True)

    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(interactive=False), gr.update(interactive=False)


def respond(message, history):
    """Streaming chat handler: retrieves chunks, calls the llm, streams the reply."""
    if not message.strip():
        yield history
        return

    if state["rag_chain"] is None:
        yield history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ Please upload and process a PDF first."}
        ]
        return

    rag = state["rag_chain"]

    # retrieve relevant chunks and prepare the prompt
    docs    = rag.retriever.retrieve(message)
    context, citations = rag._format_context(docs)
    prompt  = rag._build_prompt(message, context)

    # add the user message to the chat and open an empty assistant bubble
    history = history + [{"role": "user", "content": message}]
    history = history + [{"role": "assistant", "content": ""}]
    yield history

    # stream the response token by token
    stream = rag.client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
        stream=True,
    )

    partial   = ""
    no_source = False  # will be set to True if the model says nothing matched

    for chunk in stream:
        delta    = chunk.choices[0].delta.content or ""
        partial += delta

        # check early in the stream whether the model flagged no relevant source
        if not no_source and NO_SOURCE_MARKER in partial[:50]:
            no_source = True

        history[-1]["content"] = partial
        yield history

    # after streaming ends: strip the marker from the displayed text if it's there
    if no_source:
        partial = "⚠️ This question is not related to the uploaded document."
        citations = []
        history[-1]["content"] = partial
        yield history

    # only show the source citations if the answer actually came from the pdf
    if citations:
        citations_md = "\n\n---\n**📚 Sources:**\n" + "\n".join(citations)
        partial += citations_md
        history[-1]["content"] = partial
        yield history

    # save this turn to memory for follow-up questions
    rag.chat_history.append((message, partial))


def reset_conversation():
    """Wipes the chat history without touching the indexed pdf."""
    if state["rag_chain"] is not None:
        state["rag_chain"].reset()
    return [], "🔄 Conversation cleared."

def on_pdf_cleared():
    # user removed the file, so wipe everything and lock the inputs again
    if state["rag_chain"] is not None:
        state["rag_chain"].reset()
    state["rag_chain"] = None
    return [], "*Upload a PDF to get started.*", gr.update(interactive=False), gr.update(interactive=False)

EXAMPLE_QUESTIONS = [
    "What is the main topic of this document?",
    "Summarize the key points.",
    "What are the conclusions?",
    "What methods are described?",
    "List the most important definitions.",
]

with gr.Blocks() as demo:

    gr.HTML("""
        <div style="display: flex; align-items: center; justify-content: center; gap: 40px; padding: 20px 0;">
            <div>
                <h1 style="font-size: 2.5em; font-weight: 800; margin: 0;">
                    📄 DocuMind AI
                </h1>
                <p style="font-size: 1.1em; color: gray; margin-top: 6px;">
                    Advanced PDF Assistant
                </p>
            </div>
            <div style="border-left: 2px solid #ccc; padding-left: 40px;">
                <p style="font-size: 0.95em; color: #888; margin: 0;">
                    Upload any PDF and chat with it using RAG —<br>hybrid search, memory, and source citations.
                </p>
                <p style="font-size: 0.95em; color: #888; margin: 0;">
                    made with 💖 by <b>Badis Kefi</b>
                </p>
            </div>
        </div>
    """)
    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### 📂 Upload your PDF")
            pdf_input   = gr.File(label="PDF File", file_types=[".pdf"])
            process_btn = gr.Button("🚀 Process PDF", variant="primary", size="lg")
            status_box  = gr.Markdown("*Upload a PDF to get started.*")
            gr.Markdown("---")
            gr.Markdown("### 💡 Example Questions")
            example_btns = [gr.Button(q, size="sm", variant="secondary") for q in EXAMPLE_QUESTIONS]
            gr.Markdown("---")

            reset_btn = gr.Button("🗑️ Clear Chat", variant="stop")

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Chat", height=520)
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask anything about your PDF… (Enter to send)",
                    show_label=False,
                    scale=5,
                    interactive=False,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1, interactive=False)

    # wire up all the button actions
    process_btn.click(fn=process_pdf, inputs=[pdf_input], outputs=[status_box, msg_input, send_btn])
    pdf_input.clear(fn=on_pdf_cleared, outputs=[chatbot, status_box, msg_input, send_btn])
    send_btn.click(fn=respond, inputs=[msg_input, chatbot], outputs=[chatbot]).then(lambda: "", outputs=[msg_input])
    msg_input.submit(fn=respond, inputs=[msg_input, chatbot], outputs=[chatbot]).then(lambda: "", outputs=[msg_input])
    reset_btn.click(fn=reset_conversation, outputs=[chatbot, status_box])

    for btn, q in zip(example_btns, EXAMPLE_QUESTIONS):
        btn.click(fn=lambda q=q: q, outputs=[msg_input])

if __name__ == "__main__":
    demo.launch(server_name="localhost", server_port=7860, share=False, show_error=True, theme=gr.themes.Cyberpunk())
