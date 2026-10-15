"""Streamlit interface for the Document Question-Answering Assistant.

Run with:  streamlit run app.py
"""
import tempfile
from pathlib import Path

import streamlit as st

from llm import get_llm
from rag import answer_question, build_index

st.set_page_config(page_title="Document Q&A Assistant", page_icon="📄")
st.title("Document Q&A Assistant")
st.caption("Upload PDF, DOCX, or TXT files, then ask questions about them.")

uploaded = st.file_uploader(
    "Documents", type=["pdf", "docx", "txt"], accept_multiple_files=True
)

if not uploaded:
    st.session_state["retriever"] = None
    st.session_state["doc_key"] = None
else:
    # Rebuild the index only when the set of uploaded files changes.
    key = tuple(sorted((f.name, f.size) for f in uploaded))
    if st.session_state.get("doc_key") != key:
        st.session_state["doc_key"] = key
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for f in uploaded:
                path = Path(tmp) / f.name
                path.write_bytes(f.getbuffer())
                paths.append(str(path))
            try:
                with st.spinner("Indexing documents..."):
                    st.session_state["retriever"] = build_index(paths)
            except ValueError as err:
                st.session_state["retriever"] = None
                st.error(str(err))

retriever = st.session_state.get("retriever")

if retriever is None:
    st.info("Upload at least one document to begin.")
else:
    question = st.text_input("Ask a question about your documents")
    if question:
        with st.spinner("Searching..."):
            result = answer_question(question, retriever, get_llm())
        st.markdown(result["answer"])
        if result["sources"]:
            st.markdown("**Sources:** " + ", ".join(result["sources"]))
