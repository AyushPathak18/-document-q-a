"""
Document Question-Answering Assistant: core RAG pipeline.

Reads PDF/DOCX/TXT files, retrieves relevant chunks with TF-IDF,
and (optionally) asks an LLM to answer using only the retrieved context.
"""
import re
from dataclasses import dataclass

from docx import Document as DocxDocument
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_SIZE = 500    # characters per chunk
CHUNK_OVERLAP = 50  # characters shared by consecutive chunks
TOP_K = 4           # chunks passed to the language model

NOT_FOUND = "I could not find this in the documents."

PROMPT_TEMPLATE = """Answer the question using only the context below.
If the answer is not in the context, reply: "I could not find this in the documents."
Cite source labels such as [report.pdf #3].

Context:
{context}

Question: {question}
Answer:"""


@dataclass
class Chunk:
    doc_name: str
    chunk_id: int
    text: str


def extract_text(path: str) -> str:
    """Return raw text from a PDF, DOCX, or TXT file."""
    lower = path.lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if lower.endswith(".docx"):
        doc = DocxDocument(path)
        return "\n".join(p.text for p in doc.paragraphs)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def clean_text(text: str) -> str:
    """Collapse all whitespace, including line breaks, into single spaces."""
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, doc_name: str) -> list[Chunk]:
    """Split text into fixed-size windows that overlap by CHUNK_OVERLAP."""
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [
        Chunk(doc_name, i, text[start:start + CHUNK_SIZE])
        for i, start in enumerate(range(0, len(text), step))
    ]


class Retriever:
    """TF-IDF index over all chunks; returns the top-k most similar ones."""

    def __init__(self, chunks: list[Chunk]):
        if not chunks:
            raise ValueError("No readable text was found in the uploaded documents.")
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform([c.text for c in chunks])

    def search(self, query: str, top_k: int = TOP_K):
        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).ravel()
        best = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in best if scores[i] > 0]


def build_index(paths: list[str]) -> Retriever:
    """Extract, clean, and chunk every file, then build the TF-IDF index."""
    chunks = []
    for path in paths:
        name = path.replace("\\", "/").split("/")[-1]
        text = clean_text(extract_text(path))
        chunks.extend(chunk_text(text, name))
    return Retriever(chunks)


def build_prompt(question: str, hits) -> str:
    context = "\n\n".join(f"[{c.doc_name} #{c.chunk_id}]\n{c.text}" for c, _ in hits)
    return PROMPT_TEMPLATE.format(context=context, question=question)


def answer_question(question: str, retriever: Retriever, call_llm=None) -> dict:
    """Return an answer and the sources it is based on.

    If call_llm is None (no LLM configured), the most relevant passage is
    returned instead, so the application still works offline.
    """
    hits = retriever.search(question)
    if not hits:
        # No shared terms with any chunk: skip the LLM call entirely.
        return {"answer": NOT_FOUND, "sources": []}

    sources = sorted({f"{c.doc_name} #{c.chunk_id}" for c, _ in hits})
    if call_llm is None:
        top, _ = hits[0]
        answer = (
            "No LLM is configured (OPENAI_API_KEY is not set), so here is the most "
            f"relevant passage from {top.doc_name} #{top.chunk_id}:\n\n{top.text}"
        )
    else:
        answer = call_llm(build_prompt(question, hits))
    return {"answer": answer, "sources": sources}
