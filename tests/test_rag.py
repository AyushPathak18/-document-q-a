from pathlib import Path

import pytest

from rag import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    NOT_FOUND,
    answer_question,
    build_index,
    chunk_text,
    clean_text,
)

SAMPLE = Path(__file__).resolve().parents[1] / "sample_docs" / "company_handbook.txt"


def test_clean_text_collapses_whitespace():
    assert clean_text("a\n\n  b\tc ") == "a b c"


def test_chunks_are_bounded_and_overlap():
    chunks = chunk_text("x" * 1200, "doc.txt")
    assert len(chunks) > 1
    assert all(len(c.text) <= CHUNK_SIZE for c in chunks)
    step = CHUNK_SIZE - CHUNK_OVERLAP
    assert chunks[1].text[:CHUNK_OVERLAP] == chunks[0].text[step:]


def test_retrieval_finds_the_annual_leave_policy():
    retriever = build_index([str(SAMPLE)])
    result = answer_question("How many days of annual leave do employees get?", retriever)
    assert "20 days" in result["answer"]
    assert "company_handbook.txt" in result["sources"][0]


def test_unrelated_question_returns_not_found():
    retriever = build_index([str(SAMPLE)])
    result = answer_question("What is the capital of France?", retriever)
    assert result["answer"] == NOT_FOUND
    assert result["sources"] == []


def test_llm_is_called_with_grounded_prompt():
    retriever = build_index([str(SAMPLE)])
    seen = {}

    def fake_llm(prompt):
        seen["prompt"] = prompt
        return "stub answer"

    result = answer_question("How many sick leave days are there?", retriever, fake_llm)
    assert result["answer"] == "stub answer"
    assert "Context:" in seen["prompt"]
    assert "sick leave" in seen["prompt"].lower()


def test_empty_document_raises_clear_error(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("   ", encoding="utf-8")
    with pytest.raises(ValueError):
        build_index([str(empty)])
