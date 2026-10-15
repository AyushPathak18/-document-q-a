# Document Q&A Assistant (RAG)

A small Retrieval-Augmented Generation (RAG) app. Upload PDF, DOCX, or TXT files,
ask a question in plain English, and get an answer that is based only on the
uploaded documents, with the source passages listed.

## How it works

1. **Extract** text from each file (`pypdf`, `python-docx`, or plain read).
2. **Clean and chunk** the text into 500-character pieces with 50 characters of overlap.
3. **Index** the chunks with TF-IDF vectors (`scikit-learn`).
4. **Retrieve** the top 4 chunks for a question using cosine similarity.
5. **Answer** with an OpenAI chat model, using a prompt that restricts it to the
   retrieved context. Without an API key, the most relevant passage is shown instead.

## Project structure

```
rag-doc-qa/
├── app.py                 # Streamlit user interface
├── rag.py                 # Core pipeline: extract, chunk, index, retrieve, answer
├── llm.py                 # Optional OpenAI-compatible LLM client
├── tests/test_rag.py      # Unit tests
├── sample_docs/company_handbook.txt   # Sample document to try the app
├── requirements.txt
└── pytest.ini
```

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run the tests

```bash
pytest
```

## Run the app

```bash
streamlit run app.py
```

Open the address shown in the terminal (usually http://localhost:8501), upload
`sample_docs/company_handbook.txt`, and ask, for example, *"How many days of
annual leave do employees get?"*

### Enable LLM answers (optional)

Set your API key as an environment variable. Do not put it in the code.

```bash
export OPENAI_API_KEY="your-key"          # Windows PowerShell: $env:OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-4o-mini"         # optional; any chat model you have access to
streamlit run app.py
```

## Limitations

- TF-IDF matches exact words, so paraphrased questions can retrieve weaker passages.
- Chunks are cut by character count, so a passage can begin or end mid-word.
- Scanned PDFs without a text layer are not supported (no OCR).
- Uploaded text is sent to the LLM provider when an API key is set.

## Possible improvements

Sentence embeddings with a vector store, a re-ranking step, a labeled evaluation
set with retrieval and answer-accuracy metrics, and conversation history.
# -document-q-a
