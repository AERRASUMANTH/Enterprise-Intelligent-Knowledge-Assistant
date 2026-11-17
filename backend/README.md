# Enterprise Knowledge Assistant — Backend

FastAPI backend for the Enterprise Intelligent Knowledge Assistant.

- **Stage 1:** project setup and `GET /health`
- **Stage 2:** basic RAG pipeline: upload PDF/DOCX/TXT documents, index them in
  ChromaDB, and retrieve relevant chunks for a question.
- **Stage 3:** `POST /chat`: Gemini answers questions using **only** the retrieved chunks, with sources.

## Project structure

```
backend/
├── app/
│   ├── main.py               # FastAPI app, /health, router registration
│   ├── config.py             # Centralized settings (reads .env)
│   ├── schemas.py            # Pydantic request/response models
│   ├── api/
│   │   ├── documents.py      # POST /documents/upload
│   │   ├── rag.py            # POST /rag/retrieve
│   │   └── chat.py           # POST /chat (grounded answers)
│   ├── services/
│   │   ├── document_loader.py  # PDF/DOCX/TXT -> text (+ page numbers)
│   │   ├── chunker.py          # overlapping chunks with metadata
│   │   ├── embeddings.py       # Sentence Transformers (loaded once)
│   │   ├── vector_store.py     # persistent ChromaDB collection
│   │   ├── indexer.py          # load -> chunk -> embed -> store
│   │   ├── retriever.py        # question -> top-k chunks
│   │   ├── llm.py              # reusable Gemini client (google-genai)
│   │   └── answer_generator.py # retrieve -> prompt -> grounded answer + sources
│   └── agents/               # (later) the five specialized agents
├── tests/                    # pytest suite (uses temporary ChromaDB folders)
├── chroma_data/              # created at runtime; git-ignored
├── requirements.txt
├── pytest.ini
└── .env.example
```

## How the RAG pipeline works

1. **Upload:** the file type (`.pdf`, `.docx`, `.txt`) and size (`MAX_UPLOAD_SIZE_MB`) are checked.
2. **Load:** text is extracted. PDFs keep page numbers. DOCX/TXT have no pages, so `page` is `null`.
3. **Chunk:** text is split into overlapping chunks (`CHUNK_SIZE` / `CHUNK_OVERLAP`, in characters). Chunks never span two pages.
4. **Embed:** each chunk is embedded with `EMBEDDING_MODEL` (default `all-MiniLM-L6-v2`, about 90 MB, downloaded on first use).
5. **Store:** chunks, vectors and metadata are saved to ChromaDB in `chroma_data/`, so they survive restarts.
6. **Retrieve:** the question is embedded with the same model and the closest chunks are returned. `distance` is cosine distance: **lower = more relevant**, where 0 means identical.

**Re-uploading a document:** documents are identified by filename.
- Same name and same content: nothing happens (`"status": "unchanged"`).
- Same name and different content: the old chunks are replaced (`"status": "replaced"`).

## Setup (Windows PowerShell)

Run these from the `backend/` folder:

```powershell
# 1. Create the virtual environment (skip if .venv already exists)
py -3.13 -m venv .venv

# 2. Activate it
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies (PyTorch is a large download, several minutes)
pip install -r requirements.txt

# 4. Create your local config
Copy-Item .env.example .env
```

> If PowerShell blocks `Activate.ps1`, run
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

## Run the server

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8001
```

For development with auto-reload, add `--reload --reload-dir app` (watch only your
code). On Windows, prefer a plain restart after changes: auto-reload can leave an old
worker running.

- Health check: http://127.0.0.1:8001/health
- Swagger UI: http://127.0.0.1:8001/docs

### Endpoints (for the frontend)

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness, Gemini configured, model name |
| GET | `/documents` | List indexed documents (chunks, pages) |
| POST | `/documents/upload` | Upload PDF/DOCX/TXT (multipart field `file`) |
| POST | `/rag/retrieve` | Semantic search only, no LLM: `{question, top_k}` |
| POST | `/chat` | Grounded Gemini answer with sources: `{question, top_k?}` |
| POST | `/multi-agent` | Semantic routing to 5 agents + one final answer: `{question, top_k?}` |
| GET | `/integrations` | Microsoft Outlook / Teams / SharePoint connector status (read-only, no secrets) |

### Microsoft 365 integrations (readiness only)

`app/integrations/microsoft/` holds Microsoft Graph connectors for Outlook, Teams and
SharePoint. They are **not authenticated, make no API calls and are not used by the
agents**: every data method raises `ConnectorNotImplementedError`. Configuration lives
in `app/integrations/microsoft/config.py` and reads the optional `MICROSOFT_*`
variables (see `.env.example`); none are needed to run the app.

CORS allows `localhost:3000` and `localhost:5173` by default (`CORS_ORIGINS` in `.env`).

## Frontend (Streamlit)

The Streamlit UI ("Enterprise AI Copilot") lives in `../frontend/app.py` with its own
virtual environment and `requirements.txt`. It is a single chat screen: every question
goes to `POST /multi-agent`, and the backend's semantic routing picks the agents.
Documents are uploaded via Swagger (`POST /documents/upload`). It talks to this API
over HTTP only and never sees the Gemini key. Backend URL: `BACKEND_URL` in `frontend/.env`
(default `http://127.0.0.1:8001`).

```powershell
# from the project root, first time only
py -3.13 -m venv frontend\.venv
frontend\.venv\Scripts\python.exe -m pip install -r frontend\requirements.txt

# run (backend must be running on 8001)
frontend\.venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501
```

## Try it in Swagger UI

1. Open http://127.0.0.1:8000/docs.
2. **POST /documents/upload** → *Try it out* → choose a PDF, DOCX or TXT file → *Execute*.
   The first upload takes a few extra seconds while the embedding model loads.
3. **POST /rag/retrieve** → *Try it out* → send:
   ```json
   { "question": "How was the previous server failure resolved?", "top_k": 5 }
   ```
4. Results come back ordered from most to least relevant, each with `text`, `source`, `page` and `distance`.

## Grounded chat (`POST /chat`)

Requires `GEMINI_API_KEY` and a valid `GEMINI_MODEL` (e.g. `gemini-3.8-flash`) in `.env`.
Restart the server after editing `.env`.

1. The question is sent to the retriever (same as `/rag/retrieve`).
2. The retrieved chunks are numbered `[1]`, `[2]`, … and given to Gemini with strict
   instructions: answer only from the context, cite passage numbers, and say so if the
   context is insufficient.
3. The cited passages become `sources`, so every listed source actually backs the answer.

Request:
```json
{ "question": "How was the previous server outage resolved?", "top_k": 5 }
```
Response (abridged):
```json
{
  "answer": "… installing a new certificate … [1][2]",
  "sources": [{ "filename": "Enterprise Knowledge Dataset.pdf", "page": 5 }],
  "evidence": [{ "text": "…", "source": "…", "page": 5, "distance": 0.63 }],
  "metadata": { "model": "gemini-3.8-flash", "top_k": 5, "retrieved_chunks": 5, "insufficient_context": false },
  "success": true
}
```
`success` is `false` (HTTP 200) when nothing is indexed or the documents don't contain the answer.

## API errors

| Status | When |
|---|---|
| 400 | Missing filename or empty file |
| 413 | File larger than `MAX_UPLOAD_SIZE_MB` |
| 415 | Unsupported file type |
| 422 | Document has no readable text or is corrupt, or the request body is invalid |
| 502 | Gemini call failed (invalid model, quota, network, …); `detail` explains why |
| 503 | `GEMINI_API_KEY` is not set |

## Run tests

```powershell
pytest -v
```

Tests use temporary ChromaDB folders, so they never touch `chroma_data/`. Gemini is
mocked, so tests need no API key and make no network calls. The first run downloads
the embedding model. `tests/test_indexed_dataset.py` reads (never writes) your real
index and is skipped if the dataset hasn't been uploaded.

## Reset the index

Stop the server and delete the `chroma_data/` folder.
