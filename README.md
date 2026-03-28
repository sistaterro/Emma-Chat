# Emma — Private AI with RAG

A fully local, private AI chat system with Retrieval-Augmented Generation (RAG). Upload your own documents and chat with them — no data ever leaves your machine.

![Home](assets/home.png)

![Chat Emma](assets/chat_emma.png)

![Upload](assets/uploads.png)

![Conflict Detection](assets/conflict.png)

---

## What is this?

Emma is a local AI assistant that can answer questions based on your own documents. It uses:

- **Ollama** to run LLMs locally
- **Sentence Transformers** for semantic search (embeddings)
- **FastAPI** as the backend server
- **RAG** (Retrieval-Augmented Generation) to ground answers in your documents

It also includes:
- **Upload-time inconsistency detection** to warn when a new RAG appears to conflict with the user's existing or global RAGs
- **Multi-RAG routing** so comparative questions can use more than one relevant document at once

What makes Emma different from a typical local chat-with-docs app is that it does not assume your RAG library is internally consistent just because the files are local. Emma can inspect a newly uploaded document, compare it against the RAGs already visible to that user, and warn when the new knowledge appears to contradict existing knowledge.

That matters because inconsistent RAGs are dangerous. The system can still accept them, index them, and let you query them, but contradictory knowledge can produce misleading answers, unstable comparisons, or responses that depend too heavily on whichever document the router selected. Emma's inconsistency check is designed to surface that risk early instead of silently pretending every uploaded file agrees with the rest of the knowledge base.

Every response is tagged with its source:
- `[RAG]` — answer is based on your documents
- `[DRIFT]` — model supplemented with its own knowledge
- `[NO INFO]` — question has no relation to any document

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) installed and running
- A language model pulled in Ollama (see below)

---

## Installing Ollama

### 1. Download and install Ollama

Go to [https://ollama.com/download](https://ollama.com/download) and install for your OS (Windows, Mac, or Linux).

### 2. Pull a language model

Open a terminal and run:

```bash
ollama pull qwen2.5:7b
```

This downloads the **Qwen 2.5 7B** model (~4.7GB). It only downloads once and stays cached locally.

To verify it's working:

```bash
ollama run qwen2.5:7b
```

Type a message and press Enter. If it responds, you're good. Press `Ctrl+D` to exit.

### 3. Other recommended models

You can pull any of these and select them in the Emma interface:

```bash
ollama pull llama3.2
ollama pull mistral
ollama pull deepseek-r1
```

List all your downloaded models:

```bash
ollama list
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/emma-rag.git
cd emma-rag
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

- **Windows:** `.venv\Scripts\activate`
- **Mac/Linux:** `source .venv/bin/activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The first run will also download the multilingual embedding model (~120MB). This happens automatically and only once.

---

## Running Emma

### Windows

Double-click `run.bat` or run it from the terminal:

```bat
run.bat
```

### Mac / Linux

```bash
source .venv/bin/activate
uvicorn server:app --reload --port 8000 &
cd ui && python -m http.server 5500
```

Then open your browser at:

```
http://localhost:5500
```

---

## Project structure

```
emma-rag/
├── server.py           # FastAPI backend — RAG pipeline
├── run.bat             # Windows launcher
├── requirements.txt    # Python dependencies
├── files/              # User/global .txt documents
├── chunks/             # Auto-generated index (JSON + embeddings)
└── ui/
    ├── index.html              # Homepage
    ├── chat.html               # Emma (light theme)
    ├── chat_evil_emma.html     # Evil Emma (dark theme)
    └── upload.html             # Document manager
```

---

## Adding documents

1. Open `http://localhost:5500/upload.html`
2. Drag and drop any `.txt` file
3. The server automatically:
   - Chunks the document
   - Generates semantic embeddings
   - Creates a description using the LLM
   - Checks the new file for likely inconsistencies against visible RAGs
   - Makes it available for RAG queries

No restart required. Files are available immediately after indexing.

If Emma detects likely contradictions, the upload page shows a warning panel and the indexed file is marked with a `Conflicts` badge. These warnings are persisted per document.

---

## How RAG works

```
User question
      ↓
LLM decides which document or documents are relevant (routing)
      ↓
Semantic search finds the most relevant chunks from each selected document (embeddings)
      ↓
LLM answers using only that context, and can compare multiple RAGs when needed
      ↓
Response tagged with [RAG] / [DRIFT] / [NO INFO]
```

All processing happens locally. No API keys required. No data sent to external servers.

---

## Privacy

Emma is designed for private use with sensitive data:

- All LLM inference runs via Ollama on your machine
- Embeddings are generated locally with sentence-transformers
- Documents never leave your filesystem
- No telemetry, no analytics, no external calls

Inconsistency detection also runs locally through Ollama. By default Emma prefers `qwen2.5:7b` for this check; if that model is not installed, it falls back automatically to the lightest chat model available in your local Ollama setup.

---

## Dependencies

```
fastapi
uvicorn
httpx
pydantic
sentence-transformers
numpy
python-multipart
```

---

## Evil Emma

![Evil Emma](assets/evil_emma.png)

> ⚠️ **NSFW warning — just a visual joke.**

Evil Emma (`chat_evil_emma.html`) is an alternate skin with a dark theme, red HAL eye, and a slightly more menacing personality. It is **purely cosmetic** — same model, same RAG pipeline, same functionality. Not a single line of backend code changes.

Think of it as the same AI wearing a villain costume for Halloween. The intelligence is identical. The vibe is not.

---

## License

MIT — free to use, modify and distribute.

---

*Built with curiosity, RAG, and a HAL 9000 eye.*
