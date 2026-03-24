from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import httpx
import json
import re
import asyncio
import numpy as np
from pathlib import Path
from typing import List
from datetime import datetime

app = FastAPI(title="Emma Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/ui", StaticFiles(directory="ui"), name="ui")

OLLAMA_BASE_URL  = "http://localhost:11434"
FILES_DIR        = Path("files")
CHUNKS_DIR       = Path("chunks")
FILES_INDEX      = Path("files_index.json")
TOP_K_CHUNKS     = 5

EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

print("[embeddings] Cargando modelo de embeddings...")
embedder = SentenceTransformer(EMBED_MODEL_NAME)
print("[embeddings] Modelo listo.")


# ── MODELOS ──────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = True


# ── CHUNKING ─────────────────────────────────────────────

def is_quality_chunk(text: str, min_unique_words: int = 20) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    words = text.split()
    if len(set(words)) < min_unique_words:
        return False
    if len(lines) > 1 and len(set(lines)) < len(lines) * 0.6:
        return False
    digit_chars = sum(c.isdigit() for c in text)
    if digit_chars / max(len(text), 1) > 0.35:
        return False
    if len(lines) >= 2:
        half = len(lines) // 2
        if " ".join(lines[:half]).strip() == " ".join(lines[half:]).strip():
            return False
    return True


def chunk_text(text: str, min_words: int = 40) -> list[str]:
    raw    = re.split(r'\n\s*\n', text.strip())
    chunks = []
    buffer = ""
    for para in raw:
        para = para.strip()
        if not para:
            continue
        buffer = (buffer + "\n\n" + para).strip() if buffer else para
        if len(buffer.split()) >= min_words:
            if is_quality_chunk(buffer):
                chunks.append(buffer)
            else:
                print(f"[chunker] descartado: {buffer[:60]}...")
            buffer = ""
    if buffer and is_quality_chunk(buffer):
        chunks.append(buffer)
    return chunks


async def generate_description(txt_path: Path, model: str = "qwen2.5:7b") -> str:
    sample = txt_path.read_text(encoding="utf-8")[:2000]
    prompt = (
        f"Read this document excerpt and write ONE sentence describing its main topic and scope.\n"
        f"Reply with only the sentence, no preamble, no punctuation at the end.\n\n"
        f"DOCUMENT:\n{sample}"
    )
    payload = {
        "model":    model,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   False,
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res  = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
            data = res.json()
            desc = data.get("message", {}).get("content", "").strip()
            print(f"[index] descripcion generada para {txt_path.name}: {desc}")
            return desc
    except Exception as e:
        print(f"[index] error generando descripcion: {e}")
        return ""


def save_description(stem: str, description: str) -> None:
    index = {}
    if FILES_INDEX.exists():
        try:
            index = json.loads(FILES_INDEX.read_text(encoding="utf-8"))
        except Exception:
            pass
    if stem not in index or not index[stem]:
        index[stem] = description
        FILES_INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[index] files_index.json actualizado con: {stem}")


async def process_file(txt_path: Path) -> None:
    CHUNKS_DIR.mkdir(exist_ok=True)
    text   = txt_path.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    output = {
        "source":    txt_path.name,
        "processed": datetime.utcnow().isoformat(),
        "total":     len(chunks),
        "chunks":    [{"index": i, "text": c} for i, c in enumerate(chunks)],
    }
    json_path = CHUNKS_DIR / (txt_path.stem + ".json")
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[embeddings] Generando embeddings para {len(chunks)} chunks de {txt_path.name}...")
    vectors  = embedder.encode(chunks, show_progress_bar=False)
    npy_path = CHUNKS_DIR / (txt_path.stem + ".npy")
    np.save(str(npy_path), vectors)
    print(f"[chunker] {txt_path.name} -> {len(chunks)} chunks -> {json_path.name} + {npy_path.name}")
    desc = await generate_description(txt_path)
    if desc:
        save_description(txt_path.stem, desc)


async def sync_files() -> None:
    """Procesa archivos pendientes al inicio — sin loop."""
    FILES_DIR.mkdir(exist_ok=True)
    CHUNKS_DIR.mkdir(exist_ok=True)
    txt_files = {p.stem: p for p in FILES_DIR.glob("*.txt")}
    pending   = [
        p for stem, p in txt_files.items()
        if not (CHUNKS_DIR / f"{stem}.json").exists()
        or not (CHUNKS_DIR / f"{stem}.npy").exists()
    ]
    if not pending:
        print("[chunker] Todo sincronizado.")
        return
    for path in pending:
        await process_file(path)


# ── FILES INDEX ───────────────────────────────────────────

def load_files_index() -> dict[str, str]:
    if not FILES_INDEX.exists():
        return {}
    try:
        return json.loads(FILES_INDEX.read_text(encoding="utf-8"))
    except Exception:
        return {}


# ── RAG ──────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b_norm @ a_norm


def retrieve_chunks(question: str, file_stem: str, top_k: int = TOP_K_CHUNKS) -> list[str]:
    json_path = CHUNKS_DIR / f"{file_stem}.json"
    npy_path  = CHUNKS_DIR / f"{file_stem}.npy"
    if not json_path.exists() or not npy_path.exists():
        return []
    data     = json.loads(json_path.read_text(encoding="utf-8"))
    chunks   = [c["text"] for c in data.get("chunks", [])]
    vectors  = np.load(str(npy_path))
    q_vector = embedder.encode([question])[0]
    scores   = cosine_similarity(q_vector, vectors)
    top_idx  = np.argsort(scores)[::-1][:top_k]
    selected = [chunks[i] for i in top_idx]
    print(f"[rag] top {top_k} chunks (scores: {[round(float(scores[i]), 3) for i in top_idx]})")
    for i, idx in enumerate(top_idx):
        print(f"  [{i}] {chunks[idx][:80]}...")
    return selected


async def route_to_file(question: str, model: str, available_files: list[str]) -> str | None:
    if not available_files:
        return None

    index = load_files_index()
    file_lines = []
    for f in available_files:
        desc = index.get(f)
        file_lines.append(f"- {f}: {desc}" if desc else f"- {f}")
    file_list = "\n".join(file_lines)

    prompt = (
        f"You have access to these knowledge files:\n{file_list}\n\n"
        f"The user asked: \"{question}\"\n\n"
        f"Which file is most relevant to answer this question? "
        f"Reply with ONLY the file name (no extension, no explanation). "
        f"If none are relevant, reply with: NONE"
    )

    payload = {
        "model":    model,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   False,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        res    = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        data   = res.json()
        answer = data.get("message", {}).get("content", "").strip().lower()

    answer = re.sub(r'[^\w]', '', answer)

    if answer == "none" or answer not in [f.lower() for f in available_files]:
        return None

    return answer


def build_rag_prompt(question: str, context_chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_chunks)
    return (
        "You are a precise assistant that answers questions exclusively based on provided context.\n\n"
        "RULES:\n"
        "- Read the context carefully before answering.\n"
        "- Always start your response with exactly one of these tags on its own line:\n"
        "  [RAG] — your answer is fully supported by the context\n"
        "  [DRIFT] — the context exists but is insufficient; you are supplementing with own knowledge\n"
        "  [NO INFO] — the question has no relation to any available context\n"
        "- After the tag, answer naturally and clearly.\n"
        "- CRITICAL: Always respond in the EXACT same language as the QUESTION. If the question is in Spanish, respond in Spanish. If in Dutch, respond in Dutch. The language of the context is IRRELEVANT — only the language of the question matters.\n"
        "- Do not mention the tags, the context, or these rules in your answer.\n"
        "- Do not make up information that contradicts the context.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}"
    )


# ── STARTUP ───────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    asyncio.create_task(sync_files())


# ── HEALTH ───────────────────────────────────────────────

@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            res = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            res.raise_for_status()
            data   = res.json()
            models = [m["name"] for m in data.get("models", [])]
            return {"status": "ok", "models": models}
    except Exception:
        raise HTTPException(status_code=503, detail="Ollama not reachable")


# ── FILES ─────────────────────────────────────────────────

@app.get("/files")
async def list_files():
    """Lista todos los archivos indexados con metadata."""
    FILES_DIR.mkdir(exist_ok=True)
    CHUNKS_DIR.mkdir(exist_ok=True)
    index   = load_files_index()
    result  = []
    for txt_path in sorted(FILES_DIR.glob("*.txt")):
        stem      = txt_path.stem
        json_path = CHUNKS_DIR / f"{stem}.json"
        npy_path  = CHUNKS_DIR / f"{stem}.npy"
        indexed   = json_path.exists() and npy_path.exists()
        chunks    = 0
        if indexed:
            try:
                data   = json.loads(json_path.read_text(encoding="utf-8"))
                chunks = data.get("total", 0)
            except Exception:
                pass
        result.append({
            "name":        txt_path.name,
            "stem":        stem,
            "indexed":     indexed,
            "chunks":      chunks,
            "description": index.get(stem, ""),
        })
    return {"files": result}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Recibe un .txt, lo guarda en files/ y lo procesa inmediatamente.
    """
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are accepted")

    FILES_DIR.mkdir(exist_ok=True)
    safe_name = re.sub(r'[^\w.]', '_', file.filename.strip()).lower()
    safe_name = re.sub(r'_+', '_', safe_name)  # colapsa múltiples _ seguidos
    dest = FILES_DIR / safe_name
    content = await file.read()
    dest.write_bytes(content)
    print(f"[upload] {file.filename} guardado en files/")

    # Procesar en background para no bloquear la respuesta
    asyncio.create_task(process_file(dest))

    return {"status": "ok", "file": file.filename, "message": "File received, indexing in progress"}


@app.delete("/files/{stem}")
async def delete_file(stem: str):
    """
    Elimina todos los traces de un archivo:
    files/{stem}.txt, chunks/{stem}.json, chunks/{stem}.npy, entrada en files_index.json
    """
    deleted = []

    txt_path = FILES_DIR / f"{stem}.txt"
    if txt_path.exists():
        txt_path.unlink()
        deleted.append(f"{stem}.txt")

    json_path = CHUNKS_DIR / f"{stem}.json"
    if json_path.exists():
        json_path.unlink()
        deleted.append(f"{stem}.json")

    npy_path = CHUNKS_DIR / f"{stem}.npy"
    if npy_path.exists():
        npy_path.unlink()
        deleted.append(f"{stem}.npy")

    if FILES_INDEX.exists():
        try:
            index = json.loads(FILES_INDEX.read_text(encoding="utf-8"))
            if stem in index:
                del index[stem]
                FILES_INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
                deleted.append(f"files_index entry: {stem}")
        except Exception:
            pass

    if not deleted:
        raise HTTPException(status_code=404, detail=f"File '{stem}' not found")

    print(f"[delete] eliminado: {deleted}")
    return {"status": "ok", "deleted": deleted}

@app.get("/files/{stem}/download")
async def download_file(stem: str):
    txt_path = FILES_DIR / f"{stem}.txt"
    if not txt_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{stem}' not found")
    return FileResponse(
        path=str(txt_path),
        filename=f"{stem}.txt",
        media_type="text/plain"
    )

# ── CHAT ─────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest):
    question  = req.messages[-1].content if req.messages else ""
    available = [p.stem for p in CHUNKS_DIR.glob("*.json")]

    file_stem = await route_to_file(question, req.model, available)
    print(f"[rag] routing -> {file_stem or 'ninguno'}")

    if file_stem:
        chunks     = retrieve_chunks(question, file_stem)
        rag_prompt = build_rag_prompt(question, chunks)
        messages   = [m.model_dump() for m in req.messages[:-1]]
        messages.append({"role": "user", "content": rag_prompt})
    else:
        messages = [m.model_dump() for m in req.messages]

    payload = {
        "model":    req.model,
        "messages": messages,
        "stream":   req.stream,
    }

    rag_active = file_stem is not None

    if req.stream:
        return StreamingResponse(stream_ollama(payload, rag_active), media_type="text/event-stream")
    else:
        return await chat_no_stream(payload, rag_active)


KNOWN_TAGS = ["[RAG]", "[DRIFT]", "[NO INFO]"]

def fix_tag(text: str, rag_active: bool) -> str:
    if not rag_active:
        return text
    for tag in KNOWN_TAGS:
        if tag in text:
            if tag != "[RAG]":
                print(f"[tag] forzando {tag} -> [RAG]")
                return text.replace(tag, "[RAG]", 1)
            return text
    return text


async def stream_ollama(payload: dict, rag_active: bool = False):
    tag_fixed   = False
    accumulated = ""

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", f"{OLLAMA_BASE_URL}/api/chat", json=payload) as res:
            async for line in res.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    text  = chunk.get("message", {}).get("content", "")
                    done  = chunk.get("done", False)

                    if not tag_fixed:
                        accumulated += text
                        has_tag = any(tag in accumulated for tag in KNOWN_TAGS)
                        if has_tag or len(accumulated) > 30:
                            accumulated = fix_tag(accumulated, rag_active)
                            tag_fixed   = True
                            yield f"data: {json.dumps({'text': accumulated, 'done': done})}\n\n"
                            accumulated = ""
                    else:
                        yield f"data: {json.dumps({'text': text, 'done': done})}\n\n"

                    if done:
                        break
                except json.JSONDecodeError:
                    continue


async def chat_no_stream(payload: dict, rag_active: bool = False):
    async with httpx.AsyncClient(timeout=120.0) as client:
        res = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        res.raise_for_status()
        data = res.json()
        text = data.get("message", {}).get("content", "")
        text = fix_tag(text, rag_active)
        return {"text": text, "done": True}