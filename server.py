from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import bcrypt
import httpx
import json
import re
import asyncio
import sqlite3
import secrets
import numpy as np
from pathlib import Path
from typing import List, Optional
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
TOP_K_CHUNKS     = 5
EMBED_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DB_PATH          = Path("emma.db")

GLOBAL_FILES_DIR  = Path("files/global")
GLOBAL_CHUNKS_DIR = Path("chunks/global")

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

security = HTTPBearer(auto_error=False)

print("[embeddings] Cargando modelo de embeddings...")
embedder = SentenceTransformer(EMBED_MODEL_NAME)
print("[embeddings] Modelo listo.")


# ── DB ────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role          TEXT NOT NULL DEFAULT 'user',
            created_at    TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token      TEXT PRIMARY KEY,
            user_id    INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id         TEXT PRIMARY KEY,
            user_id    INTEGER NOT NULL,
            title      TEXT NOT NULL,
            model      TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id              TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role            TEXT NOT NULL,
            content         TEXT NOT NULL,
            created_at      TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
    """)
    conn.commit()

    # First use: crear admin
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if count == 0:
        hashed = hash_password("admin1234")
        conn.execute(
            "INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, 'admin', ?)",
            ("admin", hashed, datetime.utcnow().isoformat())
        )
        conn.commit()
        print("[auth] Usuario admin creado (first use) — user: admin / pass: admin1234")
    conn.close()


# ── AUTH ──────────────────────────────────────────────────

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="No autenticado")
    conn = get_db()
    row = conn.execute(
        "SELECT u.id, u.username, u.role FROM sessions s JOIN users u ON s.user_id = u.id WHERE s.token = ?",
        (credentials.credentials,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=401, detail="Token inválido")
    return {"id": row["id"], "username": row["username"], "role": row["role"]}


# ── FILE PATHS ────────────────────────────────────────────

def user_files_dir(user_id: int) -> Path:
    p = Path(f"files/{user_id}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def user_chunks_dir(user_id: int) -> Path:
    p = Path(f"chunks/{user_id}")
    p.mkdir(parents=True, exist_ok=True)
    return p


# ── FILES INDEX ───────────────────────────────────────────

def load_files_index(base_dir: Path) -> dict:
    idx = base_dir / "files_index.json"
    if not idx.exists():
        return {}
    try:
        return json.loads(idx.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_description_to_index(base_dir: Path, stem: str, description: str) -> None:
    idx_path = base_dir / "files_index.json"
    index = {}
    if idx_path.exists():
        try:
            index = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    if stem not in index or not index[stem]:
        index[stem] = description
        idx_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[index] files_index actualizado: {stem}")


# ── MODELOS ──────────────────────────────────────────────

class LoginRequest(BaseModel):
    username: str
    password: str

class ConversationCreate(BaseModel):
    title: str
    model: str

class ConversationTitleUpdate(BaseModel):
    title: str

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    stream: bool = True
    conversation_id: Optional[str] = None


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


async def process_file(txt_path: Path, chunks_dir: Path) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    text   = txt_path.read_text(encoding="utf-8")
    chunks = chunk_text(text)
    output = {
        "source":    txt_path.name,
        "processed": datetime.utcnow().isoformat(),
        "total":     len(chunks),
        "chunks":    [{"index": i, "text": c} for i, c in enumerate(chunks)],
    }
    json_path = chunks_dir / (txt_path.stem + ".json")
    json_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[embeddings] Generando embeddings para {len(chunks)} chunks de {txt_path.name}...")
    vectors  = embedder.encode(chunks, show_progress_bar=False)
    npy_path = chunks_dir / (txt_path.stem + ".npy")
    np.save(str(npy_path), vectors)
    print(f"[chunker] {txt_path.name} -> {len(chunks)} chunks")
    desc = await generate_description(txt_path)
    if desc:
        save_description_to_index(txt_path.parent, txt_path.stem, desc)


async def sync_dir(files_dir: Path, chunks_dir: Path) -> None:
    files_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    txt_files = {p.stem: p for p in files_dir.glob("*.txt")}
    pending   = [
        p for stem, p in txt_files.items()
        if not (chunks_dir / f"{stem}.json").exists()
        or not (chunks_dir / f"{stem}.npy").exists()
    ]
    if not pending:
        print(f"[chunker] {files_dir} sincronizado.")
        return
    for path in pending:
        await process_file(path, chunks_dir)


async def sync_all_files() -> None:
    await sync_dir(GLOBAL_FILES_DIR, GLOBAL_CHUNKS_DIR)
    files_base = Path("files")
    if files_base.exists():
        for user_dir in files_base.iterdir():
            if user_dir.is_dir() and user_dir.name != "global":
                await sync_dir(user_dir, Path(f"chunks/{user_dir.name}"))


# ── RAG ──────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a) + 1e-10)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-10)
    return b_norm @ a_norm


def retrieve_chunks(question: str, stem: str, chunks_dir: Path, top_k: int = TOP_K_CHUNKS) -> list[str]:
    json_path = chunks_dir / f"{stem}.json"
    npy_path  = chunks_dir / f"{stem}.npy"
    if not json_path.exists() or not npy_path.exists():
        return []
    data     = json.loads(json_path.read_text(encoding="utf-8"))
    chunks   = [c["text"] for c in data.get("chunks", [])]
    vectors  = np.load(str(npy_path))
    q_vector = embedder.encode([question])[0]
    scores   = cosine_similarity(q_vector, vectors)
    top_idx  = np.argsort(scores)[::-1][:top_k]
    selected = [chunks[i] for i in top_idx]
    print(f"[rag] top {top_k} chunks de {stem} (scores: {[round(float(scores[i]), 3) for i in top_idx]})")
    return selected


async def route_to_file(question: str, model: str, available_files: list[dict]) -> dict | None:
    """
    available_files: lista de {"key": "global/stem", "stem": str, "description": str, "chunks_dir": Path}
    Retorna el dict del archivo más relevante, o None.
    """
    if not available_files:
        return None

    file_lines = []
    for f in available_files:
        desc = f.get("description", "")
        line = f"- {f['key']}: {desc}" if desc else f"- {f['key']}"
        file_lines.append(line)
    file_list = "\n".join(file_lines)

    prompt = (
        f"You have access to these knowledge files:\n{file_list}\n\n"
        f"The user asked: \"{question}\"\n\n"
        f"Which file is most relevant to answer this question? "
        f"Reply with ONLY the file key exactly as shown (e.g. global/baroque). "
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

    answer = answer.strip()
    if answer == "none":
        return None

    # Buscar coincidencia exacta por key
    for f in available_files:
        if f["key"].lower() == answer:
            return f

    # Fallback: coincidencia por stem solamente
    answer_stem = answer.split("/")[-1]
    answer_stem = re.sub(r'[^\w]', '', answer_stem)
    for f in available_files:
        if f["stem"].lower() == answer_stem:
            return f

    return None


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


# ── DB HELPERS ────────────────────────────────────────────

def db_save_message(conversation_id: str, role: str, content: str) -> None:
    msg_id = secrets.token_urlsafe(8)
    conn   = get_db()
    conn.execute(
        "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
        (msg_id, conversation_id, role, content, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def db_save_user_message_and_maybe_title(conversation_id: str, content: str) -> None:
    conn = get_db()
    count = conn.execute(
        "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
    ).fetchone()[0]
    now = datetime.utcnow().isoformat()
    if count == 0:
        title = content[:60]
        conn.execute("UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                     (title, now, conversation_id))
    else:
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))
    msg_id = secrets.token_urlsafe(8)
    conn.execute(
        "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?, ?, 'user', ?, ?)",
        (msg_id, conversation_id, content, now)
    )
    conn.commit()
    conn.close()


# ── STARTUP ───────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    init_db()
    GLOBAL_FILES_DIR.mkdir(parents=True, exist_ok=True)
    GLOBAL_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    asyncio.create_task(sync_all_files())


# ── AUTH ENDPOINTS ────────────────────────────────────────

@app.post("/auth/login")
async def login(body: LoginRequest):
    conn = get_db()
    row  = conn.execute(
        "SELECT id, password_hash, role FROM users WHERE username = ?", (body.username,)
    ).fetchone()
    conn.close()
    if not row or not verify_password(body.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Usuario o contraseña incorrectos")
    token = secrets.token_urlsafe(32)
    conn  = get_db()
    conn.execute(
        "INSERT INTO sessions (token, user_id, created_at) VALUES (?, ?, ?)",
        (token, row["id"], datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()
    return {"token": token, "user": {"id": row["id"], "username": body.username, "role": row["role"]}}


@app.post("/auth/logout")
async def logout(
    user: dict = Depends(get_current_user),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    if credentials:
        conn = get_db()
        conn.execute("DELETE FROM sessions WHERE token = ?", (credentials.credentials,))
        conn.commit()
        conn.close()
    return {"status": "ok"}


@app.get("/auth/me")
async def me(user: dict = Depends(get_current_user)):
    return {"id": user["id"], "username": user["username"], "role": user["role"]}


# ── HEALTH ───────────────────────────────────────────────

@app.get("/health")
async def health(user: dict = Depends(get_current_user)):
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
async def list_files(user: dict = Depends(get_current_user)):
    result = []

    # Archivos globales
    global_index = load_files_index(GLOBAL_FILES_DIR)
    for txt_path in sorted(GLOBAL_FILES_DIR.glob("*.txt")):
        stem      = txt_path.stem
        json_path = GLOBAL_CHUNKS_DIR / f"{stem}.json"
        npy_path  = GLOBAL_CHUNKS_DIR / f"{stem}.npy"
        indexed   = json_path.exists() and npy_path.exists()
        chunks    = 0
        if indexed:
            try:
                data   = json.loads(json_path.read_text(encoding="utf-8"))
                chunks = data.get("total", 0)
            except Exception:
                pass
        result.append({
            "name": txt_path.name, "stem": stem, "scope": "global",
            "indexed": indexed, "chunks": chunks,
            "description": global_index.get(stem, ""),
        })

    # Archivos del usuario
    u_files  = user_files_dir(user["id"])
    u_chunks = user_chunks_dir(user["id"])
    u_index  = load_files_index(u_files)
    for txt_path in sorted(u_files.glob("*.txt")):
        stem      = txt_path.stem
        json_path = u_chunks / f"{stem}.json"
        npy_path  = u_chunks / f"{stem}.npy"
        indexed   = json_path.exists() and npy_path.exists()
        chunks    = 0
        if indexed:
            try:
                data   = json.loads(json_path.read_text(encoding="utf-8"))
                chunks = data.get("total", 0)
            except Exception:
                pass
        result.append({
            "name": txt_path.name, "stem": stem, "scope": "user",
            "indexed": indexed, "chunks": chunks,
            "description": u_index.get(stem, ""),
        })

    return {"files": result}


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    scope: str = "user",
    user: dict = Depends(get_current_user)
):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .txt")
    if scope == "global" and user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Solo el admin puede subir archivos globales")

    if scope == "global":
        dest_dir   = GLOBAL_FILES_DIR
        chunks_dir = GLOBAL_CHUNKS_DIR
    else:
        dest_dir   = user_files_dir(user["id"])
        chunks_dir = user_chunks_dir(user["id"])

    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[^\w.]', '_', file.filename.strip()).lower()
    safe_name = re.sub(r'_+', '_', safe_name)
    dest      = dest_dir / safe_name
    content   = await file.read()
    dest.write_bytes(content)
    print(f"[upload] {file.filename} guardado en {dest_dir} (scope={scope})")

    asyncio.create_task(process_file(dest, chunks_dir))
    return {"status": "ok", "file": file.filename, "scope": scope, "message": "Archivo recibido, indexando..."}


@app.delete("/files/{scope}/{stem}")
async def delete_file(scope: str, stem: str, user: dict = Depends(get_current_user)):
    if scope == "global":
        if user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Solo el admin puede eliminar archivos globales")
        files_dir  = GLOBAL_FILES_DIR
        chunks_dir = GLOBAL_CHUNKS_DIR
    else:
        files_dir  = user_files_dir(user["id"])
        chunks_dir = user_chunks_dir(user["id"])

    deleted = []
    for path in [files_dir / f"{stem}.txt", chunks_dir / f"{stem}.json", chunks_dir / f"{stem}.npy"]:
        if path.exists():
            path.unlink()
            deleted.append(path.name)

    idx_path = files_dir / "files_index.json"
    if idx_path.exists():
        try:
            index = json.loads(idx_path.read_text(encoding="utf-8"))
            if stem in index:
                del index[stem]
                idx_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Archivo '{stem}' no encontrado")
    return {"status": "ok", "deleted": deleted}


@app.get("/files/{scope}/{stem}/download")
async def download_file(scope: str, stem: str, user: dict = Depends(get_current_user)):
    if scope == "global":
        files_dir = GLOBAL_FILES_DIR
    else:
        files_dir = user_files_dir(user["id"])

    txt_path = files_dir / f"{stem}.txt"
    if not txt_path.exists():
        raise HTTPException(status_code=404, detail=f"Archivo '{stem}' no encontrado")
    return FileResponse(path=str(txt_path), filename=f"{stem}.txt", media_type="text/plain")


# ── CONVERSATIONS ─────────────────────────────────────────

@app.get("/conversations")
async def list_conversations(user: dict = Depends(get_current_user)):
    conn = get_db()
    rows = conn.execute(
        "SELECT id, title, model, created_at, updated_at FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
        (user["id"],)
    ).fetchall()
    conn.close()
    return {"conversations": [dict(r) for r in rows]}


@app.post("/conversations")
async def create_conversation(body: ConversationCreate, user: dict = Depends(get_current_user)):
    conv_id = secrets.token_urlsafe(12)
    now     = datetime.utcnow().isoformat()
    conn    = get_db()
    conn.execute(
        "INSERT INTO conversations (id, user_id, title, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
        (conv_id, user["id"], body.title, body.model, now, now)
    )
    conn.commit()
    conn.close()
    return {"id": conv_id, "title": body.title, "model": body.model, "created_at": now, "updated_at": now}


@app.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str, user: dict = Depends(get_current_user)):
    conn = get_db()
    conv = conn.execute(
        "SELECT id, title, model, created_at, updated_at FROM conversations WHERE id = ? AND user_id = ?",
        (conv_id, user["id"])
    ).fetchone()
    if not conv:
        conn.close()
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    msgs = conn.execute(
        "SELECT id, role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at",
        (conv_id,)
    ).fetchall()
    conn.close()
    return {**dict(conv), "messages": [dict(m) for m in msgs]}


@app.patch("/conversations/{conv_id}/title")
async def update_conversation_title(conv_id: str, body: ConversationTitleUpdate, user: dict = Depends(get_current_user)):
    conn = get_db()
    conn.execute(
        "UPDATE conversations SET title = ? WHERE id = ? AND user_id = ?",
        (body.title, conv_id, user["id"])
    )
    conn.commit()
    conn.close()
    return {"status": "ok"}


@app.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str, user: dict = Depends(get_current_user)):
    conn = get_db()
    row  = conn.execute(
        "SELECT id FROM conversations WHERE id = ? AND user_id = ?", (conv_id, user["id"])
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(status_code=404, detail="Conversación no encontrada")
    conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()
    return {"status": "ok"}


# ── CHAT ─────────────────────────────────────────────────

@app.post("/chat")
async def chat(req: ChatRequest, user: dict = Depends(get_current_user)):
    question = req.messages[-1].content if req.messages else ""

    # Construir lista de archivos disponibles: global + usuario
    GLOBAL_CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    u_chunks_dir = user_chunks_dir(user["id"])
    g_index      = load_files_index(GLOBAL_FILES_DIR)
    u_index      = load_files_index(user_files_dir(user["id"]))

    available = []
    for stem in [p.stem for p in GLOBAL_CHUNKS_DIR.glob("*.json")]:
        available.append({
            "key":        f"global/{stem}",
            "stem":       stem,
            "description": g_index.get(stem, ""),
            "chunks_dir": GLOBAL_CHUNKS_DIR,
        })
    for stem in [p.stem for p in u_chunks_dir.glob("*.json")]:
        available.append({
            "key":        f"user/{stem}",
            "stem":       stem,
            "description": u_index.get(stem, ""),
            "chunks_dir": u_chunks_dir,
        })

    matched = await route_to_file(question, req.model, available)
    print(f"[rag] routing -> {matched['key'] if matched else 'ninguno'}")

    if matched:
        chunks     = retrieve_chunks(question, matched["stem"], matched["chunks_dir"])
        rag_prompt = build_rag_prompt(question, chunks)
        messages   = [m.model_dump() for m in req.messages[:-1]]
        messages.append({"role": "user", "content": rag_prompt})
    else:
        messages = [m.model_dump() for m in req.messages]

    # Guardar mensaje del usuario en DB
    if req.conversation_id:
        db_save_user_message_and_maybe_title(req.conversation_id, question)

    payload    = {"model": req.model, "messages": messages, "stream": req.stream}
    rag_active = matched is not None

    if req.stream:
        return StreamingResponse(
            stream_ollama(payload, rag_active, req.conversation_id),
            media_type="text/event-stream"
        )
    else:
        return await chat_no_stream(payload, rag_active, req.conversation_id)


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


async def stream_ollama(payload: dict, rag_active: bool = False, conversation_id: str | None = None):
    tag_fixed      = False
    accumulated    = ""
    full_response  = ""

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
                            full_response += accumulated
                            yield f"data: {json.dumps({'text': accumulated, 'done': done})}\n\n"
                            accumulated = ""
                    else:
                        full_response += text
                        yield f"data: {json.dumps({'text': text, 'done': done})}\n\n"

                    if done:
                        if conversation_id and full_response:
                            db_save_message(conversation_id, "assistant", full_response)
                        break
                except json.JSONDecodeError:
                    continue


async def chat_no_stream(payload: dict, rag_active: bool = False, conversation_id: str | None = None):
    async with httpx.AsyncClient(timeout=120.0) as client:
        res = await client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        res.raise_for_status()
        data = res.json()
        text = data.get("message", {}).get("content", "")
        text = fix_tag(text, rag_active)
        if conversation_id and text:
            db_save_message(conversation_id, "assistant", text)
        return {"text": text, "done": True}
