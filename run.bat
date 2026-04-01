@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
  echo [error] Virtual environment not found.
  echo [hint] Create it first with: python -m venv .venv
  exit /b 1
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
  echo [error] Could not activate virtual environment.
  exit /b 1
)

python -c "import fastapi, uvicorn, bcrypt, sentence_transformers" >nul 2>nul
if errorlevel 1 (
  echo [error] Missing Python dependencies.
  echo [hint] Install them with: pip install -r requirements.txt
  exit /b 1
)

curl -s http://localhost:11434/api/tags >nul 2>nul
if errorlevel 1 (
  echo [warn] Ollama does not seem reachable at http://localhost:11434
  echo [warn] Emma may start, but chat and indexing will fail until Ollama is running.
)

echo [info] Starting Emma server...
start "Emma Server" cmd /k uvicorn server:app --reload --port 8000

echo [info] Waiting for backend...
powershell -NoProfile -Command ^
  "$ok=$false; for($i=0;$i -lt 30;$i++){ try { Invoke-WebRequest -UseBasicParsing http://localhost:8000/ui/login.html > $null; $ok=$true; break } catch { Start-Sleep -Seconds 1 } }; if(-not $ok){ exit 1 }"
if errorlevel 1 (
  echo [error] Backend did not become ready in time.
  exit /b 1
)

echo [info] Opening browser...
start "" http://localhost:8000/ui/login.html

exit /b 0
