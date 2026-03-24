call .venv\Scripts\activate
start uvicorn server:app --reload --port 8000
cd ui
python -m http.server 5500