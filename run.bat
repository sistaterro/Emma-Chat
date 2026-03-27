call .venv\Scripts\activate
pip install bcrypt==4.3.0 -q
start uvicorn server:app --reload --port 8000
timeout /t 2 /nobreak >nul
start http://localhost:8000/ui/login.html
