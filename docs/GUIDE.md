# IDP System – Deployment & Usage Guide

> **Intelligent Document Processing** powered by Google Gemini, FastAPI, and Streamlit.

---

## 📁 Project Structure

```
idp-system/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── config.py                 ← Settings (env vars)
│   │   ├── models.py                 ← Pydantic request/response models
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   └── documents.py          ← API endpoints
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── file_parser.py        ← PDF/DOCX/XLSX/PPTX/TXT parsing
│   │   │   ├── text_processor.py     ← Text cleaning & normalisation
│   │   │   ├── chunker.py            ← Token-aware chunking
│   │   │   └── llm_service.py        ← Gemini API integration
│   │   └── utils/
│   │       └── __init__.py
│   ├── main.py                       ← FastAPI entry point
│   ├── requirements.txt
│   ├── render.yaml                   ← Render blueprint
│   └── .env.example
├── frontend/
│   ├── app.py                        ← Streamlit UI
│   ├── requirements.txt
│   ├── .env.example
│   └── .streamlit/
│       ├── config.toml               ← Theme config
│       └── secrets.toml              ← Secrets (Streamlit Cloud)
└── docs/
    └── GUIDE.md                      ← This file
```

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.11+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey) (free tier available)

---

### Step 1 – Backend

```bash
cd idp-system/backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Open .env and set GEMINI_API_KEY=your_key_here

# Run the backend
uvicorn main:app --reload --port 8000
```

The API is now running at `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

---

### Step 2 – Frontend

```bash
# Open a new terminal
cd idp-system/frontend

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt

# Copy and configure .env
cp .env.example .env
# .env already defaults to http://localhost:8000 for local dev

# Run Streamlit
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## ☁️ Render Backend Deployment

### Option A – Using render.yaml (Recommended)

1. Push your `backend/` folder to a GitHub repository
2. Go to [dashboard.render.com](https://dashboard.render.com)
3. Click **New → Blueprint**
4. Connect your GitHub repo
5. Render will detect `render.yaml` automatically
6. In the **Environment Variables** section, add:
   - `GEMINI_API_KEY` → your Gemini API key
7. Click **Apply** → deployment starts

Your API will be live at:  
`https://idp-system-backend.onrender.com`

---

### Option B – Manual Web Service

1. Go to **New → Web Service** on Render
2. Connect your GitHub repo
3. Configure:
   | Setting | Value |
   |---------|-------|
   | **Build Command** | `pip install -r requirements.txt` |
   | **Start Command** | `uvicorn main:app --host 0.0.0.0 --port $PORT` |
   | **Python Version** | `3.11` |
4. Add Environment Variable: `GEMINI_API_KEY`
5. Click **Create Web Service**

> ⚠️ **Free tier note:** Render free instances spin down after 15 minutes of inactivity.  
> The first request after spin-down may take 30–60 seconds. Upgrade to **Starter** for production.

---

## 🌐 Streamlit Cloud Deployment

1. Push `frontend/` to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New App**
4. Select your repo and set **Main file path** to `app.py`
5. Click **Advanced settings → Secrets** and paste:
   ```toml
   BACKEND_URL = "https://your-idp-backend.onrender.com"
   ```
6. Click **Deploy**

---

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8000/api/v1/health
```
Expected response:
```json
{
  "status": "ok",
  "version": "1.0.0",
  "gemini_configured": true,
  "supported_formats": [".txt", ".pdf", ".docx", ".xlsx", ".pptx"]
}
```

### Summarization Test
```bash
curl -X POST http://localhost:8000/api/v1/process \
  -F "file=@/path/to/your/document.pdf" \
  -F "task=summarize" \
  -F "output_format=markdown"
```

### Question Answering Test
```bash
curl -X POST http://localhost:8000/api/v1/process \
  -F "file=@/path/to/your/document.pdf" \
  -F "task=qa" \
  -F "question=What are the main recommendations?" \
  -F "output_format=markdown"
```

### Key Extraction Test
```bash
curl -X POST http://localhost:8000/api/v1/process \
  -F "file=@/path/to/your/document.pdf" \
  -F "task=extract" \
  -F "output_format=json"
```

---

## 📊 Performance Optimisation

### 1. Chunking Tuning
In `backend/.env`, adjust:
```env
CHUNK_SIZE_TOKENS=1500        # Reduce for faster responses
CHUNK_OVERLAP_TOKENS=150      # Increase for better coherence
MAX_CHUNKS_PER_REQUEST=10     # Cap to avoid rate limits
```

### 2. Model Selection
- **gemini-1.5-flash** (default) – fastest, cheapest, great for most tasks
- **gemini-1.5-pro** – higher quality, used automatically for multi-chunk synthesis

### 3. Reducing Cold Start (Render)
- Add a health-check ping via an uptime monitor (e.g., UptimeRobot, free tier)
- Ping `https://your-backend.onrender.com/api/v1/health` every 14 minutes

### 4. File Size Limits
- Default: 20 MB (`MAX_FILE_SIZE_MB=20` in .env)
- Render free tier: 512 MB RAM — keep files under 20 MB
- For larger files, upgrade Render plan or implement streaming

### 5. Caching
For repeated identical documents, add Redis caching by hashing file content
and storing results. This avoids redundant Gemini API calls.

---

## 🔐 Security Checklist

- [ ] Never commit `.env` to version control (`.gitignore` includes it)
- [ ] Set `CORS allow_origins` to your Streamlit URL only (not `*`) in production
- [ ] Rotate your `GEMINI_API_KEY` regularly
- [ ] Add rate limiting via `slowapi` for public-facing deployments
- [ ] Validate file MIME types server-side (not just extension)

---

## 🐛 Common Issues

| Issue | Solution |
|-------|----------|
| `GEMINI_API_KEY is not configured` | Set the key in `.env` or Render env vars |
| `Module not found: fitz` | Run `pip install PyMuPDF` |
| `Cannot reach backend` | Check `BACKEND_URL` in frontend `.env` |
| PDF returns empty text | PDF is image-only (scanned) — OCR not supported |
| `429 Too Many Requests` | Gemini rate limit hit — reduce `MAX_CHUNKS_PER_REQUEST` |
| Render deploy fails | Check Python version is 3.11 in Render settings |

---

## 📡 API Reference

### `POST /api/v1/process`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | File | ✅ | Document to process |
| `task` | string | ✅ | `summarize` / `qa` / `extract` |
| `question` | string | QA only | Question to answer |
| `output_format` | string | ❌ | `markdown` / `text` / `json` |

### `GET /api/v1/health`
Returns backend status and configuration info.

---

*Built with ❤️ using FastAPI · Streamlit · Google Gemini*
