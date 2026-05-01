# 🧠 IDP System – Intelligent Document Processing

A production-ready document intelligence platform built with **FastAPI**, **Streamlit**, and **Google Gemini**.

## Features
- 📄 Supports PDF, DOCX, XLSX, PPTX, TXT
- 📝 Summarization, ❓ Q&A, 🔍 Structured Extraction
- 🔄 Token-aware chunking with smart overlap
- 🚀 Deployable on Render + Streamlit Cloud

## Quick Start

```bash
# Backend
cd backend && pip install -r requirements.txt
cp .env.example .env   # Add your GEMINI_API_KEY
uvicorn main:app --reload

# Frontend (new terminal)
cd frontend && pip install -r requirements.txt
streamlit run app.py
```

## Full Guide
See [`docs/GUIDE.md`](docs/GUIDE.md) for complete setup, deployment, and testing instructions.
