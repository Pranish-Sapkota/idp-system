"""
frontend/app.py
────────────────
IDP System – Streamlit Frontend

Clean, production-grade UI for Intelligent Document Processing.
Connects to the FastAPI backend for all AI operations.
"""

import json
import os
import time
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

# Backend URL: check Streamlit secrets first, then .env, then default
def get_backend_url() -> str:
    try:
        return st.secrets["BACKEND_URL"]
    except Exception:
        return os.getenv("BACKEND_URL", "http://localhost:8000")

BACKEND_URL = get_backend_url()
API_PROCESS = f"{BACKEND_URL}/api/v1/process"
API_HEALTH  = f"{BACKEND_URL}/api/v1/health"

SUPPORTED_TYPES = [".txt", ".pdf", ".docx", ".xlsx", ".pptx"]
TASK_OPTIONS = {
    "📝 Summarize":               "summarize",
    "❓ Question Answering":      "qa",
    "🔍 Key Information Extract": "extract",
}
FORMAT_OPTIONS = {
    "Markdown": "markdown",
    "Plain Text": "text",
    "JSON": "json",
}
FILE_ICONS = {
    ".pdf":  "📄",
    ".docx": "📝",
    ".xlsx": "📊",
    ".pptx": "📑",
    ".txt":  "📃",
}

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IDP System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Main layout */
.main .block-container { padding-top: 1.5rem; max-width: 1100px; }

/* Cards */
.card {
    background: #1E2130;
    border: 1px solid #2D3250;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.card-success {
    background: linear-gradient(135deg, #0d2137 0%, #1a3a2a 100%);
    border: 1px solid #2ecc71;
}
.card-warning {
    background: #2a1f0d;
    border: 1px solid #f39c12;
    border-radius: 8px;
    padding: 0.75rem 1rem;
}
.card-error {
    background: #2a0d0d;
    border: 1px solid #e74c3c;
    border-radius: 8px;
    padding: 0.75rem 1rem;
}

/* Metrics row */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.75rem;
    margin: 0.75rem 0;
}
.metric-box {
    background: #252840;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    text-align: center;
    border: 1px solid #3a3f6e;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #6C63FF;
    line-height: 1.2;
}
.metric-label {
    font-size: 0.72rem;
    color: #9ba3bf;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 2px;
}

/* Header */
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #6C63FF, #48CAE4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.25rem;
}
.hero-sub {
    color: #9ba3bf;
    font-size: 1rem;
    margin-bottom: 1.5rem;
}

/* Status badge */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
}
.badge-green  { background: #1a3a2a; color: #2ecc71; border: 1px solid #2ecc71; }
.badge-red    { background: #2a0d0d; color: #e74c3c; border: 1px solid #e74c3c; }
.badge-purple { background: #1a1440; color: #6C63FF; border: 1px solid #6C63FF; }

/* Result area */
.result-header {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.75rem;
    font-size: 1.1rem;
    font-weight: 700;
}
.copy-hint { color: #9ba3bf; font-size: 0.8rem; margin-bottom: 0.5rem; }

/* Sidebar */
.sidebar-section {
    background: #1a1f2e;
    border-radius: 10px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.75rem;
    border: 1px solid #2D3250;
}
</style>
""", unsafe_allow_html=True)


# ── Helper Functions ──────────────────────────────────────────────────────────

@st.cache_data(ttl=30, show_spinner=False)
def check_backend_health(url: str) -> dict:
    """Ping the backend health endpoint. Cached for 30 seconds."""
    try:
        r = requests.get(f"{url}/api/v1/health", timeout=8)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        return {"status": "unreachable", "gemini_configured": False}
    except Exception as e:
        return {"status": f"error: {e}", "gemini_configured": False}


def format_file_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 ** 2:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / 1024 ** 2:.2f} MB"


def render_result(result, task: str, output_format: str):
    """Render the AI result in the appropriate format with syntax highlighting."""
    if task == "extract" or output_format == "json":
        if isinstance(result, dict):
            st.json(result)
        else:
            try:
                parsed = json.loads(result)
                st.json(parsed)
            except Exception:
                st.markdown(result)
    elif output_format == "text":
        st.text_area(
            label="Result",
            value=result if isinstance(result, str) else json.dumps(result, indent=2),
            height=450,
            label_visibility="collapsed",
        )
    else:
        # Markdown (default)
        content = result if isinstance(result, str) else json.dumps(result, indent=2)
        st.markdown(content)


def process_document(
    file_bytes: bytes,
    filename: str,
    task: str,
    question: str | None,
    output_format: str,
) -> dict:
    """Send document to backend API and return response JSON."""
    files = {"file": (filename, file_bytes, "application/octet-stream")}
    data = {"task": task, "output_format": output_format}
    if question:
        data["question"] = question

    response = requests.post(
        API_PROCESS,
        files=files,
        data=data,
        timeout=180,     # 3 minutes for large documents
    )
    response.raise_for_status()
    return response.json()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 IDP System")
    st.markdown("*Intelligent Document Processing*")
    st.markdown("---")

    # Backend status
    health = check_backend_health(BACKEND_URL)
    is_online = health.get("status") == "ok"
    gemini_ok = health.get("gemini_configured", False)

    st.markdown("#### 🔌 Backend Status")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if is_online:
            st.markdown('<span class="badge badge-green">● Online</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-red">● Offline</span>', unsafe_allow_html=True)
    with col_s2:
        if gemini_ok:
            st.markdown('<span class="badge badge-purple">Gemini ✓</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-red">Gemini ✗</span>', unsafe_allow_html=True)

    if not is_online:
        st.warning(f"Cannot reach backend at:\n`{BACKEND_URL}`", icon="⚠️")

    st.markdown("---")

    # Task selection
    st.markdown("#### ⚙️ Processing Task")
    task_label = st.radio(
        "Select task",
        options=list(TASK_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    task_value = TASK_OPTIONS[task_label]

    # Question input (only for QA)
    user_question = None
    if task_value == "qa":
        st.markdown("#### ❓ Your Question")
        user_question = st.text_area(
            "Question",
            placeholder="e.g. What are the main conclusions of this report?",
            height=100,
            label_visibility="collapsed",
        )

    # Output format
    st.markdown("#### 📤 Output Format")
    output_format_label = st.selectbox(
        "Format",
        options=list(FORMAT_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed",
    )
    output_format_value = FORMAT_OPTIONS[output_format_label]

    st.markdown("---")

    # Supported formats
    st.markdown("#### 📂 Supported Formats")
    for ext, icon in FILE_ICONS.items():
        st.markdown(f"{icon} `{ext}`")

    st.markdown("---")
    st.markdown(
        "<div style='color:#9ba3bf;font-size:0.78rem;'>Powered by Google Gemini<br>"
        "Built with FastAPI + Streamlit</div>",
        unsafe_allow_html=True,
    )


# ── Main Content ──────────────────────────────────────────────────────────────

# Hero header
st.markdown(
    '<div class="hero-title">🧠 Intelligent Document Processing</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-sub">Upload any document · Extract insights · Get structured AI analysis</div>',
    unsafe_allow_html=True,
)

# ── Upload area ───────────────────────────────────────────────────────────────
st.markdown("### 📁 Upload Document")

uploaded_file = st.file_uploader(
    label="Drop your file here or click to browse",
    type=[t.lstrip(".") for t in SUPPORTED_TYPES],
    help="Supported: PDF, DOCX, XLSX, PPTX, TXT (max 20 MB)",
)

if uploaded_file:
    ext = Path(uploaded_file.name).suffix.lower()
    icon = FILE_ICONS.get(ext, "📄")
    file_bytes = uploaded_file.read()

    # File info card
    st.markdown(f"""
    <div class="card">
        <strong>{icon} {uploaded_file.name}</strong>
        &nbsp;&nbsp;
        <span class="badge badge-purple">{ext.upper()}</span>
        &emsp;
        <span style="color:#9ba3bf;font-size:0.85rem;">{format_file_size(len(file_bytes))}</span>
    </div>
    """, unsafe_allow_html=True)

    # Validation checks
    if not is_online:
        st.error("⛔ Backend is offline. Please start the backend server and refresh.", icon="🔌")
        st.stop()

    if not gemini_ok:
        st.error(
            "⛔ Gemini API key is not configured on the backend. "
            "Set GEMINI_API_KEY in your .env file.",
            icon="🔑",
        )
        st.stop()

    if task_value == "qa" and not user_question:
        st.warning("⚠️ Please enter a question in the sidebar before processing.", icon="❓")
        st.stop()

    # ── Process button ────────────────────────────────────────────────────────
    st.markdown("---")
    col_btn, col_spacer = st.columns([1, 3])
    with col_btn:
        process_btn = st.button(
            f"🚀 Process Document",
            type="primary",
            use_container_width=True,
            disabled=not is_online,
        )

    if process_btn:
        t_start = time.time()

        with st.spinner(f"Processing `{uploaded_file.name}` — this may take a moment…"):
            try:
                result_data = process_document(
                    file_bytes=file_bytes,
                    filename=uploaded_file.name,
                    task=task_value,
                    question=user_question,
                    output_format=output_format_value,
                )
            except requests.exceptions.Timeout:
                st.error(
                    "⏱️ Request timed out (180s). The document may be too large or "
                    "the server is overloaded. Try a smaller document.",
                    icon="⏱️",
                )
                st.stop()
            except requests.exceptions.HTTPError as e:
                try:
                    err_detail = e.response.json().get("detail", str(e))
                except Exception:
                    err_detail = str(e)
                st.error(f"❌ API Error: {err_detail}", icon="🚫")
                st.stop()
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}", icon="🚫")
                st.stop()

        elapsed = time.time() - t_start

        # ── Success: render results ───────────────────────────────────────────
        st.success("✅ Document processed successfully!", icon="🎉")

        # Warning banner (e.g. chunking)
        if result_data.get("warning"):
            st.markdown(
                f'<div class="card-warning">⚠️ {result_data["warning"]}</div>',
                unsafe_allow_html=True,
            )
            st.markdown("")

        # Metrics row
        doc = result_data.get("document", {})
        proc_ms = result_data.get("processing_time_ms", 0)
        model_used = result_data.get("model_used", "—")

        st.markdown("#### 📊 Processing Stats")
        cols = st.columns(5)
        metrics = [
            ("Words",   f"{doc.get('word_count', 0):,}"),
            ("Chunks",  str(doc.get('chunk_count', '—'))),
            ("Pages",   str(doc.get('page_count', '—')) if doc.get('page_count') else "—"),
            ("Size",    format_file_size(doc.get('size_bytes', 0))),
            ("Time",    f"{proc_ms / 1000:.1f}s"),
        ]
        for col, (label, value) in zip(cols, metrics):
            with col:
                st.metric(label=label, value=value)

        st.markdown(
            f'<div style="color:#9ba3bf;font-size:0.8rem;margin-bottom:1rem;">'
            f'🤖 Model: <code>{model_used}</code>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Result output
        st.markdown("---")
        task_emoji = {"summarize": "📝", "qa": "💬", "extract": "🔍"}.get(task_value, "📄")
        task_name  = {"summarize": "Summary", "qa": "Answer", "extract": "Extracted Data"}.get(task_value, "Result")

        st.markdown(f"#### {task_emoji} {task_name}")

        result = result_data.get("result", "")

        # Tabs: Rendered | Raw
        tab_render, tab_raw = st.tabs(["🖥️ Rendered", "📋 Raw / Copy"])

        with tab_render:
            render_result(result, task_value, output_format_value)

        with tab_raw:
            raw_str = result if isinstance(result, str) else json.dumps(result, indent=2)
            st.code(raw_str, language="markdown" if output_format_value == "markdown" else "json")

        # Download button
        st.markdown("")
        download_str = result if isinstance(result, str) else json.dumps(result, indent=2)
        file_suffix = ".json" if (task_value == "extract" or output_format_value == "json") else ".md"
        dl_filename = f"{Path(uploaded_file.name).stem}_{task_value}_result{file_suffix}"

        st.download_button(
            label="⬇️ Download Result",
            data=download_str,
            file_name=dl_filename,
            mime="application/json" if file_suffix == ".json" else "text/markdown",
        )

else:
    # ── Landing / empty state ─────────────────────────────────────────────────
    st.markdown("")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="card">
            <div style="font-size:2rem;margin-bottom:0.5rem;">📝</div>
            <strong>Summarization</strong>
            <p style="color:#9ba3bf;font-size:0.88rem;margin-top:0.4rem;">
                Get a concise, structured summary of any document, preserving key facts and figures.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <div style="font-size:2rem;margin-bottom:0.5rem;">❓</div>
            <strong>Question Answering</strong>
            <p style="color:#9ba3bf;font-size:0.88rem;margin-top:0.4rem;">
                Ask any question about your document and get a precise, grounded answer.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <div style="font-size:2rem;margin-bottom:0.5rem;">🔍</div>
            <strong>Information Extraction</strong>
            <p style="color:#9ba3bf;font-size:0.88rem;margin-top:0.4rem;">
                Extract entities, dates, topics, key facts, and action items as structured JSON.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")
    st.info(
        "👈 **Select a task in the sidebar**, then upload a document above to get started.",
        icon="🚀",
    )

    # How it works
    with st.expander("🔧 How It Works", expanded=False):
        st.markdown("""
        1. **Upload** your document (PDF, DOCX, XLSX, PPTX, or TXT)
        2. **Select a task** in the sidebar:
           - *Summarize* → structured summary
           - *Question Answering* → precise, cited answers
           - *Extract* → JSON with entities, topics, facts
        3. **Click Process** — the system will:
           - Parse and clean your document
           - Split it into token-aware chunks
           - Send to Google Gemini for AI analysis
           - Return structured, ready-to-use results
        4. **Download** results as Markdown or JSON

        ---
        **Supported formats:** PDF · DOCX · XLSX · PPTX · TXT  
        **Max file size:** 20 MB  
        **AI Model:** Google Gemini 1.5 Flash / Pro
        """)
