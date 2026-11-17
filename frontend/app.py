"""Enterprise AI Copilot - single-screen Streamlit chatbot.

Every question goes to the backend's automatic multi-agent orchestration:
  POST {BACKEND_URL}/multi-agent   body: {"question": "..."}
The backend decides which of the five agents (Search, Summarization, Expert Finder,
Compliance Checker, Action) to run and returns one synthesized answer with sources.

Run:  streamlit run frontend/app.py --server.port 8501
"""

import os
from html import escape
from pathlib import Path

import requests
import streamlit as st
from dotenv import load_dotenv

# Only frontend/.env is read (BACKEND_URL). The Gemini key stays in backend/.env.
load_dotenv(Path(__file__).with_name(".env"))

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8001").rstrip("/")
CONNECT_TIMEOUT = 5                                              # seconds to reach the backend
READ_TIMEOUT = int(os.getenv("BACKEND_TIMEOUT_SECONDS", "180"))  # Gemini + retries can be slow

AGENT_LABELS = {
    "search": "Search",
    "summarization": "Summarization",
    "expert_finder": "Expert Finder",
    "compliance": "Compliance Checker",
    "action": "Action",
}

st.set_page_config(
    page_title="Enterprise AI Copilot",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ======================================================================
# STYLE
# ======================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp, button, input, textarea { font-family: 'Inter', sans-serif; }

.stApp {
    background:
        radial-gradient(circle at 15% 0%, rgba(99, 102, 241, 0.18) 0%, transparent 40%),
        radial-gradient(circle at 85% 10%, rgba(56, 189, 248, 0.12) 0%, transparent 40%),
        linear-gradient(180deg, #070d1f 0%, #0a1430 55%, #0b1638 100%);
    color: #e7ecf7;
}

/* No sidebar, no Streamlit chrome */
section[data-testid="stSidebar"], [data-testid="collapsedControl"],
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }

.block-container { max-width: 820px; padding-top: 1.6rem; padding-bottom: 7rem; }

/* Brand bar */
.brand { display: flex; align-items: center; gap: 12px; }
.brand-logo {
    width: 40px; height: 40px; border-radius: 12px; display: flex; align-items: center;
    justify-content: center; font-size: 20px; color: white;
    background: linear-gradient(135deg, #6366f1 0%, #3b82f6 55%, #22d3ee 100%);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35);
}
.brand-title { font-size: 1.15rem; font-weight: 700; color: #f8faff; line-height: 1.1; }
.brand-subtitle { font-size: 0.8rem; color: #8ea3c7; margin-top: 2px; }

/* Landing hero */
.hero { text-align: center; padding: 9vh 0 3vh; }
.hero-orb {
    width: 76px; height: 76px; margin: 0 auto 22px; border-radius: 24px; display: flex;
    align-items: center; justify-content: center; font-size: 34px; color: white;
    background: linear-gradient(135deg, #6366f1 0%, #3b82f6 55%, #22d3ee 100%);
    box-shadow: 0 12px 40px rgba(99, 102, 241, 0.45);
}
.hero-title {
    font-size: clamp(2rem, 5vw, 2.8rem); font-weight: 800; letter-spacing: -0.03em;
    background: linear-gradient(90deg, #ffffff 0%, #c7d2fe 60%, #a5b4fc 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.hero-text { color: #9fb0d0; max-width: 560px; margin: 14px auto 0; font-size: 1.02rem; line-height: 1.7; }

/* Chat bubbles */
div[data-testid="stChatMessage"] {
    background: rgba(17, 29, 58, 0.75); border: 1px solid rgba(99, 125, 190, 0.22);
    border-radius: 18px; padding: 14px 18px; margin-bottom: 14px;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.18);
}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    flex-direction: row-reverse; text-align: right; margin-left: 15%;
    background: linear-gradient(135deg, rgba(79, 70, 229, 0.55) 0%, rgba(37, 99, 235, 0.45) 100%);
    border: 1px solid rgba(129, 140, 248, 0.35);
}
div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) { margin-right: 4%; }
div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li { color: #e7ecf7; line-height: 1.65; }

/* Source chips */
.sources { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 8px; }
.chip {
    font-size: 0.78rem; color: #c7d2fe; padding: 5px 11px; border-radius: 999px;
    background: rgba(99, 102, 241, 0.14); border: 1px solid rgba(129, 140, 248, 0.3);
}

/* Details expander */
[data-testid="stExpander"] {
    background: rgba(10, 20, 45, 0.6); border: 1px solid rgba(99, 125, 190, 0.2); border-radius: 12px;
}
[data-testid="stExpander"] summary p { color: #8ea3c7 !important; font-size: 0.85rem; }

/* Chat input */
[data-testid="stBottom"] > div { background: transparent; }
[data-testid="stChatInput"] {
    border-radius: 18px !important; border: 1px solid rgba(129, 140, 248, 0.35) !important;
    background: rgba(15, 27, 56, 0.92) !important; box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
}
[data-testid="stChatInput"] textarea { color: #f1f5ff !important; }
[data-testid="stChatInputSubmitButton"] {
    background: linear-gradient(135deg, #6366f1 0%, #3b82f6 100%) !important;
    border-radius: 12px !important; color: white !important;
}

/* Enterprise integrations */
.int-card {
    background: rgba(13, 24, 50, 0.85); border: 1px solid rgba(99, 125, 190, 0.22);
    border-radius: 14px; padding: 14px; height: 100%;
}
.int-head { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.int-icon {
    width: 34px; height: 34px; border-radius: 9px; display: flex; align-items: center;
    justify-content: center; font-weight: 800; color: white; font-size: 0.95rem; flex-shrink: 0;
}
.int-name { font-weight: 700; color: #f1f5ff; font-size: 0.9rem; line-height: 1.2; }
.int-desc { color: #9fb0d0; font-size: 0.8rem; line-height: 1.5; min-height: 2.4em; }
.int-status { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; }
.int-pill { font-size: 0.7rem; font-weight: 600; padding: 3px 9px; border-radius: 999px; }
.int-ready { background: rgba(34, 197, 94, 0.12); color: #86efac; border: 1px solid rgba(34, 197, 94, 0.3); }
.int-soon { background: rgba(245, 158, 11, 0.12); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.3); }
.int-unknown { background: rgba(148, 163, 184, 0.12); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.3); }
.int-off { background: rgba(148, 163, 184, 0.08); color: #94a3b8; border: 1px dashed rgba(148, 163, 184, 0.35); }
.int-foot { color: #6f83a8; font-size: 0.72rem; margin-top: 10px; }

/* New chat button */
div.stButton > button {
    background: rgba(99, 102, 241, 0.12); color: #c7d2fe; border: 1px solid rgba(129, 140, 248, 0.35);
    border-radius: 12px; font-weight: 600; font-size: 0.85rem;
}
div.stButton > button:hover { background: rgba(99, 102, 241, 0.25); color: white; border-color: #818cf8; }
</style>
""",
    unsafe_allow_html=True,
)

# ======================================================================
# BACKEND CLIENT
# ======================================================================


class BackendError(Exception):
    """A user-friendly error message about a failed backend call."""


def _detail(response: requests.Response) -> str:
    try:
        detail = response.json().get("detail")
    except ValueError:
        return response.text[:200] or response.reason
    if isinstance(detail, list):  # FastAPI validation errors
        return "; ".join(str(d.get("msg", d)) for d in detail)
    return str(detail or response.reason)


def ask_copilot(question: str) -> dict:
    """Send the question to the multi-agent orchestrator and return its JSON response."""
    try:
        response = requests.post(
            f"{BACKEND_URL}/multi-agent",
            json={"question": question},
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
    except requests.exceptions.ConnectionError:
        raise BackendError("I can't reach the knowledge service right now. Please make sure the backend is running.")
    except requests.exceptions.Timeout:
        raise BackendError("That took longer than expected. Please try again in a moment.")
    except requests.exceptions.RequestException:
        raise BackendError("Something went wrong while contacting the knowledge service.")

    if not response.ok:
        detail = _detail(response)
        if response.status_code == 422:
            raise BackendError(f"I couldn't process that question: {detail}")
        if response.status_code in (502, 503):
            raise BackendError("The AI service is temporarily unavailable. Please try again shortly.")
        raise BackendError(f"The knowledge service returned an error ({response.status_code}).")

    try:
        data = response.json()
    except ValueError:
        raise BackendError("The knowledge service returned an unexpected response.")
    if "final_answer" not in data:
        raise BackendError("The knowledge service returned an unexpected response.")
    return data


# Display info for the Microsoft connectors; live status comes from GET /integrations.
INTEGRATION_STYLE = {
    "outlook": {"short": "Outlook", "letter": "O", "color": "linear-gradient(135deg,#0a64c9,#28a8ea)"},
    "teams": {"short": "Microsoft Teams", "letter": "T", "color": "linear-gradient(135deg,#4b53bc,#7b83eb)"},
    "sharepoint": {"short": "SharePoint", "letter": "S", "color": "linear-gradient(135deg,#036c70,#37c6d0)"},
}
FALLBACK_INTEGRATIONS = [
    {"key": "outlook", "description": "Retrieve and search enterprise emails."},
    {"key": "teams", "description": "Retrieve and search enterprise team conversations."},
    {"key": "sharepoint", "description": "Access enterprise documents and knowledge repositories."},
]


@st.cache_data(ttl=30, show_spinner=False)
def get_integrations(base_url: str) -> list[dict] | None:
    """Connector status from the backend (non-secret metadata only)."""
    try:
        r = requests.get(f"{base_url}/integrations", timeout=(CONNECT_TIMEOUT, 10))
        r.raise_for_status()
        return r.json().get("integrations")
    except (requests.RequestException, ValueError):
        return None


def render_integrations():
    live = get_integrations(BACKEND_URL)
    items = live or FALLBACK_INTEGRATIONS
    with st.expander("Enterprise Integrations · Microsoft 365", expanded=False):
        cols = st.columns(3)
        for col, item in zip(cols, items):
            style = INTEGRATION_STYLE.get(item["key"], {"short": item.get("name", ""), "letter": "?", "color": "#334155"})
            if live is None:
                status = '<span class="int-pill int-unknown">Status unavailable</span>'
            elif item.get("status") == "integration_ready":
                status = '<span class="int-pill int-ready">Integration Ready</span>'
            else:
                status = '<span class="int-pill int-soon">Coming Soon</span>'
            connected = item.get("authenticated") is True
            auth = ('<span class="int-pill int-ready">● Connected</span>' if connected
                    else '<span class="int-pill int-off">○ Not authenticated</span>')
            col.markdown(
                f'<div class="int-card"><div class="int-head">'
                f'<div class="int-icon" style="background:{style["color"]}">{style["letter"]}</div>'
                f'<div class="int-name">{escape(style["short"])}</div></div>'
                f'<div class="int-desc">{escape(item.get("description", ""))}</div>'
                f'<div class="int-status">{status}{auth}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="int-foot">Planned via Microsoft Graph API. Connectors are prepared but not yet '
            "authenticated, so no Outlook, Teams or SharePoint data is used in answers today.</div>",
            unsafe_allow_html=True,
        )


# ======================================================================
# RENDERING
# ======================================================================


def render_sources(sources: list[dict]):
    if not sources:
        return
    chips = "".join(
        f'<span class="chip">📄 {escape(str(s.get("filename", "Document")))}'
        + (f' · p.{s["page"]}' if s.get("page") is not None else "")
        + "</span>"
        for s in sources
    )
    st.markdown(f'<div class="sources">{chips}</div>', unsafe_allow_html=True)


def render_details(data: dict):
    """Agent details, hidden by default."""
    results = data.get("agent_results") or []
    agents = [AGENT_LABELS.get(a, a) for a in data.get("selected_agents") or []]
    with st.expander("How this answer was built", expanded=False):
        st.caption("Agents consulted: " + ", ".join(agents))
        for r in results:
            if r["agent"] == "search":
                continue
            label = AGENT_LABELS.get(r["agent"], r["agent"])
            if r.get("success"):
                st.markdown(f"**✓ {label}**")
                st.markdown(r.get("output", ""))
            elif r.get("error"):
                st.markdown(f"**✕ {label}** — could not complete")
            else:
                st.markdown(f"**○ {label}** — no relevant information found")
        evidence = next((r.get("evidence") for r in results if r["agent"] == "search"), []) or []
        if evidence:
            st.caption("Top supporting passages")
            for e in evidence:
                page = f", p.{e['page']}" if e.get("page") is not None else ""
                text = " ".join(str(e.get("text", "")).split())
                st.caption(f"{e.get('source', 'Document')}{page}: {text[:260]}{'…' if len(text) > 260 else ''}")


def render_assistant(msg: dict):
    if msg.get("error"):
        st.markdown(f"⚠️ {msg['error']}")
        return
    data = msg["data"]
    if data.get("status") == "partial":
        st.caption("Some agents couldn't respond, so this answer may be incomplete.")
    st.markdown(data.get("final_answer", ""))
    render_sources(data.get("sources") or [])
    render_details(data)


# ======================================================================
# PAGE
# ======================================================================
st.session_state.setdefault("messages", [])

brand_col, action_col = st.columns([5, 1], vertical_alignment="center")
with brand_col:
    st.markdown(
        '<div class="brand"><div class="brand-logo">✦</div><div>'
        '<div class="brand-title">Enterprise AI Copilot</div>'
        '<div class="brand-subtitle">Your intelligent enterprise knowledge assistant.</div>'
        "</div></div>",
        unsafe_allow_html=True,
    )
with action_col:
    if st.session_state.messages and st.button("＋ New Chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()

render_integrations()

if not st.session_state.messages:
    st.markdown(
        '<div class="hero"><div class="hero-orb">✦</div>'
        '<div class="hero-title">Hello! How can I help you today?</div>'
        '<div class="hero-text">Ask anything about your enterprise knowledge. '
        "I'll find the relevant information and bring together insights from our AI agents.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "✨"):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_assistant(msg)

question = st.chat_input("Message Enterprise AI Copilot…")

if question and question.strip():
    question = question.strip()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="✨"):
        with st.spinner("Thinking and analyzing your request..."):
            try:
                reply = {"role": "assistant", "data": ask_copilot(question)}
            except BackendError as exc:
                reply = {"role": "assistant", "error": str(exc)}
        render_assistant(reply)
    st.session_state.messages.append(reply)
    st.rerun()  # refresh header (shows "New Chat") and hide the welcome hero
