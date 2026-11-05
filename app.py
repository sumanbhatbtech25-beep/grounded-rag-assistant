"""Streamlit UI:  streamlit run app.py"""
from __future__ import annotations

import base64
import html
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from rag.evaluation import load_eval_results, load_query_log, run_eval
from rag.highlight import render_highlighted_page
from rag.indexer import index_folder, index_pdf
from rag.llm import LLM, ollama_models
from rag.pipeline import Answer, RAGPipeline
from rag.report import to_markdown, to_pdf
from rag.auth import ROLES, USERNAME_RE, USERNAME_RULES, AuthError, UserStore, user_key
from rag.indexer import index_folder_report, safe_add_pdf
from rag.security import audit, read_audit, validate_pdf
from rag.store import Embedder, VectorStore
import time

st.set_page_config(page_title="CiteSight · Grounded RAG Assistant", page_icon="📑",
                   layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ theme
if "dark" not in st.session_state:
    st.session_state.dark = st.query_params.get("theme") == "dark"
DARK = st.session_state.dark

LIGHT_TOKENS = dict(bg="#F7F6F2", card="#FFFFFF", ink="#1B2430", muted="#6B7280", line="#E0DCD1",
                    soft="#EEECE5", code="#FBFAF7", hit="#FCE9B4", hitink="#1B2430", accent="#1F4B7A",
                    ln="#A3A9B3", docic="#F3E7E3", grid="#ECE9E1")
DARK_TOKENS = dict(bg="#0D1626", card="#15233A", ink="#E6EBF2", muted="#9AA8BC", line="#26395A",
                   soft="#1B2B45", code="#101C2F", hit="#5C4712", hitink="#FFE7A8", accent="#4A8AD0",
                   ln="#5D6B80", docic="#3A2622", grid="#22344F")
T = DARK_TOKENS if DARK else LIGHT_TOKENS
NAVY, NAVY_DARK, AMBER = "#1F4B7A", "#13243A", "#E0A526"

STATUS = {  # label, light (fg, bg), dark (fg, bg)
    "answered":   ("Grounded answer", ("#1E6B4F", "#E3F2EA"), ("#7EE2B4", "#16382D")),
    "refused":    ("Not in documents", ("#7A5A12", "#FBF0D5"), ("#F3CF7A", "#3B3015")),
    "no_context": ("No relevant source", ("#7A5A12", "#FBF0D5"), ("#F3CF7A", "#3B3015")),
    "unverified": ("Withheld · no valid citation", ("#9A3B1F", "#FBE4DC"), ("#F4A58C", "#3E231B")),
    "error":      ("Error", ("#9A1F1F", "#FBDCDC"), ("#F59A9A", "#3E1B1B")),
}


def status_style(s: str) -> tuple[str, str, str]:
    label, light, dark = STATUS.get(s, (s, ("#333", "#EEE"), ("#DDD", "#333")))
    fg, bg = dark if DARK else light
    return label, fg, bg


css_vars = ";".join(f"--{k}:{v}" for k, v in T.items())
st.markdown(f"""
<style>
:root {{{css_vars}}}
.block-container {{padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px;}}
h1, h2, h3 {{letter-spacing: -0.01em;}}

/* hero */
.hero {{background: linear-gradient(135deg, {NAVY_DARK} 0%, {NAVY} 100%); color:#fff;
        border-radius: 16px; padding: 22px 28px; margin-bottom: 18px;
        display:flex; align-items:center; justify-content:space-between; gap:20px; flex-wrap:wrap;
        border:1px solid rgba(255,255,255,.06)}}
.hero .brand {{display:flex; align-items:center; gap:14px}}
.hero .mark {{width:44px; height:44px; border-radius:12px; background:{AMBER}; color:{NAVY_DARK};
              display:flex; align-items:center; justify-content:center; font-weight:800; font-size:20px;
              font-family:'Source Serif 4',serif}}
.hero h1 {{color:#fff !important; font-size:1.6rem; margin:0; padding:0; line-height:1.2}}
.hero p {{margin:2px 0 0; color:#C9D5E4 !important; font-size:0.92rem}}
.chips {{display:flex; gap:8px; flex-wrap:wrap}}
.chip {{background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.18); color:#E6EBF2;
        padding:5px 11px; border-radius:999px; font-size:0.8rem; white-space:nowrap}}
.chip b {{color:#fff}}
.dot {{display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle}}

.stTabs [data-baseweb="tab-list"] {{gap:4px; border-bottom:1px solid var(--line)}}
.stTabs [data-baseweb="tab"] {{padding:8px 16px; font-weight:500}}

/* answer card */
.answer {{background:var(--card); border:1px solid var(--line); border-left:4px solid var(--accent);
          border-radius:12px; padding:16px 18px; line-height:1.65; font-size:1rem; color:var(--ink)}}
.answer.refuse {{border-left-color:{AMBER}}}
.answer.live {{border-left-color:{AMBER}}}
.caret {{display:inline-block; width:8px; height:1.05em; background:{AMBER}; vertical-align:-2px;
         margin-left:2px; animation: blink 1s steps(2) infinite}}
@keyframes blink {{50% {{opacity:0}}}}
a.cite, span.cite {{background:var(--accent); color:#fff !important; border-radius:6px; padding:1px 7px;
        font-size:0.72em; font-weight:600; margin-left:3px; vertical-align:1px; white-space:nowrap;
        text-decoration:none !important; cursor:pointer; transition: background .15s}}
a.cite:hover {{background:{AMBER}; color:{NAVY_DARK} !important}}
.meta {{display:flex; gap:8px; flex-wrap:wrap; margin:10px 0 4px}}
.pill {{border-radius:999px; padding:3px 10px; font-size:0.78rem; font-weight:500;
        background:var(--soft); color:var(--muted); white-space:nowrap}}

/* source cards  (clicking a citation targets the card -> highlight + reveal page) */
.src {{background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px 14px;
       margin-bottom:10px; scroll-margin-top:90px; transition: box-shadow .2s, border-color .2s}}
.src:target {{border-color:{AMBER}; box-shadow:0 0 0 3px rgba(224,165,38,.35)}}
.src-top {{display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:8px}}
.src-id {{background:var(--accent); color:#fff; border-radius:6px; padding:2px 8px; font-size:0.75rem; font-weight:700}}
.src-file {{font-weight:600; color:var(--ink); font-size:0.92rem; overflow:hidden; text-overflow:ellipsis}}
.src-loc {{color:var(--muted); font-size:0.82rem; white-space:nowrap}}
.bar {{height:5px; background:var(--soft); border-radius:99px; overflow:hidden; margin-top:2px}}
.bar > span {{display:block; height:100%; background:{AMBER}}}
.lines {{font-family:'JetBrains Mono',Consolas,monospace; font-size:0.8rem; line-height:1.7;
         background:var(--code); border:1px solid var(--line); border-radius:8px; padding:8px 10px;
         color:var(--ink); overflow-x:auto}}
.lines .ln {{color:var(--ln); display:inline-block; width:3em; user-select:none}}
.lines .hit {{background:var(--hit); color:var(--hitink); border-radius:3px; padding:1px 3px}}
.src-actions {{display:flex; justify-content:space-between; align-items:center; margin-top:10px; font-size:0.82rem}}
.src-actions a {{color:var(--accent) !important; text-decoration:none !important; font-weight:600}}
.src-actions a:hover {{text-decoration:underline !important}}
.src .pagebox {{display:none; margin-top:10px}}
.src:target .pagebox {{display:block}}
.src:target .open-link {{display:none}}
.src:not(:target) .close-link {{display:none}}
.pagebox img {{width:100%; border-radius:8px; border:1px solid var(--line); background:#fff}}

/* KPI + doc cards */
.kpi {{background:var(--card); border:1px solid var(--line); border-radius:12px; padding:14px 16px; min-height:118px}}
.kpi .label {{color:var(--muted); font-size:0.78rem; text-transform:uppercase; letter-spacing:.06em}}
.kpi .value {{font-family:'Source Serif 4',serif; font-size:1.7rem; font-weight:700; color:var(--ink); margin-top:2px}}
.kpi .sub {{color:var(--muted); font-size:0.78rem}}
.doc {{background:var(--card); border:1px solid var(--line); border-radius:12px; padding:12px 14px;
       display:flex; align-items:center; gap:12px}}
.doc .ic {{width:36px; height:44px; border-radius:6px; background:var(--docic); color:#B5523B;
           display:flex; align-items:center; justify-content:center; font-size:0.65rem; font-weight:800}}
.empty {{text-align:center; padding:26px 10px 10px; color:var(--muted)}}
.empty h3 {{color:var(--ink) !important; margin-bottom:4px}}
.section-label {{color:var(--muted); font-size:0.78rem; text-transform:uppercase; letter-spacing:.06em;
                 margin:14px 0 6px; font-weight:600}}
.scope {{color:var(--muted); font-size:0.85rem}}
.scope b {{color:var(--ink)}}

section[data-testid="stSidebar"] .sb-title {{font-family:'Source Serif 4',serif; font-size:1.15rem;
        font-weight:700; color:#fff; margin-bottom:4px}}
section[data-testid="stSidebar"] .sb-note {{color:#9FB0C6; font-size:0.8rem}}
.sb-user {{display:flex; gap:10px; align-items:center; background:#1D3350; border:1px solid #2A4466;
           border-radius:12px; padding:10px 12px; margin-bottom:8px}}
.sb-user .avatar {{width:34px; height:34px; border-radius:50%; background:{AMBER}; color:{NAVY_DARK};
           display:flex; align-items:center; justify-content:center; font-weight:800}}
.sb-user .name {{color:#fff; font-weight:600; font-size:0.92rem}}
.sb-user .role {{color:#9FB0C6; font-size:0.75rem; text-transform:uppercase; letter-spacing:.06em}}
.guard {{border:1px solid var(--line); background:var(--card); border-radius:12px; padding:12px 14px;
         display:flex; gap:10px; align-items:flex-start; margin-bottom:8px; color:var(--ink); min-height:92px}}
.guard .ok {{color:#2E9E6E; font-weight:800}}
.guard small {{color:var(--muted); display:block}}
</style>
""", unsafe_allow_html=True)

if DARK:  # re-skin Streamlit's own widgets in the main area
    st.markdown("""
<style>
.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] > div {background: var(--bg) !important;}
[data-testid="stMain"], [data-testid="stMain"] [data-testid="stMarkdownContainer"],
[data-testid="stMain"] h1, [data-testid="stMain"] h2, [data-testid="stMain"] h3,
[data-testid="stMain"] label, [data-testid="stMain"] [data-testid="stWidgetLabel"] p {color: var(--ink) !important;}
[data-testid="stMain"] [data-testid="stCaptionContainer"], [data-testid="stSpinner"] {color: var(--muted) !important;}
.stTabs [data-baseweb="tab"] {color: var(--muted)}
.stTabs [aria-selected="true"] {color: var(--accent) !important}
[data-testid="stChatInput"] > div {background: var(--soft) !important; border-color: var(--line) !important;}
[data-testid="stChatInput"] textarea {color: var(--ink) !important; -webkit-text-fill-color: var(--ink);}
[data-testid="stChatInput"] textarea::placeholder {color: var(--muted) !important; -webkit-text-fill-color: var(--muted);}
[data-testid="stChatMessage"] {background: transparent !important;}
[data-testid^="stChatMessageAvatar"] {background: var(--soft) !important; color: var(--ink) !important; border-color: var(--line) !important}
[data-testid^="stChatMessageAvatar"] * {color: var(--ink) !important; fill: var(--ink) !important}
[data-testid="stMain"] [data-testid="stBaseButton-secondary"], [data-testid="stPopover"] button {
    background: var(--card) !important; color: var(--ink) !important; border-color: var(--line) !important;}
[data-testid="stMain"] [data-testid="stBaseButton-secondary"]:hover {border-color: #E0A526 !important; color: #E0A526 !important}
[data-testid="stPopoverBody"] {background: var(--card) !important; color: var(--ink) !important;}
[data-testid="stPopoverBody"] [data-baseweb="select"] > div {background: var(--soft) !important; border-color: var(--line) !important; color: var(--ink) !important}
[data-testid="stFileUploaderDropzone"] {background: var(--card) !important; border-color: var(--line) !important;}
[data-testid="stFileUploaderDropzone"] * {color: var(--muted) !important}
[data-testid="stExpander"] details {background: var(--card) !important; border-color: var(--line) !important;}
[data-testid="stExpander"] summary, [data-testid="stExpander"] summary * {color: var(--ink) !important}
[data-testid="stDataFrame"] {filter: invert(0.9) hue-rotate(180deg);}
[data-testid="stMain"] [data-testid="stDownloadButton"] button {background: var(--card) !important; color: var(--ink) !important; border-color: var(--line) !important}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ resources
@st.cache_resource(show_spinner="Loading vector store...")
def get_store(embed_backend: str, embed_model: str) -> VectorStore:
    return VectorStore(embedder=Embedder(embed_backend, embed_model))


@st.cache_data(show_spinner=False, max_entries=128)
def page_b64(path: str, page: int, lines: tuple, mtime: float) -> str:
    return base64.b64encode(render_highlighted_page(path, page, list(lines), zoom=1.4)).decode()


@st.cache_data(show_spinner=False)
def sample_questions() -> list[str]:
    try:
        items = json.loads(config.EVAL_SET.read_text(encoding="utf-8"))
        return [i["question"] for i in items if i.get("answerable", True)][:4]
    except Exception:
        return []


def style_fig(fig: go.Figure, height: int = 320) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=44, b=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=T["ink"], size=12),
        title_font=dict(family="Source Serif 4, serif", size=16, color=T["ink"]),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right", font=dict(color=T["ink"])),
    )
    fig.update_xaxes(showgrid=False, linecolor=T["line"], color=T["muted"])
    fig.update_yaxes(gridcolor=T["grid"], zeroline=False, color=T["muted"])
    return fig


def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                 f'<div class="sub">{sub}&nbsp;</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ authentication
users = UserStore()
AUTH_CSS = f"""<style>
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] {{display:none}}
.auth-head {{text-align:center; margin: 6vh 0 18px}}
.auth-head .mark {{width:56px; height:56px; border-radius:14px; background:{AMBER}; color:{NAVY_DARK};
   display:inline-flex; align-items:center; justify-content:center; font-weight:800; font-size:26px;
   font-family:'Source Serif 4',serif; margin-bottom:10px}}
.auth-head h2 {{margin:0; color:var(--ink) !important}}
.auth-head p {{color:var(--muted); margin:4px 0 0}}
[data-testid="stForm"] {{background:var(--card); border:1px solid var(--line) !important; border-radius:14px; padding:22px}}
.auth-foot {{text-align:center; color:var(--muted); font-size:0.8rem; margin-top:12px}}
</style>"""


def _sign_in(username: str, role: str) -> None:
    st.session_state.auth = {"user": username, "role": role}
    st.session_state.last_active = time.time()
    st.session_state.history = []


def _rule(ok: bool, text: str) -> str:
    colour = "#2E9E6E" if ok else "var(--muted)"
    return f'<div style="color:{colour};font-size:0.84rem;margin:2px 0">{"✓" if ok else "○"} {text}</div>'


def auth_screen() -> None:
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    first_run = not users.has_users()
    _, mid, _ = st.columns([1, 1.25, 1])
    with mid:
        st.markdown(
            '<div class="auth-head"><div class="mark">C</div>'
            + ('<h2>Set up CiteSight</h2><p>Create the administrator account to get started.</p>' if first_run
               else '<h2>Sign in to CiteSight</h2><p>Your documents stay on this machine.</p>')
            + "</div>", unsafe_allow_html=True)
        if msg := st.session_state.pop("auth_msg", None):
            st.info(msg, icon=":material/info:")

        if first_run:
            with st.container(border=True):
                u = st.text_input("Your name / username", placeholder="e.g. Suman Bhat", max_chars=32)
                p1 = st.text_input("Password", type="password")
                p2 = st.text_input("Confirm password", type="password")
                key = user_key(u)
                checks = [
                    (bool(USERNAME_RE.match(key)), f"Username: {USERNAME_RULES}"),
                    (len(p1) >= 8, "Password: at least 8 characters"),
                    (bool(re.search(r"[A-Za-z]", p1)) and bool(re.search(r"\d", p1)),
                     "Password: contains a letter and a number"),
                    (bool(p1) and p1 == p2, "Passwords match"),
                ]
                st.markdown("".join(_rule(ok, t) for ok, t in checks), unsafe_allow_html=True)
                st.write("")
                ready = all(ok for ok, _ in checks)
                if st.button("Create admin account", type="primary", width="stretch", disabled=not ready):
                    try:
                        users.add_user(u, p1, "admin")
                        audit("admin_created", key)
                        _sign_in(key, "admin")
                        st.rerun()
                    except AuthError as e:
                        st.error(str(e))
                st.caption("Press Enter after typing in a box to update the checklist. "
                           "You'll sign in with this name (capital letters don't matter).")
        else:
            with st.form("login"):
                u = st.text_input("Username", autocomplete="username", placeholder="e.g. Suman Bhat")
                p = st.text_input("Password", type="password", autocomplete="current-password")
                if st.form_submit_button("Sign in", type="primary", width="stretch"):
                    if not u.strip() or not p:
                        st.error("Enter your username and password.")
                    else:
                        role, msg = users.verify(u, p)
                        if role:
                            audit("login_success", user_key(u))
                            _sign_in(user_key(u), role)
                            st.rerun()
                        else:
                            audit("login_failed", user_key(u)[:40], reason=msg.split(".")[0])
                            st.error(msg)
            st.caption("Forgot the admin password? In the project folder run "
                       "`.venv\\Scripts\\python cli.py reset-password \"<your name>\"`")
        st.markdown('<div class="auth-foot">🔒 Passwords are stored only as salted PBKDF2 hashes.</div>',
                    unsafe_allow_html=True)


def damaged_users_screen() -> None:
    st.markdown(AUTH_CSS, unsafe_allow_html=True)
    _, mid, _ = st.columns([1, 1.6, 1])
    with mid:
        st.markdown('<div class="auth-head"><div class="mark">!</div><h2>Account file is damaged</h2>'
                    '<p>CiteSight could not read <code>data/users.json</code>, so sign-in is paused '
                    'to keep the app safe.</p></div>', unsafe_allow_html=True)
        st.error("No one can sign in until the account file is repaired. Your documents and index are not affected.",
                 icon=":material/gpp_bad:")
        st.markdown("To recreate the admin account, stop the app (**Ctrl+C**) and run this in the project folder:")
        st.code('.venv\\Scripts\\python cli.py add-user "Suman Bhat" --admin', language="powershell")
        st.caption("The damaged file is kept as data/users.damaged-<date>.json so nothing is lost.")


if users.is_damaged():
    if not st.session_state.get("_damage_logged"):
        audit("users_file_damaged")
        st.session_state._damage_logged = True
    damaged_users_screen()
    st.stop()


auth = st.session_state.get("auth")
if auth:
    idle = time.time() - st.session_state.get("last_active", 0)
    if idle > config.SESSION_TIMEOUT_MINUTES * 60:
        audit("session_timeout", auth["user"])
        for k in ("auth", "history"):
            st.session_state.pop(k, None)
        st.session_state.auth_msg = "You were signed out after a period of inactivity."
        auth = None
    elif users.role_of(auth["user"]) is None:  # account deleted meanwhile
        st.session_state.pop("auth", None)
        auth = None
if not auth:
    auth_screen()
    st.stop()

st.session_state.last_active = time.time()
USER, ROLE = auth["user"], users.role_of(auth["user"])
USER_DISPLAY = users.display_name(USER)
IS_ADMIN = ROLE == "admin"


def flash(kind: str, text: str) -> None:
    st.session_state.setdefault("flash", []).append((kind, text))


def show_flash() -> None:
    for kind, text in st.session_state.pop("flash", []):
        {"success": st.success, "warning": st.warning, "error": st.error}.get(kind, st.info)(text)


# ------------------------------------------------------------------ sidebar
def _toggle_theme():
    st.session_state.dark = st.session_state.dark_toggle
    if st.session_state.dark:
        st.query_params["theme"] = "dark"
    else:
        st.query_params.pop("theme", None)


with st.sidebar:
    st.markdown(f'<div class="sb-user"><span class="avatar">{html.escape(USER_DISPLAY[:1].upper())}</span>'
                f'<div><div class="name">{html.escape(USER_DISPLAY)}</div><div class="role">{ROLE}</div></div></div>',
                unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    with a1.popover("Account", width="stretch"):
        with st.form("chpw", clear_on_submit=True):
            cur = st.text_input("Current password", type="password")
            new1 = st.text_input("New password", type="password")
            new2 = st.text_input("Confirm new password", type="password")
            if st.form_submit_button("Change password", type="primary"):
                if users.verify(USER, cur)[0] is None:
                    st.error("Current password is wrong.")
                elif new1 != new2:
                    st.error("New passwords don't match.")
                else:
                    try:
                        users.set_password(USER, new1)
                        audit("password_changed", USER)
                        st.success("Password changed.")
                    except AuthError as e:
                        st.error(str(e))
    if a2.button("Sign out", icon=":material/logout:", width="stretch"):
        audit("logout", USER)
        for k in ("auth", "history"):
            st.session_state.pop(k, None)
        st.rerun()
    st.toggle("Dark mode", value=DARK, key="dark_toggle", on_change=_toggle_theme)
    st.markdown('<div class="sb-title">Model & retrieval</div>'
                '<div class="sb-note">Everything runs locally on this machine.</div>', unsafe_allow_html=True)
    st.write("")
    backend = st.selectbox("LLM backend", ["ollama", "openai"],
                           index=0 if config.LLM_BACKEND == "ollama" else 1,
                           help="ollama = fully local. openai = any OpenAI-compatible API "
                                "(set OPENAI_API_KEY / OPENAI_BASE_URL).")
    models = ollama_models() if backend == "ollama" else []
    if backend == "ollama":
        if models:
            default = next((m for m in models if m.split(":")[0] == config.LLM_MODEL.split(":")[0]), models[0])
            model = st.selectbox("Model", models, index=models.index(default))
        else:
            st.error("Ollama is not running. Start the Ollama app, then refresh.")
            model = st.text_input("Model", config.LLM_MODEL)
    else:
        model = st.text_input("Model", config.LLM_MODEL if config.LLM_BACKEND == "openai" else "gpt-4o-mini")
        if not config.OPENAI_API_KEY:
            st.warning("OPENAI_API_KEY is not set.")

    top_k = st.slider("Sources retrieved (top-k)", 1, 10, config.TOP_K)
    min_sim = st.slider("Evidence threshold", 0.0, 1.0, config.MIN_SIMILARITY, 0.05,
                        help="Minimum similarity for a passage to count as evidence. "
                             "Below it, the assistant refuses without calling the LLM.")
    stream = st.toggle("Stream answers", value=True, help="Show the answer word by word as it is generated.")

    history = st.session_state.get("history", [])
    if history:
        st.divider()
        st.markdown('<div class="sb-title">Export conversation</div>', unsafe_allow_html=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        st.download_button("Download PDF report", data=to_pdf(history), file_name=f"citesight-{stamp}.pdf",
                           mime="application/pdf", width="stretch", icon=":material/picture_as_pdf:")
        st.download_button("Download Markdown", data=to_markdown(history), file_name=f"citesight-{stamp}.md",
                           mime="text/markdown", width="stretch", icon=":material/description:")
        if st.button("Clear conversation", width="stretch", icon=":material/delete_sweep:"):
            st.session_state.history = []
            st.rerun()
    st.divider()
    emb = "all-MiniLM-L6-v2" if config.EMBED_BACKEND == "default" else config.EMBED_MODEL
    st.markdown(f'<div class="sb-note">Embeddings · {emb}<br>Vector store · ChromaDB</div>',
                unsafe_allow_html=True)

try:
    store = get_store(config.EMBED_BACKEND, config.EMBED_MODEL)
    sources = store.sources()  # touch the index so damage shows up here, not mid-page
except Exception as e:
    st.error(f"The search index could not be opened ({type(e).__name__}). It may be damaged. "
             "Your PDFs in docs/ are safe; the index can be rebuilt from them.", icon=":material/database:")
    if IS_ADMIN and st.button("Rebuild index from docs/", type="primary", icon=":material/build:"):
        get_store.clear()
        import gc, shutil
        gc.collect()
        backup = config.DATA_DIR / f"chroma.damaged-{time.strftime('%Y%m%d-%H%M%S')}"
        try:
            shutil.move(str(config.CHROMA_DIR), backup)
        except OSError:
            shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
        fresh = get_store(config.EMBED_BACKEND, config.EMBED_MODEL)
        with st.spinner("Rebuilding…"):
            results = index_folder_report(fresh)
        audit("index_rebuilt", USER, files=sum(r.ok for r in results))
        st.rerun()
    elif not IS_ADMIN:
        st.info("Ask an admin to sign in and rebuild the index.")
    st.stop()
pipeline = RAGPipeline(store=store, llm=LLM(backend, model), top_k=top_k, min_similarity=min_sim)
pipeline.user = USER

# ------------------------------------------------------------------ hero
online = backend != "ollama" or bool(models)
st.markdown(f"""
<div class="hero">
  <div class="brand">
    <div class="mark">C</div>
    <div><h1>CiteSight</h1>
      <p>Grounded answers from your PDFs, cited to the exact page and line.</p></div>
  </div>
  <div class="chips">
    <span class="chip"><span class="dot" style="background:{'#4ADE80' if online else '#F87171'}"></span>
      {html.escape(model)}</span>
    <span class="chip"><b>{len(sources)}</b> documents</span>
    <span class="chip"><b>{sum(sources.values())}</b> passages</span>
  </div>
</div>
""", unsafe_allow_html=True)

_tabs = st.tabs([":material/forum: Ask", ":material/folder_open: Documents",
                 ":material/monitoring: Dashboard", ":material/fact_check: Evaluation"]
                + ([":material/shield_lock: Security"] if IS_ADMIN else []))
tab_ask, tab_docs, tab_dash, tab_eval = _tabs[:4]
tab_sec = _tabs[4] if IS_ADMIN else None


# ------------------------------------------------------------------ answer rendering
def render_answer_text(text: str, key: str | None, n_sources: int = 99) -> str:
    safe = html.escape(text).replace("\n", "<br>")
    if key is None:  # still streaming: plain tags
        return re.sub(r"\[S(\d+)\]", r'<span class="cite">S\1</span>', safe)

    def link(m):
        sid = int(m.group(1))
        if sid > n_sources:
            return f'<span class="cite" title="invalid citation">S{sid}</span>'
        return f'<a class="cite" href="#src-{key}-{sid}" title="Jump to source S{sid}">S{sid}</a>'
    return re.sub(r"\[S(\d+)\]", link, safe)


def render_lines(c) -> str:
    hl = set(c.highlight_lines)
    rows = []
    for j, t in enumerate(c.lines):
        n = c.line_start + j
        body = html.escape(t)
        rows.append(f'<span class="ln">L{n}</span>' + (f'<span class="hit">{body}</span>' if n in hl else body))
    return '<div class="lines">' + "<br>".join(rows) + "</div>"


def source_card(c, key: str) -> str:
    hl = c.highlight_lines
    span = f"line {hl[0]}" if len(hl) == 1 else f"lines {hl[0]}–{hl[-1]}"
    pdf = config.DOCS_DIR / c.source
    page_html, open_link = "", ""
    if pdf.exists():
        try:
            b64 = page_b64(str(pdf), c.page, tuple(hl), pdf.stat().st_mtime)
        except Exception:
            b64 = None
    if pdf.exists() and b64 is None:
        open_link = '<span class="src-loc">page preview unavailable (file damaged?)</span>'
    elif pdf.exists():
        page_html = (f'<div class="pagebox"><img alt="{html.escape(c.source)} page {c.page}" '
                     f'src="data:image/png;base64,{b64}"></div>')
        open_link = (f'<a class="open-link" href="#src-{key}-{c.sid}">View page {c.page} ↗</a>'
                     f'<a class="close-link" href="#_closed">Hide page ✕</a>')
    return f"""
<div class="src" id="src-{key}-{c.sid}">
  <div class="src-top">
    <div style="display:flex;gap:8px;align-items:center;min-width:0">
      <span class="src-id">S{c.sid}</span><span class="src-file">{html.escape(c.source)}</span></div>
    <span class="src-loc">p. {c.page} · {span}</span>
  </div>
  {render_lines(c)}
  <div style="display:flex;justify-content:space-between;margin-top:8px" class="src-loc">
    <span>match</span><span>{c.similarity:.0%}</span></div>
  <div class="bar"><span style="width:{max(0, min(1, c.similarity)) * 100:.0f}%"></span></div>
  <div class="src-actions"><span></span>{open_link}</div>
  {page_html}
</div>"""


def show_answer(ans: Answer, key: str, target=None) -> None:
    box = target or st.container()
    with box.container():
        label, fg, bg = status_style(ans.status)
        refused = ans.status != "answered"
        n_src = max((c.sid for c in ans.citations), default=0)
        st.markdown(f'<div class="answer{" refuse" if refused else ""}">'
                    f'{render_answer_text(ans.answer, key, n_src)}</div>', unsafe_allow_html=True)

        m = ans.metrics
        pills = [f'<span class="pill" style="background:{bg};color:{fg}">● {label}</span>',
                 f'<span class="pill">⏱ {m.get("total_ms", 0) / 1000:.2f} s</span>']
        if m.get("ttft_ms"):
            pills.append(f'<span class="pill">first word {m["ttft_ms"] / 1000:.2f} s</span>')
        if m.get("generation_ms"):
            pills.append(f'<span class="pill">retrieval {m["retrieval_ms"]:.0f} ms · '
                         f'generation {m["generation_ms"] / 1000:.1f} s</span>')
        if ans.status == "answered":
            pills.append(f'<span class="pill">grounding {m.get("grounding", 0):.0%}</span>')
        if m.get("scope") and m["scope"] != "all":
            pills.append(f'<span class="pill">🔎 {html.escape(m["scope"].replace(",", ", "))}</span>')
        st.markdown('<div class="meta">' + "".join(pills) + "</div>", unsafe_allow_html=True)
        if m.get("quarantined"):
            where = ", ".join(f'{q["source"]} p.{q["page"]}' for q in m["quarantined"])
            st.warning(f"{len(m['quarantined'])} passage(s) were hidden from the model because they contain "
                       f"instruction-like text (possible prompt injection): {where}", icon=":material/shield:")
        if m.get("unsupported_numbers"):
            st.warning("These numbers were not found in the cited text: " + ", ".join(m["unsupported_numbers"]),
                       icon=":material/report:")

        if not ans.citations:
            return
        st.markdown('<div class="section-label">Sources · click a citation tag to jump</div>',
                    unsafe_allow_html=True)
        cols = st.columns(min(len(ans.citations), 2))
        for i, c in enumerate(ans.citations):
            cols[i % len(cols)].markdown(source_card(c, key), unsafe_allow_html=True)


# ------------------------------------------------------------------ Ask
with tab_ask:
    if "history" not in st.session_state:
        st.session_state.history = []

    # document filter
    names = list(sources)
    st.session_state.scope = [s for s in st.session_state.get("scope", []) if s in names]
    f_left, f_right = st.columns([5, 1.4], vertical_alignment="center")
    with f_right.popover("Filter documents", icon=":material/filter_list:", width="stretch"):
        st.multiselect("Search only in", names, key="scope", placeholder="All documents",
                       help="Leave empty to search every indexed document.")
    scope = st.session_state.scope
    f_left.markdown(
        f'<div class="scope">Searching <b>{len(scope)} of {len(names)}</b> documents: '
        f'{html.escape(", ".join(scope))}</div>' if scope else
        f'<div class="scope">Searching <b>all {len(names)}</b> documents</div>', unsafe_allow_html=True)

    welcome = st.empty()
    conversation = st.container()
    typed = st.chat_input("Ask a question about your documents…", disabled=store.count() == 0)
    q = typed or st.session_state.pop("pending_q", None)

    if not st.session_state.history and not q:
        with welcome.container():
            if store.count() == 0:
                st.markdown('<div class="empty"><h3>No documents yet</h3>'
                            '<p>Add PDFs in the <b>Documents</b> tab to start asking questions.</p></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="empty"><h3>Ask anything about your documents</h3>'
                            '<p>Every answer is backed by a highlighted line from your PDFs, '
                            'or it is refused.</p></div>', unsafe_allow_html=True)
                qs = sample_questions()
                if qs:
                    st.markdown('<div class="section-label" style="text-align:center">Try one of these</div>',
                                unsafe_allow_html=True)
                    cols = st.columns(len(qs))
                    for col, sq in zip(cols, qs):
                        if col.button(sq, width="stretch", key=f"sq_{sq}"):
                            st.session_state.pending_q = sq
                            st.rerun()

    with conversation:
        for i, (hq, ans) in enumerate(st.session_state.history):
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(f"**{html.escape(hq)}**")
            with st.chat_message("assistant", avatar=":material/menu_book:"):
                show_answer(ans, key=f"a{i}")
        if q:
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(f"**{html.escape(q)}**")
            with st.chat_message("assistant", avatar=":material/menu_book:"):
                slot = st.empty()
                slot.markdown('<div class="answer live"><span class="scope">Searching your documents…</span>'
                              '<span class="caret"></span></div>', unsafe_allow_html=True)

                def on_token(text: str) -> None:
                    if "NOT_FOUND".startswith(text.strip()[:9]) or text.strip().startswith("NOT_FOUND"):
                        body = '<span class="scope">Checking the sources…</span>'
                    else:
                        body = render_answer_text(text, None)
                    slot.markdown(f'<div class="answer live">{body}<span class="caret"></span></div>',
                                  unsafe_allow_html=True)

                ans = pipeline.ask(q, sources=scope or None, on_token=on_token if stream else None)
                show_answer(ans, key=f"a{len(st.session_state.history)}", target=slot)
            st.session_state.history.append((q, ans))
            st.rerun()  # refresh sidebar export buttons


# ------------------------------------------------------------------ Documents
with tab_docs:
    show_flash()
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Library")
        if not sources:
            st.caption("Nothing indexed yet.")
        for name, n in sources.items():
            a, b = st.columns([6, 1], vertical_alignment="center")
            a.markdown(f'<div class="doc"><div class="ic">PDF</div><div><div class="src-file">'
                       f'{html.escape(name)}</div><div class="src-loc">{n} passages indexed</div></div></div>',
                       unsafe_allow_html=True)
            if IS_ADMIN and b.button("", key=f"rm_{name}", icon=":material/delete:",
                                     help=f"Remove {name} from the index"):
                store.delete_source(name)
                audit("document_removed", USER, file=name)
                st.rerun()
    with right:
        if not IS_ADMIN:
            st.subheader("Add documents")
            st.info("Only admins can add or remove documents.", icon=":material/lock:")
        else:
            st.subheader("Add documents")
            files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True,
                                     label_visibility="collapsed",
                                     help=f"PDF only · up to {config.MAX_UPLOAD_MB} MB and {config.MAX_PAGES} pages each")
            if files and st.button(f"Index {len(files)} file(s)", type="primary", width="stretch",
                                   icon=":material/upload_file:"):
                for f in files:
                    data = f.getvalue()
                    with st.spinner(f"Checking and indexing {f.name}…"):
                        r = safe_add_pdf(store, data, f.name)
                    if not r.ok:
                        flash("error", f"{f.name}: not added: {r.reason}. Nothing was changed.")
                        audit("upload_rejected", USER, file=r.name, reason=r.reason)
                        continue
                    audit("document_added", USER, file=r.name, passages=r.passages, flagged=r.flagged,
                          size_kb=round(len(data) / 1024))
                    flash("success", f"{r.name}: {r.passages} passages indexed.")
                    if r.flagged:
                        flash("warning", f"{r.name}: {r.flagged} passage(s) contain instruction-like text "
                                         "(possible prompt injection). They will never be shown to the model.")
                st.rerun()
            st.write("")
            c1, c2 = st.columns(2)
            if c1.button("Re-index docs/", width="stretch", icon=":material/sync:"):
                with st.spinner("Indexing…"):
                    results = index_folder_report(store)
                good = [r for r in results if r.ok]
                audit("reindex", USER, files=len(good), rejected=len(results) - len(good))
                flash("success", f"Indexed {len(good)} PDF(s), {sum(r.passages for r in good)} passages.")
                for r in results:
                    if not r.ok:
                        flash("warning", f"{r.name}: damaged, so it was moved to docs/_rejected/ "
                                         f"and left out ({r.reason}).")
                st.rerun()
            if c2.button("Reset index", width="stretch", icon=":material/restart_alt:"):
                store.reset()
                audit("index_reset", USER)
                st.rerun()
            st.caption(f"Files are checked before indexing: real PDF, not password-protected, "
                       f"≤ {config.MAX_UPLOAD_MB} MB, ≤ {config.MAX_PAGES} pages.")


# ------------------------------------------------------------------ Dashboard
with tab_dash:
    df = load_query_log()
    if df.empty:
        st.markdown('<div class="empty"><h3>No queries yet</h3>'
                    '<p>Ask a few questions and latency and quality metrics will appear here.</p></div>',
                    unsafe_allow_html=True)
    else:
        answered = df[df.status == "answered"]
        k = st.columns(5)
        kpi(k[0], "Queries", f"{len(df)}")
        kpi(k[1], "Avg latency", f"{df.total_ms.mean() / 1000:.2f}s", "end to end")
        if "ttft_ms" in df and df.ttft_ms.notna().any():
            kpi(k[2], "Avg first word", f"{df.ttft_ms.mean() / 1000:.2f}s", "time to first token")
        else:
            kpi(k[2], "p95 latency", f"{df.total_ms.quantile(.95) / 1000:.2f}s")
        kpi(k[3], "Answer rate", f"{(df.status == 'answered').mean():.0%}", "rest were refused")
        kpi(k[4], "Avg grounding",
            f"{answered.grounding.mean():.0%}" if "grounding" in df and len(answered) else "–",
            "answer words found in sources")
        st.write("")

        d = df.reset_index(drop=True).copy()
        d["query"] = d.index + 1
        fig = go.Figure()
        fig.add_bar(x=d["query"], y=d["retrieval_ms"], name="Retrieval", marker_color="#6C9BC4")
        fig.add_bar(x=d["query"], y=d["generation_ms"], name="Generation", marker_color=T["accent"])
        fig.update_layout(barmode="stack", title="Latency per query (ms)", xaxis_title="query #",
                          bargap=0.55 if len(d) < 8 else 0.25)
        st.plotly_chart(style_fig(fig, 340), width="stretch")

        a, b = st.columns(2)
        with a:
            counts = df.status.map(lambda s: status_style(s)[0]).value_counts().reset_index()
            counts.columns = ["outcome", "count"]
            colour = {status_style(s)[0]: status_style(s)[1] for s in STATUS}
            fig = px.bar(counts, x="count", y="outcome", orientation="h", title="Query outcomes",
                         color="outcome", color_discrete_map=colour)
            fig.update_layout(showlegend=False, bargap=0.45)
            fig.update_yaxes(title=None)
            fig.update_xaxes(dtick=1)
            st.plotly_chart(style_fig(fig, 300), width="stretch")
        with b:
            if len(answered) and "grounding" in answered:
                fig = px.histogram(answered, x="grounding", nbins=10, range_x=[0, 1],
                                   title="Grounding score of answers")
                fig.update_traces(marker_color=AMBER)
                st.plotly_chart(style_fig(fig, 300), width="stretch")

        st.subheader("Recent queries")
        cols = [c for c in ["ts", "question", "status", "total_ms", "ttft_ms", "retrieval_ms", "generation_ms",
                            "grounding", "top_similarity", "scope", "citations", "model"] if c in df]
        st.dataframe(
            df[cols].iloc[::-1], width="stretch", hide_index=True,
            column_config={
                "ts": st.column_config.DatetimeColumn("Time", format="DD MMM, HH:mm"),
                "question": st.column_config.TextColumn("Question", width="large"),
                "total_ms": st.column_config.NumberColumn("Total (ms)", format="%.0f"),
                "ttft_ms": st.column_config.NumberColumn("First word (ms)", format="%.0f"),
                "retrieval_ms": st.column_config.NumberColumn("Retrieval (ms)", format="%.0f"),
                "generation_ms": st.column_config.NumberColumn("Generation (ms)", format="%.0f"),
                "grounding": st.column_config.ProgressColumn("Grounding", min_value=0, max_value=1, format="percent"),
                "top_similarity": st.column_config.NumberColumn("Top match", format="%.2f"),
            })
        if IS_ADMIN and st.button("Clear query log", icon=":material/delete_sweep:"):
            audit("query_log_cleared", USER)
            config.QUERY_LOG.unlink(missing_ok=True)
            st.rerun()


# ------------------------------------------------------------------ Evaluation
with tab_eval:
    top_l, top_r = st.columns([4, 1], vertical_alignment="center")
    top_l.markdown("Runs every question in `eval/questions.json`. Answerable questions check that the right page "
                   "is retrieved and cited; unanswerable ones check that the assistant **refuses**.")
    if top_r.button("Run evaluation", type="primary", width="stretch", icon=":material/play_arrow:",
                    disabled=store.count() == 0 or not IS_ADMIN,
                    help=None if IS_ADMIN else "Only admins can run the evaluation."):
        audit("evaluation_run", USER)
        with st.spinner("Running evaluation…"):
            run_eval(pipeline)
    res = load_eval_results()
    if config.EVAL_RESULTS.exists() and res is None:
        st.warning("The saved evaluation results are damaged and were ignored. Run the evaluation again.")
    if res:
        s = res["summary"]
        pct = lambda v: "–" if v is None else f"{v:.0%}"
        k = st.columns(6)
        kpi(k[0], "Hit @ k", pct(s["retrieval_hit_rate"]), "right page retrieved")
        kpi(k[1], "MRR", "–" if s["mrr"] is None else f"{s['mrr']:.2f}", "rank of right page")
        kpi(k[2], "Citation acc.", pct(s["citation_accuracy"]), "right page cited")
        kpi(k[3], "Refusal acc.", pct(s["refusal_accuracy"]), "off-topic refused")
        kpi(k[4], "Grounding", pct(s["avg_grounding"]), "avg of answers")
        kpi(k[5], "p95 latency", f"{s['latency_p95_ms'] / 1000:.1f}s", f"avg {s['latency_avg_ms'] / 1000:.1f}s")
        st.caption(f"Last run {s['run_at']} · model `{s['model']}` · {s['n_questions']} questions")
        rows = pd.DataFrame(res["rows"])
        rows["status"] = rows["status"].map(lambda x: status_style(x)[0])
        st.dataframe(
            rows[["question", "answerable", "status", "behaviour_correct", "retrieval_hit",
                  "citation_correct", "grounding", "total_ms", "answer"]],
            width="stretch", hide_index=True,
            column_config={
                "question": st.column_config.TextColumn("Question", width="medium"),
                "answerable": st.column_config.CheckboxColumn("Answerable"),
                "behaviour_correct": st.column_config.CheckboxColumn("Correct behaviour"),
                "retrieval_hit": st.column_config.CheckboxColumn("Retrieved"),
                "citation_correct": st.column_config.CheckboxColumn("Cited right page"),
                "grounding": st.column_config.ProgressColumn("Grounding", min_value=0, max_value=1, format="percent"),
                "total_ms": st.column_config.NumberColumn("Latency (ms)", format="%.0f"),
                "answer": st.column_config.TextColumn("Answer", width="large"),
            })
    else:
        st.markdown('<div class="empty"><h3>No evaluation run yet</h3>'
                    '<p>Click <b>Run evaluation</b> to score retrieval, citations and refusals.</p></div>',
                    unsafe_allow_html=True)


# ------------------------------------------------------------------ Security (admin only)
if tab_sec is not None:
    with tab_sec:
        events = read_audit(limit=1000)
        now = pd.Timestamp.now(tz="UTC")
        recent = [e for e in events if (now - pd.Timestamp(e["ts"])) < pd.Timedelta(hours=24)]
        all_users = users.list_users()
        qlog = load_query_log()
        k = st.columns(4)
        kpi(k[0], "Accounts", f"{len(all_users)}", f"{sum(u['role'] == 'admin' for u in all_users)} admin(s)")
        kpi(k[1], "Failed sign-ins", f"{sum(e['event'] == 'login_failed' for e in recent)}", "last 24 hours")
        kpi(k[2], "Locked accounts", f"{sum(u['locked'] for u in all_users)}",
            f"after {config.LOGIN_MAX_ATTEMPTS} wrong passwords")
        kpi(k[3], "Injection blocks",
            f"{int(qlog['quarantined'].sum()) if 'quarantined' in qlog else 0}", "passages hidden from model")
        st.write("")

        left, right = st.columns([3, 2], gap="large")
        with left:
            st.subheader("Users")
            st.dataframe(pd.DataFrame(all_users).drop(columns=["key"], errors="ignore"),
                         width="stretch", hide_index=True,
                         column_config={"username": "User", "role": "Role", "created": "Created",
                                        "last_login": "Last sign-in",
                                        "locked": st.column_config.CheckboxColumn("Locked")})
            others = [u["username"] for u in all_users if u["key"] != USER]
            if others:
                m1, m2 = st.columns([2, 3], vertical_alignment="bottom")
                target = m1.selectbox("Manage user", others)
                b1, b2, b3 = m2.columns(3)
                if b1.button("Unlock", width="stretch", icon=":material/lock_open:"):
                    users.unlock(target)
                    audit("user_unlocked", USER, target=target)
                    st.rerun()
                with b2.popover("Reset", icon=":material/password:", width="stretch"):
                    with st.form("reset_pw", clear_on_submit=True):
                        npw = st.text_input(f"New password for {target}", type="password")
                        if st.form_submit_button("Set password", type="primary"):
                            try:
                                users.set_password(target, npw)
                                audit("password_reset", USER, target=target)
                                st.success("Password updated.")
                            except AuthError as e:
                                st.error(str(e))
                if b3.button("Delete", width="stretch", icon=":material/person_remove:"):
                    try:
                        users.delete_user(target)
                        audit("user_deleted", USER, target=target)
                        st.rerun()
                    except AuthError as e:
                        st.error(str(e))
        with right:
            st.subheader("Add a user")
            with st.form("add_user", clear_on_submit=True):
                nu = st.text_input("Username", placeholder="lowercase, 3-32 characters")
                npw = st.text_input("Temporary password", type="password",
                                    help="At least 8 characters with a letter and a number.")
                nr = st.selectbox("Role", ROLES, index=1,
                                  help="viewer: ask questions only · admin: also manage documents and users")
                if st.form_submit_button("Create user", type="primary", width="stretch",
                                         icon=":material/person_add:"):
                    try:
                        users.add_user(nu, npw, nr)
                        audit("user_created", USER, target=nu.strip().lower(), role=nr)
                        st.rerun()
                    except AuthError as e:
                        st.error(str(e))

        st.subheader("Protections in place")
        guards = [
            ("Sign-in required", "Salted PBKDF2-SHA256 hashes (240k rounds); passwords are never stored."),
            ("Brute-force lockout", f"{config.LOGIN_MAX_ATTEMPTS} wrong passwords lock the account for "
                                    f"{config.LOCKOUT_MINUTES} min."),
            ("Session timeout", f"Signed out after {config.SESSION_TIMEOUT_MINUTES} min of inactivity."),
            ("Role-based access", "Viewers can only ask questions; admins manage documents and users."),
            ("Upload checks", f"Real, unencrypted PDFs only · ≤ {config.MAX_UPLOAD_MB} MB · ≤ {config.MAX_PAGES} "
                              "pages · file names sanitised."),
            ("Prompt-injection guard", "Passages that try to instruct the model are never sent to it; "
                                       "sources are fenced as untrusted data." if config.INJECTION_GUARD else "Off"),
            ("Private logs", "Emails, phone and ID numbers are masked in the query and audit logs."
             if config.REDACT_LOGS else "Off"),
            ("Local-only server", "Bound to localhost: other devices on your network can't open the app."),
            ("XSS-safe rendering", "All document text and answers are HTML-escaped before display."),
        ]
        g = st.columns(3)
        for i, (title, desc) in enumerate(guards):
            g[i % 3].markdown(f'<div class="guard"><span class="ok">✓</span><div><b>{title}</b>'
                              f'<small>{desc}</small></div></div>', unsafe_allow_html=True)

        st.subheader("Audit log")
        if events:
            adf = pd.DataFrame(events)
            kinds = sorted(adf["event"].unique())
            pick = st.multiselect("Show events", kinds, placeholder="All events")
            if pick:
                adf = adf[adf["event"].isin(pick)]
            adf["ts"] = pd.to_datetime(adf["ts"])
            st.dataframe(adf, width="stretch", hide_index=True,
                         column_config={"ts": st.column_config.DatetimeColumn("Time", format="DD MMM, HH:mm:ss"),
                                        "event": "Event", "user": "User"})
        else:
            st.caption("No events recorded yet.")
