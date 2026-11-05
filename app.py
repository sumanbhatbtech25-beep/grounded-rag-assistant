"""Streamlit UI:  streamlit run app.py"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
from rag.evaluation import load_query_log, run_eval
from rag.highlight import render_highlighted_page
from rag.indexer import index_folder, index_pdf
from rag.llm import LLM, ollama_models
from rag.pipeline import Answer, RAGPipeline
from rag.store import Embedder, VectorStore

st.set_page_config(page_title="CiteSight · Grounded RAG Assistant", page_icon="📑",
                   layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ design tokens
NAVY, NAVY_DARK, AMBER, INK, MUTED, PAPER, CARD, LINE = (
    "#1F4B7A", "#13243A", "#E0A526", "#1B2430", "#6B7280", "#F7F6F2", "#FFFFFF", "#E0DCD1")
STATUS = {  # label, text colour, background
    "answered":   ("Grounded answer", "#1E6B4F", "#E3F2EA"),
    "refused":    ("Not in documents", "#7A5A12", "#FBF0D5"),
    "no_context": ("No relevant source", "#7A5A12", "#FBF0D5"),
    "unverified": ("Withheld · no valid citation", "#9A3B1F", "#FBE4DC"),
    "error":      ("Error", "#9A1F1F", "#FBDCDC"),
}

st.markdown(f"""
<style>
.block-container {{padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px;}}
h1, h2, h3 {{letter-spacing: -0.01em;}}

/* hero */
.hero {{background: linear-gradient(135deg, {NAVY_DARK} 0%, {NAVY} 100%); color:#fff;
        border-radius: 16px; padding: 22px 28px; margin-bottom: 18px;
        display:flex; align-items:center; justify-content:space-between; gap:20px; flex-wrap:wrap}}
.hero .brand {{display:flex; align-items:center; gap:14px}}
.hero .mark {{width:44px; height:44px; border-radius:12px; background:{AMBER}; color:{NAVY_DARK};
              display:flex; align-items:center; justify-content:center; font-weight:800; font-size:20px;
              font-family:'Source Serif 4',serif}}
.hero h1 {{color:#fff; font-size:1.6rem; margin:0; padding:0; line-height:1.2}}
.hero p {{margin:2px 0 0; color:#C9D5E4; font-size:0.92rem}}
.chips {{display:flex; gap:8px; flex-wrap:wrap}}
.chip {{background:rgba(255,255,255,.10); border:1px solid rgba(255,255,255,.18); color:#E6EBF2;
        padding:5px 11px; border-radius:999px; font-size:0.8rem; white-space:nowrap}}
.chip b {{color:#fff}}
.dot {{display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle}}

/* tabs as a segmented nav */
.stTabs [data-baseweb="tab-list"] {{gap:4px; border-bottom:1px solid {LINE}}}
.stTabs [data-baseweb="tab"] {{padding:8px 16px; font-weight:500}}

/* answer card */
.answer {{background:{CARD}; border:1px solid {LINE}; border-left:4px solid {NAVY};
          border-radius:12px; padding:16px 18px; line-height:1.65; font-size:1rem; color:{INK}}}
.answer.refuse {{border-left-color:{AMBER}}}
.cite {{background:{NAVY}; color:#fff; border-radius:6px; padding:1px 7px; font-size:0.72em;
        font-weight:600; margin-left:3px; vertical-align:1px; white-space:nowrap}}
.meta {{display:flex; gap:8px; flex-wrap:wrap; margin:10px 0 4px}}
.pill {{border-radius:999px; padding:3px 10px; font-size:0.78rem; font-weight:500;
        background:#EEECE5; color:#3C4655; white-space:nowrap}}

/* source cards */
.src {{background:{CARD}; border:1px solid {LINE}; border-radius:12px; padding:12px 14px; margin-bottom:6px}}
.src-top {{display:flex; justify-content:space-between; align-items:center; gap:10px; margin-bottom:8px}}
.src-id {{background:{NAVY}; color:#fff; border-radius:6px; padding:2px 8px; font-size:0.75rem; font-weight:700}}
.src-file {{font-weight:600; color:{INK}; font-size:0.92rem; overflow:hidden; text-overflow:ellipsis}}
.src-loc {{color:{MUTED}; font-size:0.82rem; white-space:nowrap}}
.bar {{height:5px; background:#EEECE5; border-radius:99px; overflow:hidden; margin-top:2px}}
.bar > span {{display:block; height:100%; background:{AMBER}}}
.lines {{font-family:'JetBrains Mono',Consolas,monospace; font-size:0.8rem; line-height:1.7;
         background:#FBFAF7; border:1px solid {LINE}; border-radius:8px; padding:8px 10px}}
.lines .ln {{color:#A3A9B3; display:inline-block; width:3em; user-select:none}}
.lines .hit {{background:#FCE9B4; border-radius:3px; padding:1px 3px; color:{INK}}}

/* KPI + doc cards */
.kpi {{background:{CARD}; border:1px solid {LINE}; border-radius:12px; padding:14px 16px; min-height:118px}}
.kpi .label {{color:{MUTED}; font-size:0.78rem; text-transform:uppercase; letter-spacing:.06em}}
.kpi .value {{font-family:'Source Serif 4',serif; font-size:1.7rem; font-weight:700; color:{INK}; margin-top:2px}}
.kpi .sub {{color:{MUTED}; font-size:0.78rem}}
.doc {{background:{CARD}; border:1px solid {LINE}; border-radius:12px; padding:12px 14px;
       display:flex; align-items:center; gap:12px}}
.doc .ic {{width:36px; height:44px; border-radius:6px; background:#F3E7E3; color:#B5523B;
           display:flex; align-items:center; justify-content:center; font-size:0.65rem; font-weight:800}}
.empty {{text-align:center; padding:26px 10px 10px; color:{MUTED}}}
.empty h3 {{color:{INK}; margin-bottom:4px}}
.section-label {{color:{MUTED}; font-size:0.78rem; text-transform:uppercase; letter-spacing:.06em;
                 margin:14px 0 6px; font-weight:600}}

/* sidebar */
section[data-testid="stSidebar"] .sb-title {{font-family:'Source Serif 4',serif; font-size:1.15rem;
        font-weight:700; color:#fff; margin-bottom:4px}}
section[data-testid="stSidebar"] .sb-note {{color:#9FB0C6; font-size:0.8rem}}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ resources
@st.cache_resource(show_spinner="Loading vector store...")
def get_store(embed_backend: str, embed_model: str) -> VectorStore:
    return VectorStore(embedder=Embedder(embed_backend, embed_model))


@st.cache_data(show_spinner=False, max_entries=64)
def page_png(path: str, page: int, lines: tuple, mtime: float) -> bytes:
    return render_highlighted_page(path, page, list(lines))


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
        font=dict(family="Inter, sans-serif", color=INK, size=12),
        title_font=dict(family="Source Serif 4, serif", size=16),
        legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
        colorway=[NAVY, AMBER, "#6C9BC4", "#9A7B4F"],
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE)
    fig.update_yaxes(gridcolor="#ECE9E1", zeroline=False)
    return fig


def kpi(col, label: str, value: str, sub: str = "") -> None:
    col.markdown(f'<div class="kpi"><div class="label">{label}</div><div class="value">{value}</div>'
                 f'<div class="sub">{sub}&nbsp;</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
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
    st.divider()
    emb = "all-MiniLM-L6-v2" if config.EMBED_BACKEND == "default" else config.EMBED_MODEL
    st.markdown(f'<div class="sb-note">Embeddings · {emb}<br>Vector store · ChromaDB</div>',
                unsafe_allow_html=True)
    if st.session_state.get("history") and st.button("Clear conversation", width="stretch"):
        st.session_state.history = []
        st.rerun()

store = get_store(config.EMBED_BACKEND, config.EMBED_MODEL)
pipeline = RAGPipeline(store=store, llm=LLM(backend, model), top_k=top_k, min_similarity=min_sim)
sources = store.sources()

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

tab_ask, tab_docs, tab_dash, tab_eval = st.tabs(
    [":material/forum: Ask", ":material/folder_open: Documents",
     ":material/monitoring: Dashboard", ":material/fact_check: Evaluation"])


# ------------------------------------------------------------------ answer rendering
def render_answer_text(text: str) -> str:
    safe = html.escape(text).replace("\n", "<br>")
    return re.sub(r"\[S(\d+)\]", r'<span class="cite">S\1</span>', safe)


def render_lines(c) -> str:
    hl = set(c.highlight_lines)
    rows = []
    for j, t in enumerate(c.lines):
        n = c.line_start + j
        body = html.escape(t)
        rows.append(f'<span class="ln">L{n}</span>' + (f'<span class="hit">{body}</span>' if n in hl else body))
    return '<div class="lines">' + "<br>".join(rows) + "</div>"


def show_answer(ans: Answer, key: str) -> None:
    label, fg, bg = STATUS.get(ans.status, (ans.status, INK, "#EEE"))
    refused = ans.status != "answered"
    st.markdown(f'<div class="answer{" refuse" if refused else ""}">{render_answer_text(ans.answer)}</div>',
                unsafe_allow_html=True)

    m = ans.metrics
    pills = [f'<span class="pill" style="background:{bg};color:{fg}">● {label}</span>',
             f'<span class="pill">⏱ {m.get("total_ms", 0) / 1000:.2f} s</span>']
    if m.get("generation_ms"):
        pills.append(f'<span class="pill">retrieval {m["retrieval_ms"]:.0f} ms · '
                     f'generation {m["generation_ms"] / 1000:.1f} s</span>')
    if ans.status == "answered":
        pills.append(f'<span class="pill">grounding {m.get("grounding", 0):.0%}</span>')
        pills.append(f'<span class="pill">{len(ans.citations)} source(s)</span>')
    st.markdown('<div class="meta">' + "".join(pills) + "</div>", unsafe_allow_html=True)
    if m.get("unsupported_numbers"):
        st.warning("These numbers were not found in the cited text: " + ", ".join(m["unsupported_numbers"]),
                   icon=":material/report:")

    if not ans.citations:
        return
    st.markdown('<div class="section-label">Sources</div>', unsafe_allow_html=True)
    cols = st.columns(min(len(ans.citations), 2))
    for i, c in enumerate(ans.citations):
        hl = c.highlight_lines
        span = f"line {hl[0]}" if len(hl) == 1 else f"lines {hl[0]}–{hl[-1]}"
        with cols[i % len(cols)]:
            st.markdown(f"""
            <div class="src">
              <div class="src-top">
                <div style="display:flex;gap:8px;align-items:center;min-width:0">
                  <span class="src-id">S{c.sid}</span><span class="src-file">{html.escape(c.source)}</span></div>
                <span class="src-loc">p. {c.page} · {span}</span>
              </div>
              {render_lines(c)}
              <div style="display:flex;justify-content:space-between;margin-top:8px" class="src-loc">
                <span>match</span><span>{c.similarity:.0%}</span></div>
              <div class="bar"><span style="width:{max(0, min(1, c.similarity)) * 100:.0f}%"></span></div>
            </div>""", unsafe_allow_html=True)
            pdf = config.DOCS_DIR / c.source
            if pdf.exists():
                with st.expander(f"View page {c.page} with highlight", icon=":material/description:"):
                    st.image(page_png(str(pdf), c.page, tuple(hl), pdf.stat().st_mtime), width="stretch")


# ------------------------------------------------------------------ Ask
with tab_ask:
    if "history" not in st.session_state:
        st.session_state.history = []

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
                show_answer(ans, key=f"h{i}")
        if q:
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(f"**{html.escape(q)}**")
            with st.chat_message("assistant", avatar=":material/menu_book:"):
                with st.spinner("Searching your documents and drafting a cited answer…"):
                    ans = pipeline.ask(q)
                show_answer(ans, key="new")
            st.session_state.history.append((q, ans))


# ------------------------------------------------------------------ Documents
with tab_docs:
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
            if b.button("", key=f"rm_{name}", icon=":material/delete:", help=f"Remove {name} from the index"):
                store.delete_source(name)
                st.rerun()
    with right:
        st.subheader("Add documents")
        files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True,
                                 label_visibility="collapsed")
        if files and st.button(f"Index {len(files)} file(s)", type="primary", width="stretch",
                               icon=":material/upload_file:"):
            for f in files:
                dest = config.DOCS_DIR / Path(f.name).name
                dest.write_bytes(f.getbuffer())
                with st.spinner(f"Indexing {dest.name}…"):
                    n = index_pdf(store, dest)
                if n:
                    st.success(f"{dest.name}: {n} passages indexed")
                else:
                    st.warning(f"{dest.name}: no text found (scanned PDF? run OCR first)")
            st.rerun()
        st.write("")
        c1, c2 = st.columns(2)
        if c1.button("Re-index docs/", width="stretch", icon=":material/sync:"):
            with st.spinner("Indexing…"):
                res = index_folder(store)
            st.toast(f"Indexed {len(res)} PDF(s), {sum(res.values())} passages")
            st.rerun()
        if c2.button("Reset index", width="stretch", icon=":material/restart_alt:"):
            store.reset()
            st.rerun()
        st.caption("PDFs are stored in the project's `docs/` folder. Scanned PDFs need OCR first.")


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
        fig.add_bar(x=d["query"], y=d["generation_ms"], name="Generation", marker_color=NAVY)
        fig.update_layout(barmode="stack", title="Latency per query (ms)", xaxis_title="query #",
                          bargap=0.55 if len(d) < 8 else 0.25)
        st.plotly_chart(style_fig(fig, 340), width="stretch")

        a, b = st.columns(2)
        with a:
            counts = df.status.map(lambda s: STATUS.get(s, (s,))[0]).value_counts().reset_index()
            counts.columns = ["outcome", "count"]
            colour = {v[0]: v[1] for v in STATUS.values()}
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
        cols = [c for c in ["ts", "question", "status", "total_ms", "retrieval_ms", "generation_ms",
                            "grounding", "top_similarity", "citations", "model"] if c in df]
        st.dataframe(
            df[cols].iloc[::-1], width="stretch", hide_index=True,
            column_config={
                "ts": st.column_config.DatetimeColumn("Time", format="DD MMM, HH:mm"),
                "question": st.column_config.TextColumn("Question", width="large"),
                "total_ms": st.column_config.NumberColumn("Total (ms)", format="%.0f"),
                "retrieval_ms": st.column_config.NumberColumn("Retrieval (ms)", format="%.0f"),
                "generation_ms": st.column_config.NumberColumn("Generation (ms)", format="%.0f"),
                "grounding": st.column_config.ProgressColumn("Grounding", min_value=0, max_value=1, format="percent"),
                "top_similarity": st.column_config.NumberColumn("Top match", format="%.2f"),
            })
        if st.button("Clear query log", icon=":material/delete_sweep:"):
            config.QUERY_LOG.unlink(missing_ok=True)
            st.rerun()


# ------------------------------------------------------------------ Evaluation
with tab_eval:
    top_l, top_r = st.columns([4, 1], vertical_alignment="center")
    top_l.markdown("Runs every question in `eval/questions.json`. Answerable questions check that the right page "
                   "is retrieved and cited; unanswerable ones check that the assistant **refuses**.")
    if top_r.button("Run evaluation", type="primary", width="stretch", icon=":material/play_arrow:",
                    disabled=store.count() == 0):
        with st.spinner("Running evaluation…"):
            run_eval(pipeline)
    if config.EVAL_RESULTS.exists():
        res = json.loads(config.EVAL_RESULTS.read_text(encoding="utf-8"))
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
        rows["status"] = rows["status"].map(lambda x: STATUS.get(x, (x,))[0])
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
