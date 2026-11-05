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

st.set_page_config(page_title="Grounded RAG Assistant", page_icon="📑", layout="wide")

st.markdown("""
<style>
.cite {background:#2563eb;color:#fff;border-radius:6px;padding:1px 6px;font-size:0.78em;
       font-weight:600;margin-left:2px;white-space:nowrap}
.src-head {font-weight:600;margin-bottom:4px}
.lines {font-family:ui-monospace,Consolas,monospace;font-size:0.85em;line-height:1.5;
        border:1px solid rgba(128,128,128,.25);border-radius:8px;padding:8px 10px}
.lines .ln {color:#888;display:inline-block;width:3.2em}
.lines mark {background:#fde68a;color:#111;border-radius:3px;padding:0 2px}
.status {font-size:0.8em;opacity:.75}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ resources
@st.cache_resource(show_spinner="Loading vector store...")
def get_store(embed_backend: str, embed_model: str) -> VectorStore:
    return VectorStore(embedder=Embedder(embed_backend, embed_model))


@st.cache_data(show_spinner=False, max_entries=64)
def page_png(path: str, page: int, lines: tuple, mtime: float) -> bytes:
    return render_highlighted_page(path, page, list(lines))


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    backend = st.selectbox("LLM backend", ["ollama", "openai"],
                           index=0 if config.LLM_BACKEND == "ollama" else 1,
                           help="ollama = fully local. openai = any OpenAI-compatible API (set OPENAI_API_KEY / OPENAI_BASE_URL).")
    if backend == "ollama":
        models = ollama_models()
        if models:
            st.success(f"Ollama running · {len(models)} model(s)")
            default = next((m for m in models if m.split(":")[0] == config.LLM_MODEL.split(":")[0]), models[0])
            model = st.selectbox("Model", models, index=models.index(default))
        else:
            st.error("Ollama not reachable at " + config.OLLAMA_URL)
            st.caption("Install from ollama.com, then run:\n\n`ollama pull llama3.2`")
            model = st.text_input("Model", config.LLM_MODEL)
    else:
        model = st.text_input("Model", config.LLM_MODEL if config.LLM_BACKEND == "openai" else "gpt-4o-mini")
        if not config.OPENAI_API_KEY:
            st.warning("OPENAI_API_KEY is not set.")

    top_k = st.slider("Chunks retrieved (top-k)", 1, 10, config.TOP_K)
    min_sim = st.slider("Min. similarity to count as evidence", 0.0, 1.0, config.MIN_SIMILARITY, 0.05,
                        help="If no chunk reaches this score the assistant refuses without asking the LLM.")
    st.caption(f"Embeddings: `{config.EMBED_BACKEND}`"
               + (f" · `{config.EMBED_MODEL}`" if config.EMBED_BACKEND == "ollama" else " · all-MiniLM-L6-v2"))

store = get_store(config.EMBED_BACKEND, config.EMBED_MODEL)
pipeline = RAGPipeline(store=store, llm=LLM(backend, model), top_k=top_k, min_similarity=min_sim)

st.title("📑 Grounded RAG Assistant")
st.caption("Answers only from your PDFs. Every claim is cited to the exact page and line, or the assistant refuses.")

tab_ask, tab_docs, tab_dash, tab_eval = st.tabs(["💬 Ask", "📂 Documents", "📊 Dashboard", "🧪 Evaluation"])


# ------------------------------------------------------------------ helpers
def render_answer_text(text: str) -> str:
    safe = html.escape(text).replace("\n", "<br>")
    return re.sub(r"\[S(\d+)\]", r'<span class="cite">S\1</span>', safe)


def render_lines(c) -> str:
    hl = set(c.highlight_lines)
    rows = []
    for j, t in enumerate(c.lines):
        n = c.line_start + j
        body = html.escape(t)
        if n in hl:
            body = f"<mark>{body}</mark>"
        rows.append(f'<span class="ln">L{n}</span>{body}')
    return '<div class="lines">' + "<br>".join(rows) + "</div>"


def show_answer(ans: Answer, key: str) -> None:
    st.markdown(render_answer_text(ans.answer), unsafe_allow_html=True)
    m = ans.metrics
    bits = [f"status: **{ans.status}**", f"⏱ {m.get('total_ms', 0):.0f} ms"]
    if m.get("generation_ms"):
        bits.append(f"retrieval {m['retrieval_ms']:.0f} ms · generation {m['generation_ms']:.0f} ms")
    if ans.status == "answered":
        bits.append(f"grounding {m.get('grounding', 0):.0%}")
    st.caption(" · ".join(bits))
    if m.get("unsupported_numbers"):
        st.warning("Numbers not found in the cited text: " + ", ".join(m["unsupported_numbers"]))

    for c in ans.citations:
        hl = c.highlight_lines
        span = f"line {hl[0]}" if len(hl) == 1 else f"lines {hl[0]}–{hl[-1]}"
        with st.expander(f"S{c.sid} · {c.source} · page {c.page} · {span}  (similarity {c.similarity:.2f})"):
            left, right = st.columns([1, 1])
            with left:
                st.markdown(render_lines(c), unsafe_allow_html=True)
            with right:
                pdf = config.DOCS_DIR / c.source
                if pdf.exists():
                    st.image(page_png(str(pdf), c.page, tuple(hl), pdf.stat().st_mtime),
                             caption=f"{c.source} — page {c.page}", width="stretch")
                else:
                    st.info("Original PDF not found in docs/.")


# ------------------------------------------------------------------ Ask
with tab_ask:
    if store.count() == 0:
        st.info("No documents indexed yet. Add PDFs in the **📂 Documents** tab.")
    if "history" not in st.session_state:
        st.session_state.history = []

    for i, (q, ans) in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            show_answer(ans, key=f"h{i}")

    if q := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating a cited answer..."):
                ans = pipeline.ask(q)
            show_answer(ans, key="new")
        st.session_state.history.append((q, ans))

    if st.session_state.history and st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()


# ------------------------------------------------------------------ Documents
with tab_docs:
    st.subheader("Add documents")
    files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    if files and st.button(f"Index {len(files)} file(s)", type="primary"):
        for f in files:
            dest = config.DOCS_DIR / Path(f.name).name
            dest.write_bytes(f.getbuffer())
            with st.spinner(f"Indexing {dest.name}..."):
                n = index_pdf(store, dest)
            if n:
                st.success(f"{dest.name}: {n} chunks indexed")
            else:
                st.warning(f"{dest.name}: no text found (scanned PDF? run OCR first)")

    c1, c2 = st.columns(2)
    if c1.button("Re-index everything in docs/"):
        with st.spinner("Indexing..."):
            res = index_folder(store)
        st.success(f"Indexed {len(res)} PDF(s), {sum(res.values())} chunks")
    if c2.button("Reset index"):
        store.reset()
        st.success("Index cleared")

    st.subheader("Indexed documents")
    src = store.sources()
    if not src:
        st.caption("Nothing indexed yet.")
    for name, n in src.items():
        a, b, c = st.columns([5, 2, 1])
        a.write(f"📄 {name}")
        b.caption(f"{n} chunks")
        if c.button("Remove", key=f"rm_{name}"):
            store.delete_source(name)
            st.rerun()


# ------------------------------------------------------------------ Dashboard
with tab_dash:
    df = load_query_log()
    if df.empty:
        st.info("Ask a few questions and the latency/quality dashboard will appear here.")
    else:
        answered = df[df.status == "answered"]
        k = st.columns(5)
        k[0].metric("Queries", len(df))
        k[1].metric("Avg latency", f"{df.total_ms.mean():.0f} ms")
        k[2].metric("p95 latency", f"{df.total_ms.quantile(.95):.0f} ms")
        k[3].metric("Answer rate", f"{(df.status == 'answered').mean():.0%}")
        k[4].metric("Avg grounding", f"{answered.grounding.mean():.0%}" if "grounding" in df and len(answered) else "–")

        d = df.reset_index(drop=True).copy()
        d["query #"] = d.index + 1
        fig = go.Figure()
        fig.add_bar(x=d["query #"], y=d["retrieval_ms"], name="Retrieval")
        fig.add_bar(x=d["query #"], y=d["generation_ms"], name="Generation")
        fig.update_layout(barmode="stack", title="Latency per query (ms)", height=340,
                          margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")

        a, b = st.columns(2)
        with a:
            counts = df.status.value_counts().reset_index()
            counts.columns = ["status", "count"]
            st.plotly_chart(px.bar(counts, x="status", y="count", title="Outcome of each query",
                                   height=300), width="stretch")
        with b:
            if len(answered) and "grounding" in answered:
                st.plotly_chart(px.histogram(answered, x="grounding", nbins=10, range_x=[0, 1],
                                             title="Grounding score of answers", height=300),
                                width="stretch")
            if "top_similarity" in df:
                st.caption(f"Avg top retrieval similarity: {df.top_similarity.mean():.2f}")

        st.subheader("Recent queries")
        cols = [c for c in ["ts", "question", "status", "total_ms", "retrieval_ms", "generation_ms",
                            "grounding", "top_similarity", "citations", "model"] if c in df]
        st.dataframe(df[cols].iloc[::-1], width="stretch", hide_index=True)
        if st.button("Clear query log"):
            config.QUERY_LOG.unlink(missing_ok=True)
            st.rerun()


# ------------------------------------------------------------------ Evaluation
with tab_eval:
    st.write("Runs every question in `eval/questions.json` through the pipeline. Answerable questions check "
             "that the right page is retrieved and cited; unanswerable ones check that the assistant refuses.")
    if st.button("▶ Run evaluation", type="primary", disabled=store.count() == 0):
        with st.spinner("Running evaluation..."):
            run_eval(pipeline)
    if config.EVAL_RESULTS.exists():
        res = json.loads(config.EVAL_RESULTS.read_text(encoding="utf-8"))
        s = res["summary"]
        st.caption(f"Last run: {s['run_at']} · model `{s['model']}` · {s['n_questions']} questions")
        fmt = lambda v: "–" if v is None else f"{v:.0%}"
        k = st.columns(6)
        k[0].metric("Retrieval hit@k", fmt(s["retrieval_hit_rate"]))
        k[1].metric("MRR", "–" if s["mrr"] is None else f"{s['mrr']:.2f}")
        k[2].metric("Citation accuracy", fmt(s["citation_accuracy"]))
        k[3].metric("Refusal accuracy", fmt(s["refusal_accuracy"]))
        k[4].metric("Avg grounding", fmt(s["avg_grounding"]))
        k[5].metric("p95 latency", f"{s['latency_p95_ms']:.0f} ms")
        st.dataframe(pd.DataFrame(res["rows"]), width="stretch", hide_index=True)
