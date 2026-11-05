"""Offline evaluation + query-log analytics for the dashboard.

Eval set format (eval/questions.json):
[
  {"question": "...", "answerable": true,
   "expected_source": "club_rules.pdf", "expected_page": 2,
   "expected_keywords": ["500", "membership"]},
  {"question": "Who won the 2018 World Cup?", "answerable": false}
]
"""
from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import pandas as pd

import config
from rag.pipeline import RAGPipeline


def load_query_log(path: Path = config.QUERY_LOG) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:  # a damaged line (e.g. power cut mid-write) is skipped, not fatal
            rec = json.loads(line)
            if isinstance(rec, dict) and "ts" in rec and "status" in rec:
                rows.append(rec)
        except ValueError:
            continue
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        for col in ("total_ms", "retrieval_ms", "generation_ms", "grounding", "top_similarity", "ttft_ms",
                    "quarantined"):
            if col in df:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _p95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]


def load_eval_results(path: Path = config.EVAL_RESULTS) -> dict | None:
    """Last evaluation results, or None if missing/damaged."""
    try:
        res = json.loads(Path(path).read_text(encoding="utf-8"))
        res["summary"]["model"], res["rows"][0]  # basic shape check
        return res
    except Exception:
        return None


def run_eval(pipeline: RAGPipeline, eval_path: Path = config.EVAL_SET,
             out_path: Path | None = config.EVAL_RESULTS) -> dict:
    items = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    rows = []
    for it in items:
        ans = pipeline.ask(it["question"], log=False)
        answerable = it.get("answerable", True)
        exp = (it.get("expected_source"), it.get("expected_page"))

        ranks = [i for i, r in enumerate(ans.retrieved, start=1)
                 if (r["source"], r["page"]) == exp]
        cited_pages = {(c.source, c.page) for c in ans.citations}
        kws = [k.lower() for k in it.get("expected_keywords", [])]
        answered = ans.status == "answered"

        rows.append({
            "question": it["question"],
            "answerable": answerable,
            "status": ans.status,
            "retrieval_hit": bool(ranks) if answerable else None,
            "reciprocal_rank": (1 / ranks[0] if ranks else 0.0) if answerable else None,
            "citation_correct": (exp in cited_pages) if answerable else None,
            "keyword_recall": (sum(k in ans.answer.lower() for k in kws) / len(kws)
                               if (answerable and kws) else None),
            # correct behaviour = answer the answerable ones, refuse the rest
            "behaviour_correct": answered if answerable else not answered,
            "grounding": ans.metrics.get("grounding"),
            "unsupported_numbers": len(ans.metrics.get("unsupported_numbers", []) or []),
            "retrieval_ms": ans.metrics.get("retrieval_ms"),
            "generation_ms": ans.metrics.get("generation_ms"),
            "total_ms": ans.metrics.get("total_ms"),
            "answer": ans.answer,
        })

    df = pd.DataFrame(rows)
    ans_df = df[df.answerable]
    unans_df = df[~df.answerable]

    def mean(s):
        s = s.dropna()
        return round(float(s.astype(float).mean()), 3) if len(s) else None

    summary = {
        "n_questions": len(df),
        "retrieval_hit_rate": mean(ans_df.retrieval_hit),
        "mrr": mean(ans_df.reciprocal_rank),
        "citation_accuracy": mean(ans_df.citation_correct),
        "keyword_recall": mean(ans_df.keyword_recall),
        "answer_rate": mean(ans_df.status.eq("answered")),
        "refusal_accuracy": mean(unans_df.behaviour_correct) if len(unans_df) else None,
        "avg_grounding": mean(df[df.status == "answered"].grounding),
        "latency_avg_ms": round(statistics.mean(df.total_ms), 1) if len(df) else 0,
        "latency_p95_ms": round(_p95(df.total_ms.tolist()), 1),
        "model": pipeline.llm.model,
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    result = {"summary": summary, "rows": rows}
    if out_path:
        Path(out_path).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
