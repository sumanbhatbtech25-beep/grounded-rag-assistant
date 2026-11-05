"""Retrieve -> grounded prompt -> generate -> verify citations -> pinpoint lines.

Hallucination guards, in order:
  1. Relevance gate: if no chunk clears MIN_SIMILARITY, refuse without calling the LLM.
  2. Strict prompt: answer only from numbered sources, cite [S#] on every sentence,
     or reply NOT_FOUND.
  3. Verification: citations must point at real sources; an answer with no valid
     citation is withheld. Numbers in the answer that don't appear in the cited
     text are flagged.
  4. Pinpointing: each cited chunk is narrowed to the exact lines that support
     the sentence, so the UI can highlight page + line.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

import config
from rag.llm import LLM
from rag.store import Retrieved, VectorStore

REFUSAL = "I couldn't find this in the indexed documents, so I won't guess."

SYSTEM_PROMPT = """You are a strict document assistant. You answer ONLY from the numbered SOURCES given to you.

Rules:
1. Use only facts stated in the SOURCES. Never use outside knowledge.
2. End every sentence with a citation to the source(s) that support it, like [S1] or [S2][S3].
3. Quote exact wording for rules, numbers, dates and deadlines.
4. If the SOURCES do not contain the answer, reply with exactly: NOT_FOUND
5. Be concise: at most 5 sentences or bullet points."""

STOPWORDS = set("""a an the and or but if of to in on at by for with from as is are was were be been being
this that these those it its into than then there their they them he she his her we our you your i
not no can may must shall will would should could do does did done have has had any all each per such
which who whom what when where why how also only about over under up out so very more most other""".split())

CITE_RE = re.compile(r"\[S(\d+)\]")
NUM_RE = re.compile(r"\d+(?:[.,:]\d+)*")
# Split after . ! ? (or a closing citation) when the next word starts a new sentence.
# Abbreviations like "Rs." "Dr." "No." "e.g." are not sentence ends.
SENT_SPLIT_RE = re.compile(
    r"(?<!\bRs\.)(?<!\bDr\.)(?<!\bMr\.)(?<!\bMs\.)(?<!\bNo\.)(?<!\bSt\.)(?<!\be\.g\.)(?<!\bi\.e\.)(?<!\bvs\.)"
    r"(?<=[.!?\]])\s+(?=[A-Z\"'(*\[-])"
)


# ----------------------------------------------------------------- data types
@dataclass
class Citation:
    sid: int
    source: str
    page: int
    line_start: int
    line_end: int
    highlight_lines: list[int]
    similarity: float
    lines: list[str]


@dataclass
class Answer:
    question: str
    answer: str
    status: str                      # answered | refused | no_context | unverified | error
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    raw_output: str = ""


# ----------------------------------------------------------------- helpers
def tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 1}


def split_sentences(text: str) -> list[str]:
    parts: list[str] = []
    for block in re.split(r"\n+", text.strip()):
        for s in SENT_SPLIT_RE.split(block.strip()):
            s = s.strip(" -*•\t")
            if not s:
                continue
            # a fragment that is only citations belongs to the previous sentence
            if parts and not CITE_RE.sub("", s).strip(" .;,"):
                parts[-1] += " " + s
            else:
                parts.append(s)
    return parts


def format_context(hits: list[Retrieved]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        numbered = "\n".join(f"L{c.line_start + j}: {t}" for j, t in enumerate(c.lines))
        blocks.append(f"[S{i}] {c.source} | page {c.page} | lines {c.line_start}-{c.line_end}\n{numbered}")
    return "\n\n".join(blocks)


def pinpoint_lines(sentence: str, hit: Retrieved) -> list[int]:
    """Line numbers inside the chunk that best support `sentence`."""
    st = tokens(CITE_RE.sub("", sentence))
    scored = []
    for j, line in enumerate(hit.chunk.lines):
        lt = tokens(line)
        if not lt:
            continue
        overlap = len(st & lt)
        scored.append((overlap / len(lt), overlap, hit.chunk.line_start + j))
    if not scored:
        return []
    best_overlap = max(ov for _, ov, _ in scored)
    if best_overlap == 0:
        return []
    # keep the strongest line(s): lines whose overlap is close to the best one
    cutoff = max(1, 0.6 * best_overlap)
    return sorted(ln for ratio, ov, ln in scored
                  if ov >= cutoff and (ov >= 2 or ratio >= 0.5))


# ----------------------------------------------------------------- pipeline
class RAGPipeline:
    def __init__(self, store: VectorStore | None = None, llm: LLM | None = None,
                 top_k: int = config.TOP_K, min_similarity: float = config.MIN_SIMILARITY,
                 log_path=config.QUERY_LOG):
        self.store = store or VectorStore()
        self.llm = llm or LLM()
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.log_path = log_path

    def ask(self, question: str, log: bool = True) -> Answer:
        t0 = time.perf_counter()
        hits = self.store.query(question, k=self.top_k)
        t_retrieval = time.perf_counter() - t0

        evidence = [h for h in hits if h.similarity >= self.min_similarity]
        retrieved = [{"source": h.chunk.source, "page": h.chunk.page,
                      "line_start": h.chunk.line_start, "line_end": h.chunk.line_end,
                      "similarity": h.similarity} for h in hits]
        metrics = {
            "retrieval_ms": round(t_retrieval * 1000, 1),
            "generation_ms": 0.0,
            "top_similarity": hits[0].similarity if hits else 0.0,
            "n_retrieved": len(hits),
            "n_evidence": len(evidence),
            "model": self.llm.model,
        }

        if not evidence:
            ans = Answer(question, REFUSAL, "no_context", retrieved=retrieved, metrics=metrics)
            return self._finish(ans, t0, log)

        user_prompt = f"SOURCES:\n\n{format_context(evidence)}\n\nQUESTION: {question}\n\nANSWER (with [S#] citations):"
        t1 = time.perf_counter()
        try:
            res = self.llm.generate(SYSTEM_PROMPT, user_prompt)
        except Exception as e:  # LLM down, model missing, ...
            metrics["error"] = str(e)
            ans = Answer(question, f"LLM error: {e}", "error", retrieved=retrieved, metrics=metrics)
            return self._finish(ans, t0, log)
        metrics["generation_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        metrics["prompt_tokens"] = res.prompt_tokens
        metrics["output_tokens"] = res.output_tokens

        ans = self.verify(question, res.text, evidence)
        ans.retrieved = retrieved
        ans.metrics.update(metrics)
        return self._finish(ans, t0, log)

    # ------------------------------------------------------------ verification
    def verify(self, question: str, raw: str, evidence: list[Retrieved]) -> Answer:
        text = raw.strip()
        if not text or "NOT_FOUND" in text.upper().replace(" ", "_")[:40]:
            return Answer(question, REFUSAL, "refused", raw_output=raw,
                          metrics={"grounding": 0.0, "n_citations": 0})

        sentences = split_sentences(text)
        n = len(evidence)
        cited: dict[int, set[int]] = {}
        invalid, uncited, grounding_scores, unsupported_numbers = 0, 0, [], []

        for s in sentences:
            ids = [int(x) for x in CITE_RE.findall(s)]
            valid = [i for i in ids if 1 <= i <= n]
            invalid += len(ids) - len(valid)
            if not valid:
                uncited += 1
                grounding_scores.append(0.0)
                continue
            support_text = " ".join(evidence[i - 1].chunk.text for i in valid)
            st = tokens(CITE_RE.sub("", s))
            grounding_scores.append(len(st & tokens(support_text)) / len(st) if st else 1.0)
            for num in NUM_RE.findall(CITE_RE.sub("", s)):
                if num not in support_text:
                    unsupported_numbers.append(num)
            for i in valid:
                cited.setdefault(i, set()).update(pinpoint_lines(s, evidence[i - 1]))

        metrics = {
            "n_sentences": len(sentences),
            "n_citations": len(cited),
            "invalid_citations": invalid,
            "uncited_sentences": uncited,
            "unsupported_numbers": unsupported_numbers,
            "grounding": round(sum(grounding_scores) / len(grounding_scores), 3) if grounding_scores else 0.0,
        }

        if not cited:
            return Answer(question, REFUSAL, "unverified", raw_output=raw, metrics=metrics)

        citations = []
        for i in sorted(cited):
            h = evidence[i - 1]
            c = h.chunk
            hl = sorted(cited[i]) or list(range(c.line_start, c.line_end + 1))
            citations.append(Citation(i, c.source, c.page, c.line_start, c.line_end, hl, h.similarity, c.lines))
        return Answer(question, text, "answered", citations=citations, metrics=metrics, raw_output=raw)

    # ------------------------------------------------------------ logging
    def _finish(self, ans: Answer, t0: float, log: bool) -> Answer:
        ans.metrics["total_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        if log and self.log_path:
            rec = {
                "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "question": ans.question,
                "status": ans.status,
                "citations": [f"{c.source} p{c.page} L{c.highlight_lines[0]}-{c.highlight_lines[-1]}"
                              for c in ans.citations if c.highlight_lines],
                **{k: v for k, v in ans.metrics.items() if not isinstance(v, list)},
                "unsupported_numbers": len(ans.metrics.get("unsupported_numbers", [])),
            }
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        return ans


def answer_to_dict(ans: Answer) -> dict:
    return asdict(ans)
