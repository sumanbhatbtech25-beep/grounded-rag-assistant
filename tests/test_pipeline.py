"""Offline tests: no Ollama and no model downloads needed.

    pytest -q
"""
import hashlib
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.indexer import index_folder  # noqa: E402
from rag.ingest import chunk_lines, extract_lines, Line  # noqa: E402
from rag.highlight import render_highlighted_page  # noqa: E402
from rag.llm import LLMResult  # noqa: E402
from rag.pipeline import RAGPipeline, REFUSAL, split_sentences  # noqa: E402
from rag.store import VectorStore  # noqa: E402

DOCS = ROOT / "docs"


class HashEmbedder:
    """Bag-of-words hashing embedder: deterministic and offline."""
    name = "test-hash"

    def __call__(self, texts):
        out = []
        for t in texts:
            v = [0.0] * 256
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 256] += 1
            n = math.sqrt(sum(x * x for x in v)) or 1
            out.append([x / n for x in v])
        return out


class FakeLLM:
    model = "fake"

    def __init__(self, reply):
        self.reply = reply

    def generate(self, system, user):
        self.last_prompt = user
        return LLMResult(self.reply(user) if callable(self.reply) else self.reply)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    if not (DOCS / "sample_club_rulebook.pdf").exists():
        import runpy
        runpy.run_path(str(ROOT / "scripts" / "make_sample_docs.py"), run_name="__main__")
    s = VectorStore(path=tmp_path_factory.mktemp("chroma"), embedder=HashEmbedder())
    index_folder(s, DOCS)
    return s


def test_line_numbers_are_per_page_and_contiguous():
    lines = extract_lines(DOCS / "sample_club_rulebook.pdf")
    for page in {l.page for l in lines}:
        nums = [l.line_no for l in lines if l.page == page]
        assert nums == list(range(1, len(nums) + 1))
    fee = next(l for l in lines if "Rs. 500" in l.text)
    assert fee.page == 1


def test_chunks_never_cross_pages():
    lines = [Line(f"p{p} l{i}", p, i, (0, 0, 1, 1)) for p in (1, 2) for i in range(1, 12)]
    chunks = chunk_lines(lines, "x.pdf", size=5, overlap=2)
    assert all(c.lines[0].startswith(f"p{c.page} ") and c.lines[-1].startswith(f"p{c.page} ") for c in chunks)
    assert chunks[0].line_start == 1 and chunks[0].line_end == 5
    assert chunks[1].line_start == 4  # overlap of 2
    assert max(c.line_end for c in chunks if c.page == 1) == 11


def test_split_sentences_keeps_trailing_citations():
    s = split_sentences("The fee is Rs. 500. [S1] Meetings are monthly [S2].\n- Bullet point [S1]")
    assert len(s) == 3 and s[0].endswith("[S1]")


def test_retrieval_finds_right_page(store):
    hits = store.query("annual membership fee", k=3)
    assert hits[0].chunk.source == "sample_club_rulebook.pdf" and hits[0].chunk.page == 1


def test_answer_is_cited_and_pinpointed(store, tmp_path):
    llm = FakeLLM("The annual membership fee is Rs. 500, payable before 31 August. [S1]")
    p = RAGPipeline(store=store, llm=llm, min_similarity=0.0, log_path=tmp_path / "log.jsonl")
    ans = p.ask("What is the annual membership fee?")
    assert ans.status == "answered"
    c = ans.citations[0]
    assert (c.source, c.page) == ("sample_club_rulebook.pdf", 1)
    lines = extract_lines(DOCS / c.source)
    fee_line = next(l.line_no for l in lines if l.page == 1 and "Rs. 500" in l.text)
    assert fee_line in c.highlight_lines
    assert ans.metrics["grounding"] > 0.8
    assert (tmp_path / "log.jsonl").read_text().count("\n") == 1


def test_uncited_answer_is_withheld(store, tmp_path):
    p = RAGPipeline(store=store, llm=FakeLLM("The fee is probably around Rs. 1000."),
                    min_similarity=0.0, log_path=None)
    ans = p.ask("What is the annual membership fee?")
    assert ans.status == "unverified" and ans.answer == REFUSAL


def test_not_found_and_invalid_citation(store):
    p = RAGPipeline(store=store, llm=FakeLLM("NOT_FOUND"), min_similarity=0.0, log_path=None)
    assert p.ask("Who won the 2018 World Cup?").status == "refused"
    p = RAGPipeline(store=store, llm=FakeLLM("Something made up. [S99]"), min_similarity=0.0, log_path=None)
    ans = p.ask("annual fee")
    assert ans.status == "unverified" and ans.metrics["invalid_citations"] == 1


def test_relevance_gate_skips_llm(store):
    llm = FakeLLM(lambda u: pytest.fail("LLM should not be called"))
    p = RAGPipeline(store=store, llm=llm, min_similarity=0.99, log_path=None)
    assert p.ask("zzzz qqqq").status == "no_context"


def test_unsupported_numbers_flagged(store):
    p = RAGPipeline(store=store, llm=FakeLLM("The membership fee is Rs. 900. [S1]"),
                    min_similarity=0.0, log_path=None)
    ans = p.ask("What is the annual membership fee?")
    assert "900" in ans.metrics["unsupported_numbers"]


def test_highlight_renders_png():
    png = render_highlighted_page(DOCS / "sample_club_rulebook.pdf", 1, [4])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
