"""Damaged / hostile files must be rejected cleanly and never break the app or the index."""
import json
import sys
from pathlib import Path

import pymupdf
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from rag.auth import UserDBError, UserStore  # noqa: E402
from rag.evaluation import load_eval_results, load_query_log  # noqa: E402
from rag.indexer import index_folder_report, safe_add_pdf  # noqa: E402
from rag.security import read_audit  # noqa: E402
from rag.store import VectorStore  # noqa: E402
from test_pipeline import HashEmbedder  # noqa: E402


def make_pdf(*pages: str) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def store(tmp_path):
    return VectorStore(path=tmp_path / "db", embedder=HashEmbedder())


DAMAGED = {
    "truncated": lambda good: good[: len(good) // 3],
    "garbage after header": lambda good: b"%PDF-1.7\n" + b"\x00\xff garbage " * 200,
    "html renamed to pdf": lambda good: b"<html><script>alert(1)</script></html>",
    "empty file": lambda good: b"",
    "zip renamed to pdf": lambda good: b"PK\x03\x04" + b"\x00" * 100,
}


@pytest.mark.parametrize("kind", DAMAGED)
def test_damaged_upload_changes_nothing(store, tmp_path, kind):
    docs = tmp_path / "docs"
    good = make_pdf("The library closes at 8 PM.")
    assert safe_add_pdf(store, good, "rules.pdf", docs_dir=docs).ok
    before_files, before_count = sorted(p.name for p in docs.iterdir()), store.count()

    bad = DAMAGED[kind](make_pdf("Some other text."))
    r = safe_add_pdf(store, bad, "broken.pdf", docs_dir=docs)
    assert not r.ok and r.reason
    assert sorted(p.name for p in docs.iterdir()) == before_files  # file NOT saved
    assert store.count() == before_count                            # index untouched
    assert "broken.pdf" not in store.sources()


def test_damaged_replacement_keeps_old_version(store, tmp_path):
    docs = tmp_path / "docs"
    assert safe_add_pdf(store, make_pdf("Version one: fee is Rs. 100."), "fees.pdf", docs_dir=docs).ok
    r = safe_add_pdf(store, make_pdf("x")[:50], "fees.pdf", docs_dir=docs)
    assert not r.ok
    assert "Version one" in store.query("fee", k=1)[0].chunk.text
    assert pymupdf.open(docs / "fees.pdf").page_count == 1  # old file still intact


def test_scanned_pdf_without_text_is_not_added(store, tmp_path):
    doc = pymupdf.open()
    doc.new_page()  # blank page, no text layer
    r = safe_add_pdf(store, doc.tobytes(), "scan.pdf", docs_dir=tmp_path / "docs")
    assert not r.ok and "no selectable text" in r.reason


def test_indexing_failure_rolls_back(tmp_path):
    class FailingEmbedder(HashEmbedder):
        def __call__(self, texts):
            raise RuntimeError("embedding model crashed")

    s = VectorStore(path=tmp_path / "db", embedder=FailingEmbedder())
    docs = tmp_path / "docs"
    r = safe_add_pdf(s, make_pdf("Hello world."), "a.pdf", docs_dir=docs)
    assert not r.ok and s.count() == 0
    assert not (docs / "a.pdf").exists()


def test_folder_reindex_quarantines_damaged_files(store, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "good.pdf").write_bytes(make_pdf("Meetings are on Fridays."))
    (docs / "bad.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
    results = {r.name: r for r in index_folder_report(store, docs)}
    assert results["good.pdf"].ok and results["good.pdf"].passages > 0
    assert not results["bad.pdf"].ok
    assert not (docs / "bad.pdf").exists()
    assert (docs / "_rejected" / "bad.pdf").exists()
    assert (docs / "_rejected" / "bad.pdf.reason.txt").exists()
    assert list(store.sources()) == ["good.pdf"]


def test_friendly_usernames(tmp_path):
    us = UserStore(tmp_path / "u.json")
    us.add_user("  Suman   Bhat ", "Secret123", "admin")
    assert us.verify("suman bhat", "Secret123")[0] == "admin"
    assert us.verify("SUMAN BHAT", "Secret123")[0] == "admin"
    assert us.display_name("suman bhat") == "Suman Bhat"


def test_damaged_user_file_is_detected_and_recoverable(tmp_path):
    path = tmp_path / "users.json"
    path.write_text('{"suman": {"role": "admin", "salt": "ab"', encoding="utf-8")  # cut off mid-write
    us = UserStore(path)
    assert us.is_damaged()
    with pytest.raises(UserDBError):
        us.verify("suman", "x")
    backup = us.backup_damaged()
    assert backup.exists() and not path.exists()
    us.add_user("Suman Bhat", "Secret123", "admin")
    assert us.verify("Suman Bhat", "Secret123")[0] == "admin"


def test_damaged_log_lines_are_skipped(tmp_path):
    log = tmp_path / "q.jsonl"
    log.write_text('{"ts": "2026-11-05T09:00:00+00:00", "status": "answered", "total_ms": 900}\n'
                   '{"ts": "2026-11-05T09:01:00+00:00", "sta\n'
                   'not json at all\n'
                   '{"ts": "2026-11-05T09:02:00+00:00", "status": "refused", "total_ms": "oops"}\n',
                   encoding="utf-8")
    df = load_query_log(log)
    assert len(df) == 2 and df.total_ms.isna().sum() == 1

    audit = tmp_path / "a.jsonl"
    audit.write_text('{"ts": "x", "event": "login_success", "user": "a"}\n{broken\n', encoding="utf-8")
    assert len(read_audit(audit)) == 1


def test_damaged_eval_results_ignored(tmp_path):
    p = tmp_path / "eval.json"
    p.write_text('{"summary": {', encoding="utf-8")
    assert load_eval_results(p) is None
    p.write_text(json.dumps({"summary": {"model": "m"}, "rows": [{}]}), encoding="utf-8")
    assert load_eval_results(p) is not None
