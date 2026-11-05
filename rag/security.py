"""Upload validation, prompt-injection detection, PII redaction and audit logging."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pymupdf

import config

# ---------------------------------------------------------------- uploads
SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")


def safe_filename(name: str) -> str:
    """Strip any path, replace odd characters, force a .pdf extension."""
    base = Path(name.replace("\\", "/")).name
    stem = SAFE_NAME_RE.sub("_", Path(base).stem).strip(" ._") or "document"
    return f"{stem[:100]}.pdf"


def validate_pdf(data: bytes, name: str, max_mb: int = config.MAX_UPLOAD_MB,
                 max_pages: int = config.MAX_PAGES) -> tuple[bool, str, str]:
    """Returns (ok, reason, safe_name)."""
    safe = safe_filename(name)
    if len(data) > max_mb * 1024 * 1024:
        return False, f"larger than {max_mb} MB", safe
    if not data[:1024].lstrip().startswith(b"%PDF-"):
        return False, "not a real PDF file", safe
    try:
        with pymupdf.open(stream=data, filetype="pdf") as doc:
            if doc.needs_pass or doc.is_encrypted:
                return False, "password-protected PDFs are not supported", safe
            if doc.page_count == 0:
                return False, "the PDF has no pages", safe
            if doc.page_count > max_pages:
                return False, f"more than {max_pages} pages", safe
    except Exception:
        return False, "the PDF is damaged or unreadable", safe
    return True, "ok", safe


# ---------------------------------------------------------------- prompt injection
# Text inside a document that tries to give the model instructions.
INJECTION_PATTERNS = [
    r"\b(ignore|disregard|forget|override)\b.{0,40}\b(previous|prior|above|earlier|all|system)\b.{0,20}"
    r"\b(instructions?|prompts?|rules?|messages?)\b",
    r"\byou are now (an?|the|in)\b",
    r"\b(new|updated) (system )?instructions? (for|to) (the |this )?(ai|assistant|model|llm|chatbot)\b",
    r"\bsystem prompt\b",
    r"\b(reveal|print|show|repeat)\b.{0,30}\b(system prompt|your instructions|password|api key|secret key)\b",
    r"\bdo not (cite|reveal) (the |any )?(sources?|documents?|citations?)\b",
    r"\b(ai|assistant|model|chatbot|llm)\b.{0,30}\b(must|should) (now )?(ignore|answer|say|reply)\b",
    r"\byou (must|should|will) (now )?(act|pretend|behave) as\b",
    r"<\|?(im_start|im_end|system)\|?>",
    r"\[/?INST\]",
    r"^\s*#{2,}\s*(system|instruction)",
]
INJECTION_RE = re.compile("|".join(f"(?:{p})" for p in INJECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)


def detect_injection(text: str) -> list[str]:
    return [m.group(0)[:80] for m in INJECTION_RE.finditer(text)]


# ---------------------------------------------------------------- PII redaction
PII_PATTERNS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "[email]"),
    (re.compile(r"\b(?:\d[ -]?){13,16}\b"), "[card-number]"),
    (re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"), "[id-number]"),
    (re.compile(r"(?:\+?91[ -]?)?\b[6-9]\d{9}\b"), "[phone]"),
]


def redact(text: str) -> str:
    for rx, repl in PII_PATTERNS:
        text = rx.sub(repl, text)
    return text


# ---------------------------------------------------------------- audit log
def audit(event: str, user: str | None = None, path: Path = config.AUDIT_LOG, **details) -> None:
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, "user": user or "-"}
    rec.update({k: (redact(v) if isinstance(v, str) else v) for k, v in details.items()})
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass


def read_audit(path: Path = config.AUDIT_LOG, limit: int = 300) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    for line in reversed(Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]):
        try:
            rec = json.loads(line)
            if isinstance(rec, dict) and "event" in rec and "ts" in rec:
                out.append(rec)
        except ValueError:
            continue
    return out
