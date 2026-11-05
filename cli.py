"""Command line interface.

    python cli.py ingest [folder]      index every PDF in docs/ (or a folder)
    python cli.py ask "question"       ask one question
    python cli.py eval                 run the evaluation set
    python cli.py add-user NAME [--admin]   create an account
    python cli.py reset-password NAME       set a new password (also unlocks the account)
"""
import argparse
import getpass
import json

import config
from rag.evaluation import run_eval
from rag.indexer import index_folder
from rag.pipeline import RAGPipeline
from rag.store import VectorStore


def main() -> None:
    p = argparse.ArgumentParser(description="Grounded RAG assistant")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest")
    i.add_argument("folder", nargs="?", default=str(config.DOCS_DIR))
    a = sub.add_parser("ask")
    a.add_argument("question")
    sub.add_parser("eval")
    u = sub.add_parser("add-user")
    u.add_argument("name")
    u.add_argument("--admin", action="store_true")
    r = sub.add_parser("reset-password")
    r.add_argument("name")
    args = p.parse_args()

    if args.cmd in ("add-user", "reset-password"):
        from rag.auth import AuthError, UserStore
        from rag.security import audit
        users = UserStore()
        pw = getpass.getpass("New password: ")
        if pw != getpass.getpass("Confirm password: "):
            raise SystemExit("Passwords don't match.")
        try:
            if args.cmd == "add-user":
                users.add_user(args.name, pw, "admin" if args.admin else "viewer")
                audit("user_created", "cli", target=args.name)
            else:
                users.set_password(args.name.lower(), pw)
                audit("password_reset", "cli", target=args.name)
        except AuthError as e:
            raise SystemExit(str(e))
        print("Done.")
        return

    store = VectorStore()
    if args.cmd == "ingest":
        for name, n in index_folder(store, args.folder).items():
            print(f"{name}: {n} chunks" + ("" if n else "  (no text - scanned PDF?)"))
        print(f"Total chunks in index: {store.count()}")
    elif args.cmd == "ask":
        ans = RAGPipeline(store=store).ask(args.question)
        print(ans.answer, "\n")
        for c in ans.citations:
            hl = c.highlight_lines
            print(f"  [S{c.sid}] {c.source}  page {c.page}  lines {hl[0]}-{hl[-1]}")
        print(f"\nstatus={ans.status}  total={ans.metrics['total_ms']:.0f} ms  "
              f"grounding={ans.metrics.get('grounding', 0):.2f}")
    elif args.cmd == "eval":
        res = run_eval(RAGPipeline(store=store))
        print(json.dumps(res["summary"], indent=2))


if __name__ == "__main__":
    main()
