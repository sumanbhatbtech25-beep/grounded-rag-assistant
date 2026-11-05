"""Local user accounts: salted PBKDF2-SHA256 password hashes, roles, lockout.

Stored in data/users.json (git-ignored). Plain-text passwords are never written anywhere.
Roles:
  admin  - ask, upload/remove documents, run evaluation, manage users, view audit log
  viewer - ask questions and view the dashboard only
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import time
from pathlib import Path

import config

ITERATIONS = 240_000
ROLES = ("admin", "viewer")
USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9 ._-]{2,31}$")
USERNAME_RULES = "3-32 characters: letters, numbers, spaces, . _ -"


def normalize_username(name: str) -> str:
    """'  Suman   Bhat ' -> 'Suman Bhat' (display form). Matching is case-insensitive."""
    return " ".join((name or "").split())


def user_key(name: str) -> str:
    return normalize_username(name).lower()


def hash_password(password: str, salt: bytes | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS)
    return salt.hex(), dk.hex()


def password_problems(password: str) -> list[str]:
    problems = []
    if len(password) < 8:
        problems.append("at least 8 characters")
    if not re.search(r"[A-Za-z]", password):
        problems.append("a letter")
    if not re.search(r"\d", password):
        problems.append("a number")
    return problems


class AuthError(ValueError):
    pass


class UserDBError(RuntimeError):
    """users.json exists but can't be read (damaged / edited by hand)."""


class UserStore:
    def __init__(self, path: Path = config.USERS_FILE,
                 max_attempts: int = config.LOGIN_MAX_ATTEMPTS, lockout_minutes: int = config.LOCKOUT_MINUTES):
        self.path = Path(path)
        self.max_attempts = max_attempts
        self.lockout_seconds = lockout_minutes * 60

    # ---------------------------------------------------------------- storage
    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, dict) or not all(
                    isinstance(u, dict) and {"role", "salt", "hash"} <= u.keys() for u in data.values()):
                raise ValueError("unexpected structure")
            return data
        except (OSError, ValueError) as e:
            raise UserDBError(f"{self.path.name} is damaged ({type(e).__name__})") from e

    def is_damaged(self) -> bool:
        try:
            self._load()
            return False
        except UserDBError:
            return True

    def backup_damaged(self) -> Path | None:
        """Move a damaged users.json aside (kept for inspection) so accounts can be recreated."""
        if not self.path.exists() or not self.is_damaged():
            return None
        dest = self.path.with_name(f"users.damaged-{time.strftime('%Y%m%d-%H%M%S')}.json")
        os.replace(self.path, dest)
        return dest

    def _save(self, users: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(users, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)  # atomic

    # ---------------------------------------------------------------- queries
    def has_users(self) -> bool:
        return bool(self._load())

    def list_users(self) -> list[dict]:
        out = []
        for name, u in sorted(self._load().items()):
            out.append({"username": u.get("display", name), "key": name, "role": u["role"],
                        "created": u.get("created"),
                        "last_login": u.get("last_login"),
                        "locked": u.get("locked_until", 0) > time.time()})
        return out

    def role_of(self, username: str) -> str | None:
        u = self._load().get(user_key(username))
        return u["role"] if u else None

    def display_name(self, username: str) -> str:
        u = self._load().get(user_key(username))
        return u.get("display", username) if u else username

    # ---------------------------------------------------------------- changes
    def add_user(self, username: str, password: str, role: str = "viewer") -> None:
        display = normalize_username(username)
        username = display.lower()
        if not USERNAME_RE.match(username):
            raise AuthError(f"Username must be {USERNAME_RULES}.")
        if role not in ROLES:
            raise AuthError(f"Role must be one of {ROLES}")
        if problems := password_problems(password):
            raise AuthError("Password needs " + ", ".join(problems))
        users = self._load()
        if username in users:
            raise AuthError("That username already exists")
        salt, h = hash_password(password)
        users[username] = {"display": display, "role": role, "salt": salt, "hash": h,
                           "created": time.strftime("%Y-%m-%d %H:%M"), "failed": 0, "locked_until": 0}
        self._save(users)

    def set_password(self, username: str, new_password: str) -> None:
        username = user_key(username)
        if problems := password_problems(new_password):
            raise AuthError("Password needs " + ", ".join(problems))
        users = self._load()
        if username not in users:
            raise AuthError("Unknown user")
        users[username]["salt"], users[username]["hash"] = hash_password(new_password)
        users[username]["failed"], users[username]["locked_until"] = 0, 0
        self._save(users)

    def delete_user(self, username: str) -> None:
        username = user_key(username)
        users = self._load()
        if username not in users:
            raise AuthError("Unknown user")
        admins = [n for n, u in users.items() if u["role"] == "admin"]
        if users[username]["role"] == "admin" and len(admins) == 1:
            raise AuthError("You can't delete the last admin")
        del users[username]
        self._save(users)

    def unlock(self, username: str) -> None:
        username = user_key(username)
        users = self._load()
        if username in users:
            users[username]["failed"], users[username]["locked_until"] = 0, 0
            self._save(users)

    # ---------------------------------------------------------------- login
    def verify(self, username: str, password: str) -> tuple[str | None, str]:
        """Returns (role, message). role is None when login fails."""
        username = user_key(username)
        users = self._load()
        u = users.get(username)
        if u is None:
            hash_password(password, b"\0" * 16)  # same work as a real check: no user enumeration by timing
            return None, "Wrong username or password."

        now = time.time()
        if u.get("locked_until", 0) > now:
            mins = int((u["locked_until"] - now) // 60) + 1
            return None, f"Too many failed attempts. Try again in {mins} min."

        _, h = hash_password(password, bytes.fromhex(u["salt"]))
        if hmac.compare_digest(h, u["hash"]):
            u["failed"], u["locked_until"] = 0, 0
            u["last_login"] = time.strftime("%Y-%m-%d %H:%M")
            self._save(users)
            return u["role"], "ok"

        u["failed"] = u.get("failed", 0) + 1
        if u["failed"] >= self.max_attempts:
            u["locked_until"], u["failed"] = now + self.lockout_seconds, 0
            self._save(users)
            return None, f"Too many failed attempts. Account locked for {self.lockout_seconds // 60} min."
        self._save(users)
        left = self.max_attempts - u["failed"]
        return None, f"Wrong username or password. {left} attempt(s) left before lockout."
