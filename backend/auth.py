"""Accounts for the dashboard: email + password sign-up (with an email OTP step) / login and "Sign in
with Google".

Small on purpose: users, sessions and pending sign-ups live in one SQLite file (env AUTH_DB, default
backend/data/auth.db), passwords are hashed with scrypt, and a session is a random bearer token of which
only the SHA-256 is stored. Google sign-in: the browser gets an ID token from Google, and this module
checks it with Google's tokeninfo endpoint (signature, expiry) and that it was issued for OUR client id
(env GOOGLE_CLIENT_ID) and a verified email.

Email sign-up is two calls: POST /auth/signup validates the form, hashes the password, stores a pending
row keyed by email and emails a 6-digit code (send_otp_email, SMTP via env vars below); POST
/auth/verify-signup takes that code and only then creates the account and starts a session. A pending
row that is never verified expires (OTP_TTL_SECONDS) and is pruned lazily; wrong codes are capped
(OTP_MAX_ATTEMPTS) and a new one is rate-limited (OTP_RESEND_COOLDOWN_SECONDS).

The model endpoints (/analyze, /preview, /report) require a signed-in user through the `require_user`
dependency (401 otherwise); env REQUIRE_SIGN_IN=0 switches that off for local development. /health, /auth/* and
the static /reports files stay open (report folders are unguessable uuid4 names, and <img>/<a> links cannot send
a bearer header).
"""
import hashlib
import hmac
import json
import os
import re
import secrets
import smtplib
import sqlite3
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import closing
from email.message import EmailMessage
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, status
from pydantic import BaseModel

router = APIRouter(prefix="/auth", tags=["auth"])

SESSION_DAYS = 14
MIN_PASSWORD = 8
MAX_PASSWORD = 128
TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_GOOGLE_ISSUERS = {"accounts.google.com", "https://accounts.google.com"}

# Guesses, not validated against real abuse traffic.
OTP_TTL_SECONDS = 10 * 60
OTP_MAX_ATTEMPTS = 5
OTP_RESEND_COOLDOWN_SECONDS = 30
# Password guessing: after LOGIN_MAX_FAILURES wrong logins for one email within LOGIN_WINDOW_SECONDS, that email
# is refused (429) until the window ends. Counted per email (known or not, so it reveals nothing); a right
# password clears the count.
LOGIN_MAX_FAILURES = 5
LOGIN_WINDOW_SECONDS = 15 * 60


def google_client_id() -> str:
    return os.environ.get("GOOGLE_CLIENT_ID", "").strip()


def _db_path() -> Path:
    configured = os.environ.get("AUTH_DB", "").strip()
    return Path(configured) if configured else Path(__file__).parent / "data" / "auth.db"


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        "CREATE TABLE IF NOT EXISTS users ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT NOT NULL UNIQUE, name TEXT NOT NULL,"
        " password_hash TEXT, google_sub TEXT, created_at REAL NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS sessions ("
        " token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL, expires_at REAL NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS pending_signups ("
        " email TEXT PRIMARY KEY, name TEXT NOT NULL, password_hash TEXT NOT NULL,"
        " otp_hash TEXT NOT NULL, expires_at REAL NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,"
        " last_sent_at REAL NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS reports ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL, session_id TEXT NOT NULL UNIQUE,"
        " query TEXT NOT NULL, task TEXT, created_at REAL NOT NULL, title TEXT)"
    )
    _ensure_column(conn, "reports", "title", "TEXT")   # added after the table already shipped; migrates in place
    conn.execute(
        "CREATE TABLE IF NOT EXISTS login_failures ("
        " email TEXT PRIMARY KEY, failures INTEGER NOT NULL, first_at REAL NOT NULL)"
    )
    return conn


def _login_locked(conn: sqlite3.Connection, email: str) -> Optional[int]:
    """Seconds until this email may try again, or None if it is not locked."""
    row = conn.execute("SELECT failures, first_at FROM login_failures WHERE email = ?", (email,)).fetchone()
    if row is None:
        return None
    remaining = row["first_at"] + LOGIN_WINDOW_SECONDS - time.time()
    if remaining <= 0:
        conn.execute("DELETE FROM login_failures WHERE email = ?", (email,))
        conn.commit()
        return None
    return int(remaining) + 1 if row["failures"] >= LOGIN_MAX_FAILURES else None


def _record_login_failure(conn: sqlite3.Connection, email: str) -> None:
    conn.execute(
        "INSERT INTO login_failures (email, failures, first_at) VALUES (?, 1, ?)"
        " ON CONFLICT(email) DO UPDATE SET failures = failures + 1",
        (email, time.time()),
    )
    conn.commit()


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, decl: str) -> None:
    """ALTER TABLE ... ADD COLUMN, but only if it's not already there (SQLite has no IF NOT EXISTS for
    columns). Safe to call every connection; a no-op once the column exists."""
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


# ------------------------------------------------------------------ passwords and tokens (pure)

def hash_password(password: str, salt: Optional[bytes] = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=32)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: Optional[str]) -> bool:
    """Constant-time check; a missing hash still costs one scrypt so timing does not reveal which emails exist."""
    if not stored or not stored.startswith("scrypt$"):
        hash_password(password)
        return False
    try:
        _, salt_hex, digest_hex = stored.split("$")
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.scrypt(password.encode("utf-8"), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1, dklen=len(expected))
    except ValueError:
        return False
    return hmac.compare_digest(actual, expected)


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def normalize_email(email: str) -> str:
    return email.strip().lower()


# ------------------------------------------------------------------ Google

def verify_google_credential(credential: str) -> dict:
    """Ask Google to validate the ID token. Returns its claims, or raises HTTPException. Replaced in tests."""
    client_id = google_client_id()
    if not client_id:
        raise HTTPException(status_code=503, detail="Google sign-in is not set up on this server (GOOGLE_CLIENT_ID is empty).")
    url = f"{TOKENINFO_URL}?{urllib.parse.urlencode({'id_token': credential})}"
    try:
        with urllib.request.urlopen(url, timeout=8) as response:
            claims = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError:
        raise HTTPException(status_code=401, detail="Google did not accept this sign-in. Try again.")
    except (urllib.error.URLError, TimeoutError, ValueError):
        raise HTTPException(status_code=502, detail="Could not reach Google to check the sign-in. Try again in a moment.")
    if claims.get("aud") != client_id or claims.get("iss") not in _GOOGLE_ISSUERS:
        raise HTTPException(status_code=401, detail="This Google sign-in was not issued for SatQuery-AI.")
    if str(claims.get("email_verified")).lower() != "true" or not claims.get("email"):
        raise HTTPException(status_code=401, detail="Your Google account's email address is not verified.")
    if float(claims.get("exp", 0)) < time.time():
        raise HTTPException(status_code=401, detail="This Google sign-in has expired. Try again.")
    return claims


# ------------------------------------------------------------------ email OTP

def _generate_otp() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def _otp_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _prune_pending(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM pending_signups WHERE expires_at < ?", (time.time(),))


def send_otp_email(to_email: str, name: str, code: str) -> None:
    """Sends the verification code over SMTP (env SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASSWORD/SMTP_FROM/
    SMTP_USE_TLS). Raises HTTPException if SMTP isn't configured or sending fails. Replaced in tests."""
    host = os.environ.get("SMTP_HOST", "").strip()
    if not host:
        raise HTTPException(status_code=503, detail="Email sign-up is not set up on this server (SMTP_HOST is empty).")
    port = int(os.environ.get("SMTP_PORT", "587").strip() or "587")
    user = os.environ.get("SMTP_USER", "").strip()
    password = os.environ.get("SMTP_PASSWORD", "")
    sender = os.environ.get("SMTP_FROM", "").strip() or user
    use_tls = os.environ.get("SMTP_USE_TLS", "1").strip().lower() not in ("0", "false", "no")

    message = EmailMessage()
    message["Subject"] = "Your SatQuery AI verification code"
    message["From"] = sender
    message["To"] = to_email
    message.set_content(
        f"Hi {name},\n\nYour SatQuery AI verification code is: {code}\n\n"
        f"It expires in {OTP_TTL_SECONDS // 60} minutes. If you did not request this, ignore this email."
    )
    try:
        with smtplib.SMTP(host, port, timeout=10) as smtp:
            if use_tls:
                smtp.starttls()
            if user:
                smtp.login(user, password)
            smtp.send_message(message)
    except (smtplib.SMTPException, OSError) as e:
        raise HTTPException(status_code=502, detail="Could not send the verification email. Try again in a moment.") from e


# ------------------------------------------------------------------ sessions

def _public(row: sqlite3.Row) -> dict:
    return {"id": row["id"], "email": row["email"], "name": row["name"], "google": bool(row["google_sub"])}


def _start_session(conn: sqlite3.Connection, user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    now = time.time()
    conn.execute("DELETE FROM sessions WHERE expires_at < ?", (now,))
    conn.execute("INSERT INTO sessions (token_hash, user_id, expires_at) VALUES (?, ?, ?)",
                 (_token_hash(token), user_id, now + SESSION_DAYS * 86400))
    conn.commit()
    return token


def _bearer(authorization: Optional[str]) -> Optional[str]:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip() or None
    return None


def _user_for_token(conn: sqlite3.Connection, token: Optional[str]) -> Optional[sqlite3.Row]:
    if not token:
        return None
    return conn.execute(
        "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token_hash = ? AND s.expires_at > ?",
        (_token_hash(token), time.time()),
    ).fetchone()


def sign_in_required() -> bool:
    """On unless env REQUIRE_SIGN_IN is 0/false/no/off (local development, the CLI without an account)."""
    return os.environ.get("REQUIRE_SIGN_IN", "1").strip().lower() not in {"0", "false", "no", "off"}


def require_user(authorization: Optional[str] = Header(default=None)) -> Optional[dict]:
    """FastAPI dependency for the model endpoints (/analyze, /preview, /report): the signed-in user
    ({id, email, name}), or 401. Returns None only when sign-in is switched off (REQUIRE_SIGN_IN=0)."""
    if not sign_in_required():
        return None
    with closing(_connect()) as conn:
        user = _user_for_token(conn, _bearer(authorization))
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Please sign in again: you are not signed in, or your session has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"id": user["id"], "email": user["email"], "name": user["name"]}


def user_owns_report(user_id: int, session_id: str) -> bool:
    with closing(_connect()) as conn:
        row = conn.execute("SELECT 1 FROM reports WHERE session_id = ? AND user_id = ?", (session_id, user_id)).fetchone()
    return row is not None


# ------------------------------------------------------------------ report history

def save_report_for_token(token: Optional[str], session_id: str, query: str, task: Optional[str]) -> bool:
    """Links a finished /analyze run to whoever is signed in, so it shows up in their account's report
    history. A no-op (returns False) when nobody is signed in: analysis works the same either way, this
    only adds to the account's history on top. Never raises: a DB hiccup must not break /analyze."""
    if not token:
        return False
    try:
        with closing(_connect()) as conn:
            user = _user_for_token(conn, token)
            if not user:
                return False
            conn.execute(
                "INSERT OR REPLACE INTO reports (user_id, session_id, query, task, created_at) VALUES (?, ?, ?, ?, ?)",
                (user["id"], session_id, query[:2000], task, time.time()),
            )
            conn.commit()
            return True
    except Exception as e:
        print(f"[{session_id}] Could not save report to account history: {e}")
        return False


MAX_REPORT_TITLE = 200


def list_reports_for_token(token: Optional[str], limit: int = 100) -> list:
    with closing(_connect()) as conn:
        user = _user_for_token(conn, token)
        if not user:
            raise HTTPException(status_code=401, detail="Not signed in.")
        rows = conn.execute(
            "SELECT session_id, query, task, title, created_at FROM reports WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user["id"], limit),
        ).fetchall()
        return [
            {
                "session_id": r["session_id"],
                # A custom name if one was set (rename_report_for_token), else the original query text.
                "title": r["title"] or r["query"],
                "query": r["query"],
                "task": r["task"],
                "created_at": r["created_at"],
                "visual_evidence_url": f"/reports/{r['session_id']}/evidence.png",
                "report_download_url": f"/reports/{r['session_id']}/report.pdf",
            }
            for r in rows
        ]


def rename_report_for_token(token: Optional[str], session_id: str, title: str) -> None:
    """Renames one of the caller's own saved reports. Raises HTTPException on any problem (not signed
    in, bad title, no such report in this account)."""
    title = title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="Enter a name for the report.")
    if len(title) > MAX_REPORT_TITLE:
        raise HTTPException(status_code=422, detail=f"Keep the name to {MAX_REPORT_TITLE} characters or fewer.")
    with closing(_connect()) as conn:
        user = _user_for_token(conn, token)
        if not user:
            raise HTTPException(status_code=401, detail="Not signed in.")
        row = conn.execute("SELECT user_id FROM reports WHERE session_id = ?", (session_id,)).fetchone()
        if not row or row["user_id"] != user["id"]:
            raise HTTPException(status_code=404, detail="No such report in your account.")
        conn.execute("UPDATE reports SET title = ? WHERE session_id = ?", (title, session_id))
        conn.commit()


# ------------------------------------------------------------------ endpoints

class SignupBody(BaseModel):
    name: str
    email: str
    password: str


class LoginBody(BaseModel):
    email: str
    password: str


class GoogleBody(BaseModel):
    credential: str


class VerifySignupBody(BaseModel):
    email: str
    code: str


class ResendOtpBody(BaseModel):
    email: str


class RenameReportBody(BaseModel):
    title: str


@router.get("/config", summary="What the sign-in page needs to know")
def auth_config():
    return {"google_client_id": google_client_id() or None, "min_password_length": MIN_PASSWORD}


@router.post("/signup", summary="Start creating an account: validates the form and emails a verification code")
def signup(body: SignupBody):
    name = body.name.strip()
    email = normalize_email(body.email)
    if not name or len(name) > 80:
        raise HTTPException(status_code=422, detail="Enter your name (up to 80 characters).")
    if not _EMAIL.match(email) or len(email) > 254:
        raise HTTPException(status_code=422, detail="Enter a valid email address.")
    if not MIN_PASSWORD <= len(body.password) <= MAX_PASSWORD:
        raise HTTPException(status_code=422, detail=f"The password must be {MIN_PASSWORD} to {MAX_PASSWORD} characters.")
    with closing(_connect()) as conn:
        existing = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            hint = " It was created with Google: use “Continue with Google”." if not existing["password_hash"] else " Log in instead."
            raise HTTPException(status_code=409, detail="An account with this email already exists." + hint)
        _prune_pending(conn)
        code = _generate_otp()
        now = time.time()
        conn.execute(
            "INSERT INTO pending_signups (email, name, password_hash, otp_hash, expires_at, attempts, last_sent_at)"
            " VALUES (?, ?, ?, ?, ?, 0, ?)"
            " ON CONFLICT(email) DO UPDATE SET name = excluded.name, password_hash = excluded.password_hash,"
            " otp_hash = excluded.otp_hash, expires_at = excluded.expires_at, attempts = 0, last_sent_at = excluded.last_sent_at",
            (email, name, hash_password(body.password), _otp_hash(code), now + OTP_TTL_SECONDS, now),
        )
        conn.commit()
        send_otp_email(email, name, code)
        return {"pending": True, "email": email, "expires_in": OTP_TTL_SECONDS}


@router.post("/verify-signup", status_code=status.HTTP_201_CREATED, summary="Finish sign-up with the emailed code")
def verify_signup(body: VerifySignupBody):
    email = normalize_email(body.email)
    code = body.code.strip()
    with closing(_connect()) as conn:
        _prune_pending(conn)
        pending = conn.execute("SELECT * FROM pending_signups WHERE email = ?", (email,)).fetchone()
        if not pending:
            raise HTTPException(status_code=404, detail="No pending sign-up for this email, or the code expired. Start again.")
        if pending["attempts"] >= OTP_MAX_ATTEMPTS:
            conn.execute("DELETE FROM pending_signups WHERE email = ?", (email,))
            conn.commit()
            raise HTTPException(status_code=429, detail="Too many wrong codes. Start again to get a new one.")
        if not hmac.compare_digest(_otp_hash(code), pending["otp_hash"]):
            conn.execute("UPDATE pending_signups SET attempts = attempts + 1 WHERE email = ?", (email,))
            conn.commit()
            raise HTTPException(status_code=401, detail="Incorrect code.")
        existing = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if existing:
            conn.execute("DELETE FROM pending_signups WHERE email = ?", (email,))
            conn.commit()
            raise HTTPException(status_code=409, detail="An account with this email already exists. Log in instead.")
        cursor = conn.execute(
            "INSERT INTO users (email, name, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (email, pending["name"], pending["password_hash"], time.time()),
        )
        conn.execute("DELETE FROM pending_signups WHERE email = ?", (email,))
        conn.commit()
        user = conn.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return {"token": _start_session(conn, user["id"]), "user": _public(user)}


@router.post("/resend-otp", summary="Send a new verification code for a pending sign-up")
def resend_otp(body: ResendOtpBody):
    email = normalize_email(body.email)
    with closing(_connect()) as conn:
        _prune_pending(conn)
        pending = conn.execute("SELECT * FROM pending_signups WHERE email = ?", (email,)).fetchone()
        if not pending:
            raise HTTPException(status_code=404, detail="No pending sign-up for this email. Start again.")
        wait = OTP_RESEND_COOLDOWN_SECONDS - (time.time() - pending["last_sent_at"])
        if wait > 0:
            raise HTTPException(status_code=429, detail=f"Wait {int(wait) + 1}s before requesting another code.")
        code = _generate_otp()
        now = time.time()
        conn.execute(
            "UPDATE pending_signups SET otp_hash = ?, expires_at = ?, attempts = 0, last_sent_at = ? WHERE email = ?",
            (_otp_hash(code), now + OTP_TTL_SECONDS, now, email),
        )
        conn.commit()
        send_otp_email(email, pending["name"], code)
        return {"pending": True, "email": email, "expires_in": OTP_TTL_SECONDS}


@router.post("/login", summary="Log in with an email and password")
def login(body: LoginBody):
    email = normalize_email(body.email)
    with closing(_connect()) as conn:
        wait = _login_locked(conn, email)
        if wait is not None:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many wrong attempts. Try again in {max(1, round(wait / 60))} minute(s).",
                headers={"Retry-After": str(wait)},
            )
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        ok = verify_password(body.password, user["password_hash"] if user else None)
        if not user or not ok:
            _record_login_failure(conn, email)
            if user and not user["password_hash"]:
                raise HTTPException(status_code=401, detail="This account uses Google. Use “Continue with Google”.")
            raise HTTPException(status_code=401, detail="Wrong email or password.")
        conn.execute("DELETE FROM login_failures WHERE email = ?", (email,))
        return {"token": _start_session(conn, user["id"]), "user": _public(user)}


@router.post("/google", summary="Sign in (or sign up) with a Google ID token")
def google_login(body: GoogleBody):
    claims = verify_google_credential(body.credential)
    email = normalize_email(claims["email"])
    with closing(_connect()) as conn:
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        if user is None:
            name = (claims.get("name") or email.split("@")[0]).strip()[:80]
            cursor = conn.execute(
                "INSERT INTO users (email, name, google_sub, created_at) VALUES (?, ?, ?, ?)",
                (email, name, claims.get("sub"), time.time()),
            )
            conn.commit()
            user = conn.execute("SELECT * FROM users WHERE id = ?", (cursor.lastrowid,)).fetchone()
        elif not user["google_sub"]:
            # Google vouches for this address, so link it to the existing email/password account
            conn.execute("UPDATE users SET google_sub = ? WHERE id = ?", (claims.get("sub"), user["id"]))
            conn.commit()
            user = conn.execute("SELECT * FROM users WHERE id = ?", (user["id"],)).fetchone()
        return {"token": _start_session(conn, user["id"]), "user": _public(user)}


@router.get("/me", summary="Who is signed in")
def me(authorization: Optional[str] = Header(default=None)):
    with closing(_connect()) as conn:
        user = _user_for_token(conn, _bearer(authorization))
        if not user:
            raise HTTPException(status_code=401, detail="Not signed in.")
        return {"user": _public(user)}


@router.get("/reports", summary="This account's past analysis reports, newest first")
def my_reports(authorization: Optional[str] = Header(default=None)):
    return {"reports": list_reports_for_token(_bearer(authorization))}


@router.patch("/reports/{session_id}", summary="Rename one of this account's saved reports")
def rename_report(session_id: str, body: RenameReportBody, authorization: Optional[str] = Header(default=None)):
    rename_report_for_token(_bearer(authorization), session_id, body.title)
    return {"ok": True}


@router.post("/logout", summary="End this session")
def logout(authorization: Optional[str] = Header(default=None)):
    token = _bearer(authorization)
    if token:
        with closing(_connect()) as conn:
            conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
            conn.commit()
    return {"ok": True}
