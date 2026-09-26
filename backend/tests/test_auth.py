"""Sign-up, login, sessions and Google sign-in, through the real FastAPI app (Google's own check is replaced)."""
import auth
import pytest
from fastapi import HTTPException

from tests.test_endpoint_wiring import client  # noqa: F401  (shared fixture)


@pytest.fixture()
def sent(monkeypatch):
    """Captures the OTP send instead of hitting SMTP: {email: code}."""
    codes = {}
    monkeypatch.setattr(auth, "send_otp_email", lambda email, name, code: codes.__setitem__(email, code))
    return codes


@pytest.fixture()
def api(client, tmp_path, monkeypatch):
    monkeypatch.setenv("AUTH_DB", str(tmp_path / "auth.db"))
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "client-123")
    return client


def start_signup(api, email="Ada@Example.com", password="correct horse", name="Ada"):
    return api.post("/auth/signup", json={"name": name, "email": email, "password": password})


def signup(api, sent, email="Ada@Example.com", password="correct horse", name="Ada"):
    """Full sign-up: start it, then verify with the code the (stubbed) email carried. Returns the
    verify-signup response, matching what the old one-call signup used to return."""
    r = start_signup(api, email=email, password=password, name=name)
    if r.status_code != 200:
        return r
    code = sent[auth.normalize_email(email)]
    return api.post("/auth/verify-signup", json={"email": email, "code": code})


def test_signup_then_me_then_logout(api, sent):
    r = signup(api, sent)
    assert r.status_code == 201
    body = r.json()
    assert body["user"]["email"] == "ada@example.com"          # emails are stored lower-case
    assert "password" not in str(body).lower().replace("password_hash", "")
    headers = {"Authorization": f"Bearer {body['token']}"}
    assert api.get("/auth/me", headers=headers).json()["user"]["name"] == "Ada"
    assert api.post("/auth/logout", headers=headers).status_code == 200
    assert api.get("/auth/me", headers=headers).status_code == 401       # the session is gone


def test_password_is_hashed_at_rest(api, sent, tmp_path):
    signup(api, sent, password="correct horse")
    raw = (tmp_path / "auth.db").read_bytes()
    assert b"correct horse" not in raw and b"scrypt$" in raw


@pytest.mark.parametrize("payload", [
    {"name": "", "email": "a@b.co", "password": "longenough"},
    {"name": "A", "email": "not-an-email", "password": "longenough"},
    {"name": "A", "email": "a@b.co", "password": "short"},
])
def test_signup_validation(api, payload):
    assert api.post("/auth/signup", json=payload).status_code == 422


def test_duplicate_email_is_refused_regardless_of_case(api, sent):
    assert signup(api, sent).status_code == 201
    r = signup(api, sent, email="ADA@example.com")
    assert r.status_code == 409 and "already exists" in r.json()["detail"]


def test_login_right_and_wrong(api, sent):
    signup(api, sent)
    ok = api.post("/auth/login", json={"email": " ada@example.com ", "password": "correct horse"})
    assert ok.status_code == 200 and ok.json()["token"]
    bad = api.post("/auth/login", json={"email": "ada@example.com", "password": "wrong password"})
    unknown = api.post("/auth/login", json={"email": "nobody@example.com", "password": "correct horse"})
    assert bad.status_code == unknown.status_code == 401
    assert bad.json()["detail"] == unknown.json()["detail"]               # does not reveal which emails exist


def _login(api, password, email="ada@example.com"):
    return api.post("/auth/login", json={"email": email, "password": password})


def test_repeated_wrong_passwords_lock_the_email(api, sent, monkeypatch):
    signup(api, sent)
    for _ in range(auth.LOGIN_MAX_FAILURES):
        assert _login(api, "wrong password").status_code == 401
    locked = _login(api, "correct horse")                                  # even the right password waits
    assert locked.status_code == 429 and int(locked.headers["Retry-After"]) > 0
    # Unknown emails lock the same way, so the lock reveals nothing.
    for _ in range(auth.LOGIN_MAX_FAILURES):
        _login(api, "x" * 8, email="nobody@example.com")
    assert _login(api, "x" * 8, email="nobody@example.com").status_code == 429
    # Once the window has passed, the right password works again.
    later = auth.time.time() + auth.LOGIN_WINDOW_SECONDS + 1
    monkeypatch.setattr(auth.time, "time", lambda: later)
    assert _login(api, "correct horse").status_code == 200


def test_a_right_password_clears_earlier_failures(api, sent):
    signup(api, sent)
    for _ in range(auth.LOGIN_MAX_FAILURES - 1):
        _login(api, "wrong password")
    assert _login(api, "correct horse").status_code == 200
    for _ in range(auth.LOGIN_MAX_FAILURES - 1):
        assert _login(api, "wrong password").status_code == 401            # the count started again


def test_me_without_or_with_a_bad_token(api):
    assert api.get("/auth/me").status_code == 401
    assert api.get("/auth/me", headers={"Authorization": "Bearer nonsense"}).status_code == 401


def test_expired_session_is_refused(api, sent, monkeypatch):
    token = signup(api, sent).json()["token"]
    real = auth.time.time
    monkeypatch.setattr(auth.time, "time", lambda: real() + (auth.SESSION_DAYS + 1) * 86400)
    assert api.get("/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401


def test_config_reports_the_google_client_id(api, monkeypatch):
    assert api.get("/auth/config").json()["google_client_id"] == "client-123"
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "")
    assert api.get("/auth/config").json()["google_client_id"] is None


def _claims(**extra):
    return {"email": "grace@gmail.com", "name": "Grace Hopper", "sub": "sub-1", **extra}


def test_google_creates_an_account_and_signs_in(api, monkeypatch):
    monkeypatch.setattr(auth, "verify_google_credential", lambda c: _claims())
    r = api.post("/auth/google", json={"credential": "x"})
    assert r.status_code == 200
    assert r.json()["user"] == {"id": 1, "email": "grace@gmail.com", "name": "Grace Hopper", "google": True}
    again = api.post("/auth/google", json={"credential": "x"})
    assert again.json()["user"]["id"] == 1                                 # same account, not a second one


def test_google_links_an_existing_password_account_and_password_still_works(api, sent, monkeypatch):
    signup(api, sent, email="grace@gmail.com", password="correct horse", name="Grace")
    monkeypatch.setattr(auth, "verify_google_credential", lambda c: _claims())
    assert api.post("/auth/google", json={"credential": "x"}).json()["user"]["google"] is True
    assert api.post("/auth/login", json={"email": "grace@gmail.com", "password": "correct horse"}).status_code == 200


def test_google_only_account_cannot_log_in_with_a_password(api, sent, monkeypatch):
    monkeypatch.setattr(auth, "verify_google_credential", lambda c: _claims())
    api.post("/auth/google", json={"credential": "x"})
    r = api.post("/auth/login", json={"email": "grace@gmail.com", "password": "anything at all"})
    assert r.status_code == 401 and "Google" in r.json()["detail"]
    assert "Google" in signup(api, sent, email="grace@gmail.com").json()["detail"]


def test_google_without_a_client_id_is_503(api, monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "")
    assert api.post("/auth/google", json={"credential": "x"}).status_code == 503


def test_signup_without_smtp_configured_is_503(api, monkeypatch):
    monkeypatch.delenv("SMTP_HOST", raising=False)
    r = start_signup(api)
    assert r.status_code == 503 and "SMTP_HOST" in r.json()["detail"]


def test_signup_sends_a_pending_code_not_a_token(api, sent):
    r = start_signup(api)
    assert r.status_code == 200
    assert r.json() == {"pending": True, "email": "ada@example.com", "expires_in": auth.OTP_TTL_SECONDS}
    assert sent["ada@example.com"] and len(sent["ada@example.com"]) == 6


def test_wrong_code_is_refused_and_capped(api, sent):
    start_signup(api)
    for _ in range(auth.OTP_MAX_ATTEMPTS):
        r = api.post("/auth/verify-signup", json={"email": "Ada@Example.com", "code": "000000"})
        assert r.status_code == 401
    # one more wrong try (or the right one) now finds the pending sign-up gone
    over = api.post("/auth/verify-signup", json={"email": "Ada@Example.com", "code": sent["ada@example.com"]})
    assert over.status_code in (404, 429)


def test_expired_code_is_refused(api, sent, monkeypatch):
    start_signup(api)
    real = auth.time.time
    monkeypatch.setattr(auth.time, "time", lambda: real() + auth.OTP_TTL_SECONDS + 1)
    r = api.post("/auth/verify-signup", json={"email": "Ada@Example.com", "code": sent["ada@example.com"]})
    assert r.status_code == 404


def test_resend_gets_a_new_code_and_is_rate_limited(api, sent, monkeypatch):
    start_signup(api)
    first_code = sent["ada@example.com"]
    cooldown = api.post("/auth/resend-otp", json={"email": "Ada@Example.com"})
    assert cooldown.status_code == 429

    real = auth.time.time
    monkeypatch.setattr(auth.time, "time", lambda: real() + auth.OTP_RESEND_COOLDOWN_SECONDS + 1)
    ok = api.post("/auth/resend-otp", json={"email": "Ada@Example.com"})
    assert ok.status_code == 200
    assert sent["ada@example.com"] != first_code

    # the old code no longer works, the new one does
    assert api.post("/auth/verify-signup", json={"email": "Ada@Example.com", "code": first_code}).status_code == 401
    assert api.post("/auth/verify-signup", json={"email": "Ada@Example.com", "code": sent["ada@example.com"]}).status_code == 201


def test_verifying_an_unknown_email_is_404(api):
    assert api.post("/auth/verify-signup", json={"email": "nobody@example.com", "code": "123456"}).status_code == 404
    assert api.post("/auth/resend-otp", json={"email": "nobody@example.com"}).status_code == 404


class _Resp:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        import json
        return json.dumps(self.payload).encode()


def _tokeninfo(monkeypatch, **claims):
    base = {"aud": "client-123", "iss": "https://accounts.google.com", "email": "a@gmail.com",
            "email_verified": "true", "exp": str(auth.time.time() + 600)}
    monkeypatch.setattr(auth.urllib.request, "urlopen", lambda url, timeout=0: _Resp({**base, **claims}))


def test_google_token_checks(api, monkeypatch):
    _tokeninfo(monkeypatch)
    assert auth.verify_google_credential("t")["email"] == "a@gmail.com"
    for bad in ({"aud": "someone-else"}, {"iss": "evil.example"}, {"email_verified": "false"}, {"exp": "1"}):
        _tokeninfo(monkeypatch, **bad)
        with pytest.raises(HTTPException) as err:
            auth.verify_google_credential("t")
        assert err.value.status_code == 401, bad
