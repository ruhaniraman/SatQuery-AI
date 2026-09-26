"""The model endpoints (/analyze, /preview, /report) need a signed-in user; /health and /auth stay open, and a
report can only be rebuilt by the account that ran it. Drives the real app with the stub engine."""
import unittest.mock

import auth
import numpy as np
import pytest

from tests.test_endpoint_wiring import client, png_bytes  # noqa: F401  (shared fixture + helper)


@pytest.fixture()
def api(client, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setenv("AUTH_DB", str(tmp_path / "auth.db"))
    monkeypatch.setenv("REQUIRE_SIGN_IN", "1")
    return client


def token_for(api, email):
    codes = {}
    with unittest.mock.patch.object(auth, "send_otp_email", lambda e, name, code: codes.__setitem__(e, code)):
        api.post("/auth/signup", json={"name": "Ada", "email": email, "password": "correct horse"})
        r = api.post("/auth/verify-signup", json={"email": email, "code": codes[email]})
    return r.json()["token"]


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def image():
    return ("scene.png", png_bytes(np.full((64, 64, 3), 120, dtype=np.uint8)), "image/png")


def analyze(api, headers=None):
    return api.post("/analyze", data={"query": "what is here?"}, files={"images": image()}, headers=headers or {})


def preview(api, headers=None):
    return api.post("/preview", data={"modality": "optical"}, files={"image": image()}, headers=headers or {})


def report(api, session_id, headers=None):
    return api.post("/report", data={"session_id": session_id, "chat_history": "[]"}, headers=headers or {})


@pytest.mark.parametrize("headers", [None, {"Authorization": "Bearer nonsense"}, {"Authorization": "Basic abc"}])
def test_model_endpoints_refuse_without_a_valid_session(api, headers):
    for r in (analyze(api, headers), preview(api, headers), report(api, "0" * 8 + "-0000-4000-8000-" + "0" * 12, headers)):
        assert r.status_code == 401, r.text
        assert r.headers["WWW-Authenticate"] == "Bearer" and "sign in" in r.json()["detail"]
    assert api.agent.calls == []                                     # the model never ran


def test_signed_in_user_can_use_them(api):
    headers = bearer(token_for(api, "ada@example.com"))
    r = analyze(api, headers)
    assert r.status_code == 200, r.text
    assert preview(api, headers).status_code == 200
    session_id = r.json()["agent_execution_trace"]["pipeline_id"]
    rebuilt = report(api, session_id, headers)
    assert rebuilt.status_code == 200 and rebuilt.json()["report_download_url"].endswith("report_chat.pdf")


def test_only_the_owner_can_rebuild_a_report(api):
    owner, other = bearer(token_for(api, "ada@example.com")), bearer(token_for(api, "bob@example.com"))
    session_id = analyze(api, owner).json()["agent_execution_trace"]["pipeline_id"]
    stolen = report(api, session_id, other)
    assert stolen.status_code == 404 and stolen.json()["detail"] == "No such analysis run."


def test_open_endpoints_stay_open(api):
    assert api.get("/health").status_code == 200
    assert api.get("/auth/config").status_code == 200


def test_cors_origins_come_from_the_environment(api, monkeypatch):
    import main_api
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert main_api.cors_origins() == ["http://localhost:5173", "http://127.0.0.1:5173"]
    monkeypatch.setenv("CORS_ORIGINS", " https://satquery.example.org/ , http://localhost:5173 ,")
    assert main_api.cors_origins() == ["https://satquery.example.org", "http://localhost:5173"]


def test_sign_in_can_be_switched_off_for_local_use(api, monkeypatch):
    monkeypatch.setenv("REQUIRE_SIGN_IN", "0")
    assert analyze(api).status_code == 200 and preview(api).status_code == 200
