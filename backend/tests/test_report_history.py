"""An account's report history: /analyze links a finished run to whoever is signed in (bearer token),
and GET /auth/reports lists them back, newest first. Anonymous /analyze calls are unaffected."""
import auth
import numpy as np
import pytest

from tests.test_endpoint_wiring import client, png_bytes  # noqa: F401  (shared fixture + helper)


@pytest.fixture()
def api(client, tmp_path, monkeypatch):  # noqa: F811 (fixture name shadowing is the pytest pattern here)
    monkeypatch.setenv("AUTH_DB", str(tmp_path / "auth.db"))
    return client


def signed_up_token(api):
    codes = {}
    import unittest.mock
    with unittest.mock.patch.object(auth, "send_otp_email", lambda email, name, code: codes.__setitem__(email, code)):
        api.post("/auth/signup", json={"name": "Ada", "email": "ada@example.com", "password": "correct horse"})
        r = api.post("/auth/verify-signup", json={"email": "ada@example.com", "code": codes["ada@example.com"]})
    return r.json()["token"]


def analyze(api, token=None, query="what is here?"):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    img = ("scene.png", png_bytes(np.full((64, 64, 3), 120, dtype=np.uint8)), "image/png")
    return api.post("/analyze", data={"query": query}, files={"images": img}, headers=headers)


def test_reports_requires_sign_in(api):
    assert api.get("/auth/reports").status_code == 401
    assert api.get("/auth/reports", headers={"Authorization": "Bearer nonsense"}).status_code == 401


def test_analyze_without_a_token_does_not_appear_anywhere(api):
    token = signed_up_token(api)
    assert analyze(api).status_code == 200                                  # anonymous call still works
    assert api.get("/auth/reports", headers={"Authorization": f"Bearer {token}"}).json()["reports"] == []


def test_signed_in_analyze_is_added_to_the_account_history(api):
    token = signed_up_token(api)
    r = analyze(api, token=token, query="describe this scene")
    assert r.status_code == 200
    session_id = r.json()["agent_execution_trace"]["pipeline_id"]

    reports = api.get("/auth/reports", headers={"Authorization": f"Bearer {token}"}).json()["reports"]
    assert len(reports) == 1
    entry = reports[0]
    assert entry["session_id"] == session_id
    assert entry["query"] == "describe this scene"
    assert entry["title"] == "describe this scene"       # no custom name set yet: defaults to the query
    assert entry["task"] == "SINGLE_IMAGE_VQA"
    assert entry["visual_evidence_url"] == f"/reports/{session_id}/evidence.png"
    assert entry["report_download_url"] == f"/reports/{session_id}/report.pdf"
    # the PDF this /analyze call built is really there for that URL to serve
    assert api.get(entry["report_download_url"]).status_code == 200


def test_history_is_newest_first(api):
    token = signed_up_token(api)
    analyze(api, token=token, query="first")
    analyze(api, token=token, query="second")
    reports = api.get("/auth/reports", headers={"Authorization": f"Bearer {token}"}).json()["reports"]
    assert [r["query"] for r in reports] == ["second", "first"]


def test_two_accounts_do_not_see_each_other_s_history(api):
    import unittest.mock
    codes = {}
    with unittest.mock.patch.object(auth, "send_otp_email", lambda email, name, code: codes.__setitem__(email, code)):
        api.post("/auth/signup", json={"name": "Ada", "email": "ada@example.com", "password": "correct horse"})
        token_a = api.post("/auth/verify-signup", json={"email": "ada@example.com", "code": codes["ada@example.com"]}).json()["token"]
        api.post("/auth/signup", json={"name": "Bea", "email": "bea@example.com", "password": "correct horse"})
        token_b = api.post("/auth/verify-signup", json={"email": "bea@example.com", "code": codes["bea@example.com"]}).json()["token"]

    analyze(api, token=token_a, query="ada's query")
    reports_a = api.get("/auth/reports", headers={"Authorization": f"Bearer {token_a}"}).json()["reports"]
    reports_b = api.get("/auth/reports", headers={"Authorization": f"Bearer {token_b}"}).json()["reports"]
    assert len(reports_a) == 1 and reports_b == []


def test_renaming_a_report_changes_its_title_but_not_its_original_query(api):
    token = signed_up_token(api)
    session_id = analyze(api, token=token, query="original question").json()["agent_execution_trace"]["pipeline_id"]
    headers = {"Authorization": f"Bearer {token}"}

    r = api.patch(f"/auth/reports/{session_id}", json={"title": "  Mining scan, north pit  "}, headers=headers)
    assert r.status_code == 200

    entry = api.get("/auth/reports", headers=headers).json()["reports"][0]
    assert entry["title"] == "Mining scan, north pit"      # trimmed
    assert entry["query"] == "original question"           # the original question is preserved separately


@pytest.mark.parametrize("bad_title", ["", "   ", "x" * 201])
def test_renaming_rejects_empty_or_too_long_names(api, bad_title):
    token = signed_up_token(api)
    session_id = analyze(api, token=token).json()["agent_execution_trace"]["pipeline_id"]
    r = api.patch(f"/auth/reports/{session_id}", json={"title": bad_title}, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 422


def test_renaming_requires_sign_in_and_ownership(api):
    token = signed_up_token(api)
    session_id = analyze(api, token=token).json()["agent_execution_trace"]["pipeline_id"]

    anon = api.patch(f"/auth/reports/{session_id}", json={"title": "New name"})
    assert anon.status_code == 401

    import unittest.mock
    codes = {}
    with unittest.mock.patch.object(auth, "send_otp_email", lambda email, name, code: codes.__setitem__(email, code)):
        api.post("/auth/signup", json={"name": "Bea", "email": "bea@example.com", "password": "correct horse"})
        other_token = api.post("/auth/verify-signup", json={"email": "bea@example.com", "code": codes["bea@example.com"]}).json()["token"]
    stolen = api.patch(f"/auth/reports/{session_id}", json={"title": "Not mine"}, headers={"Authorization": f"Bearer {other_token}"})
    assert stolen.status_code == 404

    missing = api.patch("/auth/reports/no-such-session", json={"title": "x"}, headers={"Authorization": f"Bearer {token}"})
    assert missing.status_code == 404
