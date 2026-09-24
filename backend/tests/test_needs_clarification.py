"""An opening question this vague ("Hi", "?", a bare pronoun) gets a clarification message instead of
a guess - and, cheaper too, never reaches the model. Any question once the conversation has a prior
turn is treated as a real follow-up and always goes to the model."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from agent_manager.clarification import needs_clarification  # noqa: E402
from test_endpoint_wiring import client, png_bytes  # noqa: E402,F401

GRAY = lambda: png_bytes(__import__("numpy").full((64, 64, 3), 90, "uint8"))  # noqa: E731


def one_image(client, **form):
    files = [("images", ("a.png", GRAY(), "image/png"))]
    return client.post("/analyze", data=form, files=files)


# ------------------------------------------------------------------ pure function

@pytest.mark.parametrize("query", ["hi", "Hi!", "  HEY  ", "hello?", "ok", "thanks", "?", "  ", "what about it"])
def test_vague_openers_ask_for_more(query):
    assert needs_clarification(query, []) is not None


@pytest.mark.parametrize("query", ["what changed?", "is there water here", "describe the mining area", "roads?"])
def test_real_questions_are_left_alone(query):
    assert needs_clarification(query, []) is None


def test_any_prior_turn_lets_a_short_follow_up_through():
    history = [{"role": "user", "content": "what is here?"}, {"role": "ai", "content": "A forest."}]
    assert needs_clarification("and this?", history) is None
    assert needs_clarification("hi", history) is None  # even a greeting, once there is real context


# ------------------------------------------------------------------ wired into /analyze

def test_vague_opener_gets_a_clarification_answer_and_never_calls_the_model(client):
    r = one_image(client, query="hi", adapter="general", chat_history="[]")
    assert r.status_code == 200
    body = r.json()
    assert "more specific" in body["answer"]
    assert client.agent.calls == []  # the model was never invoked

    trace = body["agent_execution_trace"]
    assert trace["nodes_traversed"] == ["RuleBasedRouter", "InputPreprocessor", "ClarificationGate"]
    assert trace["telemetry"]["model_used"] is None
    assert trace["telemetry"]["active_adapter"] is None
    assert "vague" in trace["telemetry"]["inference"]


def test_a_real_question_still_calls_the_model(client):
    r = one_image(client, query="what land cover is visible?", adapter="general", chat_history="[]")
    assert r.status_code == 200
    assert len(client.agent.calls) == 1
    trace = r.json()["agent_execution_trace"]
    assert trace["nodes_traversed"] == ["RuleBasedRouter", "InputPreprocessor", "SingleImageSpecialist"]
    assert trace["telemetry"]["model_used"]


def test_a_scan_is_never_gated_by_clarification(client):
    # adapter != "general": the query is a fixed scan prompt, not a free-text question, so it is never
    # ambiguous in the sense this gate cares about.
    r = one_image(client, query="hi", adapter="mining", chat_history="[]")
    assert r.status_code == 200
    assert len(client.agent.calls) == 1
    assert r.json()["agent_execution_trace"]["nodes_traversed"] == ["RuleBasedRouter", "InputPreprocessor", "SingleImageSpecialist"]


def test_a_vague_opener_with_real_conversation_history_still_reaches_the_model(client):
    history = '[{"role": "user", "content": "what is here?"}, {"role": "ai", "content": "A forest."}]'
    r = one_image(client, query="hi", adapter="general", chat_history=history)
    assert r.status_code == 200
    assert len(client.agent.calls) == 1
