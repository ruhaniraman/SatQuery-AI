"""Closed questions (yes/no, counts, rural/urban) get their short answer from the remote-sensing VQA adapter, and the
trace says so; everything else is unchanged. Drives the real /analyze with the stub engine."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_manager.closed_questions import adapter_question_space, closed_question_space, display_answer  # noqa: E402
from test_endpoint_wiring import client, png_bytes, post  # noqa: E402,F401

VQA_INFO = {"base_model": "Qwen/Qwen2-VL-2B-Instruct", "lora_rank": 16, "lora_alpha": 32,
            "training_data": "RSVQA-LR train split: 57223 questions on 572 images"}


def _with_adapter(client, reply="Yes", confidence=0.93):
    calls = []

    def short_answer(images, prompt, adapter=None, max_new_tokens=16):
        calls.append({"images": images, "prompt": prompt, "adapter": adapter})
        return reply, confidence

    client.agent.adapter_info = {"vqa": VQA_INFO}
    client.agent.short_answer = short_answer
    return calls


def _ask(client, query, **form):
    img = png_bytes(np.full((64, 64, 3), 90, np.uint8))
    r = post(client, [("images", ("a.png", img, "image/png"))], query=query, **form)
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------- which questions are closed

def test_question_kinds():
    assert closed_question_space("Is there a water body in this image?").kind == "closed"
    assert closed_question_space("are the fields irrigated").options == ["no", "yes"]
    assert closed_question_space("How many buildings are there?").kind == "numeric"
    assert closed_question_space("What is the number of roads?").kind == "numeric"
    assert closed_question_space("Is it a rural or an urban area?").options == ["rural", "urban"]
    for open_question in ("Describe the land cover.", "What changed here?", "Where is the river?",
                          "Is it forest or farmland?", "Which area is flooded?", ""):
        assert closed_question_space(open_question) is None, open_question


def test_polite_requests_and_mixed_questions_are_not_yes_no():
    for request in ("Can you describe the land cover in this image?", "Could you tell me what is going on here?",
                    "Would you explain the scene?", "Will you check the image", "Do you see anything unusual? Describe it.",
                    "Are there buildings near the water body, and where?", "Is there water, and which side is it on?"):
        assert closed_question_space(request) is None, request
    # Still yes/no: "can" / "could" about the scene itself, not a request to the assistant.
    assert closed_question_space("Can a road be seen near the river?").kind == "closed"
    assert closed_question_space("Has the built-up area increased?").kind == "closed"


def test_counts_are_not_sent_to_the_adapter():
    assert closed_question_space("How many ponds are there?").kind == "numeric"      # recognised...
    assert adapter_question_space("How many ponds are there?") is None               # ...but not routed
    assert adapter_question_space("Is there a pond?").kind == "closed"
    assert adapter_question_space("Is this area rural or urban?").options == ["rural", "urban"]


def test_display_answer_only_accepts_a_real_answer():
    yes_no = closed_question_space("Is there water?")
    assert display_answer(yes_no, "yes") == "Yes" and display_answer(yes_no, "No.") == "No"
    assert display_answer(yes_no, "I cannot tell") is None
    count = closed_question_space("How many ponds?")
    assert display_answer(count, "3") == "3" and display_answer(count, "several") is None


# ---------------------------------------------------------------- /analyze

def test_closed_question_uses_the_adapter_and_says_so(client):
    calls = _with_adapter(client, reply="Yes", confidence=0.93)
    data = _ask(client, "Is there a water body?")
    assert data["answer"].startswith("Answer: Yes\n\nA plain answer.")
    (call,) = calls
    assert call["adapter"] == "vqa" and len(call["images"]) == 1
    assert call["prompt"].endswith("Answer with yes or no only.")
    # The explanation pass is told the adapted answer, so it explains it instead of contradicting it blindly.
    assert client.agent.calls[-1]["answer_hint"] == "Yes"
    trace = data["agent_execution_trace"]
    assert trace["nodes_traversed"] == ["RuleBasedRouter", "InputPreprocessor", "AdaptedVQASpecialist", "SingleImageSpecialist"]
    tel = trace["telemetry"]
    assert tel["active_adapter"] == "vqa" and tel["short_answer"] == "Yes"
    assert "LoRA adapter 'vqa'" in tel["model_used"] and "base weights for the explanation" in tel["model_used"]
    assert tel["adapter_training_data"] == VQA_INFO["training_data"]
    assert tel["confidence"] == 0.93                        # the short answer's probability, not the paragraph's
    assert "VQA adapter gave its short answer" in tel["confidence_method"]


def test_open_question_is_unchanged(client):
    calls = _with_adapter(client)
    data = _ask(client, "Describe the land cover.")
    assert calls == [] and data["answer"] == "A plain answer."
    tel = data["agent_execution_trace"]["telemetry"]
    assert tel["active_adapter"] is None and tel["confidence"] == 0.87 and "short_answer" not in tel


def test_unusable_adapter_reply_falls_back_to_the_general_answer(client):
    _with_adapter(client, reply="hard to say", confidence=0.4)
    data = _ask(client, "Are there buildings?")
    assert data["answer"] == "A plain answer."
    tel = data["agent_execution_trace"]["telemetry"]
    assert tel["confidence"] == 0.87 and "short_answer" not in tel and tel["active_adapter"] is None
    assert client.agent.calls[-1]["answer_hint"] == ""


def test_polite_request_and_count_take_the_general_path(client):
    calls = _with_adapter(client)
    for query in ("Can you describe the land cover in this image?", "How many buildings are there?"):
        data = _ask(client, query)
        assert data["answer"] == "A plain answer.", query
    assert calls == []


def test_sar_images_never_use_the_optical_adapter(client):
    calls = _with_adapter(client)
    data = _ask(client, "Is there a water body?", modality_a="sar")
    assert calls == [] and not data["answer"].startswith("Answer:")


def test_without_the_adapter_closed_questions_take_the_general_path(client):
    data = _ask(client, "Is there a water body?")         # the stub engine has no adapter_info at all
    assert data["answer"] == "A plain answer."
