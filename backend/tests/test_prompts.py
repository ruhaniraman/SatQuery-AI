"""Prompt quality guards: bounded replies, compact history, token budgets, and the real VLM bridge."""
import contextlib
import os
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from agent_manager.prompts import (  # noqa: E402
    MAX_HISTORY_CHARS_PER_TURN,
    REPLY_WORD_LIMIT,
    general_system_prompt,
    history_for_model,
)
from change_detection.cdvqa_engine import build_change_prompt  # noqa: E402
from fusion.sar_optical_fusion import fusion_system_prompt  # noqa: E402
from test_endpoint_wiring import client  # noqa: E402,F401

TOKENIZER = os.path.join(os.path.dirname(__file__), "..", "models", "tokenizer.json")
REGIONS = [(10, 10, 80, 80, 0.06), (300, 200, 60, 60, 0.03)]


# ------------------------------------------------------------------ every prompt is bounded and honest

def test_every_free_text_prompt_asks_for_a_bounded_reply():
    for text in (general_system_prompt(), general_system_prompt("sar"), fusion_system_prompt(),
                 build_change_prompt([], (512, 512)), build_change_prompt(REGIONS, (512, 512), modality="sar")):
        assert f"under {REPLY_WORD_LIMIT} words" in text


def test_prompts_ask_for_words_not_invented_numbers():
    assert "Amounts in words" in general_system_prompt()
    assert "in words" in fusion_system_prompt() and "not numbers" in fusion_system_prompt()


def test_prompts_keep_the_grounding_structure_and_an_abstain_option():
    for text in (general_system_prompt(), fusion_system_prompt()):
        assert "OBSERVATIONS" in text and "ASSESSMENT" in text and "cannot answer" in text
    assert "never assume" in general_system_prompt()


def test_change_prompt_stays_consistent_with_the_pixel_analysis():
    assert "flagged 2 region(s)" in build_change_prompt(REGIONS, (512, 512))
    assert "no region of substantial change" in build_change_prompt([], (512, 512))
    assert "Also answer: any new roads?" in build_change_prompt([], (512, 512), "any new roads?")
    unknown = build_change_prompt([], (512, 512))
    assert "capture dates are unknown" in unknown and "BEFORE" in unknown
    dated = build_change_prompt([], (512, 512), capture_dates=("2019-01-10", "2024-01-10"))
    assert "2019-01-10" in dated and "2024-01-10" in dated and "unknown" not in dated


# ------------------------------------------------------------------ history compaction

LONG = "OBSERVATIONS: " + "A river bends through farmland. " * 30 + "ASSESSMENT: About a third of the scene is water."


def test_assistant_turns_are_reduced_to_their_conclusion():
    turns = history_for_model([{"role": "user", "content": "how much water?"}, {"role": "ai", "content": LONG},
                               {"role": "user", "content": "and north?"}], "and north?")
    assert turns[1]["content"] == "ASSESSMENT: About a third of the scene is water."
    assert "river bends" not in str(turns)


def test_long_turns_are_clipped_but_short_ones_untouched():
    long_user = "why " * 400
    turns = history_for_model([{"role": "user", "content": long_user}, {"role": "ai", "content": "x" * 3000}], "q")
    assert all(len(t["content"]) <= MAX_HISTORY_CHARS_PER_TURN for t in turns)
    assert turns[0]["content"].endswith("…")
    short = history_for_model([{"role": "user", "content": "hi"}, {"role": "ai", "content": "hello there"}], "q")
    assert [t["content"] for t in short] == ["hi", "hello there"]


def test_assistant_turn_without_a_marker_is_just_clipped():
    turns = history_for_model([{"role": "ai", "content": "Scan complete. Found 2 potential region(s)."}], "q")
    assert turns[0]["content"] == "Scan complete. Found 2 potential region(s)."


# ------------------------------------------------------------------ token budgets (real Qwen tokenizer)

@pytest.fixture(scope="module")
def count():
    tokenizers = pytest.importorskip("tokenizers")
    if not os.path.exists(TOKENIZER):
        pytest.skip("models/tokenizer.json not present")
    tok = tokenizers.Tokenizer.from_file(TOKENIZER)
    return lambda text: len(tok.encode(text).ids)


def test_prompt_token_budgets(count):
    """Regression guard: measured at 91 / 158 / 112 / 135 / 165 when written (were 110 / 164 / 162 /
    193 / 221). Generous headroom, but a prompt that balloons again fails here."""
    assert count(general_system_prompt("optical")) <= 105
    assert count(general_system_prompt("sar")) <= 165
    assert count(build_change_prompt(REGIONS, (512, 512))) <= 130
    assert count(build_change_prompt(REGIONS, (512, 512), modality="sar")) <= 150
    assert count(fusion_system_prompt()) <= 180


def test_six_long_turns_cost_a_small_fraction_of_what_they_used_to(count):
    hist = []
    for i in range(6):
        hist += [{"role": "user", "content": f"question {i} about the scene"}, {"role": "ai", "content": LONG}]
    hist.append({"role": "user", "content": "next?"})
    compact = " ".join(t["content"] for t in history_for_model(hist, "next?"))
    verbatim = " ".join(t["content"] for t in hist[-7:-1])
    assert count(compact) < 0.25 * count(verbatim)


# ------------------------------------------------------------------ the real VLM bridge

def test_vlm_bridge_sends_prompt_unwrapped_with_system_role_and_model_device(client, monkeypatch):
    import main_api

    class Inputs(dict):
        input_ids = [[1, 2]]

        def to(self, device):
            self["device"] = device
            return self

    calls = {}

    class Processor:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            calls["messages"] = messages
            return "TEXT"

        def __call__(self, text, images, padding, return_tensors):
            calls["n_images"] = len(images)
            return Inputs()

        def batch_decode(self, ids, skip_special_tokens):
            calls["decoded"] = ids
            return ["the answer"]

    class Model:
        device = "cuda:1"

        @contextlib.contextmanager
        def disable_adapter(self):
            calls["adapters_disabled"] = True
            yield

        def generate(self, **kw):
            calls["max_new_tokens"] = kw["max_new_tokens"]
            return [[1, 2, 3, 4]]

    import threading
    engine = types.SimpleNamespace(processor=Processor(), model=Model(), lock=threading.Lock())
    monkeypatch.setattr(main_api, "get_agent", lambda: engine)

    out = client.real_vlm("Describe the scene.", np.zeros((8, 8, 3), np.uint8), system="SYS")
    assert out == "the answer"
    msgs = calls["messages"]
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1]["content"][-1] == {"type": "text", "text": "Describe the scene."}   # no wrapper header
    assert calls["n_images"] == 1 and calls["adapters_disabled"] and calls["max_new_tokens"] == 1024
    assert calls["decoded"] == [[3, 4]]
