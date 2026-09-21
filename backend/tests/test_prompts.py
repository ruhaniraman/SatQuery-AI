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
    QUADRANT_NAMES,
    REPLY_WORD_LIMIT,
    build_general_messages,
    build_question_text,
    describe_first_enabled,
    describe_quadrant_prompt,
    describe_region_prompt,
    format_scene_description,
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
    assert "Measured facts" in general_system_prompt()
    assert "in words" in fusion_system_prompt() and "not numbers" in fusion_system_prompt()


def test_prompts_keep_the_grounding_structure_and_an_abstain_option():
    for text in (general_system_prompt(), fusion_system_prompt()):
        assert "OBSERVATIONS" in text and "ASSESSMENT" in text and "cannot answer" in text
    assert "never assume" in general_system_prompt().lower()


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
    """Regression guard. Change/fusion prompts: measured at 112 / 135 / 165 (were 162 / 193 / 221).
    The general prompt was 91 tokens; it is deliberately longer now (concrete-detail instructions plus a
    style example) to fight vague answers. Generous headroom, but a prompt that balloons fails here."""
    assert count(general_system_prompt("optical")) <= 330
    assert count(general_system_prompt("sar")) <= 390
    assert count(build_change_prompt(REGIONS, (512, 512))) <= 230
    assert count(build_change_prompt(REGIONS, (512, 512), modality="sar")) <= 250
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


# ------------------------------------------------------------------ concrete-answer prompt + describe-first pass

def test_general_prompt_asks_for_located_concrete_detail_not_brevity():
    text = general_system_prompt()
    assert "WHERE" in text and "specific" in text.lower()
    assert REPLY_WORD_LIMIT >= 150
    assert "under 120 words" not in text


def test_question_text_carries_facts_and_description_when_given():
    plain = build_question_text("what is here?")
    assert plain == "what is here?"
    full = build_question_text("what is here?", "green 40%", "upper-left: river")
    assert "Measured facts" in full and "green 40%" in full
    assert "upper-left: river" in full and full.rstrip().endswith("Question: what is here?")


def test_general_messages_put_facts_in_the_image_turn_only():
    msgs = build_general_messages("q?", True, None, "optical", scene_facts="F", scene_description="D")
    assert msgs[0]["role"] == "system" and "F" not in msgs[0]["content"]
    last = msgs[-1]["content"]
    assert last[0] == {"type": "image"} and "F" in last[1]["text"] and "D" in last[1]["text"]
    no_image = build_general_messages("q?", False, None, "optical", scene_facts="F", scene_description="D")
    assert no_image[-1] == {"role": "user", "content": "q?"}


def test_scene_description_lists_quadrants_in_order_and_skips_blanks():
    text = format_scene_description("A river valley.", {"lower-right": "houses", "upper-left": "forest", "upper-right": "  "})
    assert text.splitlines() == ["A river valley.", "upper-left: forest", "lower-right: houses"]
    assert format_scene_description("", None) == ""
    assert "upper-left" in describe_quadrant_prompt(QUADRANT_NAMES[0])


def test_describe_first_can_be_switched_off(monkeypatch):
    monkeypatch.delenv("DESCRIBE_FIRST", raising=False)
    assert describe_first_enabled()
    monkeypatch.setenv("DESCRIBE_FIRST", "0")
    assert not describe_first_enabled()


def test_the_region_prompt_asks_for_prose_and_does_not_prime_the_model_with_things_to_find():
    text = describe_region_prompt("centre")
    assert "centre area" in text and "plain prose" in text and "no list, no numbering" in text
    assert "two to four sentences" in text                              # fuller than the old 'two or three short'
    for priming in ("water", "buildings", "roads", "excavated"):
        assert priming not in text.lower(), priming
    assert "brightness patterns, not colours" in describe_region_prompt("centre", "sar")
