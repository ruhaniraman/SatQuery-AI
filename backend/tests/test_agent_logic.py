"""Prompt construction, yes/no logit scoring, tile merging, and the real grid scan driven by a fake model."""
import contextlib
import io
import importlib
import os
import sys
import types

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent_manager.grid_scan import SCAN_YES_THRESHOLD, merge_positive_tiles, tile_windows, yes_probability  # noqa: E402
from agent_manager.prompts import (  # noqa: E402
    DEFAULT_MAX_NEW_TOKENS,
    build_general_messages,
    general_system_prompt,
    history_for_model,
)


# ------------------------------------------------------------------ prompts / history

def test_history_drops_current_turn_system_pseudo_turns_and_maps_roles():
    hist = [
        {"role": "user", "content": "[System]: Initiating mining scan..."},
        {"role": "ai", "content": "Scan complete. Found 2 potential region(s)."},
        {"role": "user", "content": "what about the north side?"},
        {"role": "ai", "content": "The north is forest."},
        {"role": "user", "content": "and the south?"},          # the CURRENT question, as the UI sends it
    ]
    turns = history_for_model(hist, "and the south?")
    assert [t["role"] for t in turns] == ["assistant", "user", "assistant"]
    assert all("[System]" not in t["content"] for t in turns)
    assert turns[-1]["content"] == "The north is forest."


def test_history_is_bounded_and_tolerates_garbage():
    hist = [{"role": "user" if i % 2 == 0 else "ai", "content": f"m{i}"} for i in range(40)]
    assert len(history_for_model(hist, "zzz", max_turns=6)) == 6
    assert history_for_model(None, "q") == [] and history_for_model("nope", "q") == []
    assert history_for_model([1, {"role": "user"}, {"role": "x", "content": "a"}], "q") == []


def test_general_messages_use_system_role_and_attach_image_to_last_turn():
    msgs = build_general_messages("how much water?", True,
                                  [{"role": "user", "content": "hi"}, {"role": "ai", "content": "hello"},
                                   {"role": "user", "content": "how much water?"}])
    assert msgs[0]["role"] == "system" and isinstance(msgs[0]["content"], str)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[-1]["content"][0] == {"type": "image"}
    assert "[SYSTEM]" not in str(msgs)
    assert build_general_messages("q", False)[-1]["content"] == "q"


def test_general_prompt_is_scene_agnostic_and_sar_aware():
    p = general_system_prompt("optical").lower()
    for forced in ("building density", "bare land", "color/state of the water"):
        assert forced not in p
    assert "backscatter" in general_system_prompt("sar") and "backscatter" not in general_system_prompt("optical")


def test_single_token_budget_is_generous():
    assert DEFAULT_MAX_NEW_TOKENS >= 512


# ------------------------------------------------------------------ pure grid helpers

def test_yes_probability_from_logits():
    logits = np.full((3, 30), -5.0)
    logits[0, [10, 11]] = 4.0      # yes-like dominant
    logits[1, [20, 21]] = 4.0      # no-like dominant
    logits[2, 10] = 2.0
    logits[2, 20] = 2.0            # tie
    logits[:, 3] = 50.0            # huge mass on an unrelated token must be ignored
    p = yes_probability(logits, [10, 11], [20, 21])
    assert p[0] > 0.99 and p[1] < 0.01 and abs(p[2] - 0.5) < 1e-6


def test_yes_probability_is_numerically_stable():
    logits = np.zeros((1, 10))
    logits[0, 1], logits[0, 2] = 1000.0, -1000.0
    assert yes_probability(logits, [1], [2])[0] == pytest.approx(1.0)


def test_merge_adjacent_tiles_but_not_diagonals():
    g = np.zeros((4, 4), bool)
    g[0, 0] = g[0, 1] = True            # one horizontal pair
    g[2, 3] = True                      # a separate cell
    g[3, 2] = True                      # touches (2,3) only diagonally -> separate region
    boxes = merge_positive_tiles(g)
    assert boxes == [[0.0, 0.0, 0.25, 0.5], [0.5, 0.75, 0.75, 1.0], [0.75, 0.5, 1.0, 0.75]]
    assert merge_positive_tiles(np.zeros((4, 4), bool)) == []


def test_merge_l_shape_gives_bounding_box():
    g = np.zeros((4, 4), bool)
    g[1, 1] = g[2, 1] = g[2, 2] = True
    assert merge_positive_tiles(g) == [[0.25, 0.25, 0.75, 0.75]]


def test_tile_windows_cover_image_with_context_clamped():
    w = tile_windows(103, 50, 2, 3)
    assert len(w) == 6
    for _, _, (l, t, r, b) in w:
        assert 0 <= l < r <= 103 and 0 <= t < b <= 50
    corner = w[0][2]
    assert corner[0] == 0 and corner[1] == 0 and corner[2] > 103 // 3   # padded past its own cell


# ------------------------------------------------------------------ real controller, fake model

class FakeArr:
    def __init__(self, a): self.a = np.asarray(a)
    def __getitem__(self, k): return FakeArr(self.a[k])
    def float(self): return self
    def cpu(self): return self
    def numpy(self): return self.a


class FakeTokenizer:
    padding_side = "right"
    _ids = {"yes": [10], "Yes": [11], " yes": [12], " Yes": [13], "YES": [14],
            "no": [20], "No": [21], " no": [22], " No": [23], "NO": [24]}
    def encode(self, w, add_special_tokens=False): return self._ids[w]


class FakeInputs(dict):
    def to(self, _device): return self


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
    load_kwargs = {}
    @classmethod
    def from_pretrained(cls, *a, **k):
        FakeProcessor.load_kwargs = k
        return cls()
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True): return "TEXT"
    def __call__(self, text, images, return_tensors, padding):
        assert len(text) == len(images)
        return FakeInputs(n=len(images))


class FakeModel:
    positive = set()
    accept_logits_to_keep = True
    device = "cpu"
    def __init__(self): self.seen, self.batch_sizes = 0, []
    @classmethod
    def from_pretrained(cls, *a, **k): return cls()
    def load_adapter(self, *a, **k): pass
    def __call__(self, n, **kw):
        if "logits_to_keep" in kw and not self.accept_logits_to_keep:
            raise TypeError("unexpected keyword logits_to_keep")
        self.batch_sizes.append(n)
        logits = np.full((n, 3, 30), -5.0)
        for i in range(n):
            idx = self.seen + i
            logits[i, -1, 10 if idx in self.positive else 20] = 5.0
        self.seen += n
        return types.SimpleNamespace(logits=FakeArr(logits))


@pytest.fixture()
def controller(monkeypatch):
    torch = types.ModuleType("torch")
    torch.float16 = "f16"
    torch.no_grad = contextlib.contextmanager(lambda: (yield))
    tf = types.ModuleType("transformers")
    tf.AutoProcessor, tf.Qwen2VLForConditionalGeneration = FakeProcessor, FakeModel
    tf.BitsAndBytesConfig = lambda **k: None
    peft = types.ModuleType("peft")
    peft.PeftModel = FakeModel
    saved = sys.modules.pop("agent_manager.agent_controller", None)
    for name, mod in (("torch", torch), ("transformers", tf), ("peft", peft)):
        monkeypatch.setitem(sys.modules, name, mod)
    mod = importlib.import_module("agent_manager.agent_controller")
    FakeModel.positive, FakeModel.accept_logits_to_keep = set(), True
    yield mod
    sys.modules.pop("agent_manager.agent_controller", None)
    if saved is not None:
        sys.modules["agent_manager.agent_controller"] = saved


def test_processor_is_set_to_left_padding(controller):
    assert controller.load_agent().processor.tokenizer.padding_side == "left"


def test_grid_scan_batches_scores_and_merges(controller):
    FakeModel.positive = {0, 1, 11}                     # cells (0,0),(0,1) and (2,3)
    eng = controller.load_agent()
    boxes = eng._grid_classification(Image.new("RGB", (400, 400)), "is there a pit?")
    assert boxes == [[0.0, 0.0, 0.25, 0.5], [0.5, 0.75, 0.75, 1.0]]
    assert eng.model.batch_sizes == [4, 4, 4, 4]         # 16 cells in 4 batched passes, not 16 calls


def test_grid_scan_no_positive_cells(controller):
    assert controller.load_agent()._grid_classification(Image.new("RGB", (64, 64)), "q") == []


def test_grid_scan_falls_back_when_model_lacks_logits_to_keep(controller):
    FakeModel.accept_logits_to_keep = False
    FakeModel.positive = {5}
    boxes = controller.load_agent()._grid_classification(Image.new("RGB", (64, 64)), "q")
    assert boxes == [[0.25, 0.25, 0.5, 0.5]]


def test_grid_scan_threshold_is_tunable(controller):
    FakeModel.positive = {0}                             # P(yes) ~= 0.99995 for that cell
    eng = controller.load_agent()
    assert eng._grid_classification(Image.new("RGB", (64, 64)), "q", threshold=0.99) == [[0.0, 0.0, 0.25, 0.25]]
    FakeModel.positive = {0}
    eng.model.seen = 0
    assert eng._grid_classification(Image.new("RGB", (64, 64)), "q", threshold=0.9999999) == []


def test_processor_is_capped_so_large_images_cannot_oom(controller):
    controller.load_agent()
    assert FakeProcessor.load_kwargs["max_pixels"] == controller.MAX_PIXELS == 1280 * 28 * 28
    assert FakeProcessor.load_kwargs["min_pixels"] == controller.MIN_PIXELS   # tiny images get upscaled


def test_importing_the_module_does_not_load_the_model(controller):
    assert controller.get_agent() is None                # nothing loaded until load_agent() is called
    first = controller.load_agent()
    assert controller.load_agent() is first and controller.get_agent() is first


def test_route_intent_dead_code_is_gone(controller):
    assert not hasattr(controller.SatQueryEngine, "route_intent")


# ------------------------------------------------------------------ describe-first general path

class GenProcessor:
    """Records every chat template call and every batch; decodes row i of a batch to a numbered caption."""
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.templates, self.batches = [], []
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.templates.append(messages)
        return "TEXT"
    def __call__(self, text, images, return_tensors, padding):
        self.batches.append(len(images))
        return FakeInputs(input_ids=types.SimpleNamespace(shape=(1, 5)), n=len(images))
    def decode(self, ids, skip_special_tokens=True):
        return "caption-" + str(ids)


class GenModel:
    device = "cpu"
    def __init__(self):
        self.generate_calls, self.adapter_disabled = [], 0
    @contextlib.contextmanager
    def disable_adapter(self):
        self.adapter_disabled += 1
        yield
    def generate(self, **kw):
        self.generate_calls.append(kw)
        n = kw["n"]
        sequences = [list(range(5)) + [100 + i] for i in range(n)]
        if kw.get("return_dict_in_generate"):
            # One generated token per row here; scores is deliberately shaped so the real _query()
            # code path (torch.softmax et al) is exercised, not stubbed around.
            return types.SimpleNamespace(sequences=sequences, scores=(FakeArr(np.full((n, 30), -5.0)),))
        return sequences


def general_engine(controller):
    eng = controller.load_agent()
    eng.processor, eng.model = GenProcessor(), GenModel()
    return eng


def test_describe_scene_runs_one_overview_and_four_quadrants_with_adapters_off(controller):
    eng = general_engine(controller)
    text = eng._describe_scene(Image.new("RGB", (64, 64)))
    assert eng.processor.batches == [1, 4]
    assert eng.model.adapter_disabled == 2
    assert all(kw["max_new_tokens"] > 0 and kw["repetition_penalty"] == 1.05 for kw in eng.model.generate_calls)
    lines = text.splitlines()
    assert lines[0].startswith("caption-") and [l.split(":")[0] for l in lines[1:]] == [
        "upper-left", "upper-right", "lower-left", "lower-right"]


def test_general_query_feeds_description_and_facts_to_the_answer_pass(controller, monkeypatch):
    monkeypatch.setenv("DESCRIBE_FIRST", "1")
    eng = general_engine(controller)
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, "PNG")
    eng.query("what is here?", image_bytes=buf.getvalue(), explicit_adapter="general", scene_facts="FACTS-XYZ")
    assert eng.processor.batches == [1, 4, 1]                      # overview, quadrants, then the answer
    final_user = eng.processor.templates[-1][-1]["content"][-1]["text"]
    assert "FACTS-XYZ" in final_user and "upper-left: caption-" in final_user and "what is here?" in final_user


def test_describe_first_off_answers_in_a_single_pass(controller, monkeypatch):
    monkeypatch.setenv("DESCRIBE_FIRST", "0")
    eng = general_engine(controller)
    buf = io.BytesIO()
    Image.new("RGB", (64, 64)).save(buf, "PNG")
    eng.query("what is here?", image_bytes=buf.getvalue(), explicit_adapter="general", scene_facts="FACTS-XYZ")
    assert eng.processor.batches == [1]
    final_user = eng.processor.templates[-1][-1]["content"][-1]["text"]
    assert "FACTS-XYZ" in final_user and "upper-left" not in final_user


# ------------------------------------------------------------------ LoRA scan scores (used to compare two dates)

def test_scan_scores_returns_the_probability_grid_under_the_named_adapter(controller):
    FakeModel.positive = {5}
    eng = controller.load_agent()
    chosen = []
    eng.model.set_adapter = chosen.append
    grid = eng.scan_scores(np.zeros((64, 64, 3), np.uint8), "mining")
    assert chosen == ["mining"] and grid.shape == (4, 4)
    assert grid[1, 1] > 0.99 and grid.sum() < 1.5                     # only cell 5 == (1, 1) is positive


def test_scan_scores_rejects_unknown_adapters_and_shares_the_scan_questions(controller):
    eng = controller.load_agent()
    with pytest.raises(ValueError):
        eng.scan_scores(np.zeros((8, 8, 3), np.uint8), "general")
    assert set(controller.ADAPTER_QUESTIONS) == {"mining", "deforestation", "agriculture"}
    assert all("'yes' or 'no'" in q for q in controller.ADAPTER_QUESTIONS.values())


def test_scan_threshold_keeps_a_mixed_scene_from_being_boxed_whole():
    # Real mining-adapter P(yes) grid for a scene of two pits inside a town: at 0.5 ten tiles are
    # positive, 4-connected into ONE group whose bounding box is the whole image.
    p = np.array([[0.84, 0.41, 0.15, 0.12], [0.91, 0.25, 0.88, 0.27],
                  [0.91, 0.59, 0.97, 0.35], [0.82, 0.62, 0.88, 0.85]])
    assert merge_positive_tiles(p >= 0.5) == [[0.0, 0.0, 1.0, 1.0]]
    boxes = merge_positive_tiles(p >= SCAN_YES_THRESHOLD)
    assert boxes == [[0.0, 0.0, 1.0, 0.25], [0.25, 0.5, 1.0, 1.0]]
