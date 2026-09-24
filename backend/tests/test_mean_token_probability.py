"""_mean_token_probability's real math. Importing agent_manager.agent_controller needs the real
torch/transformers/peft packages (skipped where they are not installed, e.g. CI's light dependency
set - see .github/workflows/ci.yml), but this file never calls real torch at TEST-execution time: it
patches the `torch` name inside agent_controller's own namespace with a tiny softmax-only stand-in.
That sidesteps a real, pre-existing fragility elsewhere in this suite (test_endpoint_wiring.py's
`client` fixture does a raw sys.modules.pop("torch", ...) that is not tracked/restored by monkeypatch,
so a later genuine `import torch` can re-run torch's C-extension init mid-process, which torch does
not support - "Only a single TORCH_LIBRARY can be used..." - unrelated to the logic under test here)."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
agent_controller = pytest.importorskip("agent_manager.agent_controller")

_mean_token_probability = agent_controller._mean_token_probability


class FakeScalar:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


class FakeRow:
    """One generation step's per-vocab values (post- or pre-softmax)."""
    def __init__(self, values):
        self.values = list(values)

    def float(self):
        return self

    def __getitem__(self, i):
        return FakeScalar(self.values[i])


class FakeIds:
    def __init__(self, ids):
        self.ids = list(ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return FakeScalar(self.ids[i])


class FakeTorch:
    @staticmethod
    def softmax(row, dim=-1):
        m = max(row.values)
        exps = [math.exp(v - m) for v in row.values]
        s = sum(exps)
        return FakeRow([e / s for e in exps])


@pytest.fixture(autouse=True)
def fake_torch(monkeypatch):
    monkeypatch.setattr(agent_controller, "torch", FakeTorch)


def test_confident_generation_scores_near_one():
    logits = FakeRow([-10.0, -10.0, 10.0, -10.0, -10.0])   # token 2 holds almost all the mass
    conf = _mean_token_probability([[logits]], FakeIds([2]))
    assert conf > 0.99


def test_uniform_logits_score_at_chance():
    logits = FakeRow([0.0] * 5)
    conf = _mean_token_probability([[logits]], FakeIds([0]))
    assert abs(conf - 0.2) < 1e-4   # 1-in-5, uniform


def test_mean_is_taken_over_every_generation_step():
    near_certain = FakeRow([10.0, -10.0, -10.0])
    uniform = FakeRow([0.0, 0.0, 0.0])
    conf = _mean_token_probability([[near_certain], [uniform]], FakeIds([0, 1]))
    assert 0.6 < conf < 0.7   # mean of ~1.0 and ~0.333


def test_nothing_generated_or_no_scores_returns_none():
    assert _mean_token_probability([], FakeIds([1])) is None
    assert _mean_token_probability([[FakeRow([0.0, 0.0, 0.0])]], FakeIds([])) is None
    assert _mean_token_probability(None, FakeIds([1])) is None
