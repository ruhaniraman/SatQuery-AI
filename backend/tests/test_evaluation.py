"""The benchmark harness (evaluation/, run_eval.py): loaders on miniature copies of the real annotation layouts,
answer matching, metrics, and the runner with a fake predictor (no model, no GPU)."""
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.answer_space import AnswerSpace, benchmark_prompt, build_answer_spaces, normalize  # noqa: E402
from evaluation.benchmarks import EvalItem, load_benchmark, stratified_sample  # noqa: E402
from evaluation.metrics import calibration, majority_baseline, summarize, wilson_interval  # noqa: E402
from evaluation.runner import run_benchmark  # noqa: E402
import run_eval  # noqa: E402


def _write(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def _triplet(root, prefix, rows, inactive_extra=True):
    """rows: (qid, img_id, type, question, answer). Writes the RSVQA/CDVQA three-file layout, plus one inactive
    question (must be ignored) the way the real test split has them."""
    questions = [{"id": q, "img_id": i, "type": t, "question": text, "answers_ids": [q], "active": True}
                 for q, i, t, text, _ in rows]
    answers = [{"id": q, "question_id": q, "answer": a, "active": True} for q, _, _, _, a in rows]
    if inactive_extra:
        # Complete apart from active=False, so only the active flag can keep it out.
        questions.append({"id": 999, "img_id": rows[0][1], "type": rows[0][2], "question": "old?", "answers_ids": [999],
                          "active": False})
        answers.append({"id": 999, "question_id": 999, "answer": "yes", "active": True})
    images = [{"id": i, "file_name": f"{i:05d}.png", "active": True} for i in sorted({r[1] for r in rows})]
    _write(os.path.join(root, f"{prefix}_questions.json"), {"questions": questions})
    _write(os.path.join(root, f"{prefix}_answers.json"), {"answers": answers})
    _write(os.path.join(root, f"{prefix}_images.json"), {"images": images})


# ---------------------------------------------------------------- loaders

def test_rsvqa_loader_joins_the_three_files_and_skips_inactive(tmp_path):
    _triplet(str(tmp_path), "LR_split_test", [
        (1, 7, "presence", "Is there a road?", "yes"),
        (2, 7, "rural_urban", "Is it a rural or an urban area", "urban"),
    ])
    items = load_benchmark("rsvqa-lr", str(tmp_path))
    assert [(i.item_id, i.qtype, i.answer) for i in items] == [("1", "presence", "yes"), ("2", "rural_urban", "urban")]
    assert items[0].image_paths == [os.path.join(str(tmp_path), "Images_LR", "7.tif")]


def test_cdvqa_loader_builds_the_before_after_pair(tmp_path):
    _triplet(str(tmp_path), "Test", [(0, 3, "change_or_not", "Did the regions of trees change?", "no")])
    (item,) = load_benchmark("cdvqa", str(tmp_path), image_dir="SECOND")
    assert item.image_paths == [os.path.join("SECOND", "im1", "00003.png"), os.path.join("SECOND", "im2", "00003.png")]


def test_vrsbench_loader_reads_the_flat_list(tmp_path):
    _write(str(tmp_path / "VRSBench_EVAL_vqa.json"), [
        {"image_id": "P0003_0002.png", "question": "How many small vehicles?", "ground_truth": "2",
         "dataset": "RSBench", "question_id": 1, "type": "object quantity"},
    ])
    (item,) = load_benchmark("vrsbench-vqa", str(tmp_path))
    assert (item.item_id, item.answer, item.qtype) == ("1", "2", "object quantity")
    assert item.image_paths == [os.path.join(str(tmp_path), "Images_val", "P0003_0002.png")]


def test_loader_reports_a_wrong_file_shape(tmp_path):
    _write(str(tmp_path / "LR_split_test_questions.json"), {"items": []})
    _write(str(tmp_path / "LR_split_test_answers.json"), {"answers": []})
    _write(str(tmp_path / "LR_split_test_images.json"), {"images": []})
    with pytest.raises(ValueError, match="expected a 'questions' list"):
        load_benchmark("rsvqa-lr", str(tmp_path))


def _items(spec):
    """spec: {qtype: count} -> EvalItems with alternating yes/no answers."""
    out = []
    for qtype, count in spec.items():
        for k in range(count):
            out.append(EvalItem("rsvqa-lr", f"{qtype}-{k}", "q?", "yes" if k % 2 else "no", qtype, ["x.tif"]))
    return out


def test_stratified_sample_keeps_type_shares_and_is_reproducible():
    items = _items({"presence": 60, "comp": 30, "rural_urban": 10})
    sample = stratified_sample(items, 20, seed=3)
    counts = {t: sum(1 for i in sample if i.qtype == t) for t in ("presence", "comp", "rural_urban")}
    assert counts == {"presence": 12, "comp": 6, "rural_urban": 2}
    assert [i.item_id for i in stratified_sample(items, 20, seed=3)] == [i.item_id for i in sample]
    assert [i.item_id for i in stratified_sample(items, 20, seed=4)] != [i.item_id for i in sample]


def test_stratified_sample_gives_rare_types_one_item_and_never_exceeds_the_limit():
    items = _items({"common": 97, "rare1": 1, "rare2": 1, "rare3": 1})
    sample = stratified_sample(items, 5, seed=0)
    assert len(sample) == 5
    assert {i.qtype for i in sample} == {"common", "rare1", "rare2", "rare3"}
    assert stratified_sample(items, 0) == items and stratified_sample(items, 1000) == items


# ---------------------------------------------------------------- answer matching

def test_yes_no_space_reads_the_first_answer_word():
    space = build_answer_spaces([("presence", "yes"), ("presence", "no")])["presence"]
    assert space.instruction() == "Answer with yes or no only."
    assert space.is_correct("Yes.", "yes")
    assert space.is_correct("No, there is no water.", "no")
    assert not space.is_correct("Yes, but no roads", "no")
    assert not space.is_correct("I cannot tell", "yes")
    assert not space.is_correct("I do not know", "no")      # whole words only: "not"/"know" are not "no"


def test_numeric_space_accepts_digits_and_number_words():
    space = build_answer_spaces([("count", "0"), ("count", "12"), ("count", "3")])["count"]
    assert space.kind == "numeric"
    assert space.is_correct("12", "12") and space.is_correct("There are twelve.", "12")
    assert not space.is_correct("13", "12")
    assert space.is_correct("three", "3") and space.is_correct("There are 3 buildings.", "3")
    assert space.is_correct("None.", "0")


def test_cdvqa_classes_are_shown_as_words_and_mapped_back():
    pairs = [("change_to_what", a) for a in ("NVG_surface", "buildings", "low_vegetation", "trees", "water", "playgrounds")]
    space = build_answer_spaces(pairs)["change_to_what"]
    assert "non-vegetated ground surface" in space.instruction() and "NVG_surface" not in space.instruction()
    assert space.canonical("Non-vegetated ground surface.") == "NVG_surface"
    assert space.canonical("It changed to low vegetation") == "low_vegetation"
    # 'vegetation' alone must not be read as either vegetation class by accident.
    assert space.canonical("buildings") == "buildings"


def test_ratio_buckets_are_ordered_and_not_confused_with_each_other():
    pairs = [("change_ratio_types", a) for a in ("0", "0_to_10", "10_to_20", "20_to_30")]
    space = build_answer_spaces(pairs)["change_ratio_types"]
    assert space.options == ["0", "0_to_10", "10_to_20", "20_to_30"]
    assert "0%, 0-10%, 10-20%, 20-30%" in space.instruction()
    assert space.canonical("10-20%") == "10_to_20"
    assert space.canonical("0%") == "0"
    assert space.canonical("About 20-30% of the area") == "20_to_30"


def test_large_vocabularies_are_open_and_matched_strictly():
    pairs = [("object color", c) for c in "red blue green white black gray yellow orange brown pink purple teal cyan".split()]
    space = build_answer_spaces(pairs)["object color"]
    assert space.kind == "open"
    assert space.is_correct("White.", "white") and not space.is_correct("It is white", "white")
    assert AnswerSpace("open", []).is_correct("2 vehicles", "2")


def test_normalize_drops_articles_and_punctuation():
    assert normalize("  The Buildings! ") == "buildings"


def test_pair_prompt_states_the_date_order():
    space = AnswerSpace("closed", ["no", "yes"])
    prompt = benchmark_prompt("Did the regions of trees change", space, pair=True)
    assert prompt.startswith("The first image is the earlier date (before)")
    assert "change?" in prompt and prompt.endswith("Answer with yes or no only.")


# ---------------------------------------------------------------- metrics

def test_wilson_interval_matches_the_textbook_value():
    lo, hi = wilson_interval(8, 10)
    assert math.isclose(lo, 0.4902, abs_tol=1e-3) and math.isclose(hi, 0.9433, abs_tol=1e-3)
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_summary_counts_errors_as_wrong_and_averages_over_types():
    records = [
        {"qtype": "a", "correct": True}, {"qtype": "a", "correct": True},
        {"qtype": "b", "correct": False}, {"qtype": "b", "correct": False, "error": "boom"},
    ]
    s = summarize(records)
    assert s["overall"]["accuracy"] == 0.5 and s["errors"] == 1
    assert s["average_over_types"] == 0.5
    assert summarize(records, exclude_types_from_aa=["b"])["average_over_types"] == 1.0


def test_calibration_of_a_perfectly_calibrated_and_an_overconfident_model():
    good = [{"confidence": 0.75, "correct": k < 3} for k in range(4)]
    assert calibration(good)["ece"] == pytest.approx(0.0)
    bad = [{"confidence": 0.95, "correct": False} for _ in range(4)]
    assert calibration(bad)["ece"] == pytest.approx(0.95)
    assert calibration([{"confidence": None, "correct": True}]) is None


def test_majority_baseline_is_the_answer_prior():
    records = [{"qtype": "p", "answer": "yes"}] * 3 + [{"qtype": "p", "answer": "no"}]
    assert majority_baseline(records)["overall"]["accuracy"] == 0.75


# ---------------------------------------------------------------- runner

def _dataset(tmp_path, with_images=True):
    root = str(tmp_path / "data")
    _triplet(root, "LR_split_test", [
        (1, 1, "presence", "Is there a road?", "yes"),
        (2, 1, "presence", "Is there water?", "no"),
        (3, 2, "rural_urban", "Is it a rural or an urban area", "urban"),
        (4, 2, "count", "How many buildings are there?", "4"),
    ])
    if with_images:
        os.makedirs(os.path.join(root, "Images_LR"))
        for i in (1, 2):
            open(os.path.join(root, "Images_LR", f"{i}.tif"), "wb").close()
    return root


def _config(**over):
    return {"benchmark": "rsvqa-lr", "split": None, "limit": 0, "seed": 0, "predictor": "fake", "adapter": None, **over}


def test_runner_scores_writes_files_and_reports_the_paper_protocol(tmp_path):
    items = load_benchmark("rsvqa-lr", _dataset(tmp_path))
    replies = {"1": ("Yes.", 0.9), "2": ("Yes", 0.6), "3": ("urban", 0.8), "4": ("four", 0.4)}
    prompts = {}

    def fake(item, prompt):
        prompts[item.item_id] = prompt
        return replies[item.item_id]

    out = str(tmp_path / "out")
    summary = run_benchmark(items, fake, out, _config(), log=lambda _: None)
    assert summary["model"]["overall"]["correct"] == 3          # item 2 wrong
    assert summary["model"]["per_type"]["count"]["accuracy"] == 1.0
    assert summary["paper_protocol"]["excludes"] == ["area", "count"]
    assert summary["paper_protocol"]["model"]["overall"]["n"] == 3
    assert "Answer with a single number" in prompts["4"] and "rural, urban" not in prompts["1"]
    lines = [json.loads(line) for line in open(os.path.join(out, "predictions.jsonl"), encoding="utf-8")]
    assert {r["item_id"]: r["predicted"] for r in lines} == {"1": "yes", "2": "yes", "3": "urban", "4": "4"}
    md = open(os.path.join(out, "summary.md"), encoding="utf-8").read()
    assert "**Overall accuracy: 75.0%**" in md and "| presence | 2 | 50.0% |" in md
    assert json.load(open(os.path.join(out, "config.json")))["benchmark"] == "rsvqa-lr"


def test_runner_resumes_without_asking_again(tmp_path):
    items = load_benchmark("rsvqa-lr", _dataset(tmp_path))
    out = str(tmp_path / "out")
    calls = []

    def flaky(item, prompt):
        calls.append(item.item_id)
        if len(calls) == 3:
            raise KeyboardInterrupt        # the user stops the run part-way
        return ("yes", None)

    with pytest.raises(KeyboardInterrupt):
        run_benchmark(items, flaky, out, _config(), log=lambda _: None)
    calls.clear()
    summary = run_benchmark(items, lambda item, prompt: calls.append(item.item_id) or ("yes", None), out, _config(),
                            log=lambda _: None)
    assert calls == ["3", "4"]                  # the two answered before the stop are not asked again
    assert summary["model"]["overall"]["n"] == 4


def test_runner_refuses_to_mix_two_different_runs_in_one_folder(tmp_path):
    items = load_benchmark("rsvqa-lr", _dataset(tmp_path))
    out = str(tmp_path / "out")
    run_benchmark(items, lambda i, p: ("yes", None), out, _config(), log=lambda _: None)
    with pytest.raises(ValueError, match="holds a different run"):
        run_benchmark(items, lambda i, p: ("yes", None), out, _config(seed=1), log=lambda _: None)


def test_missing_images_and_predictor_crashes_are_recorded_as_wrong(tmp_path):
    items = load_benchmark("rsvqa-lr", _dataset(tmp_path, with_images=False))
    summary = run_benchmark(items, lambda i, p: ("yes", None), str(tmp_path / "out"), _config(), log=lambda _: None)
    assert summary["model"]["errors"] == 4 and summary["model"]["overall"]["correct"] == 0
    first = json.loads(open(os.path.join(str(tmp_path / "out"), "predictions.jsonl")).readline())
    assert first["error"].startswith("FileNotFoundError: image not found")


def test_cli_prior_run_needs_no_model(tmp_path, monkeypatch, capsys):
    root = _dataset(tmp_path)
    monkeypatch.chdir(tmp_path)
    summary = run_eval.main(["rsvqa-lr", "--root", root, "--limit", "0", "--predictor", "prior"])
    # prior answers: presence -> most common of {yes, no} (tie: either), rural_urban -> urban, count -> 4
    assert summary["model"]["per_type"]["rural_urban"]["accuracy"] == 1.0
    assert os.path.exists(os.path.join("eval_results", "rsvqa-lr-default-prior-nall-s0", "summary.md"))
    assert "Overall accuracy" in capsys.readouterr().out


def test_cli_exits_readably_when_the_files_are_missing(tmp_path):
    with pytest.raises(SystemExit, match="Could not load rsvqa-lr"):
        run_eval.main(["rsvqa-lr", "--root", str(tmp_path / "nope"), "--predictor", "prior"])
