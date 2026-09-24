"""training/vqa_data.py: packing, targets, and the loss mask (pure; the training script itself needs a GPU)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation.answer_space import benchmark_prompt, build_answer_spaces  # noqa: E402
from evaluation.benchmarks import EvalItem  # noqa: E402
from training.vqa_data import assistant_label_mask, build_packs, pack_messages, target_text  # noqa: E402

IM_START, IM_END, HEADER = 1, 2, [7, 8]      # stand-ins for <|im_start|>, <|im_end|>, "assistant\n"


def _items():
    out = []
    for img in range(3):
        for k in range(10):
            qtype, answer = ("count", str(k)) if k % 2 else ("presence", "yes" if k % 4 else "no")
            out.append(EvalItem("rsvqa-lr", f"{img}-{k}", f"Q{img}-{k}?", answer, qtype, [f"{img}.tif"]))
    return out


def test_every_question_is_packed_once_with_its_own_image():
    items = _items()
    spaces = build_answer_spaces((i.qtype, i.answer) for i in items)
    packs = build_packs(items, spaces, per_pack=4, seed=0)
    assert len(packs) == 3 * 3                       # 10 questions per image -> 4 + 4 + 2
    seen = []
    for pack in packs:
        (path,) = pack["image_paths"]
        for prompt, _ in pack["turns"]:
            question = prompt.split("\n")[0]
            assert question.startswith(f"Q{path[0]}-")       # never mixed with another image's questions
            seen.append(question)
    assert sorted(seen) == sorted(i.question for i in items)
    assert build_packs(items, spaces, per_pack=4, seed=0) == packs
    assert build_packs(items, spaces, per_pack=4, seed=1) != packs


def test_prompts_and_targets_match_what_the_harness_asks_and_accepts():
    items = _items()
    spaces = build_answer_spaces((i.qtype, i.answer) for i in items)
    packs = build_packs(items, spaces, per_pack=10)
    by_question = {i.question: i for i in items}
    for pack in packs:
        for prompt, target in pack["turns"]:
            item = by_question[prompt.split("\n")[0]]
            space = spaces[item.qtype]
            assert prompt == benchmark_prompt(item.question, space)
            assert space.is_correct(target, item.answer)     # the target itself scores as correct


def test_cdvqa_codes_are_trained_as_the_words_the_prompt_shows():
    spaces = build_answer_spaces([("change_to_what", a) for a in ("NVG_surface", "buildings", "water")])
    assert target_text("NVG_surface", spaces["change_to_what"]) == "non-vegetated ground surface"


def test_only_the_first_turn_carries_the_image():
    pack = {"image_paths": ["a.tif", "b.tif"], "turns": [("q1", "yes"), ("q2", "3")]}
    msgs = pack_messages(pack)
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant"]
    assert [c["type"] for c in msgs[0]["content"]] == ["image", "image", "text"]
    assert [c["type"] for c in msgs[2]["content"]] == ["text"]
    assert msgs[3]["content"][0]["text"] == "3"


def test_loss_mask_covers_each_reply_and_its_end_token_only():
    # system ... <im_end> | user ... <im_end> | <im_start> assistant\n 50 51 <im_end> | user | assistant\n 60 <im_end>
    ids = [IM_START, 9, 30, IM_END, IM_START, 5, 31, IM_END,
           IM_START, 7, 8, 50, 51, IM_END, 10,
           IM_START, 5, 32, IM_END, IM_START, 7, 8, 60, IM_END, 10]
    mask = assistant_label_mask(ids, IM_START, IM_END, HEADER)
    assert [t for t, m in zip(ids, mask) if m] == [50, 51, IM_END, 60, IM_END]


def test_loss_mask_ignores_a_user_turn_that_mentions_the_header_tokens():
    ids = [IM_START, 5, 7, 8, 40, IM_END]          # "user" then tokens that look like the header mid-turn
    assert not any(assistant_label_mask(ids, IM_START, IM_END, HEADER))
