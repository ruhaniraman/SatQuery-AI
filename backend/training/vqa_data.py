"""Training examples for the VQA LoRA (pure: no torch). The prompts are EXACTLY the evaluation harness's
(evaluation.answer_space.benchmark_prompt with the split's answer vocabulary), so the adapter learns the format it
is scored in, and the target is the answer as the harness shows it to the model (readable() for closed answers).

Several questions about one image are packed into one multi-turn conversation (the image only in the first user
turn), so the ~250 visual tokens are processed once per pack instead of once per question."""
import random
from collections import defaultdict
from typing import Dict, List, Sequence

from evaluation.answer_space import AnswerSpace, benchmark_prompt, readable
from evaluation.benchmarks import EvalItem


def target_text(answer: str, space: AnswerSpace) -> str:
    if space.kind == "closed":
        return readable(answer)
    if space.kind == "numeric":
        return str(int(answer))
    return answer


def build_packs(items: Sequence[EvalItem], spaces: Dict[str, AnswerSpace], per_pack: int = 8, seed: int = 0) -> List[dict]:
    """Group by image, shuffle each image's questions, cut into packs of `per_pack`, shuffle the packs.
    Returns [{"image_paths": [...], "turns": [(prompt, target), ...]}]; every question appears exactly once."""
    rng = random.Random(seed)
    by_image = defaultdict(list)
    for item in items:
        by_image[tuple(item.image_paths)].append(item)
    packs = []
    for image_paths in sorted(by_image):
        group = by_image[image_paths]
        rng.shuffle(group)
        for start in range(0, len(group), per_pack):
            chunk = group[start:start + per_pack]
            turns = []
            for item in chunk:
                space = spaces[item.qtype]
                turns.append((benchmark_prompt(item.question, space, pair=len(image_paths) == 2), target_text(item.answer, space)))
            packs.append({"image_paths": list(image_paths), "turns": turns})
    rng.shuffle(packs)
    return packs


def pack_messages(pack: dict) -> List[dict]:
    """Chat messages for one pack: images + first prompt in the first user turn, text-only turns after."""
    messages = []
    for k, (prompt, target) in enumerate(pack["turns"]):
        content = ([{"type": "image"} for _ in pack["image_paths"]] if k == 0 else []) + [{"type": "text", "text": prompt}]
        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return messages


def assistant_label_mask(input_ids: Sequence[int], im_start: int, im_end: int, assistant_header: Sequence[int]) -> List[bool]:
    """True for the tokens the loss is computed on: every assistant reply's tokens plus its closing <|im_end|>
    (so the model learns to stop). A reply starts after `<|im_start|>` + the tokens of "assistant\\n"."""
    mask = [False] * len(input_ids)
    header = list(assistant_header)
    i = 0
    n = len(input_ids)
    while i < n:
        if input_ids[i] == im_start and list(input_ids[i + 1:i + 1 + len(header)]) == header:
            j = i + 1 + len(header)
            while j < n and input_ids[j] != im_end:
                mask[j] = True
                j += 1
            if j < n:
                mask[j] = True          # the <|im_end|> itself
            i = j + 1
        else:
            i += 1
    return mask
