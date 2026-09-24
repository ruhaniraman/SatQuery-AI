"""Loaders for the public benchmarks SIH26167 names (RSVQA, VRSBench, CDVQA). Pure: json + os only, so the
tests and a dry run need no model. Each loader returns EvalItems; image files are resolved to paths but not
opened (a missing image is reported by the runner per item, not by the loader).

File layouts (checked against the real downloads, Sept 2026):
- RSVQA-LR / RSVQA-HR (zenodo.org/records/6344334 and 6344367): `<prefix>_split_<split>_{questions,answers,images}.json`,
  each a dict with one list under "questions" / "answers" / "images". A question has id, img_id, type, question,
  answers_ids, active; an answer has id, question_id, answer, active. Only active entries count (the test split
  file lists 33,212 questions, 10,004 of them active). Images are `<image_dir>/<img_id>.tif`.
- CDVQA (github.com/YZHJessica/CDVQA): the same three-file layout, named `<Split>_{questions,answers,images}.json`
  (Train, Val, Test, Test2). An image entry has file_name (e.g. "07308.png"); the pair is the SECOND dataset's
  `im1/<file_name>` (before) and `im2/<file_name>` (after), which are NOT in that repo (downloaded separately).
- VRSBench (huggingface.co/datasets/xiang709/VRSBench): `VRSBench_EVAL_vqa.json` is a plain list of
  {image_id, question, ground_truth, question_id, type, dataset}; images are in Images_val.zip.
"""
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

BENCHMARKS = ("rsvqa-lr", "rsvqa-hr", "cdvqa", "vrsbench-vqa")


@dataclass
class EvalItem:
    benchmark: str
    item_id: str
    question: str
    answer: str
    qtype: str
    image_paths: List[str]                     # one image, or [before, after] for a pair
    meta: Dict[str, str] = field(default_factory=dict)


def _load_list(path: str, key: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if key not in data:
        raise ValueError(f"{path}: expected a '{key}' list, found keys {sorted(data)}")
    return data[key]


def _triplet(questions_path: str, answers_path: str, images_path: str):
    """The RSVQA/CDVQA three-file layout, joined: yields (question, answer_text, image_entry) for active
    questions whose answer and image are present and active."""
    answers = {a["id"]: a for a in _load_list(answers_path, "answers") if a.get("active", True)}
    images = {i["id"]: i for i in _load_list(images_path, "images") if i.get("active", True)}
    for q in _load_list(questions_path, "questions"):
        if not q.get("active", True):
            continue
        image = images.get(q.get("img_id"))
        answer_ids = [a for a in q.get("answers_ids", []) if a in answers]
        if image is None or not answer_ids:
            continue
        yield q, answers[answer_ids[0]]["answer"], image


def load_rsvqa(root: str, variant: str = "lr", split: str = "test", image_dir: Optional[str] = None) -> List[EvalItem]:
    prefix = variant.upper()
    base = os.path.join(root, f"{prefix}_split_{split}")
    image_dir = image_dir or os.path.join(root, f"Images_{prefix}")
    items = []
    for q, answer, image in _triplet(f"{base}_questions.json", f"{base}_answers.json", f"{base}_images.json"):
        items.append(EvalItem(
            benchmark=f"rsvqa-{variant.lower()}", item_id=str(q["id"]), question=q["question"], answer=str(answer),
            qtype=q.get("type", "unknown"), image_paths=[os.path.join(image_dir, f"{image['id']}.tif")],
        ))
    return items


def load_cdvqa(root: str, split: str = "Test", image_root: Optional[str] = None) -> List[EvalItem]:
    image_root = image_root or os.path.join(root, "images", split)
    base = os.path.join(root, split)
    items = []
    for q, answer, image in _triplet(f"{base}_questions.json", f"{base}_answers.json", f"{base}_images.json"):
        name = image["file_name"]
        items.append(EvalItem(
            benchmark="cdvqa", item_id=str(q["id"]), question=q["question"], answer=str(answer),
            qtype=q.get("type", "unknown"),
            image_paths=[os.path.join(image_root, "im1", name), os.path.join(image_root, "im2", name)],
            meta={"pair": name},
        ))
    return items


def load_vrsbench_vqa(eval_json: str, image_dir: str) -> List[EvalItem]:
    with open(eval_json, encoding="utf-8") as f:
        rows = json.load(f)
    return [
        EvalItem(
            benchmark="vrsbench-vqa", item_id=str(r["question_id"]), question=r["question"], answer=str(r["ground_truth"]),
            qtype=r.get("type", "unknown"), image_paths=[os.path.join(image_dir, r["image_id"])],
        )
        for r in rows
    ]


def load_benchmark(name: str, root: str, split: Optional[str] = None, image_dir: Optional[str] = None) -> List[EvalItem]:
    if name in ("rsvqa-lr", "rsvqa-hr"):
        return load_rsvqa(root, name.split("-")[1], split or "test", image_dir)
    if name == "cdvqa":
        return load_cdvqa(root, split or "Test", image_dir)
    if name == "vrsbench-vqa":
        return load_vrsbench_vqa(os.path.join(root, "VRSBench_EVAL_vqa.json"), image_dir or os.path.join(root, "Images_val"))
    raise ValueError(f"Unknown benchmark {name!r}; choose from {', '.join(BENCHMARKS)}")


def stratified_sample(items: List[EvalItem], limit: Optional[int], seed: int = 0) -> List[EvalItem]:
    """A reproducible subset of `limit` items that keeps each question type's share (every type gets at least
    one item when limit allows), so a quick run is not, say, all yes/no questions. None/0 or a limit at least
    the size of the set returns everything, in the original order."""
    if not limit or limit >= len(items):
        return list(items)
    by_type = defaultdict(list)
    for item in items:
        by_type[item.qtype].append(item)
    rng = random.Random(seed)
    quotas = {t: max(1, round(limit * len(group) / len(items))) for t, group in by_type.items()}
    # Rounding can overshoot: take the excess back from the largest quotas (never below one).
    while sum(quotas.values()) > limit:
        biggest = max((t for t in quotas if quotas[t] > 1), key=lambda t: quotas[t], default=None)
        if biggest is None:
            break
        quotas[biggest] -= 1
    chosen = set()
    for t, group in by_type.items():
        for item in rng.sample(group, min(quotas[t], len(group))):
            chosen.add(id(item))
    return [item for item in items if id(item) in chosen]
