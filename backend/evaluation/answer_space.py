"""Short-answer prompts and answer matching for benchmark questions (pure).

Benchmarks score an exact answer ("yes", "urban", "12", "NVG_surface", "10_to_20"), while the product's
general path writes a paragraph. For each question type we build an AnswerSpace from the answers that
occur for that type in the WHOLE loaded split (the benchmark's fixed answer vocabulary, as a classifier head
would have; never the answer of the item being asked). A small closed vocabulary becomes "answer with
exactly one of: ..."; an all-numeric one becomes "answer with a single number"; anything else is open
("a single word or short phrase"). The model's reply is then mapped back onto the vocabulary.
"""
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

MAX_CLOSED_OPTIONS = 12

# CDVQA's answer codes, shown to the model as plain words and mapped back.
_READABLE = {
    "NVG_surface": "non-vegetated ground surface",
    "low_vegetation": "low vegetation",
}
_NUMBER_WORDS = {
    "zero": 0, "none": 0, "no": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}
_RATIO = re.compile(r"^(\d+)_to_(\d+)$")


def normalize(text: str) -> str:
    """Lowercase, drop punctuation (keeping digits, letters, '_' and '%'), articles and extra spaces."""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s%]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def readable(code: str) -> str:
    """How an answer code is shown to the model: CDVQA's class codes as words, ratio buckets as ranges."""
    if code in _READABLE:
        return _READABLE[code]
    m = _RATIO.match(code)
    if m:
        return f"{m.group(1)}-{m.group(2)}%"
    if code == "0":
        return "0%"
    return code


def _is_number(text: str) -> bool:
    return re.fullmatch(r"\d+", normalize(text)) is not None


@dataclass
class AnswerSpace:
    kind: str                       # "closed" | "numeric" | "open"
    options: List[str]              # the answer codes (closed only)

    def instruction(self) -> str:
        if self.kind == "closed":
            if sorted(normalize(o) for o in self.options) == ["no", "yes"]:
                return "Answer with yes or no only."
            shown = ", ".join(readable(o) for o in self.options)
            return f"Answer with exactly one of: {shown}. Reply with the answer only."
        if self.kind == "numeric":
            return "Answer with a single number in digits only."
        return "Answer with a single word or a short phrase only."

    def canonical(self, reply: str) -> str:
        """Map a model reply onto this space. Closed: the option whose code or readable form matches exactly, else
        the one whose readable form appears earliest in the reply (longest wins a tie, so '10-20%' is not read as
        '0%'). Numeric: the first number, in digits or as a word. Open: the normalized reply."""
        text = normalize(reply)
        if self.kind == "numeric":
            m = re.search(r"\d+", text)
            if m:
                return str(int(m.group(0)))
            for word in text.split():
                if word in _NUMBER_WORDS:
                    return str(_NUMBER_WORDS[word])
            return text
        if self.kind == "closed":
            forms = {o: {normalize(o), normalize(readable(o))} for o in self.options}
            for option, names in forms.items():
                if text in names:
                    return option
            best = None
            for option, names in forms.items():
                for name in names:
                    m = re.search(rf"(?<![\w-]){re.escape(name)}(?![\w-])", text)
                    if m and (best is None or (m.start(), -len(name)) < (best[0], -best[1])):
                        best = (m.start(), len(name), option)
            return best[2] if best else text
        return text

    def is_correct(self, reply: str, answer: str) -> bool:
        predicted = self.canonical(reply)
        if self.kind == "closed":
            return predicted == answer or normalize(predicted) == normalize(answer)
        if self.kind == "numeric":
            return predicted == str(int(normalize(answer))) if _is_number(answer) else predicted == normalize(answer)
        # Open answers (VRSBench): strict match after normalization, numbers compared as numbers. The official
        # VRSBench VQA score is GPT-judged (synonyms count), so this is a LOWER bound, not the official metric.
        gold = normalize(answer)
        if gold.isdigit():
            return AnswerSpace("numeric", []).canonical(reply) == str(int(gold))
        return predicted == gold


def build_answer_spaces(pairs: Iterable) -> Dict[str, AnswerSpace]:
    """pairs: (qtype, answer) for every item of the loaded split."""
    vocab = defaultdict(set)
    for qtype, answer in pairs:
        vocab[qtype].add(str(answer))
    spaces = {}
    for qtype, answers in vocab.items():
        if all(_is_number(a) for a in answers):
            spaces[qtype] = AnswerSpace("numeric", [])
        elif len({normalize(a) for a in answers}) <= MAX_CLOSED_OPTIONS:
            spaces[qtype] = AnswerSpace("closed", sorted(answers, key=_option_order))
        else:
            spaces[qtype] = AnswerSpace("open", [])
    return spaces


def _option_order(code: str):
    """Ratio buckets in numeric order, everything else alphabetically."""
    m = _RATIO.match(code)
    if m:
        return (0, int(m.group(1)), "")
    if code == "0":
        return (0, -1, "")
    return (1, 0, normalize(code))


def benchmark_prompt(question: str, space: AnswerSpace, pair: bool = False, note: Optional[str] = None) -> str:
    """The text sent with the image(s). For a pair the first image is the earlier date."""
    lead = ("The first image is the earlier date (before) and the second image is the later date (after) "
            "of the same area. " if pair else "")
    question = question.strip()
    if not question.endswith("?"):
        question += "?"
    parts = [lead + question, space.instruction()]
    if note:
        parts.append(note)
    return "\n".join(parts)
