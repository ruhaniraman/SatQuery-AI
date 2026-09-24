"""Which chat questions have a short, checkable answer (pure).

Yes/no questions, counts and "rural or urban" are the question kinds the VQA LoRA was trained on (RSVQA-LR, see
models/adapters/vqa/training_info.json). For those, /analyze first asks the adapted model for the short answer (its
confidence is the probability of that answer) and then lets the general path explain it. Anything else ("describe",
"what", "where", "which", either/or questions) returns None and goes to the general path unchanged.
"""
import re
from typing import Optional

from evaluation.answer_space import AnswerSpace

_YES_NO_OPENERS = {
    "is", "are", "was", "were", "does", "do", "did", "can", "could", "has", "have", "had", "will", "would", "should",
}
_COUNT = re.compile(r"^(how many|what is the (number|amount|count) of|what's the number of|number of|count (the|how))\b")
_RURAL_URBAN = re.compile(r"\b(rural|urban)\b.*\b(rural|urban)\b")


def closed_question_space(question: str) -> Optional[AnswerSpace]:
    text = re.sub(r"\s+", " ", question.strip().lower())
    if not text:
        return None
    if _RURAL_URBAN.search(text):
        return AnswerSpace("closed", ["rural", "urban"])
    if _COUNT.match(text):
        return AnswerSpace("numeric", [])
    first = re.split(r"[\s,?]", text, maxsplit=1)[0]
    # "Is it forest or farmland?" asks to choose, not yes/no.
    if first in _YES_NO_OPENERS and not re.search(r"\bor\b", text):
        return AnswerSpace("closed", ["no", "yes"])
    return None


def display_answer(space: AnswerSpace, reply: str) -> Optional[str]:
    """The adapted model's reply as shown to the user ('Yes', '12', 'Urban'), or None if it gave no usable answer
    (then only the general explanation is shown)."""
    value = space.canonical(reply)
    if space.kind == "numeric":
        return value if value.isdigit() else None
    if value in space.options:
        return value.capitalize()
    return None
