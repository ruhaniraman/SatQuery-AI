"""The single-image general (free-text) VQA path's clarification gate: pure, no heavy imports (no
torch/transformers), so it is safe to import directly at test module scope. See main_api.py's use of
needs_clarification() for how this is wired into /analyze."""
import re
from typing import Optional

# A free-text question this vague, with no prior turn in the conversation to supply the missing
# context, is asked instead of guessed at (saves a GPU call too). Deliberately a short, exact-match
# list: a real question - even a short or wrong one, like "water?" - must still reach the model. Only
# an unmistakably empty opener (a bare greeting, filler, or dangling pronoun) is caught here.
CLARIFICATION_TRIGGERS = {
    "hi", "hii", "hiii", "hey", "heyy", "hello", "yo", "sup", "hola", "namaste",
    "test", "testing", "ok", "okay", "k", "cool", "nice", "thanks", "thank you",
    # These grammatically presuppose something said earlier ("it", "this" with no noun in the query
    # itself, or "more"/"go on" continuing a reply that never happened) - unlike "what is this?" or
    # "explain this", which stand on their own as a question about the whole image.
    "and this", "go on", "more info", "tell me more", "what about it",
}


def needs_clarification(query: str, history: list) -> Optional[str]:
    """None if the query should go to the model as normal; otherwise the clarification message to show
    instead. Only applies to a conversation's opening question (any prior turn means the model already
    has context for a short follow-up like "and this?")."""
    if history:
        return None
    text = re.sub(r"[?!.,\s]+$", "", query.strip().lower())
    text = re.sub(r"^[?!.,\s]+", "", text)
    if not text:
        return "Please type a question about the image."
    if text in CLARIFICATION_TRIGGERS:
        return ("Could you ask something more specific about the image — the land cover, a feature you "
                "can see, or a comparison to something?")
    return None
