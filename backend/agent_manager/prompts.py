"""Prompt/message construction for the single-image agent. No torch/transformers imports, so it is
unit-testable without the model.

Design notes (token counts measured with the real Qwen tokenizer, see CLAUDE.md):
  * The two-part OBSERVATIONS -> ASSESSMENT reply is kept on purpose: describing before answering is
    what keeps a 2B model grounded in the image. It is stated once, tersely.
  * Every free-text prompt asks for a bounded reply. Output tokens are generated one at a time, so a
    length cap in the prompt saves far more than trimming prompt words does.
  * Amounts are requested in words ("about half"), never as numbers: the model cannot measure areas
    and precise-looking figures would be invented.
  * The LoRA yes/no question templates in agent_controller are deliberately NOT touched here: they
    are tied to how the adapters were fine-tuned, which the repo does not record.
"""
from typing import Any, Dict, List, Optional

# Ceiling for every free-text path (single-image, change detection, fusion). It is only a ceiling:
# the prompts ask for short replies. 256 used to truncate the OBSERVATIONS / ASSESSMENT structure.
DEFAULT_MAX_NEW_TOKENS = 1024

# Word budget requested in the prompts (the model is asked, not forced, to stay under it).
REPLY_WORD_LIMIT = 120

MAX_HISTORY_TURNS = 6
# Prior turns are context, not the answer: cap each so long replies don't dominate the prompt.
MAX_HISTORY_CHARS_PER_TURN = 500

_SAR_NOTE = (
    " This is a SAR (radar) image: brightness is backscatter, not colour. Bright = strong reflectors"
    " (buildings, rough ground); dark = smooth surfaces or calm water. Do not describe colours."
)


def general_system_prompt(modality: str = "optical") -> str:
    prompt = (
        "You are a satellite-imagery analyst. Use only what is visible; never assume a feature "
        f"exists. Reply in two parts, under {REPLY_WORD_LIMIT} words total.\n"
        "OBSERVATIONS: visible features relevant to the question; only those present.\n"
        "ASSESSMENT: the answer, from those observations only; say so if the image cannot answer. "
        "Amounts in words (\"about half\"), not numbers. No generic advice, place names or dates."
    )
    return prompt + (_SAR_NOTE if modality == "sar" else "")


def _clip(text: str, limit: int = MAX_HISTORY_CHARS_PER_TURN) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _compress_assistant_turn(text: str) -> str:
    """A follow-up needs the earlier conclusion, not the whole description: keep from ASSESSMENT on."""
    marker = text.find("ASSESSMENT:")
    return _clip(text[marker:] if marker != -1 else text)


def history_for_model(history: Any, current_prompt: str, max_turns: int = MAX_HISTORY_TURNS) -> List[Dict[str, str]]:
    """Turn the UI's chat_history into compact, model-ready text turns.

    The UI history (a) already ends with the CURRENT user message, (b) uses role "ai", and (c)
    contains synthetic system entries for button scans. Those are dropped/mapped here so the model
    sees each real turn once, only the most recent `max_turns`, each capped in length (assistant
    turns are reduced to their ASSESSMENT).
    """
    if not isinstance(history, list):
        return []
    turns: List[Dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        role = {"user": "user", "ai": "assistant", "assistant": "assistant"}.get(item.get("role"))
        if role is None or not isinstance(content, str) or not content.strip():
            continue
        if content.lstrip().startswith("[System]"):
            continue
        turns.append({"role": role, "content": content})

    if turns and turns[-1]["role"] == "user" and turns[-1]["content"].strip() == current_prompt.strip():
        turns.pop()
    return [
        {"role": t["role"], "content": _compress_assistant_turn(t["content"]) if t["role"] == "assistant" else _clip(t["content"])}
        for t in turns[-max_turns:]
    ]


def build_general_messages(
    prompt: str,
    has_image: bool,
    history: Optional[list] = None,
    modality: str = "optical",
) -> List[Dict[str, Any]]:
    """System role + prior turns (text only) + the current question, with the image attached to the
    current question."""
    messages: List[Dict[str, Any]] = [{"role": "system", "content": general_system_prompt(modality)}]
    messages.extend(history_for_model(history, prompt))
    if has_image:
        messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]})
    else:
        messages.append({"role": "user", "content": prompt})
    return messages
