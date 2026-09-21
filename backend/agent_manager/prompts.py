"""Prompt/message construction for the single-image agent. No torch/transformers imports, so it is
unit-testable without the model.

Design notes (token counts measured with the real Qwen tokenizer, see CLAUDE.md):
  * The two-part OBSERVATIONS -> ASSESSMENT reply is kept on purpose: describing before answering is
    what keeps a 2B model grounded in the image.
  * Replies used to be capped at 120 words with "say so if the image cannot answer" and a ban on
    anything that sounded generic; that made a 2B model brief and evasive. The cap is now roomier
    and the prompt asks for concrete, located detail instead. Output tokens still cost latency.
  * Amounts are requested in words ("about half"). The only numbers the model may quote are the
    ones in the "Measured facts" block, which are computed from the pixels (agent_manager.scene_stats).
  * The general path can run a describe-first pass (whole image + quadrants) whose text is given to
    the answering pass. It is a prompt-side aid for a small model, not new evidence: the prompt says so.
  * The LoRA yes/no question templates in agent_controller are deliberately NOT touched here: they
    are tied to how the adapters were fine-tuned, which the repo does not record.
"""
import os
from typing import Any, Dict, List, Optional

# Ceiling for every free-text path (single-image, change detection, fusion). It is only a ceiling:
# the prompts ask for bounded replies. 256 used to truncate the OBSERVATIONS / ASSESSMENT structure.
DEFAULT_MAX_NEW_TOKENS = 1024

# Word budget requested in the prompts (the model is asked, not forced, to stay under it).
REPLY_WORD_LIMIT = 200

MAX_HISTORY_TURNS = 6
# Prior turns are context, not the answer: cap each so long replies don't dominate the prompt.
MAX_HISTORY_CHARS_PER_TURN = 500

# Token ceilings for the describe-first pass (whole image, then one line per quadrant).
DESCRIBE_MAX_NEW_TOKENS = 220
QUADRANT_MAX_NEW_TOKENS = 100

_SAR_NOTE = (
    " This is a SAR (radar) image: brightness is backscatter, not colour. Bright = strong reflectors"
    " (buildings, rough ground); dark = smooth surfaces or calm water. Do not describe colours."
)

_STYLE_EXAMPLE = (
    "Style example.\n"
    "Q: What is in this image?\n"
    "OBSERVATIONS: Upper half: a wide river with a sandbar, banks lined with dense trees. Lower-left: "
    "a grid of rectangular fields, some green, some bare brown. Lower-right: a small cluster of "
    "buildings along a straight road.\n"
    "ASSESSMENT: A river valley with mixed farmland and a small settlement; most of the land is "
    "cultivated, and the river is the dominant feature."
)


def describe_first_enabled() -> bool:
    """Env DESCRIBE_FIRST (default on): run a describe pass before answering on the general path."""
    return os.environ.get("DESCRIBE_FIRST", "1").strip().lower() not in {"0", "false", "no", "off"}


def general_system_prompt(modality: str = "optical") -> str:
    prompt = (
        "You are an expert satellite-imagery analyst. Be specific and concrete: name the land-cover "
        "types you see (water, forest, cropland, bare ground, buildings, roads...), say WHERE they are "
        "(upper-left, centre, along the bottom edge...), and note shapes, textures, colours and "
        "patterns. Never assume a feature exists; if you are unsure of one, say it is unclear rather "
        "than guessing. "
        f"Reply in two parts, under {REPLY_WORD_LIMIT} words total.\n"
        "OBSERVATIONS: the visible features relevant to the question, each with its location.\n"
        "ASSESSMENT: a direct answer to the question, from those observations; if the image cannot "
        "answer it, say what it does show. Amounts in words (\"about half\", \"a handful\"). Quote "
        "a number only if it is in the Measured facts.\n"
        + _STYLE_EXAMPLE
    )
    return prompt + (_SAR_NOTE if modality == "sar" else "")


# ---- describe-first pass ---------------------------------------------------------------------

QUADRANT_NAMES = ("upper-left", "upper-right", "lower-left", "lower-right")


def describe_scene_prompt(modality: str = "optical") -> str:
    """Whole-image description request for the first pass (adapters disabled by the caller)."""
    extra = " Describe brightness patterns, not colours." if modality == "sar" else ""
    return (
        "Describe this satellite image in detail: the main land-cover types, their layout (where each "
        "is), water, vegetation, built-up areas, roads or other linear features, and any unusual "
        "structures. Only what is visible; about 120 words." + extra
    )


def describe_region_prompt(location: str, modality: str = "optical") -> str:
    """Crop of one area flagged by the pixel analysis (used per date in change detection).

    Deliberately does NOT name things to look for: a prompt that listed "water, buildings, roads" made the
    2B model list exactly those, present or not. It also asks for prose, because it otherwise answers in a
    numbered list of headings that then have to be stripped."""
    extra = " Describe brightness patterns, not colours." if modality == "sar" else ""
    return (
        f"This is a crop of the {location} area of a satellite image. In two to four sentences of plain prose "
        "(no list, no numbering), describe the main ground cover, its texture and tone, and anything "
        "distinctive about it. Mention only what you can actually see; do not say what is absent." + extra
    )


def describe_quadrant_prompt(name: str, modality: str = "optical") -> str:
    extra = " Describe brightness, not colours." if modality == "sar" else ""
    return (
        f"This is the {name} part of a satellite image. In two or three short sentences say what "
        "land cover and features are visible. Only what is visible." + extra
    )


def format_scene_description(overview: str, quadrants: Optional[Dict[str, str]] = None) -> str:
    lines = [overview.strip()] if overview and overview.strip() else []
    for name in QUADRANT_NAMES:
        text = (quadrants or {}).get(name, "")
        if text and text.strip():
            lines.append(f"{name}: {' '.join(text.split())}")
    return "\n".join(lines)


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


def build_question_text(prompt: str, scene_facts: str = "", scene_description: str = "") -> str:
    """The current user turn: the question, preceded by any measured facts / first-pass description."""
    blocks = []
    if scene_facts:
        blocks.append(f"Measured facts (computed from the pixels; they describe pixels, not materials): {scene_facts}")
    if scene_description:
        blocks.append(
            "Notes from a first look at the image and its quadrants (may contain mistakes; "
            f"trust the image over them):\n{scene_description}"
        )
    if not blocks:
        return prompt
    return "\n\n".join(blocks + [f"Question: {prompt}"])


def build_general_messages(
    prompt: str,
    has_image: bool,
    history: Optional[list] = None,
    modality: str = "optical",
    scene_facts: str = "",
    scene_description: str = "",
) -> List[Dict[str, Any]]:
    """System role + prior turns (text only) + the current question, with the image attached to the
    current question."""
    messages: List[Dict[str, Any]] = [{"role": "system", "content": general_system_prompt(modality)}]
    messages.extend(history_for_model(history, prompt))
    question = build_question_text(prompt, scene_facts, scene_description) if has_image else prompt
    if has_image:
        messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]})
    else:
        messages.append({"role": "user", "content": question})
    return messages
