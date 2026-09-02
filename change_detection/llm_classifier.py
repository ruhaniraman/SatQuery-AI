"""
llm_classifier.py
===================
Real LLM-backed intent classifier to replace agent_controller.py's
`_mock_classifier`. Uses the Anthropic API with a structured (tool-call)
schema so the output is always a valid TaskIntent -- no free-text parsing,
no hallucinated categories.

Usage:
    from llm_classifier import LLMIntentClassifier
    from agent_controller import AgentController

    classifier = LLMIntentClassifier()          # reads ANTHROPIC_API_KEY from env
    controller = AgentController(classifier_fn=classifier.classify)

If no API key is set (or the call fails for any reason -- rate limit, network,
etc.), this degrades to the same keyword-based mock agent_controller.py
already ships with, so the rest of the team is never blocked by a missing key.
"""

from __future__ import annotations

import os
from typing import Optional

from agent_controller import TaskIntent

try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False


CLASSIFY_TOOL = {
    "name": "classify_intent",
    "description": "Classify a remote-sensing query into exactly one task category.",
    "input_schema": {
        "type": "object",
        "properties": {
            "intent": {
                "type": "string",
                "enum": [t.value for t in TaskIntent],
                "description": (
                    "SINGLE_IMAGE_VQA: open-ended question about one image. "
                    "CHANGE_DETECTION: what changed between two images/dates. "
                    "GROUNDING: locate/find/highlight a specific object in one image. "
                    "OPTICAL_SAR_FUSION: combine optical + radar/SAR data."
                ),
            }
        },
        "required": ["intent"],
    },
}

SYSTEM_PROMPT = (
    "You are a routing classifier for a remote-sensing AI backend. Given a "
    "user's query, call `classify_intent` with exactly one category. Do not "
    "explain your reasoning -- only call the tool."
)


class LLMIntentClassifier:
    def __init__(self, model: str = "claude-sonnet-4-5", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self._client = None

        if ANTHROPIC_SDK_AVAILABLE and self.api_key:
            self._client = anthropic.Anthropic(api_key=self.api_key)

    def classify(self, query: str) -> TaskIntent:
        """
        Drop-in replacement for AgentController's classifier_fn.
        Falls back to the keyword mock on any failure so routing never breaks.
        """
        if self._client is None:
            return self._fallback(query)

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=64,
                system=SYSTEM_PROMPT,
                tools=[CLASSIFY_TOOL],
                tool_choice={"type": "tool", "name": "classify_intent"},
                messages=[{"role": "user", "content": query}],
            )
            for block in response.content:
                if block.type == "tool_use" and block.name == "classify_intent":
                    return TaskIntent(block.input["intent"])
            return self._fallback(query)
        except Exception as e:
            print(f"[LLMIntentClassifier] LLM call failed, falling back to keyword classifier: {e}")
            return self._fallback(query)

    def _fallback(self, query: str) -> TaskIntent:
        # Reuse the exact same keyword logic as agent_controller's mock,
        # so behavior is predictable and testable without an API key.
        import re
        q = query.lower()
        if re.search(r"\bchange|difference|compare (two|between)|before.*after\b", q):
            return TaskIntent.CHANGE_DETECTION
        if re.search(r"\bsar|radar|fusion|optical.*sar|combine\b", q):
            return TaskIntent.OPTICAL_SAR_FUSION
        if re.search(r"\bwhere is|locate|find the|bounding box|highlight\b", q):
            return TaskIntent.GROUNDING
        return TaskIntent.SINGLE_IMAGE_VQA


# --------------------------------------------------------------------------- #
# Standalone smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    classifier = LLMIntentClassifier()
    print(f"API key detected: {classifier._client is not None}")

    test_queries = [
        "What is visible in the top-left corner of this image?",
        "What changed between these two dates?",
        "Locate all buildings in this scene",
        "Fuse the optical and SAR data to show structural density",
    ]
    for q in test_queries:
        intent = classifier.classify(q)
        print(f"'{q}' -> {intent.value}")
