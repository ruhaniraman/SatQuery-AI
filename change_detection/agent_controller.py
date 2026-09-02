"""
agent_controller.py
====================
Member 1: Agent Controller & Routing Logic (Stage 2 of the pipeline)

Classifies a user's natural-language query into one of four task types,
validates that the right number of images were supplied, routes to the
appropriate specialist module, and emits an auditable execution trace.

Uses Pydantic for schema enforcement. LLM classification is pluggable
(swap `_mock_classifier` for a real GPT-4o-mini / Llama-3 call + `instructor`
once you have API keys wired up) so Members 2/3/5/6 aren't blocked.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class TaskIntent(str, Enum):
    SINGLE_IMAGE_VQA = "SINGLE_IMAGE_VQA"
    CHANGE_DETECTION = "CHANGE_DETECTION"
    GROUNDING = "GROUNDING"
    OPTICAL_SAR_FUSION = "OPTICAL_SAR_FUSION"


# Tasks that require exactly two images.
TWO_IMAGE_TASKS = {TaskIntent.CHANGE_DETECTION, TaskIntent.OPTICAL_SAR_FUSION}


class RoutingRequest(BaseModel):
    query: str
    num_images_provided: int = Field(ge=0, le=2)

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be empty")
        return v


class ValidationError_(BaseModel):
    """Structured error returned when input validation fails (e.g. missing 2nd image)."""
    error: bool = True
    reason: str
    required_images: int
    images_provided: int


class ExecutionTrace(BaseModel):
    task_selected: Optional[TaskIntent]
    models_used: list[str]
    inputs_validated: dict[str, bool]
    execution_timestamp: str
    tool_result: Optional[dict] = None
    validation_error: Optional[ValidationError_] = None


# --------------------------------------------------------------------------- #
# Mock tool registry (Stage 3 specialists) -- swap for real imports:
#   from cdvqa_engine import ChangeDetectionEngine
#   from single_image_vqa_engine import VQAGroundingEngine
#   from optical_sar_fusion_model import FusionEngine
# --------------------------------------------------------------------------- #

def tool_single_image_vqa(**kwargs) -> dict:
    print("Executing Task: SINGLE_IMAGE_VQA")
    return {"status": "success", "answer": "Executing Task X (dummy VQA answer)"}


def tool_change_detection(**kwargs) -> dict:
    print("Executing Task: CHANGE_DETECTION")
    return {"status": "success", "answer": "Executing Task X (dummy change-detection result)"}


def tool_grounding(**kwargs) -> dict:
    print("Executing Task: GROUNDING")
    return {"status": "success", "answer": "Executing Task X (dummy grounding boxes)"}


def tool_optical_sar_fusion(**kwargs) -> dict:
    print("Executing Task: OPTICAL_SAR_FUSION")
    return {"status": "success", "answer": "Executing Task X (dummy fusion composite)"}


TOOL_REGISTRY: dict[TaskIntent, Callable[..., dict]] = {
    TaskIntent.SINGLE_IMAGE_VQA: tool_single_image_vqa,
    TaskIntent.CHANGE_DETECTION: tool_change_detection,
    TaskIntent.GROUNDING: tool_grounding,
    TaskIntent.OPTICAL_SAR_FUSION: tool_optical_sar_fusion,
}

MODEL_MAP: dict[TaskIntent, list[str]] = {
    TaskIntent.SINGLE_IMAGE_VQA: ["Qwen2-VL"],
    TaskIntent.GROUNDING: ["Florence-2"],
    TaskIntent.CHANGE_DETECTION: ["OpenCV-SSIM", "Qwen2-VL (mock)"],
    TaskIntent.OPTICAL_SAR_FUSION: ["OpenCV-Fusion", "Qwen2-VL (mock)"],
}


# --------------------------------------------------------------------------- #
# Controller
# --------------------------------------------------------------------------- #

class AgentController:
    def __init__(self, classifier_fn: Optional[Callable[[str], TaskIntent]] = None):
        """
        Args:
            classifier_fn: optional callable(query: str) -> TaskIntent.
                Defaults to a keyword-based mock classifier so this module
                runs standalone with zero API keys. Swap in a real LLM call
                (via `instructor` + GPT-4o-mini/Llama-3) when ready.
        """
        self.classify = classifier_fn or self._mock_classifier

    def route(self, query: str, num_images_provided: int) -> ExecutionTrace:
        request = RoutingRequest(query=query, num_images_provided=num_images_provided)
        intent = self.classify(request.query)

        required_images = 2 if intent in TWO_IMAGE_TASKS else 1
        images_ok = request.num_images_provided >= required_images

        inputs_validated = {
            "query_provided": bool(request.query.strip()),
            "image_count_sufficient": images_ok,
        }

        if not images_ok:
            validation_error = ValidationError_(
                reason=(
                    f"Task '{intent.value}' requires {required_images} image(s), "
                    f"but only {request.num_images_provided} were provided."
                ),
                required_images=required_images,
                images_provided=request.num_images_provided,
            )
            return ExecutionTrace(
                task_selected=intent,
                models_used=MODEL_MAP.get(intent, []),
                inputs_validated=inputs_validated,
                execution_timestamp=self._timestamp(),
                tool_result=None,
                validation_error=validation_error,
            )

        tool_fn = TOOL_REGISTRY[intent]
        tool_result = tool_fn(query=request.query, num_images=request.num_images_provided)

        return ExecutionTrace(
            task_selected=intent,
            models_used=MODEL_MAP.get(intent, []),
            inputs_validated=inputs_validated,
            execution_timestamp=self._timestamp(),
            tool_result=tool_result,
            validation_error=None,
        )

    # ------------------------------------------------------------------ #

    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _mock_classifier(self, query: str) -> TaskIntent:
        """Keyword-based stand-in for the real LLM classifier."""
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
    controller = AgentController()

    test_cases = [
        ("What is visible in the top-left corner of this image?", 1),
        ("What changed between these two dates?", 2),
        ("What changed between these two dates?", 1),  # should trigger validation error
        ("Locate all buildings in this scene", 1),
        ("Fuse the optical and SAR data to show structural density", 2),
    ]

    for query, n_images in test_cases:
        trace = controller.route(query, n_images)
        print(f"\nQuery: '{query}' | images: {n_images}")
        print(trace.model_dump_json(indent=2))
