from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    SINGLE_IMAGE_VQA = "SINGLE_IMAGE_VQA"
    CHANGE_DETECTION = "CHANGE_DETECTION"
    OPTICAL_SAR_FUSION = "OPTICAL_SAR_FUSION"


class ImageInput(BaseModel):
    path: str
    modality: str = "unknown"
    # Real capture date only (None when none could be read reliably; never a placeholder)
    date: Optional[str] = None
    date_source: Optional[str] = None


class ExecutionTrace(BaseModel):
    """What actually happened for one request. Returned to the client, stored in data.json and
    printed in the PDF, so every field must describe real behaviour, not intent."""
    pipeline_id: str
    task: TaskType
    # The task is picked by fixed rules (image count + user-selected sensor types), not by a model
    routing: str = "rule-based"
    routing_reason: str
    # Stages that really executed, in order
    nodes_traversed: List[str]
    # Stage-specific facts: model/adapter used, method, counts, dates, etc.
    telemetry: Dict[str, Any] = Field(default_factory=dict)
    # Non-fatal problems the user should know about (band order guessed, dates disagree, ...)
    warnings: List[str] = Field(default_factory=list)
    validation_status: str = "not performed"
    execution_status: str = "completed"
