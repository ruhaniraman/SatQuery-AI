from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class TaskType(str, Enum):
    SINGLE_IMAGE_VQA = "SINGLE_IMAGE_VQA"
    CHANGE_DETECTION = "CHANGE_DETECTION"
    GROUNDING = "GROUNDING"
    OPTICAL_SAR_FUSION = "OPTICAL_SAR_FUSION"


class ClassificationResult(BaseModel):
    task: TaskType
    confidence: float = Field(ge=0.0, le=1.0)
    
class ImageInput(BaseModel):
    path: str
    modality: str = "unknown"
    date: Optional[str] = None
    
class ExecutionTrace(BaseModel):
    query: str
    task: TaskType
    confidence: float
    validation_status: str
    execution_status: str