"""
single_image_vqa_engine.py
===========================
Member 2: Single-Image VQA & Grounding (Stage 3 specialist)

Loads Florence-2 (grounding / bounding boxes) and Qwen2-VL (open-ended VQA)
in zero-shot mode. Falls back to deterministic mocks if the models aren't
downloaded/available yet (e.g. no GPU / no network to HuggingFace in this
environment) -- so the module runs standalone per the Hour 0-12 plan and
Member 6 can build against it immediately.

Integration contract (matches cdvqa_engine.py / geotiff_loader.py):
    image input  -> HxWx3 uint8 numpy array (RGB)
    ground_object output -> (modified_image_array, list[BoundingBox])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class BoundingBox:
    label: str
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float = 1.0  # Florence-2 zero-shot grounding has no native score;
                              # kept for schema parity with Member 6's API contract.

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "box": [self.x1, self.y1, self.x2, self.y2],
            "confidence": self.confidence,
        }


@dataclass
class GroundingResult:
    image_with_boxes: np.ndarray
    boxes: list[BoundingBox] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "boxes": [b.to_dict() for b in self.boxes],
            "image_shape": list(self.image_with_boxes.shape),
        }


class VQAGroundingEngine:
    """
    Zero-shot single-image VQA (Qwen2-VL) + text-guided grounding (Florence-2).
    Both models are lazy-loaded on first use so importing this module is cheap.
    """

    def __init__(
        self,
        florence_model_id: str = "microsoft/Florence-2-base",
        qwen_model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: Optional[str] = None,
    ):
        self.florence_model_id = florence_model_id
        self.qwen_model_id = qwen_model_id
        self.device = device or ("cuda" if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu")

        self._florence_model = None
        self._florence_processor = None
        self._qwen_model = None
        self._qwen_processor = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def answer_question(self, image_array: np.ndarray, query: str) -> str:
        """
        Args:
            image_array: HxWx3 uint8 RGB numpy array.
            query: natural-language question about the image.

        Returns:
            Text answer string.
        """
        if not TRANSFORMERS_AVAILABLE:
            return self._mock_answer(query)

        try:
            self._load_qwen()
            return self._qwen_infer(image_array, query)
        except Exception as e:
            # Zero-shot foundation models can be heavy / fail to download in a
            # resource-constrained hackathon environment -- degrade gracefully
            # rather than blocking the whole pipeline.
            print(f"[VQAGroundingEngine] Qwen2-VL inference failed, falling back to mock: {e}")
            return self._mock_answer(query)

    def ground_object(self, image_array: np.ndarray, query: str) -> GroundingResult:
        """
        Args:
            image_array: HxWx3 uint8 RGB numpy array.
            query: text description of the object(s) to locate, e.g. "the airport runway".

        Returns:
            GroundingResult with the image annotated (bright red boxes) and
            structured box coordinates.
        """
        if not TRANSFORMERS_AVAILABLE:
            return self._mock_ground(image_array, query)

        try:
            self._load_florence()
            boxes = self._florence_infer(image_array, query)
        except Exception as e:
            print(f"[VQAGroundingEngine] Florence-2 inference failed, falling back to mock: {e}")
            return self._mock_ground(image_array, query)

        annotated = self._draw_boxes(image_array, boxes)
        return GroundingResult(image_with_boxes=annotated, boxes=boxes)

    # ------------------------------------------------------------------ #
    # Model loaders (lazy)
    # ------------------------------------------------------------------ #

    def _load_florence(self):
        if self._florence_model is not None:
            return
        self._florence_processor = AutoProcessor.from_pretrained(
            self.florence_model_id, trust_remote_code=True
        )
        self._florence_model = AutoModelForCausalLM.from_pretrained(
            self.florence_model_id, trust_remote_code=True
        ).to(self.device)

    def _load_qwen(self):
        if self._qwen_model is not None:
            return
        from transformers import Qwen2VLForConditionalGeneration
        self._qwen_processor = AutoProcessor.from_pretrained(self.qwen_model_id)
        self._qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.qwen_model_id
        ).to(self.device)

    # ------------------------------------------------------------------ #
    # Real inference (requires downloaded weights + network access)
    # ------------------------------------------------------------------ #

    def _qwen_infer(self, image_array: np.ndarray, query: str) -> str:
        from PIL import Image
        pil_img = Image.fromarray(image_array)
        messages = [{
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": query}],
        }]
        text_prompt = self._qwen_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._qwen_processor(text=[text_prompt], images=[pil_img], return_tensors="pt").to(self.device)
        output_ids = self._qwen_model.generate(**inputs, max_new_tokens=128)
        generated = self._qwen_processor.batch_decode(output_ids, skip_special_tokens=True)
        return generated[0]

    def _florence_infer(self, image_array: np.ndarray, query: str) -> list[BoundingBox]:
        from PIL import Image
        pil_img = Image.fromarray(image_array)
        task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
        inputs = self._florence_processor(
            text=task_prompt + query, images=pil_img, return_tensors="pt"
        ).to(self.device)
        generated_ids = self._florence_model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=256
        )
        generated_text = self._florence_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed = self._florence_processor.post_process_generation(
            generated_text, task=task_prompt, image_size=(pil_img.width, pil_img.height)
        )

        boxes = []
        result = parsed.get(task_prompt, {})
        for bbox, label in zip(result.get("bboxes", []), result.get("labels", [])):
            x1, y1, x2, y2 = [int(v) for v in bbox]
            boxes.append(BoundingBox(label=label, x1=x1, y1=y1, x2=x2, y2=y2))
        return boxes

    # ------------------------------------------------------------------ #
    # Drawing
    # ------------------------------------------------------------------ #

    def _draw_boxes(self, image_array: np.ndarray, boxes: list[BoundingBox]) -> np.ndarray:
        annotated = image_array.copy()
        for box in boxes:
            cv2.rectangle(annotated, (box.x1, box.y1), (box.x2, box.y2), (255, 0, 0), thickness=3)
            cv2.putText(
                annotated, box.label, (box.x1, max(box.y1 - 8, 0)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2,
            )
        return annotated

    # ------------------------------------------------------------------ #
    # Mocks (Hour 0-12 fallback, no model download required)
    # ------------------------------------------------------------------ #

    def _mock_answer(self, query: str) -> str:
        return f"[MOCK VQA] Zero-shot answer placeholder for query: '{query}'. Swap in real Qwen2-VL weights to activate."

    def _mock_ground(self, image_array: np.ndarray, query: str) -> GroundingResult:
        h, w = image_array.shape[:2]
        box = BoundingBox(
            label=query[:30], x1=int(w * 0.3), y1=int(h * 0.3), x2=int(w * 0.7), y2=int(h * 0.7),
            confidence=0.0,  # 0.0 signals "mock, not a real detection" to downstream consumers
        )
        annotated = self._draw_boxes(image_array, [box])
        return GroundingResult(image_with_boxes=annotated, boxes=[box])


# --------------------------------------------------------------------------- #
# Standalone smoke test
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    test_image = np.full((256, 256, 3), 100, dtype=np.uint8)
    cv2.circle(test_image, (128, 128), 40, (0, 200, 0), thickness=-1)

    engine = VQAGroundingEngine()

    print("=== VQA Test ===")
    answer = engine.answer_question(test_image, "What color is the object in the center?")
    print("Answer:", answer)

    print("\n=== Grounding Test ===")
    result = engine.ground_object(test_image, "the green circular object")
    print("Boxes:", result.to_json())

    cv2.imwrite("mock_grounding_output.png", cv2.cvtColor(result.image_with_boxes, cv2.COLOR_RGB2BGR))
    print("\nSaved mock_grounding_output.png")
