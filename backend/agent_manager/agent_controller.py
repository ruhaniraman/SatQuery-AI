import json
import os
import threading
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from contextlib import nullcontext
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from geospatial_preprocessing.geotiff_loader import load_and_standardize_image
from agent_manager.grid_scan import SCAN_YES_THRESHOLD, merge_positive_tiles, tile_windows, yes_probability
from agent_manager.prompts import (
    DEFAULT_MAX_NEW_TOKENS,
    DESCRIBE_MAX_NEW_TOKENS,
    QUADRANT_MAX_NEW_TOKENS,
    QUADRANT_NAMES,
    build_general_messages,
    describe_first_enabled,
    describe_quadrant_prompt,
    describe_scene_prompt,
    format_scene_description,
)

# Cap on pixels per image handed to Qwen2-VL (each 28x28 patch is one visual token, so this is
# ~1280 tokens). Without it a large GeoTIFF or the stitched before|after image can OOM a small
# GPU; the processor downsizes anything bigger. A starting guess, not a tuned value.
# Env SATQUERY_MAX_PIXELS raises it (more detail on big scenes, more VRAM and latency).
MAX_PIXELS = int(os.environ.get("SATQUERY_MAX_PIXELS", 1280 * 28 * 28))
# Small images are upscaled to at least this many pixels (~256 visual tokens) so a thumbnail is
# not seen at a handful of patches.
MIN_PIXELS = 256 * 28 * 28

# The fixed yes/no question each LoRA adapter answers (tied to how the adapters were fine-tuned).
ADAPTER_QUESTIONS = {
    "mining": " Is there a massive extraction pit or mining crater in this image? Answer ONLY 'yes' or 'no'.",
    "deforestation": " Is there visible deforestation, active logging, or clear-cut land in this image? Answer ONLY 'yes' or 'no'.",
    "agriculture": " Are there distinct, green agricultural fields or cultivated crop rows in this image? Answer ONLY 'yes' or 'no'.",
}


class SatQueryEngine:
    def __init__(self, base_model_id="Qwen/Qwen2-VL-2B-Instruct", adapters_base_dir="models/adapters"):
        # One lock guards every use of the model. The active LoRA adapter (set_adapter /
        # disable_adapter) is global state on the shared model object, so two requests
        # interleaving would silently run under each other's adapter.
        self.lock = threading.Lock()

        print("Initializing base Qwen2-VL-2B and adapters...")
        self.base_model_id = base_model_id
        
        # Ensure correct pathing relative to the backend directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.isabs(adapters_base_dir):
            adapters_base_dir = os.path.join(current_dir, "..", adapters_base_dir)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        self.processor = AutoProcessor.from_pretrained(base_model_id, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
        # Batched last-token scoring needs the real final token at index -1 in every row
        self.processor.tokenizer.padding_side = "left"
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )

        # 1. Attach the primary adapter
        agri_path = os.path.join(adapters_base_dir, "agriculture")
        self.model = PeftModel.from_pretrained(self.model, agri_path, adapter_name="agriculture")

        # 2. Attach the remaining adapters into the same model instance
        defor_path = os.path.join(adapters_base_dir, "deforestation")
        mine_path = os.path.join(adapters_base_dir, "mining")
        
        self.model.load_adapter(defor_path, adapter_name="deforestation")
        self.model.load_adapter(mine_path, adapter_name="mining")
        print("All domain LoRAs attached successfully!")

        # Facts about each adapter, read from the files that were actually loaded (for the audit trace).
        # The adapter files do not record what data they were trained on, so nothing is claimed.
        self.adapter_info = {}
        for name in ("agriculture", "deforestation", "mining"):
            try:
                with open(os.path.join(adapters_base_dir, name, "adapter_config.json"), encoding="utf-8") as f:
                    cfg = json.load(f)
                self.adapter_info[name] = {
                    "base_model": cfg.get("base_model_name_or_path"),
                    "lora_rank": cfg.get("r"),
                    "lora_alpha": cfg.get("lora_alpha"),
                }
            except Exception:
                self.adapter_info[name] = {}

    def _yes_no_token_ids(self):
        """First-token ids for yes-like and no-like answers ("yes"/"Yes"/" yes"..., same for no)."""
        if getattr(self, "_yn_ids", None) is None:
            tok = self.processor.tokenizer
            def ids(words):
                out = set()
                for w in words:
                    enc = tok.encode(w, add_special_tokens=False)
                    if enc:
                        out.add(enc[0])
                return sorted(out)
            self._yn_ids = (ids(["yes", "Yes", " yes", " Yes", "YES"]), ids(["no", "No", " no", " No", "NO"]))
        return self._yn_ids

    def _grid_probs(self, img, internal_prompt, grid_size=(4, 4), batch_size=4):
        """Asks the model a yes/no question about every grid cell; returns P(yes) as a (rows, cols) array.

        Each cell is scored from the next-token logits (P(yes) vs P(no)) in ONE forward pass per
        batch, rather than free-text generation per cell, so the threshold is tunable and the 16
        cells cost 4 batched passes instead of 16 sequential generate() calls. Cells are shown with
        surrounding context.
        """
        rows, cols = grid_size
        width, height = img.size
        windows = tile_windows(width, height, rows, cols)
        yes_ids, no_ids = self._yes_no_token_ids()

        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": internal_prompt}]}]
        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        probs = np.zeros(len(windows), dtype=np.float64)
        for start in range(0, len(windows), batch_size):
            chunk = windows[start:start + batch_size]
            tiles = [img.crop(box) for _, _, box in chunk]
            inputs = self.processor(
                text=[text_input] * len(tiles), images=tiles, return_tensors="pt", padding=True
            ).to(self.model.device)
            with torch.no_grad():
                try:
                    out = self.model(**inputs, logits_to_keep=1)   # only the last position is needed
                except TypeError:
                    out = self.model(**inputs)
            # Left padding (set in __init__) puts the real last token at index -1 for every row
            last = out.logits[:, -1, :].float().cpu().numpy()
            probs[start:start + len(chunk)] = yes_probability(last, yes_ids, no_ids)

        return probs.reshape(rows, cols)

    def _grid_classification(self, img, internal_prompt, grid_size=(4, 4),
                             threshold=SCAN_YES_THRESHOLD, batch_size=4):
        """The grid scan as merged region boxes: edge-adjacent positive cells become one region."""
        probs = self._grid_probs(img, internal_prompt, grid_size, batch_size)
        print("Grid P(yes):\n" + np.array2string(probs, precision=2))
        return merge_positive_tiles(probs >= threshold)

    def scan_scores(self, img_array, adapter, prompt=None):
        """P(yes) grid (4x4 ndarray) of one LoRA's question over a whole image: the single-image scan (with the
        user's `prompt` in front of the question, as the adapters were used) and the two-date comparison (no prompt)."""
        if adapter not in ADAPTER_QUESTIONS:
            raise ValueError(f"no scan question for adapter {adapter!r}")
        question = ADAPTER_QUESTIONS[adapter]
        internal_prompt = question.strip() if prompt is None else prompt + question
        with self.lock:
            self.model.set_adapter(adapter)
            return self._grid_probs(Image.fromarray(img_array), internal_prompt)

    def _generate_texts(self, message_sets, images, max_new_tokens):
        """Batched free-text generation with the LoRA adapters disabled (base weights). One reply per
        message set. Left padding (set in __init__) makes the prompt length identical across rows."""
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in message_sets]
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            with self.model.disable_adapter():
                output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, repetition_penalty=1.05)
        prompt_len = inputs["input_ids"].shape[1]
        return [self.processor.decode(ids[prompt_len:], skip_special_tokens=True).strip() for ids in output_ids]

    def _describe_scene(self, img, modality="optical"):
        """First pass for the general path: a whole-image description plus one short caption per
        quadrant (exact quadrants, no context padding). Returns the combined text for the prompt."""
        def one(text):
            return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]

        overview = self._generate_texts([one(describe_scene_prompt(modality))], [img], DESCRIBE_MAX_NEW_TOKENS)[0]
        width, height = img.size
        windows = tile_windows(width, height, 2, 2, context=0.0)   # row-major == QUADRANT_NAMES order
        tiles = [img.crop(box) for _, _, box in windows]
        captions = self._generate_texts(
            [one(describe_quadrant_prompt(name, modality)) for name in QUADRANT_NAMES], tiles, QUADRANT_MAX_NEW_TOKENS
        )
        return format_scene_description(overview, dict(zip(QUADRANT_NAMES, captions)))

    def query(self, prompt: str, image_path: str = None, image_bytes: bytes = None, explicit_adapter: str = "general", max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, modality: str = "optical", chat_history: list = None, scene_facts: str = "") -> str:
        """Thread-safe entry point: only one request may switch adapters and generate at a time."""
        with self.lock:
            return self._query(prompt, image_path, image_bytes, explicit_adapter, max_new_tokens, modality, chat_history, scene_facts)

    def _query(self, prompt: str, image_path: str = None, image_bytes: bytes = None, explicit_adapter: str = "general", max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, modality: str = "optical", chat_history: list = None, scene_facts: str = "") -> str:
        has_image = (image_path is not None and os.path.exists(image_path)) or (image_bytes is not None)
        
        # Override with the UI button's choice
        active_adapter = explicit_adapter

        # --- 1. PROMPT AUGMENTATION (Neutral Format) ---
        internal_prompt = prompt + ADAPTER_QUESTIONS.get(active_adapter, "")
        # Single direct pass
        if active_adapter == "general":
            context_manager = self.model.disable_adapter()
        else:
            self.model.set_adapter(active_adapter)
            context_manager = nullcontext()

        # Load image (Make sure to pass 'internal_prompt' instead of the raw user prompt!)
        if has_image:
            if image_path:
                img_array, metadata = load_and_standardize_image(image_path, modality)
                img = Image.fromarray(img_array)
            elif image_bytes:
                img = Image.open(BytesIO(image_bytes)).convert("RGB")

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": internal_prompt}
                ]
            }]
            images = [img]
        else:
            messages = [{"role": "user", "content": internal_prompt}]
            images = None

        if active_adapter == "general":
            # Real system role, prior turns (so follow-ups have context), then the current question
            scene_description = ""
            if has_image and describe_first_enabled():
                scene_description = self._describe_scene(img, modality)
            messages = build_general_messages(
                prompt, has_image, chat_history, modality,
                scene_facts=scene_facts, scene_description=scene_description,
            )

        if active_adapter in ["mining", "deforestation", "agriculture"] and has_image:
            # 4x4 Grid = 16 precise squares. The AI answers Yes/No for each square.
            global_boxes = self._grid_classification(img, internal_prompt, grid_size=(4, 4))
            
            if not global_boxes:
                if active_adapter == "mining":
                    return "No surface extraction or pit mining features detected in this sector."
                elif active_adapter == "deforestation":
                    return "No active logging or canopy loss detected."
                elif active_adapter == "agriculture":
                    return "No distinct agricultural features detected."
                    
            return str(global_boxes)

        text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text_input], images=images, return_tensors="pt", padding=True).to(self.model.device)

        with torch.no_grad():
            with context_manager:
                output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, repetition_penalty=1.05)

        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        raw_response = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

        # --- 2. MULTI-WORD NEGATIVE FORMATTING ---
        # Translate the blunt 'no' into a professional UI response
        if raw_response.lower() in ["no", "none", "[]", "null"]:
            if active_adapter == "mining":
                return "No surface extraction or pit mining features detected in this sector."
            elif active_adapter == "deforestation":
                return "No active logging or canopy loss detected."
            elif active_adapter == "agriculture":
                return "No distinct agricultural features detected."
                
        return raw_response

# The engine is created by load_agent() (called from the FastAPI lifespan), never at import time,
# so a missing adapter or failed model download cannot crash the app before it starts serving.
_agent = None


def load_agent() -> SatQueryEngine:
    global _agent
    if _agent is None:
        _agent = SatQueryEngine()
    return _agent


def get_agent():
    """The loaded engine, or None if it has not been (or could not be) loaded."""
    return _agent
