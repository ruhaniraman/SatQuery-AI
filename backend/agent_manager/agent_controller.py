import os
import torch
import ast
from PIL import Image
from io import BytesIO
from contextlib import nullcontext
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from geospatial_preprocessing.geotiff_loader import load_and_standardize_image

class SatQueryEngine:
    def __init__(self, base_model_id="Qwen/Qwen2-VL-2B-Instruct", adapters_base_dir="models/adapters"):
        print("Initializing base Qwen2-VL-2B and adapters...")
        
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

        self.processor = AutoProcessor.from_pretrained(base_model_id)
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

    def route_intent(self, prompt: str, has_image: bool) -> str:
        """Determines which adapter to activate based on strict command heuristics."""
        if not has_image:
            return "general"

        text = prompt.lower()
        
        # 1. Define strict command verbs that indicate a need for technical extraction
        command_verbs = ["classify", "detect", "tag", "assess", "extract", "estimate", "evaluate", "threshold"]
        
        # 2. Check if the user is explicitly asking for a strict technical task
        has_command = any(verb in text for verb in command_verbs)
        
        # 3. Only route to specialized LoRAs if a command verb is present
        if has_command:
            if any(k in text for k in ["agriculture", "crop", "farm", "arable", "pasture"]):
                return "agriculture"
            elif any(k in text for k in ["deforestation", "logging", "clearing", "canopy loss"]):
                return "deforestation"
            elif any(k in text for k in ["mine", "mining", "quarry", "pit", "coalfield", "extraction"]):
                return "mining"
                
        # 4. Default to the conversational base model for everything else
        return "general"

    def _grid_classification(self, img, internal_prompt, grid_size=(4, 4)):
        """Splits image into a grid and asks the AI a strict Yes/No for each tile."""
        rows, cols = grid_size
        width, height = img.size
        tile_w = width // cols
        tile_h = height // rows
        all_global_boxes = []

        for row in range(rows):
            for col in range(cols):
                left = col * tile_w
                top = row * tile_h
                right = (col + 1) * tile_w
                bottom = (row + 1) * tile_h
                tile_img = img.crop((left, top, right, bottom))
                
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": internal_prompt}]}]
                text_input = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                # Keep max_new_tokens incredibly low because we only want a 1-word answer
                inputs = self.processor(text=[text_input], images=[tile_img], return_tensors="pt", padding=True).to("cuda")
                
                with torch.no_grad():
                    output_ids = self.model.generate(**inputs, max_new_tokens=10) 
                
                generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
                raw_response = self.processor.decode(generated_ids, skip_special_tokens=True).strip().lower()
                
                # If the AI says 'yes', Python draws the box matching the tile exactly
                if "yes" in raw_response:
                    global_ymin = row / rows
                    global_xmin = col / cols
                    global_ymax = (row + 1) / rows
                    global_xmax = (col + 1) / cols
                    all_global_boxes.append([
                        round(global_ymin, 3), 
                        round(global_xmin, 3), 
                        round(global_ymax, 3), 
                        round(global_xmax, 3)
                    ])
                    
        return all_global_boxes

    def query(self, prompt: str, image_path: str = None, image_bytes: bytes = None, explicit_adapter: str = "general", max_new_tokens: int = 256) -> str:
        has_image = (image_path is not None and os.path.exists(image_path)) or (image_bytes is not None)
        
        # Override with the UI button's choice
        active_adapter = explicit_adapter

        # --- 1. PROMPT AUGMENTATION (Neutral Format) ---
        internal_prompt = prompt
        if active_adapter == "mining":
            internal_prompt += " Is there a massive extraction pit or mining crater in this image? Answer ONLY 'yes' or 'no'."
        elif active_adapter == "deforestation":
            internal_prompt += " Is there visible deforestation, active logging, or clear-cut land in this image? Answer ONLY 'yes' or 'no'."
        elif active_adapter == "agriculture":
            internal_prompt += " Are there distinct, green agricultural fields or cultivated crop rows in this image? Answer ONLY 'yes' or 'no'."
        # Single direct pass
        if active_adapter == "general":
            internal_prompt = (
                "[SYSTEM]: You are a precise geospatial analyst. You must follow this exact structure:\n"
                "1. OBSERVATIONS: First, describe the visible building density, the exact color/state of the water, and the actual proportion of unbuilt bare land in the image.\n"
                "2. ASSESSMENT: Answer the user's query strictly based on those physical observations. Do not offer generic advice.\n\n"
                f"[USER QUERY]: {prompt}"
            )
            context_manager = self.model.disable_adapter()
        else:
            self.model.set_adapter(active_adapter)
            context_manager = nullcontext()

        # Load image (Make sure to pass 'internal_prompt' instead of the raw user prompt!)
        if has_image:
            if image_path:
                img_array, metadata = load_and_standardize_image(image_path)
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
        inputs = self.processor(text=[text_input], images=images, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            with context_manager:
                output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

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

# Initialize a global instance so FastAPI can import it
agent = SatQueryEngine()