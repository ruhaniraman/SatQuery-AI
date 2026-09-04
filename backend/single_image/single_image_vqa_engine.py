import torch
import os
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel

class SingleImageSpecialist:
    def __init__(self, adapter_path="../backend/models"):
        print("1. Loading 4-bit Base Model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        # Load the base weights safely into the local RTX 3050
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        print("2. Fusing Custom SatQuery Adapters...")
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.processor = AutoProcessor.from_pretrained(adapter_path)
        print("Model Ready!")

    def analyze_image(self, image_path, question):
        print(f"Analyzing {image_path}...")
        image = Image.open(image_path).convert("RGB")
        
        # Format the prompt using the standard chat template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        # Prepare inputs for the model
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        
        # Generate the response
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=512)
            
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
        response = self.processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        
        return response

if __name__ == "__main__":
    # 1. Initialize the engine
    engine = SingleImageSpecialist()
    
    # 2. Point it to your real image
    # Make sure this matches the exact name of the file you saved!
    real_img_path = "real_test.png" 
    
    if os.path.exists(real_img_path):
        # 3. Ask a proper, complex prompt
        question = "Describe the terrain of the image."
        
        print(f"\n--- Testing with Real Image ---")
        print(f"Prompt: {question}")
        answer = engine.analyze_image(real_img_path, question)
        
        print(f"\nAI Output:\n{answer}")
    else:
        print(f"Error: Could not find the image '{real_img_path}'. Please make sure it is in the same folder as this script!")