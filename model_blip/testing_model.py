import requests
from io import BytesIO
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

your_model_dir = "/home/qc/blip_disaster/checkpoints/stage2_v3/best_model"
base_model_name = "Salesforce/blip-image-captioning-base"
image_url = ""
image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSfCFz9NhM4rrkLhfq_bdNqvInIGQhYwBjSRT1ZuqxKFg&s=10"

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_image(url):
    response = requests.get(url, timeout=10)
    return Image.open(BytesIO(response.content)).convert("RGB")

def generate_caption(model_dir, image):
    processor = BlipProcessor.from_pretrained(model_dir)
    model = BlipForConditionalGeneration.from_pretrained(model_dir, use_safetensors=True).to(device)
    model.eval()

    inputs = processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(**inputs, max_length=30, num_beams=5, repetition_penalty=1.2)

    return processor.decode(output_ids[0], skip_special_tokens=True)

image = load_image(image_url)

base_caption = generate_caption(base_model_name, image)
your_caption = generate_caption(your_model_dir, image)

print("\n--- Comparison ---")
print("Original BLIP :", base_caption)
print(" Fine-tuned:", your_caption)