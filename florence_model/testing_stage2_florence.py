import requests
from io import BytesIO
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

your_model_dir = "/home/qc/blip_disaster/checkpoints/florence_stage2_best.pt"
base_model_name = "microsoft/Florence-2-base"
image_url = ""
image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSzXRFg8DDI_SAlFEV1vJHhIm3r9gNYqi3f2ql2jxH92w&s=10"

TASK_PROMPT = "<DETAILED_CAPTION>"  # must match what stage 1 + stage 2 were trained on

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_image(url):
    response = requests.get(url, timeout=10)
    return Image.open(BytesIO(response.content)).convert("RGB")


def generate_caption(model, processor, image):
    inputs = processor(text=TASK_PROMPT, images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=75,
            num_beams=5,
            repetition_penalty=1.2,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed = processor.post_process_generation(
        generated_text, task=TASK_PROMPT, image_size=(image.width, image.height)
    )
    return parsed[TASK_PROMPT]


def load_base_model():
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True).to(device)
    model.eval()
    return model, processor


def load_finetuned_model(checkpoint_path):
    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, processor


image = load_image(image_url)

base_model, base_processor = load_base_model()
base_caption = generate_caption(base_model, base_processor, image)

your_model, your_processor = load_finetuned_model(your_model_dir)
your_caption = generate_caption(your_model, your_processor, image)

print("\n--- Comparison ---")
print("Original Florence-2 (base)      :", base_caption)
print("Fine-tuned (flickr + disaster)  :", your_caption)