import requests
from io import BytesIO
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

your_model_dir = "/home/qc/blip_disaster/checkpoints/florence_flickr_best.pt"  # adjust if stage2 checkpoint exists later
base_model_name = "microsoft/Florence-2-base"
image_url = ""
image_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQqvba64KNRcVTkN50u7ZxXIjcOhdzSNezx4v-NPIgOWQ&s=10"

TASK_PROMPT = "<DETAILED_CAPTION>"  # keep in sync with what you trained on

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
            max_new_tokens=50,
            num_beams=5,
            repetition_penalty=1.2,
        )

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    # Florence-2 needs this post-processing step to strip task tokens and
    # parse the raw output into a clean string - plain tokenizer.decode()
    # like BLIP uses will leave special tokens/formatting in the output.
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
    # Base architecture + processor come from the original model id,
    # only the weights get swapped from your saved checkpoint.
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
print("Original Florence-2 :", base_caption)
print(" Fine-tuned:", your_caption)