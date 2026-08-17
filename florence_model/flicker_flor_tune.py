"""
Stage 1: Fine-tune Florence-2-base on Flickr8k captions.
Mirrors the structure of your BLIP flicker_tuning.py script.
"""

import os
import copy
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

# ---------------------------
# Config
# ---------------------------
MODEL_ID = "microsoft/Florence-2-base"
FLICKR_IMAGES_DIR = "../data/Blip_disaster_model/Flickr8k/Images"
FLICKR_CAPTIONS_FILE = "../data/Blip_disaster_model/Flickr8k/captions.txt"
CHECKPOINT_DIR = "../checkpoints"
TASK_PROMPT = "<DETAILED_CAPTION>"

BATCH_SIZE = 4
EPOCHS = 20                 # early stopping will likely cut this short
LR = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
GRAD_CLIP_NORM = 1.0
VAL_SPLIT = 0.1              # 10% held out for validation / early stopping
EARLY_STOP_PATIENCE = 3      # stop if val loss doesn't improve for N epochs
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------
# Dataset
# ---------------------------
class FlickrDataset(Dataset):
    def __init__(self, images_dir, captions_file):
        self.images_dir = images_dir
        self.samples = []  # list of (image_filename, caption)

        with open(captions_file, "r") as f:
            lines = f.readlines()

        # Assumes format: image_filename,caption  (adjust parsing if your file differs)
        for line in lines[1:]:  # skip header if present
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            img_name, caption = parts
            self.samples.append((img_name.strip(), caption.strip()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_name, caption = self.samples[idx]
        image_path = os.path.join(self.images_dir, img_name)
        image = Image.open(image_path).convert("RGB")
        return image, caption


# ---------------------------
# Collator - this is the key difference from BLIP
# ---------------------------
def make_collate_fn(processor):
    def collate_fn(batch):
        images, captions = zip(*batch)

        # Input side: image + task prompt (NOT the caption)
        prompts = [TASK_PROMPT] * len(images)
        inputs = processor(
            text=list(prompts),
            images=list(images),
            return_tensors="pt",
            padding=True,
        )

        # Label side: the actual caption, tokenized separately
        labels = processor.tokenizer(
            list(captions),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).input_ids

        # Replace pad token id with -100 so loss ignores padding
        labels[labels == processor.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels
        return inputs

    return collate_fn


def run_validation(model, dataloader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )
            total_loss += outputs.loss.item()
    model.train()
    return total_loss / len(dataloader)


# ---------------------------
# Training loop
# ---------------------------
def train():
    print(f"Loading model/processor from {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Freeze the vision encoder - same overfitting-control move you used for BLIP.
    # Florence-2's vision tower is under model.vision_tower; only unfreeze if
    # you have enough data later and want to squeeze out more performance.
    for name, param in model.named_parameters():
        if "vision_tower" in name:
            param.requires_grad = False

    model.to(DEVICE)
    model.train()

    full_dataset = FlickrDataset(FLICKR_IMAGES_DIR, FLICKR_CAPTIONS_FILE)
    print(f"Loaded {len(full_dataset)} flickr samples")

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"Train: {train_size} | Val: {val_size}")

    collate_fn = make_collate_fn(processor)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state_dict = None

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for batch in progress:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            outputs = model(
                input_ids=batch["input_ids"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, GRAD_CLIP_NORM)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = run_validation(model, val_loader)
        print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"florence_flickr_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # Early stopping logic - tracks best val loss, stops if no improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve}/{EARLY_STOP_PATIENCE} epochs")
            if epochs_no_improve >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    # Save the best checkpoint separately so it's easy to find for stage 2 / eval
    if best_state_dict is not None:
        best_path = os.path.join(CHECKPOINT_DIR, "florence_flickr_best.pt")
        torch.save(best_state_dict, best_path)
        print(f"Saved best checkpoint (val_loss={best_val_loss:.4f}): {best_path}")

    print("Flickr fine-tuning done.")


if __name__ == "__main__":
    train()