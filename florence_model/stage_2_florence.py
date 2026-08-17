"""
Stage 2: Fine-tune Florence-2 (starting from the flickr8k checkpoint) on the
disaster dataset. Mirrors model_blip/stage_2_new.py structure, adapted for
Florence-2's prompt-based captioning and the Florence-2 processor/model API.
"""

import os
import copy
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm

# ---------------------------
# Config
# ---------------------------
BASE_MODEL_ID = "microsoft/Florence-2-base"
STAGE1_CHECKPOINT = "../checkpoints/florence_flickr_best.pt"  # starting point for stage 2

CAPTIONS_FILE = "../data/Final_Dataset_Caption.xlsx"
IMAGE_DIR = "../data/disaster_images"
CHECKPOINT_DIR = "../checkpoints"

TASK_PROMPT = "<DETAILED_CAPTION>"  # keep consistent with stage 1

BATCH_SIZE = 4
EPOCHS = 20
LR = 5e-6                    # lower than stage 1 - smaller dataset, less room before overfitting
WEIGHT_DECAY = 0.05           # higher than stage 1's 0.01 - stronger regularization for small dataset
WARMUP_RATIO = 0.1
GRAD_CLIP_NORM = 1.0
VAL_SPLIT = 0.15              # slightly bigger val split since dataset is small - want a stable signal
EARLY_STOP_PATIENCE = 2       # stricter than stage 1's patience=3 - disaster data is ~2k images, overfits fast
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
class DisasterDataset(Dataset):
    def __init__(self, captions_file, image_dir):
        self.image_dir = image_dir
        df = pd.read_excel(captions_file, header=None, names=["filename", "caption"])

        self.samples = []
        for _, row in df.iterrows():
            filename = str(row["filename"]).strip()
            caption = str(row["caption"]).strip()
            if not filename or not caption or filename.lower() == "nan" or caption.lower() == "nan":
                continue
            self.samples.append((filename, caption))

        print(f"Loaded {len(self.samples)} disaster captions from {captions_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, caption = self.samples[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        return image, caption


# ---------------------------
# Collator - same prompt-based pattern as stage 1
# ---------------------------
def make_collate_fn(processor):
    def collate_fn(batch):
        images, captions = zip(*batch)

        prompts = [TASK_PROMPT] * len(images)
        inputs = processor(
            text=list(prompts),
            images=list(images),
            return_tensors="pt",
            padding=True,
        )

        labels = processor.tokenizer(
            list(captions),
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).input_ids

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
    print(f"Loading base architecture/processor from {BASE_MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    print(f"Loading stage 1 (flickr) weights from {STAGE1_CHECKPOINT} ...")
    state_dict = torch.load(STAGE1_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state_dict)

    # Freeze vision encoder - same as stage 1, disaster dataset is even
    # smaller so keeping this frozen matters even more here.
    for name, param in model.named_parameters():
        if "vision_tower" in name:
            param.requires_grad = False

    model.to(DEVICE)
    model.train()

    full_dataset = DisasterDataset(CAPTIONS_FILE, IMAGE_DIR)

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

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"florence_stage2_epoch{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

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

    if best_state_dict is not None:
        best_path = os.path.join(CHECKPOINT_DIR, "florence_stage2_best.pt")
        torch.save(best_state_dict, best_path)
        print(f"Saved best checkpoint (val_loss={best_val_loss:.4f}): {best_path}")

    print("Disaster stage 2 fine-tuning done.")


if __name__ == "__main__":
    train()