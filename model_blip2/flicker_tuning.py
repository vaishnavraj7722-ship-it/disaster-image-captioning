#!/usr/bin/env python3
"""
Stage 1: Flickr8k Fine-Tuning with BLIP (base) - fits on 8GB VRAM
================================================================
Full model fine-tuning. No quantization needed. ~400M params.

Usage:
    python model_blip2/flicker_tuning.py \
        --data_path ~/blip_disaster/data/Blip_disaster_model/Flickr8k/captions.txt \
        --image_dir ~/blip_disaster/data/Blip_disaster_model/Flickr8k/Images \
        --output_dir ~/blip_disaster/checkpoints/stage1_blip \
        --batch_size 4 --grad_accum 4 --epochs 10 --lr 5e-5
"""

import os
import sys
import argparse
import logging
import math
import random
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    BlipForConditionalGeneration,
    BlipProcessor,
    get_linear_schedule_with_warmup,
)
from PIL import Image
from tqdm import tqdm
import numpy as np


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"stage1_blip_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Dataset (same as your BLIP-2 one)
# ------------------------------------------------------------------------------
class Flickr8kDataset(Dataset):
    def __init__(self, captions_file: str, image_dir: str):
        self.image_dir = image_dir
        self.samples = []

        with open(captions_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        start_idx = 0
        if lines and ("image" in lines[0].lower() or "caption" in lines[0].lower()):
            start_idx = 1

        for line in lines[start_idx:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) != 2:
                continue
            filename = parts[0].strip()
            caption = parts[1].strip()
            if "#" in filename:
                filename = filename.split("#")[0]
            self.samples.append((filename, caption))

        unique_images = len(set(f for f, _ in self.samples))
        logger.info(f"Loaded {len(self.samples)} captions from {unique_images} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, caption = self.samples[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        return {"image": image, "caption": caption}


# ------------------------------------------------------------------------------
# Collate
# ------------------------------------------------------------------------------
def collate_fn(batch, processor, max_length):
    images = [x["image"] for x in batch]
    captions = [x["caption"] for x in batch]

    encoding = processor(
        images=images,
        text=captions,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    labels = encoding["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    encoding["labels"] = labels
    return encoding


# ------------------------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------------------------
def save_checkpoint(model, processor, optimizer, scheduler, epoch, path):
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    torch.save(
        {
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        },
        os.path.join(path, "training_state.pt"),
    )
    logger.info(f"Saved checkpoint: {path}")


def load_checkpoint(checkpoint_dir, model, optimizer, scheduler, device):
    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if os.path.exists(state_path):
        state = torch.load(state_path, map_location=device)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        logger.info(f"Resumed from epoch {state['epoch'] + 1}")
        return state["epoch"]
    return 0


# ------------------------------------------------------------------------------
# Training / Validation
# ------------------------------------------------------------------------------
def train_one_epoch(model, dataloader, optimizer, scheduler, device, scaler, grad_accum):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    progress = tqdm(dataloader, desc="Train", leave=False)
    for step, batch in enumerate(progress):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum
        progress.set_postfix({"loss": f"{loss.item() * grad_accum:.4f}"})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Valid", leave=False):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            total_loss += outputs.loss.item()

    return total_loss / len(dataloader)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stage 1: Flickr8k BLIP Fine-Tuning")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--freeze_vision", action="store_true", default=False,
                        help="Freeze vision encoder (optional)")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    global logger
    logger = setup_logging(args.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Free VRAM before load: {torch.cuda.mem_get_info()[0] / 1e9:.2f} GB")

    logger.info("Loading BLIP base model (~400M params)...")
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base",
        use_safetensors=True,
    ).to(device)

    if args.freeze_vision:
        for param in model.vision_model.parameters():
            param.requires_grad = False
        logger.info("Vision encoder frozen.")

    model.gradient_checkpointing_enable()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
    if torch.cuda.is_available():
        logger.info(f"VRAM used after load: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    dataset = Flickr8kDataset(args.data_path, args.image_dir)
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )
    logger.info(f"Train: {len(train_set)} | Val: {len(val_set)}")

    collate = partial(collate_fn, processor=processor, max_length=args.max_length)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=collate)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                   num_training_steps=total_steps)
    logger.info(f"Steps/epoch: {steps_per_epoch} | Total: {total_steps} | Warmup: {warmup_steps}")

    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler, device)
        start_epoch += 1

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_dir = os.path.join(args.output_dir, "best_model")
    latest_model_dir = os.path.join(args.output_dir, "latest_model")

    for epoch in range(start_epoch, args.epochs):
        logger.info(f"=== Epoch {epoch + 1}/{args.epochs} ===")
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device, scaler, args.grad_accum)
        val_loss = validate(model, val_loader, device)
        logger.info(f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_checkpoint(model, processor, optimizer, scheduler, epoch, best_model_dir)
            logger.info(f"*** New best val loss: {best_val_loss:.4f} ***")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= args.patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                save_checkpoint(model, processor, optimizer, scheduler, epoch, latest_model_dir)
                break

        save_checkpoint(model, processor, optimizer, scheduler, epoch, latest_model_dir)

    logger.info("Stage 1 (BLIP) complete!")
    logger.info(f"Best model at: {best_model_dir}")


if __name__ == "__main__":
    main()