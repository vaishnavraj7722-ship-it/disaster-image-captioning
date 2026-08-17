import os
import csv
import argparse
import logging
import math
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Subset, DataLoader
from torch.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration, get_linear_schedule_with_warmup
from torch.optim import AdamW

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class DisasterDataset(Dataset):
    def __init__(self, captions_file: str, image_dir: str, augment: bool = False):
        self.image_dir = image_dir
        self.samples = []
        self.augment = augment
        df = pd.read_excel(captions_file, header=None, names=["filename", "caption"])
        for _, row in df.iterrows():
            filename = str(row["filename"]).strip()
            caption = str(row["caption"]).strip()
            if not filename or not caption:
                continue
            self.samples.append((filename, caption))
        unique_images = len(set(f for f, _ in self.samples))
        logger.info(f"Loaded {len(self.samples)} captions from {unique_images} images (augment={augment})")
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, caption = self.samples[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        if self.augment:
            image = self.aug_transform(image)
        return image, caption


def collate_fn(batch, processor):
    images, captions = zip(*batch)
    inputs = processor(images=list(images), text=list(captions), return_tensors="pt", padding=True, truncation=True)
    labels = inputs["input_ids"].clone()
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    inputs["labels"] = labels
    return inputs


def apply_dropout_override(model, dropout_p):
    count = 0
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = dropout_p
            count += 1
    logger.info(f"Overrode dropout to {dropout_p} on {count} layers")


def freeze_vision_encoder(model):
    frozen_params = 0
    for param in model.vision_model.parameters():
        param.requires_grad = False
        frozen_params += param.numel()
    logger.info(f"Froze vision encoder: {frozen_params:,} parameters frozen")


def build_train_val_datasets(captions_file, image_dir, augment, val_split, seed=42):
    train_full = DisasterDataset(captions_file, image_dir, augment=augment)
    val_full = DisasterDataset(captions_file, image_dir, augment=False)
    dataset_len = len(train_full)
    assert dataset_len == len(val_full)
    val_size = int(dataset_len * val_split)
    train_size = dataset_len - val_size
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_len, generator=generator).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return Subset(train_full, train_indices), Subset(val_full, val_indices)


def save_checkpoint(model, processor, optimizer, scheduler, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    torch.save({"epoch": epoch, "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()}, os.path.join(save_dir, "training_state.pt"))
    logger.info(f"Saved checkpoint: {save_dir}")


def compute_loss_and_metrics(outputs, batch, args):
    logits = outputs.logits
    labels = batch["labels"]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    if args.label_smoothing > 0:
        loss_fct = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    else:
        loss = outputs.loss
    with torch.no_grad():
        preds = shift_logits.argmax(dim=-1)
        mask = shift_labels != -100
        num_correct = (preds == shift_labels).masked_select(mask).sum().item()
        num_tokens = mask.sum().item()
    return loss, num_correct, num_tokens


def init_metrics_log(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "metrics.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "train_ppl", "val_loss", "val_acc", "val_ppl"])
    return log_path


def append_metrics_log(log_path, epoch, train_loss, train_acc, val_loss, val_acc):
    train_ppl = math.exp(min(train_loss, 20))
    val_ppl = math.exp(min(val_loss, 20))
    with open(log_path, "a", newline="") as f:
        csv.writer(f).writerow([epoch + 1, f"{train_loss:.4f}", f"{train_acc:.4f}", f"{train_ppl:.2f}",
                                 f"{val_loss:.4f}", f"{val_acc:.4f}", f"{val_ppl:.2f}"])
    return train_ppl, val_ppl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions_file", default="data/Final_Dataset_Caption.xlsx")
    parser.add_argument("--image_dir", default="data/disaster_images")
    parser.add_argument("--stage1_model_dir", default="checkpoints/stage1_v2/best_model")
    parser.add_argument("--output_dir", default="checkpoints/stage2_v3")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--freeze_vision", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    logger.info(f"Loading Stage 1 model from {args.stage1_model_dir}")
    processor = BlipProcessor.from_pretrained(args.stage1_model_dir)
    model = BlipForConditionalGeneration.from_pretrained(args.stage1_model_dir, use_safetensors=True).to(device)

    if args.dropout is not None:
        apply_dropout_override(model, args.dropout)
    if args.freeze_vision:
        freeze_vision_encoder(model)

    train_dataset, val_dataset = build_train_val_datasets(args.captions_file, args.image_dir, args.augment, args.val_split, seed=args.seed)
    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, processor))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=lambda b: collate_fn(b, processor))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable params: {trainable_count:,} / {total_params:,} ({100*trainable_count/total_params:.1f}%)")

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    scaler = GradScaler(device=device, enabled=(device == "cuda"))

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_dir = os.path.join(args.output_dir, "best_model")
    latest_model_dir = os.path.join(args.output_dir, "latest_model")
    stopped_early = False
    metrics_log_path = init_metrics_log(args.output_dir)

    for epoch in range(args.epochs):
        logger.info(f"=== Epoch {epoch+1}/{args.epochs} ===")
        model.train()
        train_loss = 0
        train_correct = 0
        train_tokens = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1} [train]")
        for batch in train_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                outputs = model(**batch)
                loss, num_correct, num_tokens = compute_loss_and_metrics(outputs, batch, args)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
            train_correct += num_correct
            train_tokens += num_tokens
            batch_acc = num_correct / max(num_tokens, 1)
            train_bar.set_postfix(loss=loss.item(), acc=f"{batch_acc:.3f}")

        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_correct / max(train_tokens, 1)

        model.eval()
        val_loss = 0
        val_correct = 0
        val_tokens = 0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1} [val]")
        with torch.no_grad():
            for batch in val_bar:
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast(device_type=device, dtype=torch.float16, enabled=(device == "cuda")):
                    outputs = model(**batch)
                    loss, num_correct, num_tokens = compute_loss_and_metrics(outputs, batch, args)
                val_loss += loss.item()
                val_correct += num_correct
                val_tokens += num_tokens
                batch_acc = num_correct / max(num_tokens, 1)
                val_bar.set_postfix(loss=loss.item(), acc=f"{batch_acc:.3f}")

        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_correct / max(val_tokens, 1)

        train_ppl, val_ppl = append_metrics_log(metrics_log_path, epoch, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc)
        logger.info(f"Train loss: {avg_train_loss:.4f} | Train acc: {avg_train_acc:.4f} | Train ppl: {train_ppl:.2f} || "
                    f"Val loss: {avg_val_loss:.4f} | Val acc: {avg_val_acc:.4f} | Val ppl: {val_ppl:.2f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_checkpoint(model, processor, optimizer, scheduler, epoch, best_model_dir)
            logger.info(f"*** New best val loss: {best_val_loss:.4f} ***")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement for {epochs_no_improve} epoch(s)")
            if epochs_no_improve >= args.patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                stopped_early = True
                break

    save_checkpoint(model, processor, optimizer, scheduler, epoch, latest_model_dir)
    logger.info("Stage 2 complete!")
    logger.info(f"Best model at: {best_model_dir}")


if __name__ == "__main__":
    main()