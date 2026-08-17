import os
import argparse
import logging
import pandas as pd
from torch.amp import autocast, GradScaler
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, get_linear_schedule_with_warmup
from torch.optim import AdamW

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class DisasterDataset(Dataset):
    """
    Loads disaster image captions from Final_Dataset_Caption.xlsx.
    One caption per image (no deduplication needed, unlike Flickr8k).
    """
    def __init__(self, captions_file: str, image_dir: str):
        self.image_dir = image_dir
        self.samples = []

        df = pd.read_excel(captions_file, header=None, names=["filename", "caption"])

        for _, row in df.iterrows():
            filename = str(row["filename"]).strip()
            caption = str(row["caption"]).strip()
            if not filename or not caption:
                continue
            self.samples.append((filename, caption))

        unique_images = len(set(f for f, _ in self.samples))
        logger.info(f"Loaded {len(self.samples)} captions from {unique_images} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, caption = self.samples[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert("RGB")
        return image, caption


def collate_fn(batch, processor):
    images, captions = zip(*batch)
    inputs = processor(
        images=list(images),
        text=list(captions),
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs["labels"] = inputs["input_ids"].clone()
    return inputs


def save_checkpoint(model, processor, optimizer, scheduler, epoch, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    torch.save({
        "epoch": epoch,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }, os.path.join(save_dir, "training_state.pt"))
    logger.info(f"Saved checkpoint: {save_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions_file", default="data/Final_Dataset_Caption.xlsx")
    parser.add_argument("--image_dir", default="data/disaster_images")
    parser.add_argument("--stage1_model_dir", default="checkpoints/stage1_v2/best_model")
    parser.add_argument("--output_dir", default="checkpoints/stage2_v1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load Stage 1 checkpoint as starting point
    logger.info(f"Loading Stage 1 model from {args.stage1_model_dir}")
    processor = BlipProcessor.from_pretrained(args.stage1_model_dir)
    model = BlipForConditionalGeneration.from_pretrained(args.stage1_model_dir, use_safetensors=True).to(device)

    # Dataset + split
    full_dataset = DisasterDataset(args.captions_file, args.image_dir)
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    logger.info(f"Train: {train_size} | Val: {val_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, processor)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, processor)
    )


    optimizer = AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_dir = os.path.join(args.output_dir, "best_model")
    latest_model_dir = os.path.join(args.output_dir, "latest_model")

    for epoch in range(args.epochs):
        logger.info(f"=== Epoch {epoch+1}/{args.epochs} ===")

        model.train()
        train_loss = 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        logger.info(f"Train loss: {avg_train_loss:.4f} | Val loss: {avg_val_loss:.4f}")

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
                save_checkpoint(model, processor, optimizer, scheduler, epoch, latest_model_dir)
                break

    save_checkpoint(model, processor, optimizer, scheduler, epoch, latest_model_dir)
    logger.info("Stage 2 complete!")
    logger.info(f"Best model at: {best_model_dir}")


if __name__ == "__main__":
    main()