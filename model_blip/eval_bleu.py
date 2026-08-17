import os
import csv
import argparse
import torch
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

# Reuses the exact same dataset class + train/val split logic as training,
# so this evaluates on the same held-out images the model never trained on.
from stage_2_new import build_train_val_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="checkpoints/stage2_v4_fixed/best_model",
                         help="Path to the trained checkpoint to evaluate.")
    parser.add_argument("--captions_file", default="data/Final_Dataset_Caption.xlsx")
    parser.add_argument("--image_dir", default="data/disaster_images")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42,
                         help="Must match the seed used during training to get the same val split.")
    parser.add_argument("--max_new_tokens", type=int, default=30)
    parser.add_argument("--output_csv", default="bleu_eval_results.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading model from {args.model_dir}")
    processor = BlipProcessor.from_pretrained(args.model_dir)
    model = BlipForConditionalGeneration.from_pretrained(args.model_dir, use_safetensors=True).to(device)
    model.eval()

    # augment=False here doesn't matter for train split (we only use val),
    # but keep the same seed so val_dataset is identical to training-time val.
    _, val_dataset = build_train_val_datasets(
        args.captions_file, args.image_dir, augment=False, val_split=args.val_split, seed=args.seed
    )
    print(f"Evaluating on {len(val_dataset)} validation images")

    smoothie = SmoothingFunction().method4  # avoids BLEU=0 on short captions with no 4-gram overlap
    weights = (0.25, 0.25, 0.25, 0.25)  # standard BLEU-4: equal weight on 1..4-grams

    references = []
    hypotheses = []
    rows = []

    for idx in tqdm(range(len(val_dataset)), desc="Generating captions"):
        image, ref_caption = val_dataset[idx]
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        gen_caption = processor.decode(out_ids[0], skip_special_tokens=True)

        ref_tokens = ref_caption.lower().split()
        gen_tokens = gen_caption.lower().split()

        bleu4 = sentence_bleu([ref_tokens], gen_tokens, weights=weights, smoothing_function=smoothie)

        references.append([ref_tokens])
        hypotheses.append(gen_tokens)
        rows.append((ref_caption, gen_caption, f"{bleu4:.4f}"))

    corpus_bleu4 = corpus_bleu(references, hypotheses, weights=weights, smoothing_function=smoothie)

    print(f"\nCorpus BLEU-4: {corpus_bleu4:.4f}")
    print("(This is the number to report — corpus-level BLEU is far more stable than averaging per-sample scores.)")

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["reference_caption", "generated_caption", "bleu4"])
        writer.writerows(rows)
    print(f"Per-sample results saved to {args.output_csv}")


if __name__ == "__main__":
    main()