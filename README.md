# Disaster Image Captioning

Generating natural language captions for disaster imagery using vision-language models, as part of a research project targeting IEEE GLOBECOM 2026 (semantic communication for disaster response).

## Overview

This project fine-tunes and compares multiple vision-language models for captioning disaster-related images. The goal is to produce accurate, information-dense captions that can support downstream semantic communication and relevance-scoring pipelines.

Models are trained in two stages:
- Stage 1: Fine-tuning on Flickr8k (general-purpose image-caption pairs) for base captioning ability.
- Stage 2: Fine-tuning on a custom disaster imagery dataset with manually written captions.

## Models

| Model | Status | Notes |
|---|---|---|
| BLIP-base | Done (Stage 1 + Stage 2) | Baseline model |
| Florence-2-base | In progress | Stage 1 training on Flickr8k underway |
| TBD | Planned | Additional comparison model |
| TBD | Planned | Additional comparison model |

## Dataset

- Stage 1: Flickr8k (~8,091 images, image,caption pairs)
- Stage 2: Custom disaster dataset (~2,169 images, manually captioned - each image reviewed and captioned by hand, 12-15 word factual captions)

Datasets are not included in this repo due to size.

## Evaluation Metrics

Models are evaluated using standard image captioning metrics:
- BLEU-1 to BLEU-4
- METEOR
- ROUGE-L
- CIDEr

Results: in progress - table to be added once evaluation across all models is complete.

| Model | BLEU-4 | METEOR | ROUGE-L | CIDEr |
|---|---|---|---|---|
| BLIP-base | TBD | TBD | TBD | TBD |
| Florence-2-base | TBD | TBD | TBD | TBD |

## Sample Captions

TBD - example disaster images with generated captions from each model, for qualitative comparison.

## Repository Structure

disaster-image-captioning/
- florence_model/ - Florence-2 training and evaluation scripts
- model_blip/ - BLIP-base training and evaluation scripts
- model_blip2/ - BLIP-2 experiments
- requirements.txt - Python dependencies
- bleu_eval_results.csv - BLEU evaluation output

## Hardware

- GPU: RTX 4060 (8GB VRAM)
- OS: Ubuntu (lab machine)

## Status

Actively in progress. This README will be updated as training completes, metrics are finalized, and additional models are added.
