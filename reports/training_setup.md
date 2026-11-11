# Training setup: YOLOv8n vs YOLO11n

## Scope and fixed protocol

This document records preparation of the E1--E4 training pipeline. It does **not** report research results: the four full experiments have not been started.

The frozen configuration is [`configs/experiments.yaml`](../configs/experiments.yaml). All experiments use the project split defined in `reports/dataset_preparation.md` (seed 42; 1224 / 216 / 360 train/validation/test images). The test split is not used by `src.train`.

| Experiment | Pretrained model | Profile |
| --- | --- | --- |
| E1 | `yolov8n.pt` | baseline |
| E2 | `yolo11n.pt` | baseline |
| E3 | `yolov8n.pt` | augmented |
| E4 | `yolo11n.pt` | augmented |

Shared settings are: detection task, 100 epochs, patience 20, batch 16, image size 320, AdamW, `lr0=0.001`, `lrf=0.01`, `weight_decay=0.0005`, `cos_lr=false`, three warm-up epochs, `seed=42`, deterministic mode, pretrained weights, and `multi_scale=false`. `close_mosaic=0` is safe because mosaic itself is zero. Ultralytics 8.3.239 expects `multi_scale` as Boolean; this is the documented representation of the requested `0.0`.

## Actual augmentation pipeline

The installed Ultralytics version inserts an Albumentations stage by default (including low-probability transforms), even when ordinary YOLO augmentation parameters are zero. `NoAlbumentationsDetectionTrainer` replaces that stage with an identity operation for the training dataset. This is verified against a real Ultralytics dataset in `reports/metrics/augmentation_pipeline.json` and visually in `reports/figures/augmentation_pipeline_check.png` (300 DPI).

The effective baseline profile has every random image/label augmentation probability or magnitude set to zero: HSV, geometric transforms, flips, mosaic, mixup, cutmix, copy-paste, erasing and BGR. The augmented profile only enables horizontal and vertical flips (0.5 each), rotation (5 degrees), translation (0.05), scale (0.10), and value shift (`hsv_v=0.10`). Hue and saturation remain zero because NEU-DET images are grayscale. `auto_augment` remains visible as `randaugment` in Ultralytics' global argument dump, but it is a classification-only transform and is not part of the detection transform chain; it is therefore inapplicable here.

## Compatibility and smoke verification

Verified environment: Python 3.12.10, PyTorch 2.7.1+cu128, torchvision 0.22.1+cu128, Ultralytics 8.3.239, NVIDIA GeForce RTX 4070 Laptop GPU (8 GiB VRAM), CUDA build 12.8. Both `yolov8n.pt` and `yolo11n.pt` were resolved as detection models.

One isolated smoke run used E1's baseline configuration with one epoch, 2% of train (24 images after Ultralytics fraction rounding), batch 16, image size 320 and zero workers. It passed AMP, trained and validated on the validation split, and wrote `best.pt` and `last.pt`. Its output is `runs/_smoke2/`; its numerical values are deliberately not used as E1 results. The initial implementation wrote metadata before trainer initialization, so Ultralytics chose `_smoke2` rather than `_smoke`; `src/train.py` has been corrected so future guarded runs preserve the requested directory name. The resume checkpoint-selection path is covered by an isolated mocked automated test; a real resume was not run because the smoke epoch had already completed. No E1--E4 artifact exists.

## Commands (PowerShell, repository root)

```powershell
# Validate an experiment without training.
.\.venv\Scripts\python.exe -m src.train --experiment E1 --dry-run

# Inspect actual baseline and augmented loaders; creates no model run.
.\.venv\Scripts\python.exe -m src.inspect_augmentations

# Optional isolated one-epoch verification; never writes runs/E1--runs/E4.
.\.venv\Scripts\python.exe -m src.train --experiment E1 --smoke-test

# Full experiment: execute only after explicit approval.
.\.venv\Scripts\python.exe -m src.train --experiment E1

# Continue only the matching interrupted run with runs/E1/weights/last.pt.
.\.venv\Scripts\python.exe -m src.train --experiment E1 --resume
```

`src.train` refuses to start a non-empty `runs/E1`--`runs/E4` directory without `--resume`. Each run stores `effective_config.yaml`, `status.json`, Ultralytics `args.yaml`, CSV metrics and checkpoints in its own directory. Use validation for model selection, hyperparameter decisions and confidence-threshold selection; reserve the fixed test split for final reporting only.

## Safe resume

Before `--resume` changes status or appends a history entry, `src.train` requires `weights/last.pt` and the original `effective_config.yaml`. It compares the requested experiment identifier, model, augmentation profile, dataset path and fixed protocol with both the original metadata and checkpoint training arguments. It also compares the checkpoint architecture YAML (with only `nc` excluded, because NEU-DET has six classes rather than COCO's 80). A mismatch or an unreadable checkpoint stops the command without modifying the existing metadata. Validated and completed resume attempts are appended to `run_history.jsonl`; `effective_config.yaml` remains the immutable original configuration.
