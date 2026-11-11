"""Verify completed E1/E2 checkpoints on Validation and write baseline reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ("E1", "E2")
METRIC_COLUMNS = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")
LOSS_COLUMNS = ("train/box_loss", "train/cls_loss", "train/dfl_loss", "val/box_loss", "val/cls_loss", "val/dfl_loss")
COMPARABILITY_KEYS = ("data", "epochs", "patience", "batch", "imgsz", "optimizer", "lr0", "lrf", "weight_decay", "cos_lr", "warmup_epochs", "seed", "deterministic", "pretrained", "multi_scale", "close_mosaic", "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "fliplr", "flipud", "mosaic", "mixup", "cutmix", "copy_paste", "erasing", "bgr")


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping: {path}")
    return payload


def _validation_of_best(experiment: str, run_dir: Path, args: dict[str, Any]) -> dict[str, Any]:
    """Load best.pt and evaluate only the configured Validation split."""
    checkpoint = run_dir / "weights" / "best.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing best checkpoint: {checkpoint}")
    model = YOLO(str(checkpoint))
    weights_finite = all(torch.isfinite(parameter).all().item() for parameter in model.model.parameters())
    if not weights_finite:
        raise ValueError(f"Non-finite model weights: {checkpoint}")
    metrics = model.val(
        data=str(ROOT / args["data"]),
        split="val",
        imgsz=int(args["imgsz"]),
        batch=int(args["batch"]),
        device=args["device"],
        plots=False,
        project=str(ROOT / "runs" / "_verification"),
        name=f"{experiment}_best_validation",
        exist_ok=True,
        verbose=False,
    )
    return {
        "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"),
        "weights_finite": weights_finite,
        "split": "val",
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "speed_ms_per_image": {key: float(value) for key, value in metrics.speed.items()},
    }


def summarize_experiment(experiment: str) -> dict[str, Any]:
    """Read training history and independently validate its selected checkpoint."""
    run_dir = ROOT / "runs" / experiment
    required = {"best.pt": run_dir / "weights" / "best.pt", "last.pt": run_dir / "weights" / "last.pt", "results.csv": run_dir / "results.csv", "args.yaml": run_dir / "args.yaml", "effective_config.yaml": run_dir / "effective_config.yaml", "status.json": run_dir / "status.json"}
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{experiment} is incomplete: {missing}")
    status = json.loads(required["status.json"].read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        raise ValueError(f"{experiment} status is not completed: {status.get('status')}")
    config, args = _load_yaml(required["effective_config.yaml"]), _load_yaml(required["args.yaml"])
    history = pd.read_csv(required["results.csv"])
    if history.empty:
        raise ValueError(f"Empty training history: {required['results.csv']}")
    nonfinite = not bool(np.isfinite(history[list(METRIC_COLUMNS) + list(LOSS_COLUMNS)].to_numpy(dtype=float)).all())
    if nonfinite:
        raise ValueError(f"Non-finite loss or metric in {required['results.csv']}")
    best = history.loc[history["metrics/mAP50-95(B)"].idxmax()]
    final = history.iloc[-1]
    epoch_gaps = history["time"].diff().dropna()
    median_epoch_seconds = float(epoch_gaps.median()) if not epoch_gaps.empty else 0.0
    max_epoch_gap_seconds = float(epoch_gaps.max()) if not epoch_gaps.empty else 0.0
    validation = _validation_of_best(experiment, run_dir, args)
    return {
        "experiment": experiment,
        "model": config["model"],
        "profile": config["profile"],
        "status": status["status"],
        "actual_epochs": int(len(history)),
        "best_epoch": int(best["epoch"]),
        "training_history_duration_seconds": float(final["time"]),
        "median_epoch_seconds": median_epoch_seconds,
        "max_recorded_interepoch_gap_seconds": max_epoch_gap_seconds,
        "history_has_wall_clock_gap": bool(median_epoch_seconds and max_epoch_gap_seconds > 10 * median_epoch_seconds),
        "history_has_nan_or_inf": nonfinite,
        "training_history_best_validation": {column: float(best[column]) for column in METRIC_COLUMNS},
        "loss_at_best_epoch": {column: float(best[column]) for column in LOSS_COLUMNS},
        "final_epoch_validation": {column: float(final[column]) for column in METRIC_COLUMNS},
        "possible_overfitting_indicator": bool(final["metrics/mAP50-95(B)"] < best["metrics/mAP50-95(B)"] and final["train/box_loss"] < best["train/box_loss"]),
        "artifacts": {name: str(path.relative_to(ROOT)).replace("\\", "/") for name, path in required.items()},
        "best_checkpoint_validation": validation,
        "effective_config": {key: config[key] for key in COMPARABILITY_KEYS},
    }


def write_reports(summary: dict[str, Any]) -> None:
    """Write machine-readable metrics and an article-ready concise account."""
    reports = ROOT / "reports"
    (reports / "metrics").mkdir(parents=True, exist_ok=True)
    (reports / "metrics" / "baseline_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    rows = []
    for item in summary["experiments"]:
        metrics = item["best_checkpoint_validation"]
        rows.append(
            f"| {item['experiment']} | {item['model']} | {item['actual_epochs']} | {item['best_epoch']} | {metrics['precision']:.5f} | {metrics['recall']:.5f} | {metrics['map50']:.5f} | {metrics['map50_95']:.5f} | {item['training_history_duration_seconds']:.2f} |"
        )
    observations = []
    for item in summary["experiments"]:
        if item["possible_overfitting_indicator"]:
            observations.append(f"- {item['experiment']}: final Validation mAP@50:95 is below its best-history value while train box loss is lower; this is a possible late-training overfitting indicator, not a basis for retuning.")
        else:
            observations.append(f"- {item['experiment']}: no simple late-training overfitting indicator under the recorded criterion.")
    text = "\n".join([
        "# Baseline training: E1 and E2",
        "",
        "All values below are obtained on **Validation**, never on Test. Test was not loaded by training or this report.",
        "",
        "| Experiment | Model | Epochs run | Best epoch | Precision | Recall | mAP@50 | mAP@50:95 | History duration, s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *rows,
        "",
        "## Comparability",
        "",
        f"The checked shared parameters are identical: {', '.join(COMPARABILITY_KEYS)}. The only intended difference is the pretrained architecture (`yolov8n.pt` versus `yolo11n.pt`). Both use the same processed split (Train/Validation/Test = 1223/216/360), but only Train and Validation were accessed.",
        "",
        "## Integrity checks",
        "",
        "- `best.pt`, `last.pt`, `results.csv`, `args.yaml`, `effective_config.yaml`, and completed run status exist for both experiments.",
        "- Histories contain no NaN or Inf in reported losses or validation metrics.",
        "- Both `best.pt` checkpoints were loaded and independently evaluated on Validation.",
        "",
        "## Observations",
        "",
        *observations,
        "",
        "No architecture-specific retuning or additional training was performed after these observations.",
    ]) + "\n"
    (reports / "baseline_training_report.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify E1/E2 best checkpoints on Validation and write baseline reports.")
    parser.parse_args()
    experiments = [summarize_experiment(experiment) for experiment in EXPERIMENTS]
    reference = experiments[0]["effective_config"]
    mismatches = {
        experiment["experiment"]: [key for key in COMPARABILITY_KEYS if experiment["effective_config"][key] != reference[key]]
        for experiment in experiments[1:]
    }
    if any(mismatches.values()):
        raise ValueError(f"Baseline configurations are not comparable: {mismatches}")
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_split": "val",
        "test_used": False,
        "comparability_mismatches": mismatches,
        "experiments": experiments,
    }
    write_reports(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
