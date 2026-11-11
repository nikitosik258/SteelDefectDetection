"""Final, Test-only evaluation of the frozen E1--E4 NEU-DET checkpoints.

The Test split is evaluated exactly once through this script. It is never used
to select a model, threshold, augmentation or training parameter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from PIL import Image
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

from src.audit_dataset import KNOWN_CLASSES


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ("E1", "E2", "E3", "E4")
IMG_SIZE = 320
VAL_CONF = 0.001  # Ultralytics mAP protocol; not a deployment threshold.
NMS_IOU = 0.7
MAX_DET = 300
FIXED_CONF = 0.25  # Frozen before Test evaluation.
MATCH_IOU = 0.50


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return payload


def _load_manifest() -> dict[str, Any]:
    return json.loads((ROOT / "reports" / "metrics" / "dataset_split.json").read_text(encoding="utf-8"))


def preflight() -> dict[str, Any]:
    """Verify checkpoint provenance, frozen split integrity and evaluation settings."""
    data_path = ROOT / "configs" / "data.yaml"
    data = check_det_dataset(str(data_path))
    names = [data["names"][index] for index in range(data["nc"])] if isinstance(data["names"], dict) else list(data["names"])
    if names != list(KNOWN_CLASSES):
        raise ValueError(f"Class order mismatch: {names} != {KNOWN_CLASSES}")
    manifest = _load_manifest()
    assignments = manifest["assignments"]
    split_ids = {split: {item["image_id"] for item in assignments if item["split"] == split} for split in ("train", "val", "test")}
    if any(split_ids[left] & split_ids[right] for left in split_ids for right in split_ids if left < right):
        raise ValueError("Dataset manifest contains split overlap")
    test_dir = Path(data["test"])
    test_images = sorted(test_dir.glob("*.jpg"))
    if len(test_images) != len(split_ids["test"]) or len(test_images) != 360:
        raise ValueError(f"Unexpected Test image count: {len(test_images)}")
    for image in test_images:
        label = test_dir.parents[1] / "labels" / "test" / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"Missing Test label: {label}")
    configs: dict[str, dict[str, Any]] = {}
    for experiment in EXPERIMENTS:
        run = ROOT / "runs" / experiment
        checkpoint = run / "weights" / "best.pt"
        status = json.loads((run / "status.json").read_text(encoding="utf-8"))
        config = _load_yaml(run / "effective_config.yaml")
        if not checkpoint.is_file() or status.get("status") != "completed":
            raise ValueError(f"{experiment} is incomplete")
        if config["data"] != "configs/data.yaml" or config["imgsz"] != IMG_SIZE:
            raise ValueError(f"{experiment} does not match the frozen evaluation data/image size")
        configs[experiment] = config
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the configured final evaluation")
    return {
        "data_yaml": str(data_path.relative_to(ROOT)).replace("\\", "/"),
        "test_images": len(test_images),
        "class_names": names,
        "configs": configs,
        "test_ids": sorted(split_ids["test"]),
    }


def _read_ground_truth(label_path: Path, image_path: Path) -> list[dict[str, Any]]:
    with Image.open(image_path) as image:
        width, height = image.size
    entries = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        class_id, xc, yc, box_w, box_h = line.split()
        class_id = int(class_id)
        values = [float(item) for item in (xc, yc, box_w, box_h)]
        if class_id not in range(len(KNOWN_CLASSES)) or not (0 <= values[0] <= 1 and 0 <= values[1] <= 1 and 0 < values[2] <= 1 and 0 < values[3] <= 1):
            raise ValueError(f"Invalid test label: {label_path}")
        x1, y1 = (values[0] - values[2] / 2) * width, (values[1] - values[3] / 2) * height
        x2, y2 = (values[0] + values[2] / 2) * width, (values[1] + values[3] / 2) * height
        entries.append({"class_id": class_id, "xyxy": (x1, y1, x2, y2)})
    return entries


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    ix1, iy1, ix2, iy2 = max(left[0], right[0]), max(left[1], right[1]), min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (left[2] - left[0]) * (left[3] - left[1]) + (right[2] - right[0]) * (right[3] - right[1]) - intersection
    return intersection / union if union > 0 else 0.0


def match_fixed_protocol(predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Calculate P/R/F1 counts with class-aware, one-to-one IoU matching.

    A prediction is eligible only for an unused GT of the same class. This
    deliberately prevents a stronger overlap with a different class from
    stealing a valid same-class match.
    """
    statistics = {name: {"tp": 0, "fp": 0, "fn": 0} for name in KNOWN_CLASSES}
    used_gt: set[int] = set()
    for prediction in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        predicted_class = prediction["class_id"]
        candidates = [
            (index, _iou(prediction["xyxy"], target["xyxy"]))
            for index, target in enumerate(ground_truth)
            if index not in used_gt and target["class_id"] == predicted_class
        ]
        index, best_iou = max(candidates, key=lambda item: item[1], default=(-1, 0.0))
        if index >= 0 and best_iou >= MATCH_IOU:
            used_gt.add(index)
            statistics[KNOWN_CLASSES[predicted_class]]["tp"] += 1
        else:
            statistics[KNOWN_CLASSES[predicted_class]]["fp"] += 1
    for index, target in enumerate(ground_truth):
        if index not in used_gt:
            statistics[KNOWN_CLASSES[target["class_id"]]]["fn"] += 1
    return statistics


def match_confusion_matrix(predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]) -> np.ndarray:
    """Create a diagnostic matrix with independent class-agnostic matching.

    Cross-class matches are intentional here: they occupy off-diagonal cells.
    This function is never used to calculate fixed Precision, Recall or F1.
    """
    count = len(KNOWN_CLASSES)
    matrix = np.zeros((count + 1, count + 1), dtype=int)
    used_gt: set[int] = set()
    for prediction in sorted(predictions, key=lambda item: item["confidence"], reverse=True):
        candidates = [(index, _iou(prediction["xyxy"], target["xyxy"])) for index, target in enumerate(ground_truth) if index not in used_gt]
        index, best_iou = max(candidates, key=lambda item: item[1], default=(-1, 0.0))
        predicted_class = prediction["class_id"]
        if index >= 0 and best_iou >= MATCH_IOU:
            used_gt.add(index)
            matrix[ground_truth[index]["class_id"], predicted_class] += 1
        else:
            matrix[count, predicted_class] += 1
    for index, target in enumerate(ground_truth):
        if index not in used_gt:
            matrix[target["class_id"], count] += 1
    return matrix


def _metrics(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {**counts, "precision": precision, "recall": recall, "f1": f1}


def evaluate_model(experiment: str, test_images: list[Path]) -> dict[str, Any]:
    """Run standard mAP and the separate frozen P/R/F1 protocol on Test."""
    run_dir = ROOT / "runs" / experiment
    args = _load_yaml(run_dir / "args.yaml")
    model = YOLO(str(run_dir / "weights" / "best.pt"))
    if not all(torch.isfinite(parameter).all().item() for parameter in model.model.parameters()):
        raise ValueError(f"{experiment} contains non-finite best.pt weights")
    standard = model.val(data=str(ROOT / args["data"]), split="test", imgsz=IMG_SIZE, batch=int(args["batch"]), conf=VAL_CONF, iou=NMS_IOU, max_det=MAX_DET, device=args["device"], plots=False, project=str(ROOT / "runs" / "_test_evaluation"), name=f"{experiment}_map", exist_ok=True, verbose=False)
    ap50 = [float(value) for value in standard.box.ap50]
    ap50_95 = [float(value) for value in standard.box.maps]
    totals = {name: {"tp": 0, "fp": 0, "fn": 0} for name in KNOWN_CLASSES}
    confusion = np.zeros((len(KNOWN_CLASSES) + 1, len(KNOWN_CLASSES) + 1), dtype=int)
    image_records = []
    speeds = []
    label_dir = ROOT / "data" / "processed" / "labels" / "test"
    for result in model.predict(source=[str(path) for path in test_images], imgsz=IMG_SIZE, conf=FIXED_CONF, iou=NMS_IOU, max_det=MAX_DET, device=args["device"], stream=True, verbose=False):
        image_path = Path(result.path)
        predictions = [
            {"class_id": int(class_id), "confidence": float(confidence), "xyxy": tuple(map(float, box))}
            for box, confidence, class_id in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy(), strict=True)
        ]
        ground_truth = _read_ground_truth(label_dir / f"{image_path.stem}.txt", image_path)
        counts = match_fixed_protocol(predictions, ground_truth)
        matrix = match_confusion_matrix(predictions, ground_truth)
        for name in KNOWN_CLASSES:
            for key in ("tp", "fp", "fn"):
                totals[name][key] += counts[name][key]
        confusion += matrix
        speeds.append(float(result.speed.get("inference", 0.0)))
        image_records.append({"image_id": image_path.stem, "ground_truth_count": len(ground_truth), "prediction_count": len(predictions), "predictions": predictions})
    per_class = {
        name: {"ap50": ap50[index], "ap50_95": ap50_95[index], **_metrics(totals[name])}
        for index, name in enumerate(KNOWN_CLASSES)
    }
    micro = _metrics({key: sum(totals[name][key] for name in KNOWN_CLASSES) for key in ("tp", "fp", "fn")})
    return {
        "experiment": experiment,
        "model": _load_yaml(run_dir / "effective_config.yaml")["model"],
        "checkpoint": str((run_dir / "weights" / "best.pt").relative_to(ROOT)).replace("\\", "/"),
        "standard_ultralytics_map": {
            "conf": VAL_CONF,
            "nms_iou": NMS_IOU,
            "max_det": MAX_DET,
            "precision_reference": float(standard.box.mp),
            "recall_reference": float(standard.box.mr),
            "map50": float(standard.box.map50),
            "map50_95": float(standard.box.map),
        },
        "fixed_pr_f1_protocol": {"confidence": FIXED_CONF, "matching_iou": MATCH_IOU, "nms_iou": NMS_IOU, "max_det": MAX_DET, "micro": micro, "per_class": per_class},
        "mean_inference_ms_per_image": float(np.mean(speeds)) if speeds else 0.0,
        "confusion_matrix": confusion.tolist(),
        "image_records": image_records,
    }


def recompute_fixed_from_saved_predictions(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Recompute only fixed P/R/F1 from saved predictions; do not run inference."""
    label_dir = ROOT / "data" / "processed" / "labels" / "test"
    image_dir = ROOT / "data" / "processed" / "images" / "test"
    updated = payload["experiments"]
    for item in updated:
        fixed = item["fixed_pr_f1_protocol"]
        if (fixed["confidence"], fixed["matching_iou"], fixed["nms_iou"]) != (FIXED_CONF, MATCH_IOU, NMS_IOU):
            raise ValueError(f"{item['experiment']} does not use the frozen fixed-metric protocol")
        item["fixed_pr_f1_protocol_before_class_aware_fix"] = json.loads(json.dumps(fixed))
        totals = {name: {"tp": 0, "fp": 0, "fn": 0} for name in KNOWN_CLASSES}
        confusion = np.zeros((len(KNOWN_CLASSES) + 1, len(KNOWN_CLASSES) + 1), dtype=int)
        for record in item["image_records"]:
            image_path = image_dir / f"{record['image_id']}.jpg"
            ground_truth = _read_ground_truth(label_dir / f"{record['image_id']}.txt", image_path)
            counts = match_fixed_protocol(record["predictions"], ground_truth)
            matrix = match_confusion_matrix(record["predictions"], ground_truth)
            for name in KNOWN_CLASSES:
                for key in ("tp", "fp", "fn"):
                    totals[name][key] += counts[name][key]
            confusion += matrix
        per_class = {
            name: {
                "ap50": item["fixed_pr_f1_protocol"]["per_class"][name]["ap50"],
                "ap50_95": item["fixed_pr_f1_protocol"]["per_class"][name]["ap50_95"],
                **_metrics(totals[name]),
            }
            for name in KNOWN_CLASSES
        }
        micro = _metrics({key: sum(totals[name][key] for name in KNOWN_CLASSES) for key in ("tp", "fp", "fn")})
        item["fixed_pr_f1_protocol"] = {
            **{key: fixed[key] for key in ("confidence", "matching_iou", "nms_iou", "max_det")},
            "matching_method": "class-aware, score-ordered, one-to-one",
            "micro": micro,
            "per_class": per_class,
        }
        item["confusion_matrix"] = confusion.tolist()
    return updated


def _finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def write_outputs(preflight_report: dict[str, Any], results: list[dict[str, Any]]) -> None:
    reports = ROOT / "reports"
    metrics_dir, tables_dir, figures_dir = reports / "metrics", reports / "tables", reports / "figures"
    for path in (metrics_dir, tables_dir, figures_dir):
        path.mkdir(parents=True, exist_ok=True)
    payload = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "evaluation_split": "test", "test_used_for": "final evaluation only", "preflight": preflight_report, "evaluation_settings": {"imgsz": IMG_SIZE, "map_conf": VAL_CONF, "nms_iou": NMS_IOU, "max_det": MAX_DET, "fixed_confidence": FIXED_CONF, "fixed_matching_iou": MATCH_IOU}, "experiments": results}
    if not _finite(payload):
        raise ValueError("Non-finite metric detected")
    (metrics_dir / "test_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (metrics_dir / "test_image_predictions.jsonl").open("w", encoding="utf-8") as stream:
        for item in results:
            for record in item["image_records"]:
                stream.write(json.dumps({"experiment": item["experiment"], **record}, ensure_ascii=False) + "\n")
    comparison_rows, class_rows = [], []
    for item in results:
        map_metrics, fixed = item["standard_ultralytics_map"], item["fixed_pr_f1_protocol"]
        comparison_rows.append({"experiment": item["experiment"], "model": item["model"], "mAP50": map_metrics["map50"], "mAP50_95": map_metrics["map50_95"], "ultralytics_precision_reference": map_metrics["precision_reference"], "ultralytics_recall_reference": map_metrics["recall_reference"], "fixed_precision": fixed["micro"]["precision"], "fixed_recall": fixed["micro"]["recall"], "fixed_f1": fixed["micro"]["f1"], "tp": fixed["micro"]["tp"], "fp": fixed["micro"]["fp"], "fn": fixed["micro"]["fn"], "mean_inference_ms_per_image": item["mean_inference_ms_per_image"], "imgsz": IMG_SIZE, "map_conf": VAL_CONF, "fixed_confidence": FIXED_CONF, "matching_iou": MATCH_IOU, "nms_iou": NMS_IOU})
        for class_name, metric in fixed["per_class"].items():
            class_rows.append({"experiment": item["experiment"], "class": class_name, **metric})
    pd.DataFrame(comparison_rows).to_csv(tables_dir / "model_comparison.csv", index=False)
    pd.DataFrame(class_rows).to_csv(tables_dir / "per_class_metrics.csv", index=False)
    _plot_metrics(comparison_rows, figures_dir / "metrics_comparison.png")
    _plot_per_class(class_rows, figures_dir / "per_class_ap.png")
    for item in results:
        _plot_confusion(np.array(item["confusion_matrix"]), item["experiment"], figures_dir / f"confusion_matrix_{item['experiment']}.png")
    _write_markdown(comparison_rows, class_rows, results, reports / "model_evaluation_report.md")


def _plot_metrics(rows: list[dict[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows).set_index("experiment")
    values = frame[["mAP50", "mAP50_95", "fixed_precision", "fixed_recall", "fixed_f1"]]
    axis = values.plot.bar(figsize=(8.0, 4.5), ylim=(0, 1), rot=0)
    axis.set_ylabel("Metric value")
    axis.set_xlabel("Experiment")
    axis.set_title("Final Test metrics: common fixed protocols")
    axis.legend(loc="upper right", fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(path, dpi=300)
    plt.close(axis.figure)


def _plot_per_class(rows: list[dict[str, Any]], path: Path) -> None:
    frame = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for axis, column, title in zip(axes, ("ap50", "ap50_95"), ("Per-class AP@50", "Per-class AP@50:95"), strict=True):
        sns.barplot(data=frame, x="class", y=column, hue="experiment", ax=axis)
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.set_xlabel("Class")
        axis.set_ylabel("Average precision")
        axis.tick_params(axis="x", rotation=35)
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_confusion(matrix: np.ndarray, experiment: str, path: Path) -> None:
    labels = list(KNOWN_CLASSES) + ["background"]
    fig, axis = plt.subplots(figsize=(7, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=axis, cbar=False)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(f"{experiment}: fixed conf=0.25, IoU match=0.50")
    axis.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _write_markdown(rows: list[dict[str, Any]], class_rows: list[dict[str, Any]], results: list[dict[str, Any]], path: Path) -> None:
    frame, per_class = pd.DataFrame(rows), pd.DataFrame(class_rows)
    best_map = frame.loc[frame["mAP50_95"].idxmax(), "experiment"]
    best_recall = frame.loc[frame["fixed_recall"].idxmax(), "experiment"]
    comparisons = []
    for left, right, purpose in (("E1", "E2", "architecture without augmentation"), ("E3", "E4", "architecture with augmentation"), ("E1", "E3", "augmentation effect for YOLOv8n"), ("E2", "E4", "augmentation effect for YOLO11n")):
        first, second = frame.set_index("experiment").loc[left], frame.set_index("experiment").loc[right]
        comparisons.append(f"- {left} → {right} ({purpose}): ΔmAP@50:95={(second['mAP50_95']-first['mAP50_95'])*100:+.2f} pp; Δfixed F1={(second['fixed_f1']-first['fixed_f1'])*100:+.2f} pp.")
    comparisons = [
        f"- {left} to {right} ({purpose}): delta mAP@50:95={(frame.set_index('experiment').loc[right, 'mAP50_95'] - frame.set_index('experiment').loc[left, 'mAP50_95']) * 100:+.2f} pp; "
        f"delta fixed F1={(frame.set_index('experiment').loc[right, 'fixed_f1'] - frame.set_index('experiment').loc[left, 'fixed_f1']) * 100:+.2f} pp."
        for left, right, purpose in (("E1", "E2", "architecture without augmentation"), ("E3", "E4", "architecture with augmentation"), ("E1", "E3", "augmentation effect for YOLOv8n"), ("E2", "E4", "augmentation effect for YOLO11n"))
    ]
    columns = ["experiment", "model", "mAP50", "mAP50_95", "fixed_precision", "fixed_recall", "fixed_f1", "tp", "fp", "fn", "mean_inference_ms_per_image"]
    display = frame[columns].copy()
    for column in ("mAP50", "mAP50_95", "fixed_precision", "fixed_recall", "fixed_f1", "mean_inference_ms_per_image"):
        display[column] = display[column].map(lambda value: f"{value:.5f}")
    table = "| " + " | ".join(columns) + " |" + chr(10)
    table += "|" + "|".join(["---"] * len(columns)) + "|" + chr(10)
    table += chr(10).join("| " + " | ".join(str(row[column]) for column in columns) + " |" for _, row in display.iterrows())
    hard = per_class.groupby("class")["ap50_95"].mean().sort_values().head(2)
    text = "\n".join([
        "# Final Test evaluation of E1--E4",
        "",
        "## Method",
        "",
        "Test was used only for this final evaluation. Standard Ultralytics mAP uses `split=test`, `imgsz=320`, `conf=0.001`, NMS IoU=0.7 and `max_det=300` for every checkpoint. NMS IoU controls suppression of predictions; it is distinct from the IoU thresholds used to define true positives in AP calculation.",
        "",
        "The separate fixed P/R/F1 protocol uses confidence=0.25, NMS IoU=0.7 and IoU=0.50. Its one-to-one matching is score-ordered and class-aware: a prediction can match only an unused GT of the same class. Confusion matrices use a separate, class-agnostic matching procedure only to display off-diagonal label errors. Ultralytics reference Precision/Recall are retained in CSV/JSON but are not substituted for fixed-threshold P/R/F1.",
        "",
        "## Overall results",
        "",
        table,
        "",
        f"Best by Test mAP@50:95: **{best_map}**. Best by fixed-threshold Recall: **{best_recall}**.",
        "",
        "## Comparison (absolute percentage points)",
        "",
        *comparisons,
        "",
        "## Per-class observations",
        "",
        f"Lowest mean AP@50:95 across the four fixed experiments: {', '.join(f'`{name}` ({value:.3f})' for name, value in hard.items())}.",
        "Per-class AP and fixed-threshold metrics are in `reports/tables/per_class_metrics.csv`; fixed-protocol confusion matrices are stored separately for each experiment.",
        "",
        "## Validity limits",
        "",
        "Results are one run per configuration at seed=42; observed differences are not claims of statistical significance. They describe this NEU-DET split and do not guarantee production performance. No configuration, threshold, epoch budget, augmentation strategy or model was changed after inspecting Test.",
    ]) + "\n"
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the one final Test evaluation for frozen E1--E4 checkpoints.")
    parser.add_argument("--recompute-fixed-from-saved", action="store_true", help="Recalculate only fixed P/R/F1 from saved Test predictions; do not run model inference.")
    arguments = parser.parse_args()
    if arguments.recompute_fixed_from_saved:
        result_path = ROOT / "reports" / "metrics" / "test_results.json"
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        results = recompute_fixed_from_saved_predictions(payload)
        write_outputs(payload["preflight"], results)
        refreshed = json.loads(result_path.read_text(encoding="utf-8"))
        refreshed["fixed_metric_methodology"] = {
            "corrected": True,
            "procedure": "class-aware, score-ordered, one-to-one matching for fixed Precision/Recall/F1",
            "confusion_matrix": "separate class-agnostic, score-ordered, one-to-one matching for diagnostic off-diagonal errors",
            "inference_rerun": False,
            "standard_ultralytics_map_changed": False,
        }
        result_path.write_text(json.dumps(refreshed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Fixed P/R/F1 recalculated from saved predictions; no inference was run.")
        return 0
    checks = preflight()
    test_images = [ROOT / "data" / "processed" / "images" / "test" / f"{image_id}.jpg" for image_id in checks["test_ids"]]
    results = [evaluate_model(experiment, test_images) for experiment in EXPERIMENTS]
    write_outputs(checks, results)
    print(json.dumps({"evaluated": EXPERIMENTS, "test_images": len(test_images), "outputs": ["reports/metrics/test_results.json", "reports/tables/model_comparison.csv", "reports/tables/per_class_metrics.csv", "reports/model_evaluation_report.md"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
