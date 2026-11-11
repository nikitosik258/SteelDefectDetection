"""Analyse frozen Test predictions without re-running detector inference."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from matplotlib.patches import Patch

from src.audit_dataset import KNOWN_CLASSES
from src.evaluate import FIXED_CONF, MATCH_IOU, NMS_IOU, _iou, _metrics, _read_ground_truth, match_fixed_protocol


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ("E1", "E2", "E3", "E4")
COLORS = {"gt": "#007f5f", "pred": "#d95f02"}


def _markdown(frame: pd.DataFrame) -> str:
    table = frame.reset_index() if frame.index.name else frame
    columns = [str(column) for column in table.columns]
    return "| " + " | ".join(columns) + " |\n|" + "|".join(["---"] * len(columns)) + "|\n" + "\n".join("| " + " | ".join(str(row[column]) for column in table.columns) + " |" for _, row in table.iterrows())


def _load() -> dict[str, Any]:
    return json.loads((ROOT / "reports" / "metrics" / "test_results.json").read_text(encoding="utf-8"))


def _categorize(predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, dict[str, int]], list[str]]:
    """Class-aware primary accounting plus diagnostic error categories."""
    per_class = match_fixed_protocol(predictions, ground_truth)
    categories = defaultdict(int)
    notes: list[str] = []
    used: set[int] = set()
    for prediction in sorted(predictions, key=lambda row: row["confidence"], reverse=True):
        same = [(i, _iou(prediction["xyxy"], gt["xyxy"])) for i, gt in enumerate(ground_truth) if gt["class_id"] == prediction["class_id"]]
        available = [(i, score) for i, score in same if i not in used]
        index, score = max(available, key=lambda row: row[1], default=(-1, 0.0))
        if index >= 0 and score >= MATCH_IOU:
            used.add(index)
            categories["tp"] += 1
            continue
        all_candidates = [(i, _iou(prediction["xyxy"], gt["xyxy"])) for i, gt in enumerate(ground_truth)]
        cross_i, cross_score = max(all_candidates, key=lambda row: row[1], default=(-1, 0.0))
        matched_same = [(i, score) for i, score in same if i in used and score >= MATCH_IOU]
        if cross_i >= 0 and cross_score >= MATCH_IOU and ground_truth[cross_i]["class_id"] != prediction["class_id"]:
            categories["classification_fp"] += 1
            notes.append("classification")
        elif matched_same:
            categories["duplicate_fp"] += 1
            notes.append("duplicate")
        elif cross_score > 0:
            categories["overlap_fp"] += 1
            notes.append("overlap")
        else:
            categories["background_fp"] += 1
            notes.append("background")
    for i, target in enumerate(ground_truth):
        if i not in used:
            area = (target["xyxy"][2] - target["xyxy"][0]) * (target["xyxy"][3] - target["xyxy"][1])
            categories["small_object_fn" if area < 32 * 32 else "fn"] += 1
    return dict(categories), per_class, notes


def _draw(image_id: str, gt: list[dict[str, Any]], predictions: list[dict[str, Any]], title: str, axis: plt.Axes, show_gt: bool = True, show_predictions: bool = True) -> None:
    path = ROOT / "data" / "processed" / "images" / "test" / f"{image_id}.jpg"
    image = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 8)
    if show_gt:
        for item in gt:
            draw.rectangle(item["xyxy"], outline=COLORS["gt"], width=2)
            draw.text((item["xyxy"][0], max(0, item["xyxy"][1] - 6)), f"GT {KNOWN_CLASSES[item['class_id']]}", fill=COLORS["gt"], font=font)
    if show_predictions:
        for item in predictions:
            draw.rectangle(item["xyxy"], outline=COLORS["pred"], width=2)
            draw.text((item["xyxy"][0], min(image.height - 6, item["xyxy"][1] + 1)), f"{KNOWN_CLASSES[item['class_id']]} {item['confidence']:.2f}", fill=COLORS["pred"], font=font)
    axis.imshow(image)
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def _select(records: dict[str, dict[str, dict[str, Any]]], gts: dict[str, list[dict[str, Any]]], experiment: str, kind: str) -> str:
    scored = []
    for image_id, record in records[experiment].items():
        categories, _, _ = _categorize(record["predictions"], gts[image_id])
        score = {"failure": categories.get("small_object_fn", 0) + categories.get("fn", 0) + categories.get("classification_fp", 0), "success": categories.get("tp", 0) - sum(value for key, value in categories.items() if key != "tp")}[kind]
        scored.append((score, image_id))
    return max(scored)[1]


def _write_figures(records: dict[str, dict[str, dict[str, Any]]], gts: dict[str, list[dict[str, Any]]]) -> None:
    figures = ROOT / "reports" / "figures"
    success, failure = _select(records, gts, "E3", "success"), _select(records, gts, "E3", "failure")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    _draw(success, gts[success], [], f"Ground Truth: {success}", axes[0], show_predictions=False)
    _draw(success, [], records["E3"][success]["predictions"], f"E3 predictions: {success}", axes[1], show_gt=False)
    fig.legend(handles=[Patch(color=COLORS["gt"], label="Ground Truth"), Patch(color=COLORS["pred"], label="E3 prediction")], loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Representative successful E3 detection; original pixels and bbox coordinates preserved", y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93)); fig.savefig(figures / "detection_examples.png", dpi=300); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    _draw(failure, gts[failure], [], f"Ground Truth: {failure}", axes[0], show_predictions=False)
    _draw(failure, [], records["E3"][failure]["predictions"], f"E3 predictions: {failure}", axes[1], show_gt=False)
    fig.legend(handles=[Patch(color=COLORS["gt"], label="Ground Truth"), Patch(color=COLORS["pred"], label="E3 prediction")], loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Representative E3 failure case; original pixels and bbox coordinates preserved", y=0.98)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(figures / "failure_cases.png", dpi=300)
    fig.savefig(figures / "failure_cases.pdf")
    plt.close(fig)
    # A single Test image, selected to maximize augmentation-related E1/E3 difference.
    candidates = []
    for image_id in gts:
        e1 = _categorize(records["E1"][image_id]["predictions"], gts[image_id])[0]
        e3 = _categorize(records["E3"][image_id]["predictions"], gts[image_id])[0]
        candidates.append((e3.get("tp", 0) - e1.get("tp", 0), image_id))
    comparison = max(candidates)[1]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for axis, experiment in zip(axes.flat, EXPERIMENTS, strict=True):
        _draw(comparison, gts[comparison], records[experiment][comparison]["predictions"], f"{experiment}: {comparison}", axis)
    fig.suptitle("Same Test image: GT green, predictions orange", y=0.995)
    fig.tight_layout(); fig.savefig(figures / "model_prediction_comparison.png", dpi=300); plt.close(fig)


def main() -> int:
    payload = _load()
    if payload.get("evaluation_split") != "test":
        raise ValueError("Expected frozen Test evaluation results")
    records = {item["experiment"]: {row["image_id"]: row for row in item["image_records"]} for item in payload["experiments"]}
    image_ids = sorted(records["E1"])
    image_dir, label_dir = ROOT / "data" / "processed" / "images" / "test", ROOT / "data" / "processed" / "labels" / "test"
    gts = {image_id: _read_ground_truth(label_dir / f"{image_id}.txt", image_dir / f"{image_id}.jpg") for image_id in image_ids}
    small_gt = sum((target["xyxy"][2] - target["xyxy"][0]) * (target["xyxy"][3] - target["xyxy"][1]) < 32 * 32 for targets in gts.values() for target in targets)
    rows, category_rows = [], []
    image_categories: dict[str, dict[str, dict[str, int]]] = {experiment: {} for experiment in EXPERIMENTS}
    for experiment in EXPERIMENTS:
        totals = {name: {"tp": 0, "fp": 0, "fn": 0} for name in KNOWN_CLASSES}; cats = defaultdict(int); multi = defaultdict(int)
        for image_id in image_ids:
            categories, counts, _ = _categorize(records[experiment][image_id]["predictions"], gts[image_id])
            image_categories[experiment][image_id] = categories
            for key, value in categories.items(): cats[key] += value
            if len(gts[image_id]) > 1: multi["images"] += 1; multi["gt_objects"] += len(gts[image_id]); multi["errors"] += sum(value for key, value in categories.items() if key != "tp")
            for name in KNOWN_CLASSES:
                for key in ("tp", "fp", "fn"): totals[name][key] += counts[name][key]
        primary = {key: sum(totals[name][key] for name in KNOWN_CLASSES) for key in ("tp", "fp", "fn")}
        stored = next(row for row in payload["experiments"] if row["experiment"] == experiment)["fixed_pr_f1_protocol"]["micro"]
        if primary != {key: stored[key] for key in primary}: raise ValueError(f"{experiment}: TP/FP/FN mismatch with frozen evaluation")
        metric = _metrics(primary)
        rows.append({"experiment": experiment, **metric, **{f"multi_{key}": value for key, value in multi.items()}})
        category_rows.extend({"experiment": experiment, "category": key, "count": value} for key, value in sorted(cats.items()))
    pd.DataFrame(rows).to_csv(ROOT / "reports" / "tables" / "error_analysis_summary.csv", index=False)
    pd.DataFrame(category_rows).to_csv(ROOT / "reports" / "tables" / "error_categories.csv", index=False)
    comparison_cases = []
    def errors(experiment: str, image_id: str) -> int:
        return sum(value for key, value in image_categories[experiment][image_id].items() if key != "tp")
    def tp(experiment: str, image_id: str) -> int:
        return image_categories[experiment][image_id].get("tp", 0)
    for image_id in image_ids:
        if all(errors(experiment, image_id) == 0 for experiment in EXPERIMENTS): comparison_cases.append({"category": "all_models_correct", "image_id": image_id})
        if all(errors(experiment, image_id) > 0 for experiment in EXPERIMENTS): comparison_cases.append({"category": "all_models_error", "image_id": image_id})
        for v8, v11, label in (("E1", "E2", "baseline"), ("E3", "E4", "augmented")):
            if tp(v8, image_id) > tp(v11, image_id) and errors(v11, image_id) > errors(v8, image_id): comparison_cases.append({"category": f"yolov8n_better_{label}", "image_id": image_id})
            if tp(v11, image_id) > tp(v8, image_id) and errors(v8, image_id) > errors(v11, image_id): comparison_cases.append({"category": f"yolo11n_better_{label}", "image_id": image_id})
        for base, augmented, label in (("E1", "E3", "yolov8n"), ("E2", "E4", "yolo11n")):
            if tp(augmented, image_id) > tp(base, image_id) or errors(augmented, image_id) < errors(base, image_id): comparison_cases.append({"category": f"augmentation_improved_{label}", "image_id": image_id})
            if tp(augmented, image_id) < tp(base, image_id) or errors(augmented, image_id) > errors(base, image_id): comparison_cases.append({"category": f"augmentation_worsened_{label}", "image_id": image_id})
    case_frame = pd.DataFrame(comparison_cases)
    case_frame.to_csv(ROOT / "reports" / "tables" / "model_comparison_cases.csv", index=False)
    _write_figures(records, gts)
    summary = pd.DataFrame(rows).set_index("experiment")
    categories = pd.DataFrame(category_rows).pivot(index="experiment", columns="category", values="count").fillna(0).astype(int)
    (ROOT / "reports" / "error_analysis_report.md").write_text(
        "# Test error analysis\n\n"
        "Fixed protocol: confidence=0.25, class-aware one-to-one matching IoU=0.50, NMS IoU=0.70. "
        "Counts use frozen predictions; no detector inference or training was repeated. Classification/overlap/duplicate labels are diagnostic categories for unmatched predictions and do not redefine TP/FP/FN. An overlap FP is an unmatched prediction with nonzero IoU against any GT, irrespective of class; it is not necessarily a pure localization error.\n\n"
        "## Primary counts\n\n" + _markdown(summary) + "\n\n## Diagnostic categories\n\n" + _markdown(categories) + "\n\n"
        + "## Cross-model case inventory\n\n" + _markdown(case_frame.groupby("category").size().rename("images").to_frame()) + "\n\n"
        f"Case-inventory categories can overlap because each records a distinct comparison condition. A small object is defined as bbox area below 1024 square pixels in original-image coordinates; Test contains {small_gt} such objects. Results were obtained with one seed and one fixed split and are descriptive, not significance tests.\n",
        encoding="utf-8",
    )
    print(json.dumps({"test_images": len(image_ids), "experiments": list(EXPERIMENTS), "outputs": ["reports/error_analysis_report.md", "reports/tables/error_analysis_summary.csv", "reports/tables/error_categories.csv", "reports/tables/model_comparison_cases.csv"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
