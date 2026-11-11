"""Verify augmented E3/E4 checkpoints and compare them with frozen baselines."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.report_baseline_results import COMPARABILITY_KEYS, ROOT, summarize_experiment


PAIRS = (("E1", "E3"), ("E2", "E4"))
AUGMENTATION_KEYS = ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective", "fliplr", "flipud", "mosaic", "mixup", "cutmix", "copy_paste", "erasing", "bgr")
METRICS = ("precision", "recall", "map50", "map50_95")


def _read_baselines() -> dict[str, dict[str, Any]]:
    path = ROOT / "reports" / "metrics" / "baseline_results.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("evaluation_split") != "val" or payload.get("test_used") is not False:
        raise ValueError("Baseline report is not a Validation-only report")
    return {item["experiment"]: item for item in payload["experiments"]}


def _verify_pair(baseline: dict[str, Any], augmented: dict[str, Any]) -> list[str]:
    """Return only differences beyond the protocol's intended augmentation profile."""
    unexpected = []
    for key in COMPARABILITY_KEYS:
        baseline_value = baseline["effective_config"][key]
        augmented_value = augmented["effective_config"][key]
        if key not in AUGMENTATION_KEYS and baseline_value != augmented_value:
            unexpected.append(key)
    for key in AUGMENTATION_KEYS:
        if key in ("hsv_h", "hsv_s", "shear", "perspective", "mosaic", "mixup", "cutmix", "copy_paste", "erasing", "bgr") and augmented["effective_config"][key] != 0.0:
            unexpected.append(f"unexpected augmented {key}")
    return unexpected


def _comparison(baseline: dict[str, Any], augmented: dict[str, Any]) -> dict[str, Any]:
    base_metrics, aug_metrics = baseline["best_checkpoint_validation"], augmented["best_checkpoint_validation"]
    changes = {metric: (aug_metrics[metric] - base_metrics[metric]) * 100.0 for metric in METRICS}
    return {
        "baseline": baseline["experiment"],
        "augmented": augmented["experiment"],
        "validation_metric_changes_percentage_points": changes,
        "unexpected_config_differences": _verify_pair(baseline, augmented),
    }


def write_reports(summary: dict[str, Any]) -> None:
    reports = ROOT / "reports"
    metrics_path = reports / "metrics" / "augmented_results.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    experiment_rows = []
    for item in summary["augmented_experiments"]:
        metric = item["best_checkpoint_validation"]
        experiment_rows.append(f"| {item['experiment']} | {item['model']} | {item['actual_epochs']} | {item['best_epoch']} | {metric['precision']:.5f} | {metric['recall']:.5f} | {metric['map50']:.5f} | {metric['map50_95']:.5f} | {item['training_history_duration_seconds']:.2f} |")
    comparison_rows = []
    for comparison in summary["comparisons"]:
        change = comparison["validation_metric_changes_percentage_points"]
        comparison_rows.append(f"| {comparison['baseline']} → {comparison['augmented']} | {change['precision']:+.2f} | {change['recall']:+.2f} | {change['map50']:+.2f} | {change['map50_95']:+.2f} |")
    observations = []
    for comparison in summary["comparisons"]:
        value = comparison["validation_metric_changes_percentage_points"]["map50_95"]
        direction = "increased" if value > 0 else "decreased" if value < 0 else "did not change"
        observations.append(f"- {comparison['baseline']} → {comparison['augmented']}: Validation mAP@50:95 {direction} by {abs(value):.2f} percentage points. This one-run comparison does not by itself prove a general causal effect.")
    timing_notes = [
        f"- {item['experiment']}: training-history wall-clock duration includes a {item['max_recorded_interepoch_gap_seconds']:.2f} s inter-epoch gap, so its total duration must not be compared as active GPU training time."
        for item in summary["augmented_experiments"]
        if item["history_has_wall_clock_gap"]
    ]
    text = "\n".join([
        "# Augmented training: E3 and E4",
        "",
        "All reported metrics come from independent evaluation of `best.pt` on **Validation**. Test was not used.",
        "",
        "| Experiment | Model | Epochs run | Best epoch | Precision | Recall | mAP@50 | mAP@50:95 | History duration, s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        *experiment_rows,
        "",
        "## Absolute Validation changes versus baseline",
        "",
        "Values are absolute percentage-point changes, not relative percentage changes.",
        "",
        "| Pair | Δ Precision, pp | Δ Recall, pp | Δ mAP@50, pp | Δ mAP@50:95, pp |",
        "| --- | ---: | ---: | ---: | ---: |",
        *comparison_rows,
        "",
        "## Protocol checks",
        "",
        "- Same processed Train/Validation split, image size, batch size, seed, optimizer, learning rate, scheduler, weight decay, AMP, early-stopping policy and epoch budget were used in each architecture pair.",
        "- The only planned difference is the augmented profile: horizontal/vertical flips, ±5° rotation, 0.05 translation, 0.10 scale and HSV value shift 0.10. Mosaic, mixup, cutmix and implicit Albumentations remain disabled.",
        "- E3/E4 started from `yolov8n.pt`/`yolo11n.pt` respectively; no E1/E2 weights were used.",
        "- All checkpoints and histories exist, and no NaN/Inf was found in history or best-model weights.",
        "",
        "## Interpretation",
        "",
        *observations,
        *timing_notes,
        "",
        "No Test evaluation, retuning, or extra experiment was performed.",
    ]) + "\n"
    (reports / "augmentation_training_report.md").write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify E3/E4 on Validation and compare fixed augmentation pairs.")
    parser.parse_args()
    baselines = _read_baselines()
    augmented = {experiment: summarize_experiment(experiment) for experiment in ("E3", "E4")}
    comparisons = [_comparison(baselines[base], augmented[aug]) for base, aug in PAIRS]
    if any(item["unexpected_config_differences"] for item in comparisons):
        raise ValueError(f"Unexpected E1/E3 or E2/E4 configuration differences: {comparisons}")
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_split": "val",
        "test_used": False,
        "augmented_experiments": [augmented["E3"], augmented["E4"]],
        "comparisons": comparisons,
    }
    write_reports(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
