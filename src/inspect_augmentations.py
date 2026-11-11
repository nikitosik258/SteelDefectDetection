"""Inspect the real Ultralytics augmentation pipeline without training a model."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics.cfg import get_cfg
from ultralytics.data.build import build_yolo_dataset
from ultralytics.data.utils import check_det_dataset

from src.train import ROOT, experiment_settings, load_protocol
from src.training_pipeline import describe_transforms, disable_implicit_albumentations
from src.audit_dataset import KNOWN_CLASSES


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_profile_dataset(profile: str):
    """Build a real training dataset and explicitly remove implicit Albumentations."""
    protocol = load_protocol()
    experiment = "E1" if profile == "baseline" else "E3"
    settings = experiment_settings(experiment, protocol)
    for key in ("model", "profile", "experiment"):
        settings.pop(key)
    args = get_cfg(overrides=settings)
    data = check_det_dataset(str(ROOT / args.data))
    dataset = build_yolo_dataset(args, data["train"], batch=4, data=data, mode="train", rect=False, stride=32)
    replacements = disable_implicit_albumentations(dataset)
    if replacements != 1:
        raise RuntimeError(f"Expected one implicit Albumentations stage, found {replacements}")
    return dataset, describe_transforms(dataset)


def denormalized_boxes(sample: dict[str, Any]) -> np.ndarray:
    boxes = sample["bboxes"].cpu().numpy()
    height, width = sample["img"].shape[1:]
    result = boxes.copy()
    result[:, [0, 2]] *= width
    result[:, [1, 3]] *= height
    result[:, 0] -= result[:, 2] / 2
    result[:, 1] -= result[:, 3] / 2
    return result


def draw_sample(axis: plt.Axes, sample: dict[str, Any], title: str) -> None:
    image = sample["img"].permute(1, 2, 0).cpu().numpy()
    axis.imshow(image)
    classes = sample["cls"].reshape(-1).cpu().numpy().astype(int)
    for (x, y, width, height), class_id in zip(denormalized_boxes(sample), classes, strict=True):
        axis.add_patch(plt.Rectangle((x, y), width, height, fill=False, edgecolor="#ff7f0e", linewidth=1.4))
        axis.text(x + 2, y + 11, KNOWN_CLASSES[class_id], color="white", fontsize=6, bbox={"facecolor": "#ff7f0e", "edgecolor": "none", "pad": 1.0})
    axis.set_title(title, fontsize=9)
    axis.axis("off")


def validate_augmented_sample(original: dict[str, Any], augmented: dict[str, Any]) -> dict[str, Any]:
    """Check transformed boxes/classes before displaying a real loader sample."""
    original_classes = original["cls"].reshape(-1).cpu().numpy().astype(int).tolist()
    augmented_classes = augmented["cls"].reshape(-1).cpu().numpy().astype(int).tolist()
    boxes = augmented["bboxes"].cpu().numpy()
    if len(boxes) == 0 or len(boxes) != len(augmented_classes):
        raise RuntimeError("Augmentation produced empty or inconsistent annotations in a visual example")
    if any(class_id not in range(len(KNOWN_CLASSES)) for class_id in augmented_classes):
        raise RuntimeError("Augmentation produced an invalid class identifier")
    if not set(augmented_classes).issubset(set(original_classes)):
        raise RuntimeError("Augmentation changed an object class")
    xyxy = boxes.copy()
    xyxy[:, 0] -= xyxy[:, 2] / 2
    xyxy[:, 1] -= xyxy[:, 3] / 2
    xyxy[:, 2] += xyxy[:, 0]
    xyxy[:, 3] += xyxy[:, 1]
    if not (np.isfinite(boxes).all() and (boxes[:, 2:] > 0).all() and (xyxy >= 0).all() and (xyxy <= 1).all()):
        raise RuntimeError("Augmentation produced an invalid normalized bounding box")
    return {
        "boxes_before": len(original_classes),
        "boxes_after": len(augmented_classes),
        "classes_before": [KNOWN_CLASSES[index] for index in original_classes],
        "classes_after": [KNOWN_CLASSES[index] for index in augmented_classes],
    }


def inspect(output_root: Path = ROOT / "reports") -> dict[str, Any]:
    """Generate proof that baseline is deterministic and augmented changes a sample."""
    baseline_dataset, baseline_transforms = build_profile_dataset("baseline")
    augmented_dataset, augmented_transforms = build_profile_dataset("augmented")
    set_seed(42)
    baseline_first = baseline_dataset[0]
    set_seed(42)
    baseline_second = baseline_dataset[0]
    set_seed(42)
    augmented = augmented_dataset[0]
    baseline_is_deterministic = bool(torch.equal(baseline_first["img"], baseline_second["img"]) and torch.equal(baseline_first["bboxes"], baseline_second["bboxes"]))
    augmented_changes_image = not torch.equal(baseline_first["img"], augmented["img"])
    if not baseline_is_deterministic:
        raise RuntimeError("Baseline loader applied an unexpected random change")
    if not augmented_changes_image:
        raise RuntimeError("Augmented loader did not alter the selected image; inspection seed must be revised")
    if baseline_dataset.im_files[0] != augmented_dataset.im_files[0]:
        raise RuntimeError("Profile datasets selected different source samples")

    raw_image = plt.imread(baseline_dataset.im_files[0])
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.25))
    axes[0].imshow(raw_image)
    axes[0].set_title("Original source image", fontsize=9)
    axes[0].axis("off")
    draw_sample(axes[1], baseline_first, "Baseline: deterministic resize only")
    draw_sample(axes[2], augmented, "Augmented: geometric/value transforms")
    fig.suptitle("Actual Ultralytics augmentation pipeline and transformed boxes", fontsize=12, y=0.98)
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.03, top=0.84, wspace=0.08)
    figure_path = output_root / "figures" / "augmentation_pipeline_check.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    example_indices = (0, 250, 500)
    fig, axes = plt.subplots(len(example_indices), 2, figsize=(6.8, 9.6), constrained_layout=True)
    examples = []
    for row, index in enumerate(example_indices):
        set_seed(42 + index)
        original = baseline_dataset[index]
        set_seed(42 + index)
        transformed = augmented_dataset[index]
        details = validate_augmented_sample(original, transformed)
        source_name = Path(baseline_dataset.im_files[index]).name
        draw_sample(axes[row, 0], original, f"Original / resize-only\n{source_name}")
        draw_sample(axes[row, 1], transformed, f"Augmented\n{source_name}")
        examples.append({"index": index, "source_image": source_name, **details})
    fig.suptitle("Real augmented training samples: boxes and classes after Ultralytics transforms", fontsize=11)
    examples_path = output_root / "figures" / "augmentation_examples.png"
    fig.savefig(examples_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    report = {
        "seed": 42,
        "source_image": baseline_dataset.im_files[0],
        "baseline_transforms": baseline_transforms,
        "augmented_transforms": augmented_transforms,
        "implicit_albumentations_replaced": True,
        "baseline_is_deterministic": baseline_is_deterministic,
        "augmented_changes_image": augmented_changes_image,
        "baseline_boxes": int(len(baseline_first["bboxes"])),
        "augmented_boxes": int(len(augmented["bboxes"])),
        "visual_examples": examples,
    }
    metrics_path = output_root / "metrics" / "augmentation_pipeline.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    print(json.dumps(inspect(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
