"""Deterministic exploratory analysis for the canonical NEU-DET VOC fold.

The module reads images and Pascal VOC XML annotations only. It never creates
splits or changes the raw dataset.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from src.audit_dataset import KNOWN_CLASSES


DEFAULT_VOC_ROOT = Path("data/raw/NEU-DET/Neu_Det_5Flod_new/flod1/VOC2007")
SMALL_DEFECT_AREA_FRACTION = 0.01
CLASS_COLORS = dict(zip(KNOWN_CLASSES, sns.color_palette("colorblind", len(KNOWN_CLASSES))))


def configure_style() -> None:
    """Set one publication-oriented visual style for all raster figures."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight"})


def load_voc_records(voc_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read a VOC2007 directory into image-level and object-level data frames."""
    image_dir = voc_root / "JPEGImages"
    annotation_dir = voc_root / "Annotations"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(f"Expected JPEGImages and Annotations under: {voc_root}")
    image_rows: list[dict[str, Any]] = []
    object_rows: list[dict[str, Any]] = []
    for image_path in sorted(image_dir.glob("*")):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        annotation_path = annotation_dir / f"{image_path.stem}.xml"
        if not annotation_path.is_file():
            raise ValueError(f"Missing annotation for image: {image_path.name}")
        with Image.open(image_path) as image:
            width, height = image.size
            channels = len(image.getbands())
        root = ET.parse(annotation_path).getroot()
        image_rows.append({
            "image_id": image_path.stem,
            "image_path": str(image_path),
            "annotation_path": str(annotation_path),
            "width": width,
            "height": height,
            "channels": channels,
        })
        for object_index, node in enumerate(root.findall("object")):
            class_name = (node.findtext("name") or "").strip()
            box = node.find("bndbox")
            if class_name not in KNOWN_CLASSES or box is None:
                raise ValueError(f"Invalid audited annotation: {annotation_path}")
            xmin, ymin, xmax, ymax = (int(float(box.findtext(key, ""))) for key in ("xmin", "ymin", "xmax", "ymax"))
            bbox_width, bbox_height = xmax - xmin, ymax - ymin
            if bbox_width <= 0 or bbox_height <= 0 or xmin < 0 or ymin < 0 or xmax > width or ymax > height:
                raise ValueError(f"Invalid bbox in audited annotation: {annotation_path}")
            area = bbox_width * bbox_height
            object_rows.append({
                "image_id": image_path.stem,
                "image_path": str(image_path),
                "class": class_name,
                "object_index": object_index,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "bbox_width_px": bbox_width,
                "bbox_height_px": bbox_height,
                "bbox_area_px2": area,
                "bbox_area_fraction": area / (width * height),
                "bbox_area_pct": 100 * area / (width * height),
                "aspect_ratio_w_h": bbox_width / bbox_height,
                "center_x_px": (xmin + xmax) / 2,
                "center_y_px": (ymin + ymax) / 2,
                "center_x_relative": (xmin + xmax) / (2 * width),
                "center_y_relative": (ymin + ymax) / (2 * height),
            })
    return pd.DataFrame(image_rows), pd.DataFrame(object_rows)


def build_statistics(images: pd.DataFrame, objects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Compute image, class, bbox and report statistics from real annotations."""
    object_counts = objects.groupby("image_id").size().rename("objects_per_image")
    image_classes = objects.groupby("image_id")["class"].agg(lambda values: sorted(set(values)))
    images = images.join(object_counts, on="image_id").join(image_classes.rename("classes"), on="image_id")
    images["objects_per_image"] = images["objects_per_image"].fillna(0).astype(int)
    images["class_count"] = images["classes"].map(len)
    image_class_counts = Counter(class_name for values in images["classes"] for class_name in values)
    object_class_counts = objects["class"].value_counts().to_dict()

    class_rows: list[dict[str, Any]] = []
    for class_name in KNOWN_CLASSES:
        subset = objects[objects["class"] == class_name]
        class_rows.append({
            "class": class_name,
            "image_count": image_class_counts.get(class_name, 0),
            "object_count": len(subset),
            "mean_bbox_area_px2": subset["bbox_area_px2"].mean(),
            "median_bbox_area_px2": subset["bbox_area_px2"].median(),
            "mean_bbox_area_pct": subset["bbox_area_pct"].mean(),
            "median_bbox_area_pct": subset["bbox_area_pct"].median(),
            "min_bbox_width_px": subset["bbox_width_px"].min(),
            "max_bbox_width_px": subset["bbox_width_px"].max(),
            "min_bbox_height_px": subset["bbox_height_px"].min(),
            "max_bbox_height_px": subset["bbox_height_px"].max(),
            "small_object_count_under_1pct": int((subset["bbox_area_fraction"] < SMALL_DEFECT_AREA_FRACTION).sum()),
        })
    class_statistics = pd.DataFrame(class_rows)
    bbox_statistics = pd.DataFrame([
        {"metric": "bbox_width_px", "mean": objects["bbox_width_px"].mean(), "median": objects["bbox_width_px"].median(), "min": objects["bbox_width_px"].min(), "max": objects["bbox_width_px"].max()},
        {"metric": "bbox_height_px", "mean": objects["bbox_height_px"].mean(), "median": objects["bbox_height_px"].median(), "min": objects["bbox_height_px"].min(), "max": objects["bbox_height_px"].max()},
        {"metric": "bbox_area_px2", "mean": objects["bbox_area_px2"].mean(), "median": objects["bbox_area_px2"].median(), "min": objects["bbox_area_px2"].min(), "max": objects["bbox_area_px2"].max()},
        {"metric": "bbox_area_pct", "mean": objects["bbox_area_pct"].mean(), "median": objects["bbox_area_pct"].median(), "min": objects["bbox_area_pct"].min(), "max": objects["bbox_area_pct"].max()},
        {"metric": "aspect_ratio_w_h", "mean": objects["aspect_ratio_w_h"].mean(), "median": objects["aspect_ratio_w_h"].median(), "min": objects["aspect_ratio_w_h"].min(), "max": objects["aspect_ratio_w_h"].max()},
    ])
    dataset_statistics = pd.DataFrame([
        {"metric": "images", "value": len(images), "unit": "images"},
        {"metric": "annotated_objects", "value": len(objects), "unit": "objects"},
        {"metric": "images_with_multiple_objects", "value": int((images["objects_per_image"] > 1).sum()), "unit": "images"},
        {"metric": "images_with_multiple_classes", "value": int((images["class_count"] > 1).sum()), "unit": "images"},
        {"metric": "small_objects_under_1pct_image_area", "value": int((objects["bbox_area_fraction"] < SMALL_DEFECT_AREA_FRACTION).sum()), "unit": "objects"},
        {"metric": "small_object_threshold", "value": SMALL_DEFECT_AREA_FRACTION * 100, "unit": "% of image area"},
    ])
    summary = {
        "images": len(images),
        "objects": len(objects),
        "image_class_counts": dict(sorted(image_class_counts.items())),
        "object_class_counts": dict(sorted(object_class_counts.items())),
        "small_objects": int((objects["bbox_area_fraction"] < SMALL_DEFECT_AREA_FRACTION).sum()),
        "images_with_multiple_classes": int((images["class_count"] > 1).sum()),
    }
    return images, dataset_statistics, class_statistics, bbox_statistics, summary


def save_figure(figure: plt.Figure, path: Path, *, use_tight_layout: bool = True, save_pdf: bool = False) -> None:
    """Save a 300-DPI PNG and, optionally, a vector PDF counterpart."""
    if use_tight_layout:
        figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    if save_pdf:
        figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def create_figures(images: pd.DataFrame, objects: pd.DataFrame, figure_dir: Path) -> None:
    """Create the required 300-DPI, deterministic EDA figures."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    class_image_counts = Counter(class_name for values in images["classes"] for class_name in values)
    class_object_counts = objects["class"].value_counts().to_dict()

    fig, axis = plt.subplots(figsize=(10, 5.6))
    positions = np.arange(len(KNOWN_CLASSES))
    axis.bar(positions - 0.2, [class_image_counts[name] for name in KNOWN_CLASSES], 0.4, label="Images containing class")
    axis.bar(positions + 0.2, [class_object_counts.get(name, 0) for name in KNOWN_CLASSES], 0.4, label="Annotated objects")
    axis.set_xticks(positions, KNOWN_CLASSES, rotation=20, ha="right")
    axis.set_ylabel("Count")
    axis.set_title("Class distribution: images versus annotated objects")
    axis.legend()
    save_figure(fig, figure_dir / "class_distribution.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for name in KNOWN_CLASSES:
        subset = objects[objects["class"] == name]
        axes[0].hist(subset["bbox_width_px"], bins=25, alpha=0.45, label=name, color=CLASS_COLORS[name])
        axes[1].hist(subset["bbox_height_px"], bins=25, alpha=0.45, label=name, color=CLASS_COLORS[name])
    axes[0].set(xlabel="Bounding-box width (px)", ylabel="Objects", title="Bounding-box widths")
    axes[1].set(xlabel="Bounding-box height (px)", title="Bounding-box heights")
    axes[1].legend(fontsize=8)
    save_figure(fig, figure_dir / "bbox_distribution.png")

    fig, axis = plt.subplots(figsize=(9, 5.2))
    for name in KNOWN_CLASSES:
        axis.hist(objects.loc[objects["class"] == name, "bbox_area_pct"], bins=30, alpha=0.48, label=name, color=CLASS_COLORS[name])
    axis.set(xlabel="Bounding-box area (% of image)", ylabel="Objects", title="Relative bounding-box area")
    axis.legend(fontsize=8)
    save_figure(fig, figure_dir / "bbox_area_distribution.png")

    fig, axis = plt.subplots(figsize=(8.5, 5))
    maximum = int(images["objects_per_image"].max())
    axis.hist(images["objects_per_image"], bins=np.arange(0.5, maximum + 1.5), color="#4C78A8", edgecolor="white")
    axis.set(xlabel="Annotated objects per image", ylabel="Images", title="Objects per image")
    axis.set_xticks(range(1, maximum + 1))
    save_figure(fig, figure_dir / "objects_per_image.png")

    # 7.16 in is a common full-width two-column article width; extra height
    # and explicit spacing keep row titles and labels separate.
    fig, axes = plt.subplots(2, 3, figsize=(7.16, 5.75))
    for axis, name in zip(axes.flat, KNOWN_CLASSES):
        image_id = sorted(objects.loc[objects["class"] == name, "image_id"].unique())[0]
        row = images.loc[images["image_id"] == image_id].iloc[0]
        with Image.open(row["image_path"]) as image:
            axis.imshow(image)
        axis.set_title(name, fontsize=9, pad=7)
        axis.axis("off")
    fig.suptitle("Deterministic class samples", fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.025, right=0.995, bottom=0.035, top=0.895, wspace=0.06, hspace=0.32)
    save_figure(fig, figure_dir / "class_samples.png", use_tight_layout=False, save_pdf=True)

    fig, axes = plt.subplots(2, 3, figsize=(7.16, 5.75))
    selected_ids: list[str] = []
    for name in KNOWN_CLASSES:
        candidates = objects.loc[objects["class"] == name, "image_id"].unique()
        candidates_frame = images[images["image_id"].isin(candidates)].sort_values(
            ["objects_per_image", "image_id"], ascending=[False, True]
        )
        selected_ids.append(candidates_frame.iloc[0]["image_id"])
    selected = images.set_index("image_id").loc[selected_ids].reset_index()
    for axis, (_, row) in zip(axes.flat, selected.iterrows()):
        with Image.open(row["image_path"]) as image:
            axis.imshow(image)
        labelled_classes: set[str] = set()
        for _, box in objects[objects["image_id"] == row["image_id"]].iterrows():
            axis.add_patch(plt.Rectangle((box.xmin, box.ymin), box.bbox_width_px, box.bbox_height_px, fill=False, linewidth=1.2, edgecolor=CLASS_COLORS[box["class"]]))
            label = box["class"]
            if label in labelled_classes:
                continue
            labelled_classes.add(label)
            label_width = 4.2 * len(label) + 5
            label_x = min(max(float(box.xmin) + 2, 2), 200 - label_width)
            label_y = min(max(float(box.ymin) + 2, 9), 195)
            axis.text(label_x, label_y, label, color="white", fontsize=5.7, va="bottom", clip_on=True,
                      bbox={"facecolor": CLASS_COLORS[label], "alpha": 0.82, "pad": 0.7, "edgecolor": "none"})
        axis.set_title(f"{row['image_id']} ({row['objects_per_image']} objects)", fontsize=7.5, pad=7)
        axis.axis("off")
    fig.suptitle("Annotation examples", fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.025, right=0.995, bottom=0.035, top=0.895, wspace=0.06, hspace=0.32)
    save_figure(fig, figure_dir / "annotation_examples.png", use_tight_layout=False, save_pdf=True)

    heatmap, x_edges, y_edges = np.histogram2d(objects["center_y_relative"], objects["center_x_relative"], bins=20, range=[[0, 1], [0, 1]])
    fig, axis = plt.subplots(figsize=(6.3, 5.4))
    image = axis.imshow(heatmap, origin="lower", extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]], cmap="mako", aspect="equal")
    fig.colorbar(image, ax=axis, label="Annotated objects per bin")
    axis.set(xlabel="Relative bbox center x", ylabel="Relative bbox center y", title="Bounding-box center density")
    save_figure(fig, figure_dir / "bbox_centers_heatmap.png")


def write_report(path: Path, summary: dict[str, Any], class_statistics: pd.DataFrame, audit: dict[str, Any]) -> None:
    largest = class_statistics.loc[class_statistics["object_count"].idxmax(), "class"]
    smallest = class_statistics.loc[class_statistics["object_count"].idxmin(), "class"]
    highest_small = class_statistics.loc[class_statistics["small_object_count_under_1pct"].idxmax(), "class"]
    lines = [
        "# NEU-DET exploratory data analysis",
        "",
        "## Confirmed observations",
        f"- Analysis uses only canonical `flod1`: {summary['images']} images and {summary['objects']} Pascal VOC objects.",
        f"- The object count matches the dataset audit ({audit['objects']}).",
        f"- Image-level class counts: `{summary['image_class_counts']}`.",
        f"- Object-level class counts: `{summary['object_class_counts']}`. These differ because images can contain multiple objects/classes.",
        f"- {summary['images_with_multiple_classes']} images contain more than one class.",
        f"- {summary['small_objects']} objects occupy less than 1% of the 200 x 200 image area; this is an explicit EDA threshold, not a dataset-provided label.",
        f"- `{largest}` has the largest object count and `{smallest}` the smallest; count alone does not establish detection difficulty.",
        "- Audit validation errors: 0. One exact duplicate pair and near-duplicate candidates remain in raw data; none were removed.",
        "",
        "## Interpretation requiring experimental validation",
        f"- The high number of small objects, especially in `{highest_small}`, suggests testing scale-preserving augmentation and image-size choices. Its benefit must be measured on validation data only.",
        "- Unequal object counts may affect per-class recall and mAP. This is a hypothesis for per-class evaluation, not evidence that any class is intrinsically harder.",
        "- Multi-class images require multi-label detection handling; class co-occurrence should not be simplified into one image-level label.",
        "",
        "## Dataset limitations",
        "- Native images are only 200 x 200 pixels; aggressive upscaling cannot create new defect detail.",
        "- The archive contains five fold copies, not 9,000 independent samples. Any future split must use one fold or an explicitly leakage-safe protocol.",
        "- The supplied folds have train/validation lists but no separate test list. A test protocol must be specified before experiments and must not guide model selection.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_eda(voc_root: Path, output_root: Path, audit_path: Path) -> dict[str, Any]:
    """Run EDA, verify it against the audit, and write all requested artifacts."""
    configure_style()
    images, objects = load_voc_records(voc_root)
    images, dataset_statistics, class_statistics, bbox_statistics, summary = build_statistics(images, objects)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if summary["objects"] != audit["objects"] or summary["images"] != audit["images"]:
        raise ValueError("EDA totals do not match reports/metrics/dataset_audit.json")
    tables_dir, figures_dir, report_path = output_root / "tables", output_root / "figures", output_root / "eda_report.md"
    tables_dir.mkdir(parents=True, exist_ok=True)
    dataset_statistics.to_csv(tables_dir / "dataset_statistics.csv", index=False, float_format="%.6f")
    class_statistics.to_csv(tables_dir / "class_statistics.csv", index=False, float_format="%.6f")
    bbox_statistics.to_csv(tables_dir / "bbox_statistics.csv", index=False, float_format="%.6f")
    create_figures(images, objects, figures_dir)
    write_report(report_path, summary, class_statistics, audit)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic EDA for the canonical NEU-DET VOC fold.")
    parser.add_argument("--voc-root", type=Path, default=DEFAULT_VOC_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("reports"))
    parser.add_argument("--audit", type=Path, default=Path("reports/metrics/dataset_audit.json"))
    args = parser.parse_args()
    print(json.dumps(run_eda(args.voc_root, args.output_root, args.audit), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
