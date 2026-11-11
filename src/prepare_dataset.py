"""Prepare a reproducible, leakage-aware YOLO dataset from canonical NEU-DET VOC.

Only ``data/processed`` and generated reports/configuration are written. Raw
images and XML annotations are read-only inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image

from src.audit_dataset import KNOWN_CLASSES


SEED = 42
SPLIT_TARGETS = {"train": 1224, "val": 216, "test": 360}
DEFAULT_VOC_ROOT = Path("data/raw/NEU-DET/Neu_Det_5Flod_new/flod1/VOC2007")
# These two source JPEGs have the same SHA-256 but conflicting VOC boxes.  The
# source archive stays intact; only the second record is excluded from YOLO use.
DUPLICATE_IMAGE_KEEP_ID = "patches_101"
DUPLICATE_IMAGE_EXCLUDE_ID = "patches_105"
DUPLICATE_IMAGE_RULE = "Keep the lexicographically lower image identifier when identical JPEG bytes have conflicting XML annotations."


@dataclass(frozen=True)
class Sample:
    image_id: str
    image_path: Path
    annotation_path: Path
    sha256: str
    classes: tuple[str, ...]
    objects: tuple[dict[str, int | str], ...]
    width: int
    height: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_samples(voc_root: Path) -> list[Sample]:
    """Read validated canonical VOC inputs, preserving source coordinates exactly."""
    image_dir, annotation_dir = voc_root / "JPEGImages", voc_root / "Annotations"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(f"Expected JPEGImages and Annotations under {voc_root}")
    samples: list[Sample] = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        annotation_path = annotation_dir / f"{image_path.stem}.xml"
        if not annotation_path.is_file():
            raise ValueError(f"Missing annotation: {annotation_path}")
        with Image.open(image_path) as image:
            width, height = image.size
        root = ET.parse(annotation_path).getroot()
        objects: list[dict[str, int | str]] = []
        for node in root.findall("object"):
            class_name = (node.findtext("name") or "").strip()
            box = node.find("bndbox")
            if class_name not in KNOWN_CLASSES or box is None:
                raise ValueError(f"Unsupported annotation in {annotation_path}")
            xmin, ymin, xmax, ymax = (int(float(box.findtext(key, ""))) for key in ("xmin", "ymin", "xmax", "ymax"))
            if xmin < 0 or ymin < 0 or xmax > width or ymax > height or xmax <= xmin or ymax <= ymin:
                raise ValueError(f"Invalid bbox in {annotation_path}: {(xmin, ymin, xmax, ymax)}")
            objects.append({"class": class_name, "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax})
        if not objects:
            raise ValueError(f"Annotation has no objects: {annotation_path}")
        samples.append(Sample(image_path.stem, image_path, annotation_path, sha256(image_path), tuple(sorted({str(item['class']) for item in objects})), tuple(objects), width, height))
    return samples


def make_exact_duplicate_groups(samples: list[Sample]) -> list[list[Sample]]:
    grouped: defaultdict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.sha256].append(sample)
    return sorted((sorted(group, key=lambda sample: sample.image_id) for group in grouped.values()), key=lambda group: group[0].image_id)


def primary_class(sample: Sample) -> str:
    """Use the verified filename prefix as a reproducible primary stratum."""
    candidate = sample.image_id.rsplit("_", 1)[0]
    return candidate if candidate in KNOWN_CLASSES else sample.classes[0]


def stratified_group_split(samples: list[Sample], seed: int = SEED, targets: dict[str, int] = SPLIT_TARGETS) -> dict[str, list[Sample]]:
    """Stratify SHA-256 groups by their verified filename-derived primary class.

    NEU-DET's filenames encode its primary category. Secondary labels exist in
    some images, so their object distributions are reported but not used as an
    unjustified independent image split key. Exact duplicate groups stay whole.
    """
    if sum(targets.values()) != len(samples):
        raise ValueError("Split targets must sum to the sample count")
    groups = make_exact_duplicate_groups(samples)
    rng = random.Random(seed)
    groups_by_primary: defaultdict[str, list[list[Sample]]] = defaultdict(list)
    for group in groups:
        group_primaries = {primary_class(sample) for sample in group}
        if len(group_primaries) != 1:
            raise ValueError(f"Exact duplicate group crosses primary classes: {[sample.image_id for sample in group]}")
        groups_by_primary[group_primaries.pop()].append(group)
    desired: dict[str, dict[str, int]] = {split: {} for split in targets}
    for label, label_groups in groups_by_primary.items():
        count = sum(len(group) for group in label_groups)
        raw = {split: count * targets[split] / len(samples) for split in targets}
        allocated = {split: int(raw[split]) for split in targets}
        for split in sorted(targets, key=lambda name: (raw[name] - allocated[name], name), reverse=True)[: count - sum(allocated.values())]:
            allocated[split] += 1
        for split in targets:
            desired[split][label] = allocated[split]
    assigned: dict[str, list[Sample]] = {split: [] for split in targets}
    sizes = Counter()
    for label in sorted(groups_by_primary):
        label_groups = groups_by_primary[label]
        rng.shuffle(label_groups)
        label_groups.sort(key=lambda group: -len(group))
        remaining = {split: desired[split][label] for split in targets}
        for group in label_groups:
            candidates = [split for split in targets if remaining[split] >= len(group)]
            if not candidates:
                raise RuntimeError(f"Cannot place duplicate group for primary class {label}")
            chosen = max(candidates, key=lambda split: (remaining[split], targets[split] - sizes[split], split))
            assigned[chosen].extend(group)
            remaining[chosen] -= len(group)
            sizes[chosen] += len(group)
        if any(remaining.values()):
            raise RuntimeError(f"Primary class allocation incomplete for {label}: {remaining}")
    if dict(sizes) != targets:
        raise RuntimeError(f"Unexpected split sizes: {dict(sizes)}")
    return {split: sorted(values, key=lambda sample: sample.image_id) for split, values in assigned.items()}


def exclude_conflicting_duplicate_from_processed(split_samples: dict[str, list[Sample]]) -> tuple[dict[str, list[Sample]], dict[str, Any]]:
    """Remove one documented conflicting duplicate only from the derived training set.

    The complete canonical fold is still audited as 1,800 source records.  This
    policy is applied *after* the frozen 1,800-record split, so validation and
    test membership never change.
    """
    by_id = {sample.image_id: (split, sample) for split, samples in split_samples.items() for sample in samples}
    if DUPLICATE_IMAGE_KEEP_ID not in by_id or DUPLICATE_IMAGE_EXCLUDE_ID not in by_id:
        raise ValueError("Resolved duplicate pair is missing from the canonical split.")
    keep_split, keep_sample = by_id[DUPLICATE_IMAGE_KEEP_ID]
    exclude_split, exclude_sample = by_id[DUPLICATE_IMAGE_EXCLUDE_ID]
    if keep_sample.sha256 != exclude_sample.sha256:
        raise ValueError("Resolved duplicate pair no longer has identical SHA-256 values.")
    if keep_split != "train" or exclude_split != "train":
        raise ValueError("Duplicate-resolution policy would alter validation or test membership.")
    processed = {
        split: [sample for sample in samples if sample.image_id != DUPLICATE_IMAGE_EXCLUDE_ID]
        for split, samples in split_samples.items()
    }
    if {sample.image_id for sample in processed["val"]} != {sample.image_id for sample in split_samples["val"]}:
        raise AssertionError("Validation changed during duplicate resolution.")
    if {sample.image_id for sample in processed["test"]} != {sample.image_id for sample in split_samples["test"]}:
        raise AssertionError("Test changed during duplicate resolution.")
    return processed, {
        "image_id": exclude_sample.image_id,
        "kept_image_id": keep_sample.image_id,
        "split": exclude_split,
        "sha256": exclude_sample.sha256,
        "voc_annotation_rows": len(exclude_sample.objects),
        "unique_yolo_objects": len(yolo_lines(exclude_sample)),
        "reason": "Identical JPEG SHA-256 with conflicting Pascal VOC bounding boxes.",
        "rule": DUPLICATE_IMAGE_RULE,
    }


def raw_yolo_lines(sample: Sample) -> list[str]:
    """Convert every VOC object entry to one normalized YOLO row, preserving repeats."""
    lines: list[str] = []
    class_to_id = {name: index for index, name in enumerate(KNOWN_CLASSES)}
    for box in sample.objects:
        xmin, ymin, xmax, ymax = (int(box[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
        box_width, box_height = xmax - xmin, ymax - ymin
        xc = (xmin + xmax) / (2 * sample.width)
        yc = (ymin + ymax) / (2 * sample.height)
        width = box_width / sample.width
        height = box_height / sample.height
        values = (xc, yc, width, height)
        if int(box["class"] in KNOWN_CLASSES) != 1 or not all(0 < value <= 1 for value in (width, height)) or not all(0 <= value <= 1 for value in (xc, yc)):
            raise ValueError(f"Cannot normalize bbox for {sample.image_id}")
        lines.append(f"{class_to_id[str(box['class'])]} {xc:.8f} {yc:.8f} {width:.8f} {height:.8f}")
    return lines


def yolo_lines(sample: Sample) -> list[str]:
    """Return one YOLO row per unique physical object.

    Exact repeated XML entries have the same class and coordinates, therefore do
    not describe separate objects. Ultralytics 8.3.239 also removes such label
    duplicates in its cache; doing so explicitly keeps on-disk labels and the
    effective training set consistent.
    """
    return list(dict.fromkeys(raw_yolo_lines(sample)))


def duplicate_voc_objects(sample: Sample) -> list[dict[str, Any]]:
    """Describe exact repeated VOC entries; no overlapping-but-distinct boxes qualify."""
    counts = Counter(
        (str(box["class"]), int(box["xmin"]), int(box["ymin"]), int(box["xmax"]), int(box["ymax"]))
        for box in sample.objects
    )
    return [
        {
            "class": class_name,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "occurrences": occurrences,
            "redundant_rows": occurrences - 1,
        }
        for (class_name, xmin, ymin, xmax, ymax), occurrences in sorted(counts.items())
        if occurrences > 1
    ]


def write_processed_dataset(split_samples: dict[str, list[Sample]], processed_root: Path) -> None:
    """Copy source images and write YOLO labels without changing raw sources."""
    for split, samples in split_samples.items():
        image_dir, label_dir = processed_root / "images" / split, processed_root / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        cache_path = label_dir.parent / f"{split}.cache"
        if cache_path.is_file():
            cache_path.unlink()
        expected = {sample.image_path.name for sample in samples}
        existing = {path.name for path in image_dir.glob("*") if path.is_file()}
        unexpected = existing - expected
        if unexpected:
            raise RuntimeError(f"Refusing to overwrite unexpected processed files in {image_dir}: {sorted(unexpected)[:5]}")
        for sample in samples:
            shutil.copy2(sample.image_path, image_dir / sample.image_path.name)
            (label_dir / f"{sample.image_id}.txt").write_text("\n".join(yolo_lines(sample)) + "\n", encoding="utf-8")


def clear_previous_generated_files(processed_root: Path, split_report_path: Path) -> None:
    """Remove only paths explicitly declared by a prior generated split manifest."""
    if not split_report_path.is_file():
        return
    previous = json.loads(split_report_path.read_text(encoding="utf-8"))
    for item in previous.get("assignments", []):
        split, image_id = item.get("split"), item.get("image_id")
        if split not in SPLIT_TARGETS or not isinstance(image_id, str):
            continue
        for path in (processed_root / "images" / split / f"{image_id}.jpg", processed_root / "labels" / split / f"{image_id}.txt"):
            if path.is_file():
                path.unlink()


def write_data_yaml(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"path": "../data/processed", "train": "images/train", "val": "images/val", "test": "images/test", "nc": len(KNOWN_CLASSES), "names": list(KNOWN_CLASSES)}
    path.write_text("# Paths are relative to this config file and portable across Windows/Linux.\n" + yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def validate_prepared_dataset(split_samples: dict[str, list[Sample]], processed_root: Path) -> dict[str, Any]:
    """Validate disjointness, SHA groups, labels and source/YOLO object-count equality."""
    image_ids = {split: {sample.image_id for sample in samples} for split, samples in split_samples.items()}
    for left in SPLIT_TARGETS:
        for right in SPLIT_TARGETS:
            if left < right and image_ids[left] & image_ids[right]:
                raise ValueError(f"Split overlap: {left}/{right}")
    sha_splits: defaultdict[str, set[str]] = defaultdict(set)
    for split, samples in split_samples.items():
        for sample in samples:
            sha_splits[sample.sha256].add(split)
            label_path = processed_root / "labels" / split / f"{sample.image_id}.txt"
            image_path = processed_root / "images" / split / sample.image_path.name
            if not image_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(f"Missing processed pair for {sample.image_id}")
            lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(lines) != len(yolo_lines(sample)):
                raise ValueError(f"Object count mismatch for {sample.image_id}")
            for line in lines:
                class_id, *coordinates = line.split()
                if int(class_id) not in range(len(KNOWN_CLASSES)) or len(coordinates) != 4:
                    raise ValueError(f"Invalid YOLO row in {label_path}")
                values = [float(value) for value in coordinates]
                if not (0 <= values[0] <= 1 and 0 <= values[1] <= 1 and 0 < values[2] <= 1 and 0 < values[3] <= 1):
                    raise ValueError(f"Out-of-range YOLO coordinate in {label_path}")
    leaked_hashes = {digest: splits for digest, splits in sha_splits.items() if len(splits) > 1}
    if leaked_hashes:
        raise ValueError(f"Exact duplicate leakage: {leaked_hashes}")
    return {
        "split_sizes": {split: len(samples) for split, samples in split_samples.items()},
        "raw_objects": {split: sum(len(sample.objects) for sample in samples) for split, samples in split_samples.items()},
        "yolo_objects": {split: sum(len(yolo_lines(sample)) for sample in samples) for split, samples in split_samples.items()},
    }


def write_split_reports(split_samples: dict[str, list[Sample]], audit: dict[str, Any], reports_root: Path, source_record_count: int, exclusion: dict[str, Any]) -> None:
    assignments = []
    statistics = []
    for split, samples in split_samples.items():
        class_objects = Counter(str(box["class"]) for sample in samples for box in sample.objects)
        class_images = Counter(label for sample in samples for label in sample.classes)
        statistics.append({"split": split, "class": "__total__", "images": len(samples), "raw_objects": sum(len(sample.objects) for sample in samples), "yolo_objects": sum(len(yolo_lines(sample)) for sample in samples), "images_containing_class": ""})
        for label in KNOWN_CLASSES:
            statistics.append({"split": split, "class": label, "images": "", "raw_objects": class_objects[label], "yolo_objects": sum(line.startswith(f"{KNOWN_CLASSES.index(label)} ") for sample in samples for line in yolo_lines(sample)), "images_containing_class": class_images[label]})
        assignments.extend({"image_id": sample.image_id, "split": split, "source_image": str(sample.image_path).replace("\\", "/"), "sha256": sample.sha256, "classes": list(sample.classes), "object_count": len(sample.objects)} for sample in samples)
    metrics_dir, tables_dir = reports_root / "metrics", reports_root / "tables"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    near_pairs = audit.get("near_duplicate_pairs", [])
    repeated_label_rows = [
        {
            "image_id": sample.image_id,
            "split": split,
            "voc_annotation_rows": len(sample.objects),
            "unique_yolo_objects": len(yolo_lines(sample)),
            "duplicate_objects": duplicate_voc_objects(sample),
        }
        for split, samples in split_samples.items()
        for sample in samples
        if len(yolo_lines(sample)) != len(sample.objects)
    ]
    redundant_voc_rows = sum(
        duplicate["redundant_rows"]
        for record in repeated_label_rows
        for duplicate in record["duplicate_objects"]
    )
    split_by_id = {item["image_id"]: item["split"] for item in assignments}
    cross_near = [
        pair for pair in near_pairs
        if Path(pair["image_a"]).stem in split_by_id
        and Path(pair["image_b"]).stem in split_by_id
        and split_by_id[Path(pair["image_a"]).stem] != split_by_id[Path(pair["image_b"]).stem]
    ]
    payload = {
        "seed": SEED,
        "method": "custom 68/12/20 deterministic primary-class stratification over SHA-256 duplicate groups, followed by one documented train-only conflicting-annotation duplicate exclusion",
        "source_canonical_records": source_record_count,
        "source_voc_annotation_rows": audit["objects"],
        "used_images": sum(len(samples) for samples in split_samples.values()),
        "used_voc_annotation_rows": sum(len(sample.objects) for samples in split_samples.values() for sample in samples),
        "used_unique_yolo_objects": sum(len(yolo_lines(sample)) for samples in split_samples.values() for sample in samples),
        "targets_before_duplicate_exclusion": SPLIT_TARGETS,
        "used_split_sizes": {split: len(samples) for split, samples in split_samples.items()},
        "class_mapping": {str(index): name for index, name in enumerate(KNOWN_CLASSES)},
        "source_split_status": "Supplied 1440/360 fold is not treated as official; it separated the known exact duplicate pair.",
        "exact_duplicate_policy": "All files with identical SHA-256 are assigned together before the documented train-only conflicting-annotation exclusion.",
        "near_duplicate_policy": "dHash candidates are retained as warnings but not grouped automatically because broad visual-similarity candidates in repetitive texture do not prove duplicate acquisition.",
        "near_duplicate_pairs_total": len(near_pairs),
        "near_duplicate_pairs_cross_split": len(cross_near),
        "exact_duplicate_voc_annotation_rows": {
            "count": redundant_voc_rows,
            "policy": "One identical class-and-coordinate XML entry represents one physical object and is retained once in YOLO labels.",
            "records": repeated_label_rows,
        },
        "processed_exclusions": [exclusion],
        "assignments": assignments,
    }
    (metrics_dir / "dataset_split.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (tables_dir / "split_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "class", "images", "raw_objects", "yolo_objects", "images_containing_class"])
        writer.writeheader()
        writer.writerows(statistics)
    markdown = "\n".join([
        "# NEU-DET dataset preparation",
        "",
        "## Research split",
        "- This is a project-specific 68/12/20 split, not an official NEU-DET partition.",
        "- Seed: 42.",
        f"- Canonical source records: {source_record_count}; used processed images: {sum(len(samples) for samples in split_samples.values())}.",
        "- Train/validation/test used for YOLO: 1223 / 216 / 360 images.",
        "- The supplied 1440/360 lists were not reused: the known exact duplicate pair was split across them.",
        "- Primary filename class is stratified exactly; secondary labels are retained as multi-label statistics.",
        "- SHA-256 duplicate groups are kept within one split; one train-only conflicting-annotation duplicate is excluded from the processed dataset.",
        f"- Excluded processed file: `{exclusion['image_id']}.jpg` (keep `{exclusion['kept_image_id']}.jpg`): {exclusion['reason']} Rule: {exclusion['rule']}",
        f"- Used VOC annotation rows: {sum(len(sample.objects) for samples in split_samples.values() for sample in samples)}; unique YOLO objects: {sum(len(yolo_lines(sample)) for samples in split_samples.values() for sample in samples)}.",
        f"- {redundant_voc_rows} exact repeated XML rows in {len(repeated_label_rows)} images are represented once in YOLO labels. They are not unique defects; Ultralytics 8.3.239 would remove them during cache construction as well.",
        f"- dHash candidate pairs: {len(near_pairs)} total; {len(cross_near)} cross-split. They are warnings, not confirmed duplicates, and were not deleted or automatically grouped.",
        "",
        "## Conversion",
        "- Pascal VOC XML was converted to normalized YOLO class_id xc yc width height labels.",
        "- Mapping: 0=crazing, 1=inclusion, 2=patches, 3=pitted_surface, 4=rolled-in_scale, 5=scratches.",
        "- Source XML/JPG files remain unchanged.",
    ]) + "\n"
    (reports_root / "dataset_preparation.md").write_text(markdown, encoding="utf-8")
    duplicate_details = "\n".join(
        f"- `{record['image_id']}.xml` ({record['split']}): "
        + "; ".join(
            f"class={item['class']}, bbox=({item['xmin']}, {item['ymin']}, {item['xmax']}, {item['ymax']}), occurrences={item['occurrences']}"
            for item in record["duplicate_objects"]
        )
        for record in repeated_label_rows
    )
    dedup_report = "\n".join([
        "# VOC-to-YOLO object-count reconciliation",
        "",
        f"- Canonical source VOC annotation rows: {audit['objects']}.",
        f"- Excluded with `patches_105`: {exclusion['voc_annotation_rows']} rows.",
        f"- Remaining VOC annotation rows: {sum(len(sample.objects) for samples in split_samples.values() for sample in samples)}.",
        f"- Exact redundant VOC rows: {redundant_voc_rows}.",
        f"- Unique YOLO objects / effective Ultralytics objects: {sum(len(yolo_lines(sample)) for samples in split_samples.values() for sample in samples)}.",
        "",
        "The following XML entries have the same class and exact `(xmin, ymin, xmax, ymax)` coordinates within one image. They are repeated annotations of one physical defect, not overlapping distinct defects. They are written once because Ultralytics 8.3.239 de-duplicates identical YOLO rows in its label cache; retaining duplicate rows would make the label text and effective training data disagree.",
        "",
        "## Exact repeated entries",
        duplicate_details or "- None.",
        "",
        "No distinct bounding box was removed and raw XML/JPG files remain unchanged.",
    ]) + "\n"
    (reports_root / "annotation_deduplication.md").write_text(dedup_report, encoding="utf-8")


def create_conflicting_duplicate_comparison(samples: list[Sample], figure_path: Path) -> None:
    """Render both conflicting VOC annotations over the shared source pixels."""
    by_id = {sample.image_id: sample for sample in samples}
    kept, excluded = by_id[DUPLICATE_IMAGE_KEEP_ID], by_id[DUPLICATE_IMAGE_EXCLUDE_ID]
    if kept.sha256 != excluded.sha256:
        raise ValueError("Cannot render conflicting duplicate comparison: JPEG hashes differ.")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.6), constrained_layout=True)
    for axis, sample, color, title in (
        (axes[0], kept, "#1b9e77", "patches_101 — retained"),
        (axes[1], excluded, "#d95f02", "patches_105 — excluded from processed data"),
    ):
        with Image.open(kept.image_path) as image:
            axis.imshow(image, cmap="gray")
        for index, box in enumerate(sample.objects, start=1):
            axis.add_patch(plt.Rectangle((int(box["xmin"]), int(box["ymin"])), int(box["xmax"]) - int(box["xmin"]), int(box["ymax"]) - int(box["ymin"]), fill=False, linewidth=1.4, edgecolor=color))
            axis.text(int(box["xmin"]) + 2, int(box["ymin"]) + 10, str(index), color="white", fontsize=7, bbox={"facecolor": color, "edgecolor": "none", "pad": 1.2})
        axis.set_title(title, fontsize=9)
        axis.set_xlim(0, kept.width)
        axis.set_ylim(kept.height, 0)
        axis.set_axis_off()
    fig.suptitle("Identical JPEG pixels with conflicting Pascal VOC bounding boxes", fontsize=11)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_annotation_check(split_samples: dict[str, list[Sample]], figure_path: Path) -> None:
    """Render deterministic YOLO-label overlays for three samples from each split."""
    colors = dict(zip(KNOWN_CLASSES, plt.get_cmap("tab10").colors))
    fig, axes = plt.subplots(3, 3, figsize=(8.2, 8.0))
    for row_index, split in enumerate(("train", "val", "test")):
        samples = []
        for class_name in KNOWN_CLASSES:
            candidates = [sample for sample in split_samples[split] if primary_class(sample) == class_name]
            if candidates:
                samples.append(candidates[0])
            if len(samples) == 3:
                break
        for column_index, sample in enumerate(samples):
            axis = axes[row_index, column_index]
            with Image.open(sample.image_path) as image:
                axis.imshow(image)
            for box in sample.objects:
                label = str(box["class"])
                axis.add_patch(plt.Rectangle((int(box["xmin"]), int(box["ymin"])), int(box["xmax"]) - int(box["xmin"]), int(box["ymax"]) - int(box["ymin"]), fill=False, linewidth=1.0, edgecolor=colors[label]))
            axis.set_title(f"{split}: {sample.image_id}", fontsize=7)
            axis.axis("off")
    fig.suptitle("YOLO annotation check: train / validation / test", fontsize=12, y=0.99)
    fig.subplots_adjust(left=0.03, right=0.99, bottom=0.03, top=0.94, wspace=0.08, hspace=0.25)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def prepare(voc_root: Path, processed_root: Path, configs_root: Path, reports_root: Path, audit_path: Path) -> dict[str, Any]:
    samples = read_samples(voc_root)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if len(samples) != audit["images"] or sum(len(sample.objects) for sample in samples) != audit["objects"]:
        raise ValueError("Raw canonical fold does not match audit totals")
    canonical_split_samples = stratified_group_split(samples)
    split_samples, exclusion = exclude_conflicting_duplicate_from_processed(canonical_split_samples)
    clear_previous_generated_files(processed_root, reports_root / "metrics" / "dataset_split.json")
    write_processed_dataset(split_samples, processed_root)
    validation = validate_prepared_dataset(split_samples, processed_root)
    write_data_yaml(configs_root / "data.yaml")
    write_split_reports(split_samples, audit, reports_root, len(samples), exclusion)
    create_annotation_check(split_samples, reports_root / "figures" / "yolo_annotation_check.png")
    create_conflicting_duplicate_comparison(samples, reports_root / "figures" / "conflicting_duplicate_annotations.png")
    validation["source_canonical_records"] = len(samples)
    validation["used_images_total"] = sum(len(samples) for samples in split_samples.values())
    validation["processed_exclusion"] = exclusion
    validation["raw_objects_total"] = sum(len(sample.objects) for samples in split_samples.values() for sample in samples)
    validation["yolo_objects_total"] = sum(len(yolo_lines(sample)) for samples in split_samples.values() for sample in samples)
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the fixed NEU-DET YOLO research split.")
    parser.add_argument("--voc-root", type=Path, default=DEFAULT_VOC_ROOT)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--configs-root", type=Path, default=Path("configs"))
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    parser.add_argument("--audit", type=Path, default=Path("reports/metrics/dataset_audit.json"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.voc_root, args.processed_root, args.configs_root, args.reports_root, args.audit), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
