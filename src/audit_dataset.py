"""Audit an immutable NEU-DET Pascal VOC dataset.

The script only reads ``data/raw``. It never repairs, moves, or deletes data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

from PIL import Image, UnidentifiedImageError


KNOWN_CLASSES = (
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}
DEFAULT_SOURCE = {
    "requested_kaggle": "https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database",
    "requested_kaggle_status": "classification-only according to public Kaggle metadata",
    "detection_source": "https://github.com/zht8506/ETDNet",
    "detection_download": "https://drive.google.com/drive/folders/1m0qVeSlcugL2VgLMjm6xHcJdxfoPphjB",
    "expected_archive": "Neu_Det_5Flod_voc.zip",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash(image: Image.Image) -> int:
    """Return a 64-bit difference hash; no additional package is required."""
    values = list(image.convert("L").resize((9, 8)).get_flattened_data())
    result = 0
    for row in range(8):
        for col in range(8):
            result = (result << 1) | int(values[row * 9 + col] > values[row * 9 + col + 1])
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def find_annotation(image: Path, annotation_dirs: list[Path]) -> Path | None:
    # Standard Pascal VOC layout: VOC2007/JPEGImages/name.jpg -> VOC2007/Annotations/name.xml.
    if image.parent.name.lower() == "jpegimages":
        sibling = image.parent.parent / "Annotations" / f"{image.stem}.xml"
        if sibling.is_file():
            return sibling
    for directory in annotation_dirs:
        candidate = directory / f"{image.stem}.xml"
        if candidate.is_file():
            return candidate
    return None


def discover_supplied_splits(voc_roots: list[Path]) -> dict[str, dict[str, int]]:
    """Read existing VOC ImageSets files without creating or modifying a split."""
    result: dict[str, dict[str, int]] = {}
    for voc_root in voc_roots:
        split_dir = voc_root / "ImageSets" / "Main"
        if not split_dir.is_dir():
            continue
        result[voc_root.parent.name] = {
            file.stem: sum(1 for line in file.read_text(encoding="utf-8").splitlines() if line.strip())
            for file in sorted(split_dir.glob("*.txt"))
        }
    return result


def parse_voc(annotation: Path, width: int, height: int) -> tuple[list[dict[str, Any]], list[str]]:
    """Parse VOC objects and return objects plus validation error codes."""
    errors: list[str] = []
    try:
        root = ET.parse(annotation).getroot()
    except (ET.ParseError, OSError) as error:
        return [], [f"annotation_parse_error: {error}"]
    objects: list[dict[str, Any]] = []
    for index, node in enumerate(root.findall("object")):
        name = (node.findtext("name") or "").strip()
        box = node.find("bndbox")
        prefix = f"object[{index}]"
        if not name:
            errors.append(f"{prefix}: missing_class")
            continue
        if name not in KNOWN_CLASSES:
            errors.append(f"{prefix}: unknown_class:{name}")
        if box is None:
            errors.append(f"{prefix}: missing_bbox")
            continue
        try:
            xmin = int(float(box.findtext("xmin", "")))
            ymin = int(float(box.findtext("ymin", "")))
            xmax = int(float(box.findtext("xmax", "")))
            ymax = int(float(box.findtext("ymax", "")))
        except ValueError:
            errors.append(f"{prefix}: invalid_bbox_values")
            continue
        if xmax <= xmin or ymax <= ymin:
            errors.append(f"{prefix}: non_positive_bbox_area")
        if xmin < 0 or ymin < 0 or xmax > width or ymax > height:
            errors.append(f"{prefix}: bbox_out_of_bounds")
        objects.append({"class": name, "bbox": [xmin, ymin, xmax, ymax]})
    if not objects and not errors:
        errors.append("annotation_without_objects")
    return objects, errors


def audit_dataset(root: Path, source: dict[str, str] | None = None, near_threshold: int = 5) -> dict[str, Any]:
    """Audit a flat or nested image/VOC layout and return JSON-safe factual results."""
    source = source or DEFAULT_SOURCE
    if not root.is_dir():
        return {
            "status": "blocked",
            "reason": f"Dataset directory does not exist: {root}",
            "source": source,
            "images": 0,
            "annotations": 0,
            "objects": 0,
        }
    image_files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    annotation_files = sorted(root.rglob("*.xml"))
    if not image_files:
        orphan_errors = [
            {"file": annotation.relative_to(root).as_posix(), "issue": "annotation_without_image"}
            for annotation in annotation_files
        ]
        return {
            "status": "blocked",
            "reason": f"No supported image files found under: {root}",
            "source": source,
            "images": 0,
            "annotations": len(annotation_files),
            "objects": 0,
            "errors": orphan_errors,
            "records": [],
        }
    annotation_dirs = sorted({path.parent for path in annotation_files})
    voc_roots = sorted(path for path in root.rglob("VOC2007") if (path / "JPEGImages").is_dir())
    supplied_splits = discover_supplied_splits(voc_roots)
    primary_voc_root = voc_roots[0] if voc_roots else None
    primary_image_files = [
        image for image in image_files if primary_voc_root is None or primary_voc_root in image.parents
    ]
    primary_annotation_files = [
        annotation for annotation in annotation_files if primary_voc_root is None or primary_voc_root in annotation.parents
    ]
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    class_counts: Counter[str] = Counter()
    physical_class_counts: Counter[str] = Counter()
    resolution_counts: Counter[str] = Counter()
    channel_counts: Counter[int] = Counter()
    hash_groups: defaultdict[str, list[str]] = defaultdict(list)
    primary_hash_groups: defaultdict[str, list[str]] = defaultdict(list)
    image_hashes: list[tuple[str, int]] = []
    multi_object_images = 0
    multi_class_images = 0
    annotated_images = 0
    total_objects = 0
    physical_annotated_images = 0
    physical_total_objects = 0
    physical_multi_object_images = 0
    physical_multi_class_images = 0
    used_annotations: set[Path] = set()

    for image_path in image_files:
        relative = image_path.relative_to(root).as_posix()
        is_primary = primary_voc_root is None or primary_voc_root in image_path.parents
        record: dict[str, Any] = {
            "image": relative,
            "annotation": "",
            "status": "ok",
            "object_count": 0,
            "classes": "",
            "is_primary_fold": is_primary,
        }
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                width, height, mode = image.size[0], image.size[1], image.mode
                channels = len(image.getbands())
                perceptual = dhash(image)
        except (OSError, UnidentifiedImageError, ValueError) as error:
            record["status"] = "corrupt_image"
            errors.append({"file": relative, "issue": f"corrupt_image: {error}"})
            records.append(record)
            continue
        record.update({"width": width, "height": height, "channels": channels, "mode": mode, "sha256": sha256(image_path)})
        if is_primary:
            resolution_counts[f"{width}x{height}"] += 1
            channel_counts[channels] += 1
        hash_groups[record["sha256"]].append(relative)
        if is_primary:
            primary_hash_groups[record["sha256"]].append(relative)
            image_hashes.append((relative, perceptual))
        annotation = find_annotation(image_path, annotation_dirs)
        if annotation is None:
            record["status"] = "missing_annotation"
            errors.append({"file": relative, "issue": "missing_annotation"})
        else:
            record["annotation"] = annotation.relative_to(root).as_posix()
            used_annotations.add(annotation)
            physical_annotated_images += 1
            if is_primary:
                annotated_images += 1
            objects, annotation_errors = parse_voc(annotation, width, height)
            record["object_count"] = len(objects)
            record["classes"] = ";".join(sorted({item["class"] for item in objects}))
            physical_total_objects += len(objects)
            physical_class_counts.update(item["class"] for item in objects)
            if len(objects) > 1:
                physical_multi_object_images += 1
            if len({item["class"] for item in objects}) > 1:
                physical_multi_class_images += 1
            if is_primary:
                total_objects += len(objects)
                class_counts.update(item["class"] for item in objects)
                if len(objects) > 1:
                    multi_object_images += 1
                if len({item["class"] for item in objects}) > 1:
                    multi_class_images += 1
            for issue in annotation_errors:
                errors.append({"file": record["annotation"], "issue": issue})
            if annotation_errors:
                record["status"] = "annotation_error"
        records.append(record)

    for annotation in annotation_files:
        if annotation not in used_annotations:
            errors.append({"file": annotation.relative_to(root).as_posix(), "issue": "annotation_without_image"})

    exact_duplicates = [files for files in primary_hash_groups.values() if len(files) > 1]
    structural_duplicates = [files for files in hash_groups.values() if len(files) > 1]
    near_duplicates: list[dict[str, Any]] = []
    for left_index, (left_name, left_hash) in enumerate(image_hashes):
        for right_name, right_hash in image_hashes[left_index + 1 :]:
            distance = hamming_distance(left_hash, right_hash)
            if distance <= near_threshold:
                near_duplicates.append({"image_a": left_name, "image_b": right_name, "hamming_distance": distance})
    return {
        "status": "complete",
        "source": source,
        "dataset_root": str(root),
        "images": len(primary_image_files),
        "annotations": len(primary_annotation_files),
        "annotated_images": annotated_images,
        "objects": total_objects,
        "class_distribution": dict(sorted(class_counts.items())),
        "image_formats": dict(sorted(Counter(path.suffix.lower() for path in primary_image_files).items())),
        "resolutions": dict(sorted(resolution_counts.items())),
        "channels": dict(sorted(channel_counts.items())),
        "multi_object_images": multi_object_images,
        "multi_class_images": multi_class_images,
        "exact_duplicate_groups": exact_duplicates,
        "structural_duplicate_groups_across_folds": structural_duplicates,
        "near_duplicate_pairs": near_duplicates,
        "folds": [path.parent.name for path in voc_roots],
        "supplied_splits": supplied_splits,
        "physical_images": len(image_files),
        "physical_annotations": len(annotation_files),
        "physical_annotated_images": physical_annotated_images,
        "physical_objects": physical_total_objects,
        "physical_class_distribution": dict(sorted(physical_class_counts.items())),
        "physical_multi_object_images": physical_multi_object_images,
        "physical_multi_class_images": physical_multi_class_images,
        "errors": errors,
        "records": records,
    }


def _legacy_write_outputs(report: dict[str, Any], json_path: Path, csv_path: Path, markdown_path: Path) -> None:
    for path in (json_path, csv_path, markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["image", "annotation", "status", "width", "height", "channels", "mode", "sha256", "object_count", "classes"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report.get("records", []))
    if report["status"] == "blocked":
        body = "\n".join([
            "# Аудит NEU-DET",
            "",
            "Статус: **заблокирован**.",
            "",
            f"Причина: `{report['reason']}`.",
            "",
            "Источник для требуемой detection-версии:",
            f"- Репозиторий: {report['source']['detection_source']}",
            f"- Загрузка: {report['source']['detection_download']}",
            f"- Требуемый архив: `{report['source']['expected_archive']}`",
            "",
            "Указанный в задаче Kaggle-набор классифицирован как classification-only по его публичным метаданным и не используется для обучения детектора.",
            "",
        ])
    else:
        body = "\n".join([
            "# Аудит NEU-DET",
            "",
            "Статус: **завершён**.",
            f"- Изображений: {report['images']}",
            f"- XML-аннотаций: {report['annotations']}",
            f"- Объектов: {report['objects']}",
            f"- Распределение классов: `{report['class_distribution']}`",
            f"- Изображений с несколькими объектами: {report['multi_object_images']}",
            f"- Изображений с несколькими классами: {report['multi_class_images']}",
            f"- Групп точных дубликатов: {len(report['exact_duplicate_groups'])}",
            f"- Пар потенциально близких изображений: {len(report['near_duplicate_pairs'])}",
            f"- Ошибок/предупреждений: {len(report['errors'])}",
        ]) + "\n"
    markdown_path.write_text(body, encoding="utf-8")


def write_outputs(report: dict[str, Any], json_path: Path, csv_path: Path, markdown_path: Path) -> None:
    """Write machine-readable records and a concise human-readable summary."""
    for path in (json_path, csv_path, markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["image", "annotation", "status", "is_primary_fold", "width", "height", "channels", "mode", "sha256", "object_count", "classes"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(report.get("records", []))
    if report["status"] == "blocked":
        body = f"# NEU-DET audit\n\nStatus: **blocked**.\n\nReason: `{report['reason']}`.\n"
    else:
        body = "\n".join([
            "# NEU-DET audit",
            "",
            "Status: **complete**.",
            "",
            "## Logical dataset (one canonical fold)",
            f"- Images: {report['images']}",
            f"- XML annotations: {report['annotations']}",
            f"- Objects: {report['objects']}",
            f"- Class distribution: `{report['class_distribution']}`",
            f"- Multi-object images: {report['multi_object_images']}",
            f"- Multi-class images: {report['multi_class_images']}",
            f"- Exact duplicate groups within canonical fold: {len(report['exact_duplicate_groups'])}",
            f"- Potentially similar pairs within canonical fold: {len(report['near_duplicate_pairs'])}",
            "",
            "## Package structure",
            f"- Folds: `{report['folds']}`",
            f"- Supplied split files: `{report['supplied_splits']}`",
            f"- Physical image files: {report['physical_images']}",
            f"- Physical XML files: {report['physical_annotations']}",
            f"- Cross-fold SHA-256 duplicate groups: {len(report['structural_duplicate_groups_across_folds'])}",
            f"- Validation errors: {len(report['errors'])}",
            "",
            "The five folds are alternate cross-validation layouts, not 9,000 independent training samples.",
        ]) + "\n"
    markdown_path.write_text(body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit NEU-DET images and Pascal VOC annotations without modifying them.")
    parser.add_argument("--root", type=Path, default=Path("data/raw/NEU-DET"))
    parser.add_argument("--json", type=Path, default=Path("reports/metrics/dataset_audit.json"))
    parser.add_argument("--csv", type=Path, default=Path("reports/tables/dataset_audit.csv"))
    parser.add_argument("--markdown", type=Path, default=Path("reports/dataset_audit.md"))
    parser.add_argument("--near-threshold", type=int, default=5)
    args = parser.parse_args()
    report = audit_dataset(args.root, near_threshold=args.near_threshold)
    report["audited_at_utc"] = datetime.now(timezone.utc).isoformat()
    write_outputs(report, args.json, args.csv, args.markdown)
    omitted = ("records", "exact_duplicate_groups", "structural_duplicate_groups_across_folds", "near_duplicate_pairs")
    summary = {key: value for key, value in report.items() if key not in omitted}
    summary["exact_duplicate_group_count"] = len(report.get("exact_duplicate_groups", []))
    summary["structural_duplicate_group_count"] = len(report.get("structural_duplicate_groups_across_folds", []))
    summary["near_duplicate_pair_count"] = len(report.get("near_duplicate_pairs", []))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "complete" else 2


if __name__ == "__main__":
    sys.exit(main())
