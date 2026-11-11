from pathlib import Path

from PIL import Image

from src.prepare_dataset import (
    DUPLICATE_IMAGE_EXCLUDE_ID,
    DUPLICATE_IMAGE_KEEP_ID,
    Sample,
    SPLIT_TARGETS,
    duplicate_voc_objects,
    exclude_conflicting_duplicate_from_processed,
    raw_yolo_lines,
    stratified_group_split,
    validate_prepared_dataset,
    write_processed_dataset,
    yolo_lines,
)


def sample(tmp_path: Path, image_id: str, digest: str, label: str = "crazing") -> Sample:
    image_path = tmp_path / f"{image_id}.jpg"
    Image.new("RGB", (20, 20), color="gray").save(image_path)
    return Sample(image_id, image_path, tmp_path / f"{image_id}.xml", digest, (label,), ({"class": label, "xmin": 2, "ymin": 3, "xmax": 12, "ymax": 13},), 20, 20)


def test_yolo_conversion_is_normalized(tmp_path: Path) -> None:
    item = sample(tmp_path, "one", "a")
    assert yolo_lines(item) == ["0 0.35000000 0.40000000 0.50000000 0.50000000"]


def test_exact_duplicate_voc_rows_are_documented_and_not_counted_as_distinct_objects(tmp_path: Path) -> None:
    item = sample(tmp_path, "duplicate", "a")
    duplicated = Sample(
        item.image_id,
        item.image_path,
        item.annotation_path,
        item.sha256,
        item.classes,
        item.objects + item.objects,
        item.width,
        item.height,
    )
    assert len(raw_yolo_lines(duplicated)) == 2
    assert len(yolo_lines(duplicated)) == 1
    assert duplicate_voc_objects(duplicated) == [{
        "class": "crazing", "xmin": 2, "ymin": 3, "xmax": 12, "ymax": 13,
        "occurrences": 2, "redundant_rows": 1,
    }]


def test_grouped_split_is_deterministic_and_keeps_exact_duplicates(tmp_path: Path) -> None:
    items = [sample(tmp_path, f"item_{index}", "duplicate" if index < 2 else str(index), "crazing" if index < 2 or index % 2 else "scratches") for index in range(10)]
    targets = {"train": 6, "val": 2, "test": 2}
    first, second = stratified_group_split(items, seed=42, targets=targets), stratified_group_split(items, seed=42, targets=targets)
    assert {key: [item.image_id for item in value] for key, value in first.items()} == {key: [item.image_id for item in value] for key, value in second.items()}
    duplicate_splits = [name for name, values in first.items() if {"item_0", "item_1"} & {item.image_id for item in values}]
    assert len(duplicate_splits) == 1


def test_processed_validation_checks_pairs_and_counts(tmp_path: Path) -> None:
    items = [sample(tmp_path, f"item_{index}", str(index)) for index in range(3)]
    splits = {"train": [items[0]], "val": [items[1]], "test": [items[2]]}
    output = tmp_path / "processed"
    write_processed_dataset(splits, output)
    summary = validate_prepared_dataset(splits, output)
    assert summary["split_sizes"] == {"train": 1, "val": 1, "test": 1}


def test_conflicting_duplicate_policy_removes_only_the_train_copy(tmp_path: Path) -> None:
    kept = sample(tmp_path, DUPLICATE_IMAGE_KEEP_ID, "same-sha", "patches")
    excluded = sample(tmp_path, DUPLICATE_IMAGE_EXCLUDE_ID, "same-sha", "patches")
    validation = sample(tmp_path, "crazing_1", "val-sha")
    test = sample(tmp_path, "scratches_1", "test-sha", "scratches")
    original = {"train": [kept, excluded], "val": [validation], "test": [test]}

    processed, record = exclude_conflicting_duplicate_from_processed(original)

    assert [item.image_id for item in processed["train"]] == [DUPLICATE_IMAGE_KEEP_ID]
    assert processed["val"] == original["val"]
    assert processed["test"] == original["test"]
    assert record["image_id"] == DUPLICATE_IMAGE_EXCLUDE_ID
