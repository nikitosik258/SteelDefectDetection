from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image

from src.audit_dataset import audit_dataset


def write_image(path: Path) -> None:
    Image.new("L", (20, 20), color=128).save(path)


def write_xml(path: Path, objects: list[tuple[str, tuple[int, int, int, int]]]) -> None:
    root = ET.Element("annotation")
    for name, (xmin, ymin, xmax, ymax) in objects:
        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = name
        box = ET.SubElement(obj, "bndbox")
        for key, value in zip(("xmin", "ymin", "xmax", "ymax"), (xmin, ymin, xmax, ymax)):
            ET.SubElement(box, key).text = str(value)
    ET.ElementTree(root).write(path)


def test_valid_image_and_multiple_objects(tmp_path: Path) -> None:
    write_image(tmp_path / "sample.jpg")
    write_xml(tmp_path / "sample.xml", [("crazing", (1, 1, 10, 10)), ("scratches", (3, 3, 15, 15))])
    report = audit_dataset(tmp_path)
    assert report["images"] == 1
    assert report["objects"] == 2
    assert report["multi_object_images"] == 1
    assert report["multi_class_images"] == 1
    assert not report["errors"]


def test_missing_image_is_not_invented(tmp_path: Path) -> None:
    write_xml(tmp_path / "orphan.xml", [("crazing", (1, 1, 10, 10))])
    report = audit_dataset(tmp_path)
    assert report["images"] == 0
    assert report["annotations"] == 1
    assert any(error["issue"] == "annotation_without_image" for error in report["errors"])


def test_missing_annotation(tmp_path: Path) -> None:
    write_image(tmp_path / "sample.jpg")
    report = audit_dataset(tmp_path)
    assert any(error["issue"] == "missing_annotation" for error in report["errors"])


def test_unknown_class_and_invalid_boxes(tmp_path: Path) -> None:
    write_image(tmp_path / "sample.jpg")
    write_xml(tmp_path / "sample.xml", [("unknown", (1, 1, 5, 5)), ("crazing", (8, 8, 2, 2)), ("patches", (-1, 1, 30, 5))])
    report = audit_dataset(tmp_path)
    issues = [error["issue"] for error in report["errors"]]
    assert any("unknown_class" in issue for issue in issues)
    assert any("non_positive_bbox_area" in issue for issue in issues)
    assert any("bbox_out_of_bounds" in issue for issue in issues)


def test_corrupt_file(tmp_path: Path) -> None:
    (tmp_path / "broken.jpg").write_bytes(b"not an image")
    report = audit_dataset(tmp_path)
    assert any("corrupt_image" in error["issue"] for error in report["errors"])


def test_voc_folds_use_their_own_annotations_and_one_canonical_fold(tmp_path: Path) -> None:
    for fold, label in (("flod1", "crazing"), ("flod2", "scratches")):
        voc_root = tmp_path / "package" / fold / "VOC2007"
        image_dir = voc_root / "JPEGImages"
        annotation_dir = voc_root / "Annotations"
        image_dir.mkdir(parents=True)
        annotation_dir.mkdir()
        write_image(image_dir / "same_name.jpg")
        write_xml(annotation_dir / "same_name.xml", [(label, (1, 1, 10, 10))])
    report = audit_dataset(tmp_path)
    assert report["images"] == 1
    assert report["physical_images"] == 2
    assert report["class_distribution"] == {"crazing": 1}
    assert report["physical_class_distribution"] == {"crazing": 1, "scratches": 1}
    assert not report["errors"]
    assert all(record["image"].split("/")[1] == record["annotation"].split("/")[1] for record in report["records"])
