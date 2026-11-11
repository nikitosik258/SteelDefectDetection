from pathlib import Path
from xml.etree import ElementTree as ET

from PIL import Image

from src.eda import build_statistics, load_voc_records


def make_voc_sample(root: Path) -> None:
    image_dir, annotation_dir = root / "JPEGImages", root / "Annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    Image.new("RGB", (20, 20), color="gray").save(image_dir / "sample.jpg")
    annotation = ET.Element("annotation")
    for name, coords in (("crazing", (1, 1, 5, 5)), ("scratches", (5, 5, 15, 15))):
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = name
        box = ET.SubElement(obj, "bndbox")
        for key, value in zip(("xmin", "ymin", "xmax", "ymax"), coords):
            ET.SubElement(box, key).text = str(value)
    ET.ElementTree(annotation).write(annotation_dir / "sample.xml")


def test_eda_keeps_image_and_object_counts_distinct(tmp_path: Path) -> None:
    make_voc_sample(tmp_path)
    images, objects = load_voc_records(tmp_path)
    images, dataset_statistics, class_statistics, bbox_statistics, summary = build_statistics(images, objects)
    assert summary["images"] == 1
    assert summary["objects"] == 2
    assert summary["image_class_counts"] == {"crazing": 1, "scratches": 1}
    assert summary["object_class_counts"] == {"crazing": 1, "scratches": 1}
    assert dataset_statistics.loc[0, "value"] == 1
    assert len(class_statistics) == 6
    assert len(bbox_statistics) == 5
