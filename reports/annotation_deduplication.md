# VOC-to-YOLO object-count reconciliation

- Canonical source VOC annotation rows: 4189.
- Excluded with `patches_105`: 3 rows.
- Remaining VOC annotation rows: 4186.
- Exact redundant VOC rows: 3.
- Unique YOLO objects / effective Ultralytics objects: 4183.

The following XML entries have the same class and exact `(xmin, ymin, xmax, ymax)` coordinates within one image. They are repeated annotations of one physical defect, not overlapping distinct defects. They are written once because Ultralytics 8.3.239 de-duplicates identical YOLO rows in its label cache; retaining duplicate rows would make the label text and effective training data disagree.

## Exact repeated entries
- `crazing_120.xml` (train): class=crazing, bbox=(1, 81, 117, 146), occurrences=2
- `inclusion_62.xml` (train): class=inclusion, bbox=(39, 77, 59, 140), occurrences=2
- `patches_198.xml` (train): class=patches, bbox=(158, 68, 200, 158), occurrences=2

No distinct bounding box was removed and raw XML/JPG files remain unchanged.
