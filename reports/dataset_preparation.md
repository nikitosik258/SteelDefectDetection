# NEU-DET dataset preparation

## Research split
- This is a project-specific 68/12/20 split, not an official NEU-DET partition.
- Seed: 42.
- Canonical source records: 1800; used processed images: 1799.
- Train/validation/test used for YOLO: 1223 / 216 / 360 images.
- The supplied 1440/360 lists were not reused: the known exact duplicate pair was split across them.
- Primary filename class is stratified exactly; secondary labels are retained as multi-label statistics.
- SHA-256 duplicate groups are kept within one split; one train-only conflicting-annotation duplicate is excluded from the processed dataset.
- Excluded processed file: `patches_105.jpg` (keep `patches_101.jpg`): Identical JPEG SHA-256 with conflicting Pascal VOC bounding boxes. Rule: Keep the lexicographically lower image identifier when identical JPEG bytes have conflicting XML annotations.
- Used VOC annotation rows: 4186; unique YOLO objects: 4183.
- 3 exact repeated XML rows in 3 images are represented once in YOLO labels. They are not unique defects; Ultralytics 8.3.239 would remove them during cache construction as well.
- dHash candidate pairs: 4420 total; 2263 cross-split. They are warnings, not confirmed duplicates, and were not deleted or automatically grouped.

## Conversion
- Pascal VOC XML was converted to normalized YOLO class_id xc yc width height labels.
- Mapping: 0=crazing, 1=inclusion, 2=patches, 3=pitted_surface, 4=rolled-in_scale, 5=scratches.
- Source XML/JPG files remain unchanged.
