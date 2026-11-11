# NEU-DET exploratory data analysis

## Confirmed observations
- Analysis uses only canonical `flod1`: 1800 images and 4189 Pascal VOC objects.
- The object count matches the dataset audit (4189).
- Image-level class counts: `{'crazing': 300, 'inclusion': 382, 'patches': 342, 'pitted_surface': 301, 'rolled-in_scale': 300, 'scratches': 300}`.
- Object-level class counts: `{'crazing': 689, 'inclusion': 1011, 'patches': 881, 'pitted_surface': 432, 'rolled-in_scale': 628, 'scratches': 548}`. These differ because images can contain multiple objects/classes.
- 123 images contain more than one class.
- 50 objects occupy less than 1% of the 200 x 200 image area; this is an explicit EDA threshold, not a dataset-provided label.
- `inclusion` has the largest object count and `pitted_surface` the smallest; count alone does not establish detection difficulty.
- Audit validation errors: 0. One exact duplicate pair and near-duplicate candidates remain in raw data; none were removed.

## Interpretation requiring experimental validation
- The high number of small objects, especially in `inclusion`, suggests testing scale-preserving augmentation and image-size choices. Its benefit must be measured on validation data only.
- Unequal object counts may affect per-class recall and mAP. This is a hypothesis for per-class evaluation, not evidence that any class is intrinsically harder.
- Multi-class images require multi-label detection handling; class co-occurrence should not be simplified into one image-level label.

## Dataset limitations
- Native images are only 200 x 200 pixels; aggressive upscaling cannot create new defect detail.
- The archive contains five fold copies, not 9,000 independent samples. Any future split must use one fold or an explicitly leakage-safe protocol.
- The supplied folds have train/validation lists but no separate test list. A test protocol must be specified before experiments and must not guide model selection.
