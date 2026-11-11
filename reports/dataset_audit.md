# NEU-DET audit

Status: **complete**.

## Logical dataset (one canonical fold)
- Images: 1800
- XML annotations: 1800
- Objects: 4189
- Class distribution: `{'crazing': 689, 'inclusion': 1011, 'patches': 881, 'pitted_surface': 432, 'rolled-in_scale': 628, 'scratches': 548}`
- Multi-object images: 1317
- Multi-class images: 123
- Exact duplicate groups within canonical fold: 1
- Potentially similar pairs within canonical fold: 4420

## Package structure
- Folds: `['flod1', 'flod2', 'flod3', 'flod4', 'flod5']`
- Supplied split files: `{'flod1': {'train': 1440, 'trainval': 1800, 'val': 360}, 'flod2': {'train': 1440, 'trainval': 1800, 'val': 360}, 'flod3': {'train': 1440, 'trainval': 1800, 'val': 360}, 'flod4': {'train': 1440, 'trainval': 1800, 'val': 360}, 'flod5': {'train': 1440, 'trainval': 1800, 'val': 360}}`
- Physical image files: 9000
- Physical XML files: 9000
- Cross-fold SHA-256 duplicate groups: 1799
- Validation errors: 0

The five folds are alternate cross-validation layouts, not 9,000 independent training samples.
