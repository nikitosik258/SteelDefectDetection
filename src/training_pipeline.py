"""Verified Ultralytics training helpers shared by experiments and smoke tests."""

from __future__ import annotations

from typing import Any

from ultralytics.data.augment import Albumentations
from ultralytics.models.yolo.detect.train import DetectionTrainer


class IdentityAlbumentations:
    """Identity replacement for Ultralytics' implicit Albumentations stage."""

    def __call__(self, labels: dict[str, Any]) -> dict[str, Any]:
        return labels


def disable_implicit_albumentations(dataset: Any) -> int:
    """Replace only the default Albumentations stage, returning replacements made."""
    transforms = dataset.transforms.transforms
    count = 0
    for index, transform in enumerate(transforms):
        if isinstance(transform, Albumentations):
            transforms[index] = IdentityAlbumentations()
            count += 1
    return count


class NoAlbumentationsDetectionTrainer(DetectionTrainer):
    """Detection trainer that removes random implicit Albumentations transforms."""

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        dataset = super().build_dataset(img_path, mode=mode, batch=batch)
        if mode == "train":
            count = disable_implicit_albumentations(dataset)
            if count != 1:
                raise RuntimeError(f"Expected exactly one Albumentations stage, found {count}")
        return dataset


def describe_transforms(dataset: Any) -> list[str]:
    """Return effective transform class names for a built YOLO dataset."""
    return [transform.__class__.__name__ for transform in dataset.transforms.transforms]
