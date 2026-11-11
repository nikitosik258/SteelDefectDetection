from src.training_pipeline import IdentityAlbumentations, describe_transforms


class DummyDataset:
    def __init__(self) -> None:
        self.transforms = type("Transforms", (), {"transforms": [IdentityAlbumentations()]})()


def test_identity_transform_preserves_labels() -> None:
    labels = {"value": 1}
    transform = IdentityAlbumentations()
    assert transform(labels) is labels


def test_transform_description_is_stable() -> None:
    assert describe_transforms(DummyDataset()) == ["IdentityAlbumentations"]
