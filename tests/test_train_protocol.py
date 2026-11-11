from pathlib import Path
import json

import pytest
import yaml

import src.train as training
from src.train import ensure_new_run, experiment_settings, load_protocol


def test_experiment_mapping_and_profiles() -> None:
    protocol = load_protocol()
    assert experiment_settings("E1", protocol)["model"] == "yolov8n.pt"
    assert experiment_settings("E2", protocol)["model"] == "yolo11n.pt"
    assert experiment_settings("E3", protocol)["profile"] == "augmented"
    baseline = experiment_settings("E1", protocol)
    assert all(baseline[key] == 0 for key in ("hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "fliplr", "flipud", "mosaic", "mixup", "cutmix", "bgr"))


def test_fixed_protocol_and_augmentation_profile() -> None:
    protocol = load_protocol()
    e3 = experiment_settings("E3", protocol)
    assert {key: e3[key] for key in ("epochs", "batch", "imgsz", "optimizer", "lr0", "weight_decay", "seed")} == {
        "epochs": 100,
        "batch": 16,
        "imgsz": 320,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "weight_decay": 0.0005,
        "seed": 42,
    }
    assert e3["fliplr"] == e3["flipud"] == 0.5
    assert e3["hsv_v"] == 0.10
    assert e3["degrees"] == 5.0


def test_run_directory_protection(tmp_path: Path) -> None:
    run_dir = tmp_path / "E1"
    ensure_new_run(run_dir, resume=False)
    run_dir.mkdir()
    (run_dir / "result.txt").write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        ensure_new_run(run_dir, resume=False)
    with pytest.raises(FileNotFoundError):
        ensure_new_run(run_dir, resume=True)


def test_guarded_run_uses_requested_directory_and_records_effective_arguments(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    protocol = load_protocol()
    captured: dict[str, object] = {}

    class FakeYOLO:
        def __init__(self, source: str) -> None:
            captured["source"] = source

        def train(self, **kwargs: object) -> None:
            captured.update(kwargs)
            run_dir = Path(str(kwargs["project"])) / str(kwargs["name"])
            (run_dir / "weights").mkdir(parents=True, exist_ok=True)
            (run_dir / "weights" / "last.pt").write_bytes(b"checkpoint")

    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(training, "load_protocol", lambda: protocol)
    monkeypatch.setattr(training, "environment_report", lambda: {"cuda_available": True})
    monkeypatch.setattr(training, "check_det_dataset", lambda path: {"path": path})
    monkeypatch.setattr(training, "YOLO", FakeYOLO)
    monkeypatch.setattr(training, "validate_resume", lambda run_dir, settings: {"checkpoint": str(run_dir / "weights" / "last.pt"), "original_experiment": "E1"})

    result = training.run_training("E1", smoke_test=True)
    assert result == tmp_path / "runs" / "_smoke"
    assert captured["name"] == "_smoke"
    assert captured["exist_ok"] is True
    status = json.loads((result / "status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    assert status["settings"]["effective_train_args"]["batch"] == "16"

    training.run_training("E1", resume=True, smoke_test=True)
    assert captured["resume"] is True
    assert captured["source"] == str(result / "weights" / "last.pt")


def _make_resume_fixture(tmp_path: Path, settings: dict[str, object]) -> Path:
    run_dir = tmp_path / "runs" / "E1"
    (run_dir / "weights").mkdir(parents=True)
    (run_dir / "weights" / "last.pt").write_bytes(b"placeholder")
    (run_dir / "effective_config.yaml").write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    return run_dir


def test_validate_resume_accepts_matching_checkpoint(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected = experiment_settings("E1", load_protocol())
    run_dir = _make_resume_fixture(tmp_path, expected)
    checkpoint_args = dict(expected)
    fake_model = type("Model", (), {"yaml": {"nc": 6, "backbone": ["v8"]}})()
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(training, "load_checkpoint", lambda path, fuse=False: (fake_model, {"train_args": checkpoint_args}))
    monkeypatch.setattr(training, "_expected_architecture_signature", lambda model: "signature")
    monkeypatch.setattr(training, "_architecture_signature", lambda model: "signature")

    context = training.validate_resume(run_dir, expected)
    assert context["original_experiment"] == "E1"
    assert context["checkpoint"].endswith("weights\\last.pt")


def test_validate_resume_rejects_foreign_experiment_without_mutating_metadata(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected = experiment_settings("E1", load_protocol())
    foreign = dict(expected) | {"experiment": "E2", "model": "yolo11n.pt"}
    run_dir = _make_resume_fixture(tmp_path, foreign)
    before = (run_dir / "effective_config.yaml").read_bytes()
    monkeypatch.setattr(training, "ROOT", tmp_path)

    with pytest.raises(training.ResumeValidationError, match="metadata experiment"):
        training.validate_resume(run_dir, expected)
    assert (run_dir / "effective_config.yaml").read_bytes() == before


def test_validate_resume_rejects_checkpoint_from_other_architecture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected = experiment_settings("E1", load_protocol())
    run_dir = _make_resume_fixture(tmp_path, expected)
    checkpoint_args = dict(expected) | {"model": "yolo11n.pt"}
    fake_model = type("Model", (), {"yaml": {"nc": 6, "backbone": ["v11"]}})()
    monkeypatch.setattr(training, "ROOT", tmp_path)
    monkeypatch.setattr(training, "load_checkpoint", lambda path, fuse=False: (fake_model, {"train_args": checkpoint_args}))
    monkeypatch.setattr(training, "_expected_architecture_signature", lambda model: "v8")
    monkeypatch.setattr(training, "_architecture_signature", lambda model: "v11")

    with pytest.raises(training.ResumeValidationError, match="checkpoint model origin differs; checkpoint architecture differs"):
        training.validate_resume(run_dir, expected)
