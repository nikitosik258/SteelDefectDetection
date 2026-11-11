"""Run one fixed YOLO experiment or a separate one-epoch smoke test."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.nn.tasks import load_checkpoint

from src.training_pipeline import NoAlbumentationsDetectionTrainer


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "experiments.yaml"

# These values define the scientific protocol and must not silently change on resume.
RESUME_CONFIG_KEYS = (
    "task", "data", "epochs", "patience", "batch", "imgsz", "optimizer", "lr0", "lrf",
    "weight_decay", "cos_lr", "warmup_epochs", "seed", "deterministic", "pretrained",
    "multi_scale", "close_mosaic", "device", "workers", "amp", "save", "val", "plots",
    "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear", "perspective",
    "fliplr", "flipud", "mosaic", "mixup", "cutmix", "copy_paste", "erasing", "bgr",
)


class ResumeValidationError(RuntimeError):
    """Raised before training when a checkpoint is unsafe to resume."""


def load_protocol(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and minimally validate the fixed experiment protocol."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"common", "augmentation_profiles", "experiments"}
    if not required.issubset(payload):
        raise ValueError(f"Invalid experiment configuration: missing {required - set(payload)}")
    return payload


def experiment_settings(experiment: str, protocol: dict[str, Any]) -> dict[str, Any]:
    """Resolve one E1--E4 configuration without architecture-specific tuning."""
    if experiment not in protocol["experiments"]:
        raise ValueError(f"Unknown experiment {experiment}; choose one of {sorted(protocol['experiments'])}")
    entry = protocol["experiments"][experiment]
    profile = entry["profile"]
    if profile not in protocol["augmentation_profiles"]:
        raise ValueError(f"Unknown augmentation profile: {profile}")
    settings = dict(protocol["common"])
    settings.update(protocol["augmentation_profiles"][profile])
    settings.update({"model": entry["model"], "profile": profile, "experiment": experiment})
    return settings


def environment_report() -> dict[str, Any]:
    """Return factual compute capability without allocating GPU tensors."""
    cuda = torch.cuda.is_available()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2) if cuda else None,
        "amp_enabled": bool(cuda),
        "recommended_workers_windows": 2,
    }


def ensure_new_run(run_dir: Path, resume: bool) -> None:
    """Avoid accidental replacement of experiment artefacts."""
    if resume:
        checkpoint = run_dir / "weights" / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Cannot resume: checkpoint not found: {checkpoint}")
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory already contains files: {run_dir}. Use --resume only for this same run.")


def _canonical_value(key: str, value: Any) -> Any:
    """Normalise values whose equivalent textual representations differ on Windows."""
    if key == "data":
        path = Path(str(value))
        return str((ROOT / path).resolve()) if not path.is_absolute() else str(path.resolve())
    if key == "device":
        return str(value)
    return value


def _architecture_signature(model: Any) -> str:
    """Hash the model YAML except class count, which legitimately changes from COCO to NEU-DET."""
    model_yaml = dict(getattr(model, "yaml", {}))
    model_yaml.pop("nc", None)
    return json.dumps(model_yaml, sort_keys=True, separators=(",", ":"), default=str)


def _expected_architecture_signature(model_name: str) -> str:
    """Load the requested pretrained architecture only to compare its static YAML structure."""
    return _architecture_signature(YOLO(model_name).model)


def _checkpoint_model_is_expected(checkpoint_model: Any, expected_model: str, checkpoint: Path) -> bool:
    """Accept the original pretrained name or the same run's own last checkpoint on later resumes."""
    value = str(checkpoint_model)
    if Path(value).name == Path(expected_model).name:
        return True
    try:
        return Path(value).resolve() == checkpoint.resolve()
    except OSError:
        return False


def validate_resume(run_dir: Path, expected_settings: dict[str, Any]) -> dict[str, Any]:
    """Validate provenance and compatibility without changing existing run metadata."""
    checkpoint = run_dir / "weights" / "last.pt"
    config_path = run_dir / "effective_config.yaml"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Cannot resume: checkpoint not found: {checkpoint}")
    if not config_path.is_file():
        raise ResumeValidationError(f"Cannot resume safely: original metadata is missing: {config_path}")
    try:
        original = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ResumeValidationError(f"Cannot read original metadata: {error}") from error
    if not isinstance(original, dict):
        raise ResumeValidationError("Cannot resume safely: original metadata is not a mapping.")

    mismatches: list[str] = []
    for key in ("experiment", "model", "profile"):
        if original.get(key) != expected_settings.get(key):
            mismatches.append(f"metadata {key}={original.get(key)!r}, requested={expected_settings.get(key)!r}")
    for key in RESUME_CONFIG_KEYS:
        if key not in original:
            mismatches.append(f"metadata lacks {key}")
        elif _canonical_value(key, original[key]) != _canonical_value(key, expected_settings[key]):
            mismatches.append(f"metadata {key} differs")
    if mismatches:
        raise ResumeValidationError("Refusing resume because the original run differs: " + "; ".join(mismatches))

    try:
        checkpoint_model, checkpoint_data = load_checkpoint(checkpoint, fuse=False)
    except Exception as error:
        raise ResumeValidationError(f"Cannot load resume checkpoint {checkpoint}: {error}") from error
    checkpoint_args = checkpoint_data.get("train_args") if isinstance(checkpoint_data, dict) else None
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = getattr(checkpoint_model, "args", None)
    if not isinstance(checkpoint_args, dict):
        raise ResumeValidationError("Cannot resume safely: checkpoint has no training arguments.")

    checkpoint_mismatches: list[str] = []
    if not _checkpoint_model_is_expected(checkpoint_args.get("model"), expected_settings["model"], checkpoint):
        checkpoint_mismatches.append("checkpoint model origin differs")
    for key in RESUME_CONFIG_KEYS:
        if key not in checkpoint_args:
            checkpoint_mismatches.append(f"checkpoint lacks {key}")
        elif _canonical_value(key, checkpoint_args[key]) != _canonical_value(key, expected_settings[key]):
            checkpoint_mismatches.append(f"checkpoint {key} differs")
    try:
        if _architecture_signature(checkpoint_model) != _expected_architecture_signature(expected_settings["model"]):
            checkpoint_mismatches.append("checkpoint architecture differs")
    except Exception as error:
        raise ResumeValidationError(f"Cannot verify checkpoint architecture: {error}") from error
    if checkpoint_mismatches:
        raise ResumeValidationError("Refusing resume because the checkpoint differs: " + "; ".join(checkpoint_mismatches))

    return {"checkpoint": str(checkpoint), "original_experiment": original["experiment"]}


def write_run_status(run_dir: Path, settings: dict[str, Any], status: str, environment: dict[str, Any]) -> None:
    """Update current status while the immutable original configuration remains separate."""
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": status, "updated_at_utc": datetime.now(timezone.utc).isoformat(), "settings": settings, "environment": environment}
    (run_dir / "status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_run_event(run_dir: Path, event: dict[str, Any]) -> None:
    """Append an immutable event record, preserving provenance across resumes."""
    payload = {"recorded_at_utc": datetime.now(timezone.utc).isoformat(), **event}
    with (run_dir / "run_history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_initial_run_metadata(run_dir: Path, settings: dict[str, Any], environment: dict[str, Any]) -> None:
    """Write immutable origin metadata only for a newly guarded run directory."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "effective_config.yaml").write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    append_run_event(run_dir, {"event": "started", "experiment": settings["experiment"]})
    write_run_status(run_dir, settings, "started", environment)


def run_training(experiment: str, resume: bool = False, smoke_test: bool = False) -> Path:
    protocol = load_protocol()
    settings = experiment_settings(experiment, protocol)
    environment = environment_report()
    if not environment["cuda_available"]:
        raise RuntimeError("CUDA is unavailable; full experiments must not run on CPU.")
    data_path = ROOT / settings["data"]
    check_det_dataset(str(data_path))
    run_dir = ROOT / "runs" / ("_smoke" if smoke_test else experiment)
    ensure_new_run(run_dir, resume)
    train_args = {key: value for key, value in settings.items() if key not in {"data", "model", "profile", "experiment"}}
    # The directory guard above is the protection against accidental overwrites.
    # ``exist_ok=True`` then prevents Ultralytics from silently creating E12,
    # _smoke2, etc. after this script has stored run metadata in the directory.
    train_args.update({"data": str(data_path), "project": str(run_dir.parent), "name": run_dir.name, "exist_ok": True, "trainer": NoAlbumentationsDetectionTrainer})
    if smoke_test:
        # 2% is only 25 training images, while batch=16 exercises the planned
        # batch size at least once. The run remains isolated from E1--E4.
        train_args.update({"epochs": 1, "batch": 16, "fraction": 0.02, "workers": 0, "patience": 1, "plots": False, "save_period": 1})
    run_settings = settings | {
        "smoke_test": smoke_test,
        "effective_train_args": {key: str(value) for key, value in train_args.items() if key != "trainer"},
    }
    resume_info: dict[str, Any] | None = None
    if resume:
        # This must happen before appending events or changing status/config files.
        resume_info = validate_resume(run_dir, settings)
        append_run_event(run_dir, {"event": "resume_validated", "requested_experiment": experiment, **resume_info})
        write_run_status(run_dir, run_settings | {"resume": resume_info}, "resuming", environment)
    else:
        write_initial_run_metadata(run_dir, run_settings, environment)
    model_source = run_dir / "weights" / "last.pt" if resume else settings["model"]
    try:
        model = YOLO(str(model_source))
        model.train(resume=resume, **train_args)
    except Exception as error:
        append_run_event(run_dir, {"event": "resume_failed" if resume else "failed", "error": repr(error), "resume": resume_info})
        write_run_status(run_dir, run_settings | ({"resume": resume_info} if resume_info else {}), "failed", environment | {"error": repr(error)})
        raise
    artifacts = [str(path.relative_to(run_dir)) for path in run_dir.rglob("*") if path.is_file()]
    append_run_event(run_dir, {"event": "resumed_completed" if resume else "completed", "resume": resume_info})
    write_run_status(run_dir, run_settings | ({"resume": resume_info} if resume_info else {}) | {"artifacts": artifacts}, "completed", environment)
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a fixed NEU-DET YOLO experiment; no hyperparameter search is performed.")
    parser.add_argument("--experiment", choices=("E1", "E2", "E3", "E4"), default="E1")
    parser.add_argument("--resume", action="store_true", help="Resume only the same experiment from runs/<ID>/weights/last.pt.")
    parser.add_argument("--smoke-test", action="store_true", help="Run one separate epoch in runs/_smoke; never writes E1--E4.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration, dataset and hardware without training.")
    args = parser.parse_args()
    settings = experiment_settings(args.experiment, load_protocol())
    if args.dry_run:
        check_det_dataset(str(ROOT / settings["data"]))
        print(json.dumps({"settings": settings, "environment": environment_report()}, ensure_ascii=False, indent=2))
        return 0
    run_dir = run_training(args.experiment, resume=args.resume, smoke_test=args.smoke_test)
    print(f"Completed: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
