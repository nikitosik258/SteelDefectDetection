"""Controlled GPU latency benchmark for the four frozen checkpoints."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
import ultralytics


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ("E1", "E2", "E3", "E4")
IMG_SIZE, BATCH, CONF, NMS_IOU, MAX_DET = 320, 1, 0.25, 0.7, 300


def _summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {"mean_ms": float(array.mean()), "median_ms": float(np.median(array)), "p95_ms": float(np.percentile(array, 95)), "std_ms": float(array.std(ddof=0)), "calculated_fps": float(1000 / array.mean())}


def _synchronize() -> None:
    torch.cuda.synchronize(0)


def _time(callable_: Any) -> float:
    _synchronize(); start = time.perf_counter(); callable_(); _synchronize()
    return (time.perf_counter() - start) * 1000


def _tensor(image_path: Path) -> torch.Tensor:
    image = cv2.imread(str(image_path))
    if image is None: raise FileNotFoundError(image_path)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).unsqueeze(0).to(0, dtype=torch.float32).div_(255)


def _parameters(model: YOLO) -> int:
    return sum(parameter.numel() for parameter in model.model.parameters())


def _forward(model: YOLO, tensor: torch.Tensor) -> None:
    with torch.inference_mode():
        model.model(tensor)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark frozen E1--E4 models under identical CUDA conditions.")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--runs", type=int, default=100)
    arguments = parser.parse_args()
    if arguments.warmup < 30 or arguments.runs < 100: raise ValueError("Protocol requires at least 30 warmups and 100 measurements")
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU is required for the benchmark")
    test_images = sorted((ROOT / "data" / "processed" / "images" / "test").glob("*.jpg"))
    if not test_images: raise FileNotFoundError("No Test images")
    image_path, tensor = test_images[0], _tensor(test_images[0])
    rows, details = [], []
    torch.backends.cudnn.benchmark = False
    for experiment in EXPERIMENTS:
        checkpoint = ROOT / "runs" / experiment / "weights" / "best.pt"
        model = YOLO(str(checkpoint)); model.model.to(0).float().eval()
        for _ in range(arguments.warmup):
            _forward(model, tensor)
            model.predict(str(image_path), imgsz=IMG_SIZE, conf=CONF, iou=NMS_IOU, max_det=MAX_DET, device=0, verbose=False)
        inference_times = [_time(lambda: _forward(model, tensor)) for _ in range(arguments.runs)]
        end_to_end_times = [_time(lambda: model.predict(str(image_path), imgsz=IMG_SIZE, conf=CONF, iou=NMS_IOU, max_det=MAX_DET, device=0, verbose=False)) for _ in range(arguments.runs)]
        record = {"experiment": experiment, "checkpoint": str(checkpoint.relative_to(ROOT)).replace("\\", "/"), "parameters": _parameters(model), "model_inference": _summary(inference_times), "end_to_end": _summary(end_to_end_times)}
        details.append(record)
        rows.append({"experiment": experiment, "parameters": record["parameters"], **{f"inference_{key}": value for key, value in record["model_inference"].items()}, **{f"end_to_end_{key}": value for key, value in record["end_to_end"].items()}})
    properties = torch.cuda.get_device_properties(0)
    environment = {"gpu": properties.name, "total_vram_gib": round(properties.total_memory / 2**30, 2), "free_vram_gib_before_report": round((properties.total_memory - torch.cuda.memory_allocated(0)) / 2**30, 2), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "ultralytics": ultralytics.__version__, "precision": "FP32", "imgsz": IMG_SIZE, "batch": BATCH, "warmup_runs": arguments.warmup, "measured_runs": arguments.runs, "input_image": image_path.name, "nms": {"confidence": CONF, "iou": NMS_IOU, "max_det": MAX_DET}, "execution": "sequential, one model at a time; CUDA synchronized around every measurement"}
    payload = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "environment": environment, "models": details}
    metrics, tables, figures, reports = ROOT / "reports" / "metrics", ROOT / "reports" / "tables", ROOT / "reports" / "figures", ROOT / "reports"
    (metrics / "benchmark_results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    frame = pd.DataFrame(rows); frame.to_csv(tables / "inference_benchmark.csv", index=False)
    accuracy = pd.read_csv(tables / "model_comparison.csv")[["experiment", "mAP50_95"]]
    chart = frame.merge(accuracy, on="experiment")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(chart["experiment"], chart["mAP50_95"], color="#377eb8"); axes[0].set_ylim(0, 0.5); axes[0].set_ylabel("Test mAP@50:95"); axes[0].set_title("Accuracy")
    axes[1].bar(chart["experiment"], chart["end_to_end_mean_ms"], color="#e6550d"); axes[1].set_ylabel("End-to-end latency, ms/image"); axes[1].set_title("Sequential FP32 latency")
    fig.tight_layout(); fig.savefig(figures / "accuracy_speed_comparison.png", dpi=300); plt.close(fig)
    reports.joinpath("benchmark_report.md").write_text(
        "# GPU inference benchmark\n\nAll checkpoints were run sequentially on the same GPU, FP32, batch=1, imgsz=320, with 30 warmups and 100 synchronized measurements. Latency was measured repeatedly on one fixed Test image (`crazing_1.jpg`). "
        "Model-inference latency measures the model forward pass using a preprocessed resident CUDA tensor. End-to-end latency measures Ultralytics sequential prediction, including image loading, preprocessing and postprocessing; checkpoint loading is excluded. Calculated FPS is 1000 / mean latency and is not an independently measured throughput. Accuracy and error-analysis results are from one seed and one fixed split.\n\n"
        + frame.to_csv(index=False) + "\nEnvironment:\n\n```json\n" + json.dumps(environment, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    print(json.dumps({"models": list(EXPERIMENTS), "outputs": ["reports/metrics/benchmark_results.json", "reports/tables/inference_benchmark.csv"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__": raise SystemExit(main())
