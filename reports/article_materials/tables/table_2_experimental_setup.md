# Таблица 2. Конфигурации экспериментов

| Эксперимент | Архитектура | Профиль | Умеренные аугментации |
|---|---|---|---|
| E1 | yolov8n.pt | baseline | нет |
| E2 | yolo11n.pt | baseline | нет |
| E3 | yolov8n.pt | augmented | flip H/V=0.5; rot.=5; tr.=0.05; sc.=0.10; HSV-V=0.10 |
| E4 | yolo11n.pt | augmented | flip H/V=0.5; rot.=5; tr.=0.05; sc.=0.10; HSV-V=0.10 |

Общее: imgsz=320 px; batch=16; AdamW; lr0=0.001; weight decay=0.0005; epochs=100; patience=20; seed=42; AMP; mosaic/mixup/cutmix=0.

Источник: `configs/experiments.yaml`, `runs/E1…E4/effective_config.yaml`.
