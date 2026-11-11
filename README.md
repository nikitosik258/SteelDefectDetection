# Обнаружение дефектов металлических поверхностей (NEU-DET)

Учебный научно-исследовательский проект по компьютерному зрению. Сопоставляются YOLOv8n и YOLO11n для обнаружения и локализации шести типов дефектов стальных поверхностей, включая проверку влияния умеренных аугментаций.

## Цель и статус

Цель — экспериментально сравнить YOLOv8n и YOLO11n на фиксированном протоколе NEU-DET. Завершены аудит, EDA, подготовка данных, эксперименты E1--E4, единственная финальная оценка на Test, анализ ошибок, benchmark и материалы статьи. Результаты получены для одного `seed=42` и одного фиксированного разбиения; статистическая значимость различий не оценивалась.

## Данные

Использована detection-версия NEU-DET в Pascal VOC из [репозитория ETDNet](https://github.com/zht8506/ETDNet), архив `Neu_Det_5Flod_voc.zip`. Указанный ранее Kaggle-источник оказался classification-only и для детекции не использовался.

Классы YOLO: `0=crazing`, `1=inclusion`, `2=patches`, `3=pitted_surface`, `4=rolled-in_scale`, `5=scratches`.

| Состояние данных | Изображения | Объекты |
|---|---:|---:|
| Исходные Pascal VOC записи | 1800 | 4189 |
| Подготовленный набор | 1799 | 4183 |
| Train | 1223 | 2804 |
| Validation | 216 | 521 |
| Test | 360 | 858 |

`patches_105.jpg` исключён только из `data/processed/`: это точный JPEG-дубликат `patches_101.jpg` с конфликтующей XML-разметкой. При конвертации устранены три повторяющихся XML-объекта; файлы в `data/raw/` не изменялись. Обоснование: [dataset_preparation.md](reports/dataset_preparation.md), [annotation_deduplication.md](reports/annotation_deduplication.md).

## Эксперименты

| Код | Модель | Профиль |
|---|---|---|
| E1 | YOLOv8n | baseline, без случайных аугментаций |
| E2 | YOLO11n | baseline, без случайных аугментаций |
| E3 | YOLOv8n | moderate augmented |
| E4 | YOLO11n | moderate augmented |

Общие параметры: `imgsz=320`, `batch=16`, `epochs=100`, `patience=20`, AdamW, `lr0=0.001`, `weight_decay=0.0005`, `seed=42`, `deterministic=True`, AMP. В E3/E4 дополнительно включены только `fliplr=0.5`, `flipud=0.5`, `degrees=5`, `translate=0.05`, `scale=0.10`, `hsv_v=0.10`; mosaic, mixup и cutmix отключены. Конфигурации и история запусков: `runs/E1`--`runs/E4`.

## Финальные результаты на Test

Test применялся только для финальной оценки. Стандартные `mAP` рассчитаны Ultralytics при `conf=0.001`, `NMS IoU=0.70`; fixed Precision/Recall/F1 — отдельно, при `confidence=0.25`, `matching IoU=0.50` и class-aware one-to-one matching.

| Эксперимент | mAP@50 | mAP@50:95 | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| E1 | 0.59679 | 0.27960 | 0.57263 | 0.71678 | 0.63665 |
| E2 | 0.53613 | 0.27254 | 0.64320 | 0.62821 | 0.63561 |
| E3 | 0.72321 | **0.37285** | 0.60315 | **0.75991** | **0.67251** |
| E4 | 0.71143 | 0.36520 | 0.57984 | 0.75758 | 0.65690 |

E3 имеет наилучшие `mAP@50:95` и fixed F1 в этом единственном запуске. См. [test_results.json](reports/metrics/test_results.json) и [model_evaluation_report.md](reports/model_evaluation_report.md).

## Скорость инференса

Измерения: NVIDIA GeForce RTX 4070 Laptop GPU (8 GiB), FP32, `imgsz=320`, batch=1, одно Test-изображение (`crazing_1.jpg`), 30 прогревов и 100 измерений. FPS = `1000 / mean latency`; это сравнительное измерение, а не production throughput.

| Эксперимент | Параметры | Forward, ms/image | End-to-end, ms/image | End-to-end FPS |
|---|---:|---:|---:|---:|
| E1 | 3,006,818 | 5.374 | 7.974 | 125.41 |
| E2 | 2,583,322 | 7.356 | 9.685 | 103.26 |
| E3 | 3,006,818 | 5.465 | 7.979 | 125.32 |
| E4 | 2,583,322 | 7.533 | 9.931 | 100.70 |

Полные условия и значения: [benchmark_results.json](reports/metrics/benchmark_results.json).

## Окружение

Проверенное окружение: Windows 11, Python 3.12.10, PyTorch 2.7.1+cu128, Torchvision 0.22.1+cu128, CUDA 12.8, Ultralytics 8.3.239 и NVIDIA GeForce RTX 4070 Laptop GPU (8 GiB). Полный снимок: [environment_venv.json](reports/metrics/environment_venv.json).

## Структура

```text
configs/                 # data.yaml и протокол E1--E4
data/raw/                # неизменённые исходные Pascal VOC данные
data/processed/          # подготовленные изображения и YOLO labels
src/                     # воспроизводимые этапы пайплайна
tests/                   # unit- и интеграционные проверки
runs/E1...E4/            # checkpoints, effective configs, history
reports/                 # аудит, EDA, метрики, таблицы и рисунки
reports/article_materials.zip
```

## Установка

Из корня проекта в PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.check_environment --output reports/metrics/environment_venv.json
.\.venv\Scripts\python.exe -m pytest -q
```

Для иной CUDA-конфигурации выберите совместимые wheel-файлы в [официальной инструкции PyTorch](https://pytorch.org/get-started/locally/) и вновь выполните `src.check_environment`.

## Команды воспроизведения

Первые три команды перезаписывают только производные данные/отчёты и не изменяют `data/raw/`.

```powershell
# Аудит исходных Pascal VOC данных
.\.venv\Scripts\python.exe -m src.audit_dataset --root data/raw --json reports/metrics/dataset_audit.json --csv reports/tables/dataset_audit.csv --markdown reports/dataset_audit.md

# Детерминированная подготовка фиксированного split и YOLO labels
.\.venv\Scripts\python.exe -m src.prepare_dataset

# EDA
.\.venv\Scripts\python.exe -m src.eda

# Проверка обучения без вычислений
.\.venv\Scripts\python.exe -m src.train --experiment E1 --dry-run

# Дорогостоящее обучение; создаёт/обновляет артефакты указанного запуска
.\.venv\Scripts\python.exe -m src.train --experiment E1

# Безопасное возобновление только того же эксперимента
.\.venv\Scripts\python.exe -m src.train --experiment E1 --resume

# Fixed P/R/F1 из сохранённых Test-предсказаний, без инференса
.\.venv\Scripts\python.exe -m src.evaluate --recompute-fixed-from-saved

# Анализ ошибок и измерение GPU latency
.\.venv\Scripts\python.exe -m src.analyze_errors
.\.venv\Scripts\python.exe -m src.benchmark --warmup 30 --runs 100

# Материалы статьи и тесты
.\.venv\Scripts\python.exe -m src.prepare_article_materials
.\.venv\Scripts\python.exe -m pytest -q
```

`python -m src.evaluate` намеренно не включён в обычный маршрут: он повторно запускает финальный Test-инференс для E1--E4. Используйте его лишь как заранее зарегистрированное воспроизведение финальной оценки с отдельной фиксацией причины и timestamp.

## Итоговые материалы

- [Финальный аудит](reports/final_audit_report.md)
- [Подготовка датасета](reports/dataset_preparation.md)
- [EDA](reports/eda_report.md)
- [Test-оценка](reports/model_evaluation_report.md)
- [Анализ ошибок](reports/error_analysis_report.md)
- [Материалы статьи](reports/article_materials.zip)

## Ограничения

- Один `seed=42` и одно фиксированное разбиение; статистическая значимость не оценивалась.
- Исходные изображения NEU-DET имеют разрешение 200x200 px.
- dHash выявил кандидаты визуально похожих текстур между split; это не доказанная утечка без group metadata или ручной проверки.
- Latency измерена многократно на одном Test-изображении и не характеризует все производственные условия.
- Нет production-валидации на независимом внешнем наборе, в других условиях съёмки или на нескольких устройствах.
