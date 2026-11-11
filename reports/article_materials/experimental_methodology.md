# Экспериментальная методика

Использована detection-версия NEU-DET с Pascal VOC-разметкой. Исходные JPEG/XML не изменялись; XML конвертировался в YOLO. Из 1800 канонических записей исключён один конфликтный byte-identical JPEG, а три буквально повторённые XML-строки представлены однократно, что даёт 1799 изображений и 4183 уникальных объекта.

Применён project-specific split 68/12/20 (1223/216/360), seed=42, с первичной стратификацией и группировкой SHA-256-дубликатов. Test не применялся для выбора модели, порога или гиперпараметров.

Сравнивались предобученные YOLOv8n и YOLO11n. Все запуски использовали imgsz=320, batch=16, AdamW, lr0=0.001, weight decay=0.0005, максимум 100 эпох, patience=20, AMP и фиксированный seed. E1/E2 были baseline без случайных аугментаций; E3/E4 имели fliplr=flipud=0.5, degrees=5, translate=0.05, scale=0.10 и hsv_v=0.10; mosaic/mixup/cutmix были отключены. `best.pt` выбирался по Validation, ранняя остановка применялась одинаково.

Финальная оценка Test использовала Ultralytics mAP при conf=0.001, NMS IoU=0.70. Отдельный P/R/F1-протокол: conf=0.25, class-aware score-ordered one-to-one matching, IoU=0.50. Confusion matrix строилась отдельным class-agnostic matching и не переопределяла P/R/F1. Error analysis использовал сохранённые прогнозы. Latency измерялась отдельно на RTX 4070 Laptop GPU, FP32, batch=1, imgsz=320, с 30 warmups и 100 синхронизированными измерениями на одном Test-изображении.
