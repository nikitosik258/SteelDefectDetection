# Final Test evaluation of E1--E4

## Method

Test was used only for this final evaluation. Standard Ultralytics mAP uses `split=test`, `imgsz=320`, `conf=0.001`, NMS IoU=0.7 and `max_det=300` for every checkpoint. NMS IoU controls suppression of predictions; it is distinct from the IoU thresholds used to define true positives in AP calculation.

The separate fixed P/R/F1 protocol uses confidence=0.25, NMS IoU=0.7 and one-to-one matching IoU=0.50. Predictions are score-ordered; an unmatched ground-truth item can be associated with at most one prediction. A class mismatch is FP for the predicted class and FN for the true class. Ultralytics reference Precision/Recall are retained in CSV/JSON but are not substituted for fixed-threshold P/R/F1.

## Overall results

| experiment | model | mAP50 | mAP50_95 | fixed_precision | fixed_recall | fixed_f1 | tp | fp | fn | mean_inference_ms_per_image |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | yolov8n.pt | 0.59679 | 0.27960 | 0.57263 | 0.71678 | 0.63665 | 615 | 459 | 243 | 0.65773 |
| E2 | yolo11n.pt | 0.53613 | 0.27254 | 0.64200 | 0.62704 | 0.63443 | 538 | 300 | 320 | 0.61033 |
| E3 | yolov8n.pt | 0.72321 | 0.37285 | 0.60315 | 0.75991 | 0.67251 | 652 | 429 | 206 | 0.59018 |
| E4 | yolo11n.pt | 0.71143 | 0.36520 | 0.57984 | 0.75758 | 0.65690 | 650 | 471 | 208 | 0.60642 |

Best by Test mAP@50:95: **E3**. Best by fixed-threshold Recall: **E3**.

## Comparison (absolute percentage points)

- E1 to E2 (architecture without augmentation): delta mAP@50:95=-0.71 pp; delta fixed F1=-0.22 pp.
- E3 to E4 (architecture with augmentation): delta mAP@50:95=-0.76 pp; delta fixed F1=-1.56 pp.
- E1 to E3 (augmentation effect for YOLOv8n): delta mAP@50:95=+9.33 pp; delta fixed F1=+3.59 pp.
- E2 to E4 (augmentation effect for YOLO11n): delta mAP@50:95=+9.27 pp; delta fixed F1=+2.25 pp.

## Per-class observations

Lowest mean AP@50:95 across the four fixed experiments: `crazing` (0.095), `rolled-in_scale` (0.178).
Per-class AP and fixed-threshold metrics are in `reports/tables/per_class_metrics.csv`; fixed-protocol confusion matrices are stored separately for each experiment.

## Validity limits

Results are one run per configuration at seed=42; observed differences are not claims of statistical significance. They describe this NEU-DET split and do not guarantee production performance. No configuration, threshold, epoch budget, augmentation strategy or model was changed after inspecting Test.
