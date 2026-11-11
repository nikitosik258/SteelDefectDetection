# Test error analysis

Fixed protocol: confidence=0.25, class-aware one-to-one matching IoU=0.50, NMS IoU=0.70. Counts use frozen predictions; no detector inference or training was repeated. Classification/overlap/duplicate labels are diagnostic categories for unmatched predictions and do not redefine TP/FP/FN. An overlap FP is an unmatched prediction with nonzero IoU against any GT, irrespective of class; it is not necessarily a pure localization error.

## Primary counts

| experiment | tp | fp | fn | precision | recall | f1 | multi_images | multi_gt_objects | multi_errors |
|---|---|---|---|---|---|---|---|---|---|
| E1 | 615 | 459 | 243 | 0.5726256983240223 | 0.7167832167832168 | 0.6366459627329192 | 272 | 770 | 599 |
| E2 | 539 | 299 | 319 | 0.6431980906921241 | 0.6282051282051282 | 0.6356132075471699 | 272 | 770 | 562 |
| E3 | 652 | 429 | 206 | 0.603145235892692 | 0.7599067599067599 | 0.6725116039195462 | 272 | 770 | 558 |
| E4 | 650 | 471 | 208 | 0.5798394290811775 | 0.7575757575757576 | 0.6568974229408793 | 272 | 770 | 591 |

## Diagnostic categories

| experiment | background_fp | classification_fp | duplicate_fp | fn | overlap_fp | small_object_fn | tp |
|---|---|---|---|---|---|---|---|
| E1 | 63 | 8 | 81 | 213 | 307 | 30 | 615 |
| E2 | 36 | 2 | 56 | 269 | 205 | 50 | 539 |
| E3 | 69 | 0 | 53 | 181 | 307 | 25 | 652 |
| E4 | 77 | 0 | 73 | 189 | 321 | 19 | 650 |

## Cross-model case inventory

| category | images |
|---|---|
| all_models_correct | 41 |
| all_models_error | 192 |
| augmentation_improved_yolo11n | 140 |
| augmentation_improved_yolov8n | 161 |
| augmentation_worsened_yolo11n | 127 |
| augmentation_worsened_yolov8n | 110 |
| yolo11n_better_augmented | 36 |
| yolo11n_better_baseline | 32 |
| yolov8n_better_augmented | 39 |
| yolov8n_better_baseline | 59 |

Case-inventory categories can overlap because each records a distinct comparison condition. A small object is defined as bbox area below 1024 square pixels in original-image coordinates; Test contains 102 such objects. Results were obtained with one seed and one fixed split and are descriptive, not significance tests.
