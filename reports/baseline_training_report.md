# Baseline training: E1 and E2

All values below are obtained on **Validation**, never on Test. Test was not loaded by training or this report.

| Experiment | Model | Epochs run | Best epoch | Precision | Recall | mAP@50 | mAP@50:95 | History duration, s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 | yolov8n.pt | 36 | 16 | 0.68223 | 0.56356 | 0.62844 | 0.31582 | 189.98 |
| E2 | yolo11n.pt | 63 | 43 | 0.65261 | 0.49095 | 0.57122 | 0.29530 | 398.08 |

## Comparability

The checked shared parameters are identical: data, epochs, patience, batch, imgsz, optimizer, lr0, lrf, weight_decay, cos_lr, warmup_epochs, seed, deterministic, pretrained, multi_scale, close_mosaic, hsv_h, hsv_s, hsv_v, degrees, translate, scale, shear, perspective, fliplr, flipud, mosaic, mixup, cutmix, copy_paste, erasing, bgr. The only intended difference is the pretrained architecture (`yolov8n.pt` versus `yolo11n.pt`). Both use the same processed split (Train/Validation/Test = 1223/216/360), but only Train and Validation were accessed.

## Integrity checks

- `best.pt`, `last.pt`, `results.csv`, `args.yaml`, `effective_config.yaml`, and completed run status exist for both experiments.
- Histories contain no NaN or Inf in reported losses or validation metrics.
- Both `best.pt` checkpoints were loaded and independently evaluated on Validation.

## Observations

- E1: final Validation mAP@50:95 is below its best-history value while train box loss is lower; this is a possible late-training overfitting indicator, not a basis for retuning.
- E2: final Validation mAP@50:95 is below its best-history value while train box loss is lower; this is a possible late-training overfitting indicator, not a basis for retuning.

No architecture-specific retuning or additional training was performed after these observations.
