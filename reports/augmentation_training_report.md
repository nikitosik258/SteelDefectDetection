# Augmented training: E3 and E4

All reported metrics come from independent evaluation of `best.pt` on **Validation**. Test was not used.

| Experiment | Model | Epochs run | Best epoch | Precision | Recall | mAP@50 | mAP@50:95 | History duration, s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E3 | yolov8n.pt | 100 | 97 | 0.73210 | 0.72397 | 0.73739 | 0.39697 | 43769.80 |
| E4 | yolo11n.pt | 95 | 75 | 0.73306 | 0.71398 | 0.74808 | 0.39650 | 585.47 |

## Absolute Validation changes versus baseline

Values are absolute percentage-point changes, not relative percentage changes.

| Pair | Δ Precision, pp | Δ Recall, pp | Δ mAP@50, pp | Δ mAP@50:95, pp |
| --- | ---: | ---: | ---: | ---: |
| E1 → E3 | +4.99 | +16.04 | +10.90 | +8.12 |
| E2 → E4 | +8.04 | +22.30 | +17.69 | +10.12 |

## Protocol checks

- Same processed Train/Validation split, image size, batch size, seed, optimizer, learning rate, scheduler, weight decay, AMP, early-stopping policy and epoch budget were used in each architecture pair.
- The only planned difference is the augmented profile: horizontal/vertical flips, ±5° rotation, 0.05 translation, 0.10 scale and HSV value shift 0.10. Mosaic, mixup, cutmix and implicit Albumentations remain disabled.
- E3/E4 started from `yolov8n.pt`/`yolo11n.pt` respectively; no E1/E2 weights were used.
- All checkpoints and histories exist, and no NaN/Inf was found in history or best-model weights.

## Interpretation

- E1 → E3: Validation mAP@50:95 increased by 8.12 percentage points. This one-run comparison does not by itself prove a general causal effect.
- E2 → E4: Validation mAP@50:95 increased by 10.12 percentage points. This one-run comparison does not by itself prove a general causal effect.
- E3: training-history wall-clock duration includes a 39497.67 s inter-epoch gap, so its total duration must not be compared as active GPU training time.

No Test evaluation, retuning, or extra experiment was performed.
