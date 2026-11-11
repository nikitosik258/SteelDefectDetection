# GPU inference benchmark

All checkpoints were run sequentially on the same GPU, FP32, batch=1, imgsz=320, with 30 warmups and 100 synchronized measurements. Latency was measured repeatedly on one fixed Test image (`crazing_1.jpg`). Model-inference latency measures the model forward pass using a preprocessed resident CUDA tensor. End-to-end latency measures Ultralytics sequential prediction, including image loading, preprocessing and postprocessing; checkpoint loading is excluded. Calculated FPS is 1000 / mean latency and is not an independently measured throughput. Accuracy and error-analysis results are from one seed and one fixed split.

experiment,parameters,inference_mean_ms,inference_median_ms,inference_p95_ms,inference_std_ms,inference_calculated_fps,end_to_end_mean_ms,end_to_end_median_ms,end_to_end_p95_ms,end_to_end_std_ms,end_to_end_calculated_fps
E1,3006818,5.373990000225604,5.1841999957105145,6.784235016675665,0.7678766453190855,186.08147762798578,7.973575999203604,7.706399992457591,9.651059993484523,0.9503765130574323,125.41424325796599
E2,2583322,7.355952000070829,7.105199998477474,9.758270018210169,1.1597705187595726,135.9443346001131,9.684636000019964,9.357450006064028,12.059869986842386,1.13476324526794,103.2563330204603
E3,3006818,5.465474000084214,5.220200007897802,7.15471499133855,0.8224557977467666,182.96674725460073,7.979420000046957,7.718699998804368,9.580814998480491,0.999368446232538,125.32239185230445
E4,2583322,7.53324199875351,7.1338499983539805,10.151009990659077,1.4839715487086262,132.74497223976945,9.930788001511246,9.563899991917424,12.440664993482642,1.3983035492870934,100.69694367132016

Environment:

```json
{
  "gpu": "NVIDIA GeForce RTX 4070 Laptop GPU",
  "total_vram_gib": 8.0,
  "free_vram_gib_before_report": 7.92,
  "torch": "2.7.1+cu128",
  "torch_cuda": "12.8",
  "ultralytics": "8.3.239",
  "precision": "FP32",
  "imgsz": 320,
  "batch": 1,
  "warmup_runs": 30,
  "measured_runs": 100,
  "input_image": "crazing_1.jpg",
  "nms": {
    "confidence": 0.25,
    "iou": 0.7,
    "max_det": 300
  },
  "execution": "sequential, one model at a time; CUDA synchronized around every measurement"
}
```
