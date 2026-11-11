# Таблица 5. Скорость инференса

| Эксперимент | Параметры, M | Forward, ms | E2E mean, ms | E2E P95, ms |
|---|---|---|---|---|
| E1 | 3.01 | 5.37 | 7.97 | 9.65 |
| E2 | 2.58 | 7.36 | 9.68 | 12.06 |
| E3 | 3.01 | 5.47 | 7.98 | 9.58 |
| E4 | 2.58 | 7.53 | 9.93 | 12.44 |

RTX 4070 Laptop GPU, FP32, batch=1, imgsz=320; 30 warmups, 100 измерений на одном Test-изображении; CUDA synchronization. FPS = 1000 / mean latency; это расчётная величина, не независимый throughput-тест.

Источник: `reports/metrics/benchmark_results.json`.
