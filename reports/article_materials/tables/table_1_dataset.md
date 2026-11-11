# Таблица 1. Статистика NEU-DET и разбиение

| Набор | Изображения, шт. | Объекты, шт. | Назначение |
|---|---|---|---|
| Исходный канонический fold | 1800 | 4189 | VOC до очистки |
| Train | 1223 | 2804 | обучение |
| Validation | 216 | 521 | early stopping и выбор best.pt |
| Test | 360 | 858 | однократная финальная оценка |
| Итоговый processed | 1799 | 4183 | YOLO после очистки |

Источник: `reports/metrics/dataset_audit.json`, `dataset_split.json`, `test_results.json`.
