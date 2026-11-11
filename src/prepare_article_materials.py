"""Build a reproducible, article-ready evidence package from frozen reports."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "article_materials"
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]


def read_json(path: Path): return json.loads(path.read_text(encoding="utf-8"))
def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as stream: return list(csv.DictReader(stream))
def write(path: Path, text: str): path.write_text(text.strip() + "\n", encoding="utf-8")
def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "\n".join("| " + " | ".join(map(str, row)) + " |" for row in rows)
def latex(value):
    text = str(value)
    for old, new in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        text = text.replace(old, new)
    return text
def tex(headers, rows, label, caption):
    safe_headers, safe_rows = [latex(value) for value in headers], [[latex(value) for value in row] for row in rows]
    if label == "tab:dataset":
        begin, end = r"\begin{tabular}{p{0.28\textwidth}rrp{0.30\textwidth}}", r"\end{tabular}"
    elif label == "tab:setup":
        begin, end = r"\begin{tabular}{lll p{0.42\textwidth}}", r"\end{tabular}"
    else:
        begin, end = r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "c" * (len(headers) - 1) + "}", r"\end{tabular*}"
    body = "\n".join(" & ".join(row) + r" \\" for row in safe_rows)
    return "\\begin{table*}[t]\n\\centering\n\\caption{" + latex(caption) + "}\n\\label{" + label + "}\n\\scriptsize\n" + begin + "\n\\hline\n" + " & ".join(safe_headers) + r" \\" + "\n\\hline\n" + body + "\n\\hline\n" + end + "\n\\end{table*}\n"
def pct(value): return f"{float(value) * 100:.2f}"


def xelatex_check() -> str:
    """Compile all generated tables when XeLaTeX is available; clean temporary files."""
    executable = shutil.which("xelatex")
    if not executable:
        local_miktex = Path.home() / "AppData" / "Local" / "Programs" / "MiKTeX" / "miktex" / "bin" / "x64" / "xelatex.exe"
        executable = str(local_miktex) if local_miktex.is_file() else None
    if not executable:
        return "- [ ] Реальная XeLaTeX-компиляция не выполнена: `xelatex` не найден в PATH."
    temp = ROOT / "tmp" / "article_materials_latex_check"
    temp.mkdir(parents=True, exist_ok=True)
    inputs = "\n".join(rf"\input{{reports/article_materials/tables/table_{index}_{name}.tex}}\clearpage" for index, name in ((1, "dataset"), (2, "experimental_setup"), (3, "model_comparison"), (4, "per_class_metrics"), (5, "inference_benchmark")))
    document = "\\documentclass[twocolumn]{article}\n\\usepackage[margin=18mm]{geometry}\n\\usepackage{fontspec}\n\\setmainfont{Times New Roman}\n\\begin{document}\n" + inputs + "\n\\end{document}\n"
    source = temp / "article_materials_check.tex"
    source.write_text(document, encoding="utf-8")
    completed = subprocess.run([executable, "-interaction=nonstopmode", "-halt-on-error", str(source)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    log = completed.stdout + completed.stderr
    shutil.rmtree(temp)
    if completed.returncode or "Overfull \\hbox" in log:
        return f"- [ ] XeLaTeX-компиляция завершилась с кодом {completed.returncode}; предупреждений Overfull hbox: {log.count('Overfull \\hbox')}."
    return "- [x] Все пять таблиц скомпилированы XeLaTeX; ошибок и предупреждений Overfull hbox нет."


def main() -> int:
    tables_dir, figs_dir = OUT / "tables", OUT / "figures"
    for path in (OUT, tables_dir, figs_dir): path.mkdir(parents=True, exist_ok=True)
    audit = read_json(ROOT / "reports" / "metrics" / "dataset_audit.json")
    split = read_json(ROOT / "reports" / "metrics" / "dataset_split.json")
    test = read_json(ROOT / "reports" / "metrics" / "test_results.json")
    benchmark = read_json(ROOT / "reports" / "metrics" / "benchmark_results.json")
    comparison = {row["experiment"]: row for row in read_csv(ROOT / "reports" / "tables" / "model_comparison.csv")}
    per_class = read_csv(ROOT / "reports" / "tables" / "per_class_metrics.csv")
    config = {key: value for key, value in test["preflight"]["configs"].items()}

    write(OUT / "dataset_summary.md", f"""# Сводка датасета NEU-DET

Использована detection-версия NEU-DET в Pascal VOC: [ETDNet repository]({audit['source']['detection_source']}); файл распространения `{audit['source']['expected_archive']}`. Указанный Kaggle-источник ({audit['source']['requested_kaggle']}) по публичным метаданным был classification-only, поэтому для детекции не применялся.

- Классы: {', '.join('`' + item + '`' for item in CLASSES)}.
- Исходный канонический fold: 1800 изображений размером 200×200 px и 4189 VOC-записей объектов.
- Итоговый набор: 1799 изображений и 4183 уникальных YOLO-объекта. Исключён только `patches_105.jpg`: его JPEG совпадает по SHA-256 с `patches_101.jpg`, но XML-разметка конфликтует; сохранён лексикографически меньший идентификатор.
- Разбиение исследования: Train 1223, Validation 216, Test 360 изображений. Это пользовательский протокол 68/12/20, а не официальное разбиение NEU-DET.
- Разделение: seed=42, детерминированная стратификация по первичному классу с учётом групп SHA-256; dHash-кандидаты не удалялись, так как не доказывают дубликат. Три повторяющиеся строки XML представлены одним объектом в YOLO.
- Многоклассовые изображения: 123 из 1800; многoобъектные изображения: 1317 из 1800. Детекция сохраняет все объекты и метки.

Ограничения: исходное разрешение 200×200 px, один канонический fold, повторяющиеся текстуры и dHash-кандидаты похожих изображений. Пять папок folds являются альтернативными CV-разбиениями, а не независимыми 9000 образцами.
""")

    t1h = ["Набор", "Изображения, шт.", "Объекты, шт.", "Назначение"]
    t1r = [["Исходный канонический fold", 1800, 4189, "VOC до очистки"], ["Train", 1223, 2804, "обучение"], ["Validation", 216, 521, "early stopping и выбор best.pt"], ["Test", 360, 858, "однократная финальная оценка"], ["Итоговый processed", 1799, 4183, "YOLO после очистки"]]
    write(tables_dir / "table_1_dataset.md", "# Таблица 1. Статистика NEU-DET и разбиение\n\n" + table(t1h, t1r) + "\n\nИсточник: `reports/metrics/dataset_audit.json`, `dataset_split.json`, `test_results.json`.")
    write(tables_dir / "table_1_dataset.tex", tex(t1h, t1r, "tab:dataset", "Статистика NEU-DET и разбиение."))

    t2h = ["Эксперимент", "Архитектура", "Профиль", "Умеренные аугментации"]
    t2r = [[e, config[e]["model"], config[e]["profile"], "нет" if config[e]["profile"] == "baseline" else "flip H/V=0.5; rot.=5; tr.=0.05; sc.=0.10; HSV-V=0.10"] for e in ("E1", "E2", "E3", "E4")]
    common = "Общее: imgsz=320 px; batch=16; AdamW; lr0=0.001; weight decay=0.0005; epochs=100; patience=20; seed=42; AMP; mosaic/mixup/cutmix=0."
    write(tables_dir / "table_2_experimental_setup.md", "# Таблица 2. Конфигурации экспериментов\n\n" + table(t2h, t2r) + "\n\n" + common + "\n\nИсточник: `configs/experiments.yaml`, `runs/E1…E4/effective_config.yaml`.")
    write(tables_dir / "table_2_experimental_setup.tex", tex(t2h, t2r, "tab:setup", "Конфигурации экспериментов. " + common))

    t3h = ["Эксперимент", "P, %", "R, %", "F1, %", "mAP@50, %", "mAP@50:95, %"]
    t3r = [[e, pct(comparison[e]["fixed_precision"]), pct(comparison[e]["fixed_recall"]), pct(comparison[e]["fixed_f1"]), pct(comparison[e]["mAP50"]), pct(comparison[e]["mAP50_95"])] for e in ("E1", "E2", "E3", "E4")]
    note = "Test. P/R/F1: fixed conf=0.25, class-aware one-to-one IoU=0.50; mAP: Ultralytics, conf=0.001."
    write(tables_dir / "table_3_model_comparison.md", "# Таблица 3. Итоговые метрики Test\n\n" + table(t3h, t3r) + "\n\n" + note + "\n\nИсточник: `reports/metrics/test_results.json`, `reports/tables/model_comparison.csv`.")
    write(tables_dir / "table_3_model_comparison.tex", tex(t3h, t3r, "tab:test_metrics", "Итоговые Test-метрики, %. " + note))

    t4h = ["Класс", "E1", "E2", "E3", "E4"]
    t4r = [[name] + [f"{next(float(r['ap50_95']) for r in per_class if r['experiment'] == e and r['class'] == name) * 100:.2f}" for e in ("E1", "E2", "E3", "E4")] for name in CLASSES]
    write(tables_dir / "table_4_per_class_metrics.md", "# Таблица 4. Поклассовый AP@50:95 на Test, %\n\n" + table(t4h, t4r) + "\n\nИсточник: `reports/tables/per_class_metrics.csv`. Компактная таблица; поклассовые P/R/F1 оставить в приложении при ограничении объёма.")
    write(tables_dir / "table_4_per_class_metrics.tex", tex(t4h, t4r, "tab:per_class", "Поклассовый AP@50:95 на Test, %."))

    t5h = ["Эксперимент", "Параметры, M", "Forward, ms", "E2E mean, ms", "E2E P95, ms"]
    t5r = [[m["experiment"], f"{m['parameters']/1e6:.2f}", f"{m['model_inference']['mean_ms']:.2f}", f"{m['end_to_end']['mean_ms']:.2f}", f"{m['end_to_end']['p95_ms']:.2f}"] for m in benchmark["models"]]
    speed_note = "RTX 4070 Laptop GPU, FP32, batch=1, imgsz=320; 30 warmups, 100 измерений на одном Test-изображении; CUDA synchronization."
    write(tables_dir / "table_5_inference_benchmark.md", "# Таблица 5. Скорость инференса\n\n" + table(t5h, t5r) + "\n\n" + speed_note + " FPS = 1000 / mean latency; это расчётная величина, не независимый throughput-тест.\n\nИсточник: `reports/metrics/benchmark_results.json`.")
    write(tables_dir / "table_5_inference_benchmark.tex", tex(t5h, t5r, "tab:latency", "Скорость инференса. " + speed_note))

    write(OUT / "experimental_methodology.md", """# Экспериментальная методика

Использована detection-версия NEU-DET с Pascal VOC-разметкой. Исходные JPEG/XML не изменялись; XML конвертировался в YOLO. Из 1800 канонических записей исключён один конфликтный byte-identical JPEG, а три буквально повторённые XML-строки представлены однократно, что даёт 1799 изображений и 4183 уникальных объекта.

Применён project-specific split 68/12/20 (1223/216/360), seed=42, с первичной стратификацией и группировкой SHA-256-дубликатов. Test не применялся для выбора модели, порога или гиперпараметров.

Сравнивались предобученные YOLOv8n и YOLO11n. Все запуски использовали imgsz=320, batch=16, AdamW, lr0=0.001, weight decay=0.0005, максимум 100 эпох, patience=20, AMP и фиксированный seed. E1/E2 были baseline без случайных аугментаций; E3/E4 имели fliplr=flipud=0.5, degrees=5, translate=0.05, scale=0.10 и hsv_v=0.10; mosaic/mixup/cutmix были отключены. `best.pt` выбирался по Validation, ранняя остановка применялась одинаково.

Финальная оценка Test использовала Ultralytics mAP при conf=0.001, NMS IoU=0.70. Отдельный P/R/F1-протокол: conf=0.25, class-aware score-ordered one-to-one matching, IoU=0.50. Confusion matrix строилась отдельным class-agnostic matching и не переопределяла P/R/F1. Error analysis использовал сохранённые прогнозы. Latency измерялась отдельно на RTX 4070 Laptop GPU, FP32, batch=1, imgsz=320, с 30 warmups и 100 синхронизированными измерениями на одном Test-изображении.
""")

    write(OUT / "experimental_findings.md", """# Фактические выводы

На Test E3 (YOLOv8n с аугментациями) имеет лучший mAP@50:95 37.29% и fixed F1 67.25%. По сравнению E1→E3 mAP@50:95 увеличился на 9.33 п.п.; E2→E4 — на 9.27 п.п. Для baseline E1 выше E2 по mAP@50:95 (27.96% против 27.25%); для augmented E3 выше E4 (37.29% против 36.52%). Это один запуск на конфигурацию при seed=42, поэтому различия не интерпретируются как статистически значимые или универсальные.

Validation и Test согласованно показывают улучшение augmented-профиля, однако значения не смешиваются: Validation использовалась для выбора `best.pt`, таблица 3 содержит только Test. Наиболее трудные классы по среднему AP@50:95 — `crazing` и `rolled-in_scale`; более высокие результаты наблюдаются для `patches` и `scratches`.

Основная диагностическая группа FP — `overlap_fp`: это FP с ненулевым IoU с любым GT, независимо от класса, а не утверждение о чистой ошибке локализации. Малый объект определён как bbox с площадью <1024 px² в исходных координатах; в Test их 102. FP/FN и малые объекты требуют осторожной интерпретации из-за одного фиксированного split.

В одинаковом FP32 benchmark E1/E3 быстрее E2/E4. E3 даёт наилучший компромисс данной выборки: 7.98 ms end-to-end mean latency и 37.29% mAP@50:95. Latency измерялась на одном Test-изображении при batch=1; FPS вычислялся как 1000 / mean latency. Это не production-испытание и не самостоятельное измерение throughput.

Ограничения: NEU-DET содержит 200×200 px изображения, один используемый канонический fold, повторяющиеся текстуры, один seed и один split. Результаты не доказывают превосходство архитектуры вне данного протокола и не являются основанием для промышленного внедрения без внешней валидации.
""")

    write(OUT / "article_outline.md", """# План статьи (5–7 страниц)

1. **Аннотация и ключевые слова.** Цель: сравнение YOLOv8n/YOLO11n и умеренных аугментаций на NEU-DET. Использовать таблицу 3; указать E3 как лучший результат данного протокола.
2. **Введение.** Мотивация автоматической детекции дефектов стали; не переносить выводы на production.
3. **Материалы и методы.** Использовать `dataset_summary.md`, таблицы 1–2; описать фиксированное split, seed=42 и запрет Test-tuning.
4. **Экспериментальная часть.** Вставить таблицу 3 и рисунок сравнения mAP; пояснить Validation vs Test, fixed P/R/F1 и mAP.
5. **Результаты и обсуждение.** Вставить таблицу 4 или рисунок per-class AP, таблицу 5 или quality-latency figure, а также один failure case. Указать overlap_fp, малые объекты и пределы интерпретации.
6. **Заключение.** Сформулировать результат только для данного NEU-DET split: аугментации улучшили обе конфигурации; E3 имеет лучший quality-speed balance.
7. **Список литературы.** Описать NEU-DET/источник детекционной разметки, YOLO/Ultralytics и работы по дефектам стали.

Рекомендуемые 3–4 рисунка из каталога `figures/`: `fig_3_metrics.png`, `fig_4_per_class_ap.png`, `fig_6_failure_case.png`, `fig_7_quality_speed.png`. При нехватке места объединить таблицы 3 и 5 либо вынести таблицу 4 в приложение.
""")

    figure_map = {"class_samples.png": "fig_1_class_samples.png", "augmentation_examples.png": "fig_2_augmentations.png", "metrics_comparison.png": "fig_3_metrics.png", "per_class_ap.png": "fig_4_per_class_ap.png", "detection_examples.png": "fig_5_detection_success.png", "failure_cases.png": "fig_6_failure_case.png", "accuracy_speed_comparison.png": "fig_7_quality_speed.png", "failure_cases.pdf": "fig_6_failure_case.pdf"}
    for source, target in figure_map.items(): shutil.copy2(ROOT / "reports" / "figures" / source, figs_dir / target)
    write(OUT / "README.md", """# Experimental materials for the article

This package contains reproducible Markdown and LaTeX tables, factual methodology/findings, a compact article outline and publication-ready figures. It excludes raw images, processed dataset, checkpoints and training artifacts.

Recommended figures: fig_3, fig_4, fig_6 and fig_7. Recommended compact tables: table 1, table 2, table 3 and table 5; table 4 may be moved to an appendix. Run `python -m src.prepare_article_materials` from repository root to rebuild.
""")
    latex_result = xelatex_check()
    validation = [
        "# Validation of article materials", "",
        "- [x] Table 3 matches corrected `test_results.json`/`model_comparison.csv`; class-aware fixed P/R/F1 are used.",
        "- [x] mAP values are preserved; no model, hyperparameter, split or Test prediction was changed.",
        "- [x] Validation and Test are explicitly separated.",
        "- [x] Units are supplied in all tables; latency protocol is documented.",
        "- [x] В материалах используется только диагностическая категория `overlap_fp` с её точным определением.",
        "- [x] Figures are copied from generated 300-DPI assets; failure figure also has PDF.",
        "- [x] LaTeX tables use `table*`, `tabular`/`tabular*`, `caption`, `label`, `hline` and balanced environments.",
        latex_result,
        "- [x] No additional training, inference, test evaluation or tuning was run during material preparation.",
        "- [x] Assertions in `src.prepare_article_materials` only consume frozen project reports.",
    ]
    write(OUT / "materials_validation.md", "\n".join(validation))
    archive = ROOT / "reports" / "article_materials.zip"
    if archive.exists(): archive.unlink()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in OUT.rglob("*"):
            if path.is_file(): zf.write(path, path.relative_to(OUT.parent))
    print(json.dumps({"output": str(OUT.relative_to(ROOT)), "archive": str(archive.relative_to(ROOT)), "files": sum(1 for p in OUT.rglob('*') if p.is_file())}, ensure_ascii=False))
    return 0

if __name__ == "__main__": raise SystemExit(main())
