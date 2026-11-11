# Validation of article materials

- [x] Table 3 matches corrected `test_results.json`/`model_comparison.csv`; class-aware fixed P/R/F1 are used.
- [x] mAP values are preserved; no model, hyperparameter, split or Test prediction was changed.
- [x] Validation and Test are explicitly separated.
- [x] Units are supplied in all tables; latency protocol is documented.
- [x] В материалах используется только диагностическая категория `overlap_fp` с её точным определением.
- [x] Figures are copied from generated 300-DPI assets; failure figure also has PDF.
- [x] LaTeX tables use `table*`, `tabular`/`tabular*`, `caption`, `label`, `hline` and balanced environments.
- [x] Все пять таблиц скомпилированы XeLaTeX; ошибок и предупреждений Overfull hbox нет.
- [x] No additional training, inference, test evaluation or tuning was run during material preparation.
- [x] Assertions in `src.prepare_article_materials` only consume frozen project reports.
