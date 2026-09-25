# Module 2 — Analytics Pipeline

Run:

```bash
python analytics/run_analytics.py
```

The script loads Titanic once through `sns.load_dataset("titanic")`, immediately saves `analytics/titanic.csv`, and uses that saved/raw DataFrame for the complete workflow. It creates EDA plots, required bivariate/correlation analyses, four multivariate charts, standardization checks, three classifiers, confusion matrices, ROC/AUC, imbalance comparison, Random Forest GridSearchCV/OOB evaluation, multivariate linear regression, residual analysis, CSV metric tables, and a reloadable complete `joblib` pipeline.

The classification recommendation is generated from the fixed stratified split and written to `ANALYTICS_REPORT.md`.
