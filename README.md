# Credit Risk Scoring (Data Science)

This project trains and evaluates a credit default risk classifier using the German Credit dataset (OpenML id=31).
Outputs:
- models/credit_model.joblib — trained sklearn pipeline
- reports/metrics.json — holdout metrics (ROC AUC, Accuracy, F1)
- reports/roc_curve.png, reports/confusion_matrix.png

Quickstart (Windows PowerShell):
1) python -m venv .venv
2) .\.venv\Scripts\Activate.ps1
3) python -m pip install --upgrade pip
4) pip install -r requirements.txt
5) python -m src.train
6) Optional: python -m src.predict sample.json

Risk categories (by probability of default):
- low: p(default) <= 0.20
- medium: 0.20 < p(default) <= 0.50
- high: p(default) > 0.50