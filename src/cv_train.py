from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from .train import load_data, build_pipeline, RANDOM_STATE, THRESHOLDS

def main():
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    reports_dir = root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pipe = build_pipeline(num_cols, cat_cols)

    param_grid = {
        "clf__C": [0.1, 0.3, 1.0, 3.0, 10.0],
        "clf__penalty": ["l2"],  # liblinear supports l2
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
        verbose=1,
    )
    print("Running GridSearchCV...")
    gs.fit(X, y)

    # Save CV results
    cv_results = pd.DataFrame(gs.cv_results_)
    cv_results.to_csv(reports_dir / "cv_results.csv", index=False)
    (reports_dir / "best_params.json").write_text(json.dumps(gs.best_params_, indent=2))
    print("Saved:", reports_dir / "cv_results.csv")
    print("Saved:", reports_dir / "best_params.json")
    print("Best AUC:", gs.best_score_)

    # Holdout evaluation with best model
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    best = gs.best_estimator_
    best.fit(X_tr, y_tr)
    from sklearn.metrics import RocCurveDisplay, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt
    proba_bad = best.predict_proba(X_te)[:, list(best.classes_).index("bad")]
    y_pred = np.where(proba_bad >= 0.5, "bad", "good")
    metrics = {
        "roc_auc": float(roc_auc_score((y_te == "bad").astype(int), proba_bad)),
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "f1": float(f1_score(y_te, y_pred, pos_label="bad")),
        "thresholds": THRESHOLDS,
        "best_params": gs.best_params_,
    }
    (reports_dir / "metrics_cv.json").write_text(json.dumps(metrics, indent=2))
    print("Saved:", reports_dir / "metrics_cv.json")

    # Plots
    fig, ax = plt.subplots(figsize=(6,5), dpi=120)
    RocCurveDisplay.from_predictions((y_te == "bad").astype(int), proba_bad, ax=ax)
    ax.set_title("ROC curve (best model)")
    fig.tight_layout()
    fig.savefig(reports_dir / "roc_curve_cv.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5,4), dpi=120)
    ConfusionMatrixDisplay.from_predictions(y_te, y_pred, display_labels=["good", "bad"], ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix (best model, thr=0.50)")
    fig.tight_layout()
    fig.savefig(reports_dir / "confusion_matrix_cv.png")
    plt.close(fig)
    print("Saved:", reports_dir / "roc_curve_cv.png")
    print("Saved:", reports_dir / "confusion_matrix_cv.png")

    # Save best model
    bundle = {
        "pipeline": best,
        "classes_": best.classes_,
        "thresholds": THRESHOLDS,
        "feature_columns": X.columns.tolist(),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "best_params": gs.best_params_,
    }
    joblib.dump(bundle, models_dir / "credit_model_best.joblib")
    print("Saved model:", models_dir / "credit_model_best.joblib")

if __name__ == "__main__":
    main()