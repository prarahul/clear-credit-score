from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    RocCurveDisplay,
    ConfusionMatrixDisplay,
    classification_report,
)

RANDOM_STATE = 42
THRESHOLDS = {"low": 0.20, "medium": 0.50}  # >0.50 is "high"

def load_data():
    # German Credit dataset (binary: good/bad)
    data = fetch_openml(name="credit-g", version=1, as_frame=True)
    df = data.frame
    y = df["class"].astype(str).str.lower()  # good/bad
    X = df.drop(columns=["class"])
    return X, y

def build_pipeline(num_cols, cat_cols) -> Pipeline:
    num_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE)
    return Pipeline(steps=[("pre", pre), ("clf", clf)])

def evaluate_and_report(y_true, y_pred, y_proba, reports_dir: Path):
    import json
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        RocCurveDisplay,
        ConfusionMatrixDisplay,
        roc_auc_score,
        accuracy_score,
        f1_score,
    )

    metrics = {
        "roc_auc": float(roc_auc_score((y_true == "bad").astype(int), y_proba)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, pos_label="bad")),
        "thresholds": THRESHOLDS,
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ROC curve
    fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
    RocCurveDisplay.from_predictions((y_true == "bad").astype(int), y_proba, ax=ax)
    ax.set_title("ROC curve")
    fig.tight_layout()
    fig.savefig(reports_dir / "roc_curve.png")
    plt.close(fig)

    # Confusion matrix (uses y_pred you passed in at 0.50 threshold)
    fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=["good", "bad"], ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix (threshold=0.50)")
    fig.tight_layout()
    fig.savefig(reports_dir / "confusion_matrix.png")
    plt.close(fig)

    print("Saved:", reports_dir / "metrics.json")
    print("Saved:", reports_dir / "roc_curve.png")
    print("Saved:", reports_dir / "confusion_matrix.png")

def main():
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    reports_dir = root / "reports"
    models_dir.mkdir(exist_ok=True, parents=True)

    print("Loading data...")
    X, y = load_data()
    print(f"Data shape: {X.shape}, target distribution: {y.value_counts(normalize=True).to_dict()}")

    # Column splits
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    print("Building pipeline...")
    pipe = build_pipeline(num_cols, cat_cols)

    print("Train/test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    print("Training...")
    pipe.fit(X_train, y_train)

    print("Evaluating...")
    proba_bad = pipe.predict_proba(X_test)[:, list(pipe.classes_).index("bad")]
    y_pred = np.where(proba_bad >= 0.5, "bad", "good")
    evaluate_and_report(y_test, y_pred, proba_bad, reports_dir)

    # Extra reports
    cls_rep = classification_report(y_test, y_pred, output_dict=True)
    (reports_dir / "classification_report.json").write_text(json.dumps(cls_rep, indent=2))

    # Feature weights for Logistic Regression
    pre = pipe.named_steps["pre"]
    clf = pipe.named_steps["clf"]
    num_features = num_cols
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    cat_features = ohe.get_feature_names_out(cat_cols).tolist()
    feature_names = num_features + cat_features
    coefs = clf.coef_[0]
    dfw = pd.DataFrame({"feature": feature_names, "coef": coefs, "abs_coef": np.abs(coefs)}).sort_values("abs_coef", ascending=False)
    dfw.to_csv(reports_dir / "feature_weights.csv", index=False)

    print("Saved:", reports_dir / "classification_report.json")
    print("Saved:", reports_dir / "feature_weights.csv")

    model_path = models_dir / "credit_model.joblib"
    bundle = {
        "pipeline": pipe,
        "classes_": pipe.classes_,
        "thresholds": THRESHOLDS,
        "feature_columns": X.columns.tolist(),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }
    joblib.dump(bundle, model_path)
    print("Saved model:", model_path)

if __name__ == "__main__":
    main()