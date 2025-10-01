from __future__ import annotations
from pathlib import Path
import sys
import json
import joblib
import pandas as pd

def categorize(prob_default: float, thresholds: dict[str, float]) -> str:
    if prob_default <= thresholds["low"]:
        return "low"
    if prob_default <= thresholds["medium"]:
        return "medium"
    return "high"

def predict_from_dict(sample: dict, model_path: Path | None = None):
    root = Path(__file__).resolve().parents[1]
    if model_path is None:
        best = root / "models" / "credit_model_best.joblib"
        model_path = best if best.exists() else (root / "models" / "credit_model.joblib")
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    thresholds = bundle.get("thresholds", {"low": 0.20, "medium": 0.50})
    feature_columns = bundle["feature_columns"]

    X = pd.DataFrame([sample]).reindex(columns=feature_columns)
    proba_bad = pipe.predict_proba(X)[0, list(pipe.classes_).index("bad")]
    risk = categorize(float(proba_bad), thresholds)
    return {"prob_default": float(proba_bad), "risk": risk}

if __name__ == "__main__":
    # Usage: python -m src.predict sample.json
    if len(sys.argv) < 2:
        print("Usage: python -m src.predict sample.json")
        sys.exit(1)
    sample_path = Path(sys.argv[1])
    sample = json.loads(sample_path.read_text())
    result = predict_from_dict(sample)
    print(json.dumps(result, indent=2))