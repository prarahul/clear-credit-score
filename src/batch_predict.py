from __future__ import annotations
from pathlib import Path
import sys
import joblib
import pandas as pd

def main():
    # Normalize argv to handle both CLI and pytest's ["-m","src.batch_predict", ...]
    tokens = [t for t in sys.argv[1:] if t not in ("-m", "src.batch_predict")]

    if len(tokens) < 2:
        print("Usage: python -m src.batch_predict input.csv output.csv [model_path]")
        sys.exit(1)

    input_csv = Path(tokens[0])
    output_csv = Path(tokens[1])
    model_path = (
        Path(tokens[2])
        if len(tokens) > 2
        else Path(__file__).resolve().parents[1] / "models" / "credit_model.joblib"
    )

    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    thresholds = bundle.get("thresholds", {"low": 0.20, "medium": 0.50})
    feature_columns = bundle["feature_columns"]

    df = pd.read_csv(input_csv)
    X = df.reindex(columns=feature_columns)
    proba_idx = list(pipe.classes_).index("bad")
    probs = pipe.predict_proba(X)[:, proba_idx]

    def categorize(p: float) -> str:
        if p <= thresholds["low"]:
            return "low"
        if p <= thresholds["medium"]:
            return "medium"
        return "high"

    df["prob_default"] = probs
    df["risk"] = [categorize(float(p)) for p in probs]
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions to: {output_csv}")

if __name__ == "__main__":
    main()