from pathlib import Path
import joblib, json

def main():
    root = Path(__file__).resolve().parents[1]
    bundle = joblib.load(root / "models" / "credit_model.joblib")
    cols = bundle["feature_columns"]
    out = root / "data" / "template_input.csv"
    out.parent.mkdir(exist_ok=True)
    out.write_text(",".join(cols) + "\n")
    print("Wrote header to:", out)
    print(json.dumps({"feature_columns": cols}, indent=2))

if __name__ == "__main__":
    main()