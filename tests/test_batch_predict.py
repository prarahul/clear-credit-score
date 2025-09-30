from pathlib import Path
import csv
import tempfile
import joblib
import pandas as pd
import runpy

def test_batch_predict_creates_output(ensure_model, tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    bundle = joblib.load(ensure_model)
    cols = bundle["feature_columns"]

    # Create a minimal input CSV with two rows
    df = pd.DataFrame([{c: None for c in cols} for _ in range(2)])
    df["duration"] = [12, 24] if "duration" in df.columns else None
    df["credit_amount"] = [2500, 4500] if "credit_amount" in df.columns else None
    df["age"] = [35, 28] if "age" in df.columns else None
    inp = tmp_path / "input.csv"
    out = tmp_path / "out.csv"
    df.to_csv(inp, index=False)

    # Run batch module with argv
    import sys
    argv_bak = sys.argv
    sys.argv = ["-m", "src.batch_predict", str(inp), str(out), str(ensure_model)]
    try:
        runpy.run_module("src.batch_predict", run_name="__main__")
    finally:
        sys.argv = argv_bak

    assert out.exists(), "Output CSV was not created"
    out_df = pd.read_csv(out)
    assert {"prob_default", "risk"}.issubset(out_df.columns)