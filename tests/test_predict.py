from pathlib import Path
import json
from src.predict import predict_from_dict

def test_single_prediction_in_range(ensure_model):
    sample = {
        "duration": 12,
        "credit_amount": 2500,
        "age": 35,
        "housing": "own",
        "job": "skilled",
        "checking_status": "<0",
        "purpose": "car (new)"
    }
    result = predict_from_dict(sample, model_path=Path(ensure_model))
    assert "prob_default" in result and "risk" in result
    assert 0.0 <= result["prob_default"] <= 1.0
    assert result["risk"] in {"low", "medium", "high"}