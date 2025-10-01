from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .predict import categorize  # reuse same risk bands

class CreditRequest(BaseModel):
    checking_status: str
    duration: int
    credit_history: str  
    purpose: str
    credit_amount: int
    savings_status: str
    employment: str
    installment_commitment: int
    personal_status: str
    other_parties: str
    residence_since: int
    property_magnitude: str
    age: int
    other_payment_plans: str
    housing: str
    existing_credits: int
    job: str
    num_dependents: int
    own_telephone: str
    foreign_worker: str

class CreditResponse(BaseModel):
    prob_default: float
    risk: str

app = FastAPI(title="Credit Risk API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_BUNDLE: dict | None = None
_PIPE = None
_FEATURE_COLUMNS: list[str] = []
_THRESHOLDS: dict[str, float] = {}
_PROBA_IDX: int = 0

def _load_bundle():
    global _BUNDLE, _PIPE, _FEATURE_COLUMNS, _THRESHOLDS, _PROBA_IDX
    root = Path(__file__).resolve().parents[1]
    best = root / "models" / "credit_model_best.joblib"
    path = best if best.exists() else (root / "models" / "credit_model.joblib")
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}. Train the model first.")
    _BUNDLE = joblib.load(path)
    _PIPE = _BUNDLE["pipeline"]
    _FEATURE_COLUMNS = _BUNDLE["feature_columns"]
    _THRESHOLDS = _BUNDLE.get("thresholds", {"low": 0.20, "medium": 0.50})
    _PROBA_IDX = list(_PIPE.classes_).index("bad")

@app.on_event("startup")
def on_startup():
    _load_bundle()

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _BUNDLE is not None}

@app.post("/predict", response_model=CreditResponse)
def predict(sample: CreditRequest) -> CreditResponse:
    if _PIPE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    X = pd.DataFrame([sample.model_dump()]).reindex(columns=_FEATURE_COLUMNS)
    proba_bad = float(_PIPE.predict_proba(X)[0, _PROBA_IDX])
    return CreditResponse(prob_default=proba_bad, risk=categorize(proba_bad, _THRESHOLDS))

@app.post("/predict_batch", response_model=List[CreditResponse])
def predict_batch(samples: List[CreditRequest]) -> List[CreditResponse]:
    if _PIPE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not samples:
        return []
    data = [sample.model_dump() for sample in samples]
    X = pd.DataFrame(data).reindex(columns=_FEATURE_COLUMNS)
    probs = _PIPE.predict_proba(X)[:, _PROBA_IDX]
    return [CreditResponse(prob_default=float(p), risk=categorize(float(p), _THRESHOLDS)) for p in probs]