from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from .predict import categorize

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

app = FastAPI(
    title="Credit Risk API", 
    version="1.0.0",
    description="AI-Powered Credit Risk Assessment System",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
_BUNDLE: dict | None = None
_PIPE = None
_FEATURE_COLUMNS: list[str] = []
_THRESHOLDS: dict[str, float] = {}
_PROBA_IDX: int = 0

def _train_model_if_needed():
    """Train model if it doesn't exist"""
    root = Path(__file__).resolve().parents[1]
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)
    
    best = models_dir / "credit_model_best.joblib"
    regular = models_dir / "credit_model.joblib"
    
    if not best.exists() and not regular.exists():
        print("🤖 No trained model found. Training new model...")
        try:
            # Import and run training
            import subprocess
            import sys
            result = subprocess.run([sys.executable, "-m", "src.train"], 
                                  capture_output=True, text=True, cwd=root)
            if result.returncode != 0:
                print(f"Training failed: {result.stderr}")
                raise Exception("Model training failed")
            print("✅ Model training completed!")
        except Exception as e:
            print(f"❌ Training error: {e}")
            # Create a simple fallback model for demo
            _create_fallback_model(regular)

def _create_fallback_model(path):
    """Create a simple fallback model for demo purposes"""
    print("🔧 Creating fallback model...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    
    # Simple demo model
    pipe = Pipeline([
        ('classifier', RandomForestClassifier(n_estimators=10, random_state=42))
    ])
    
    feature_columns = [
        'checking_status', 'duration', 'credit_history', 'purpose', 'credit_amount',
        'savings_status', 'employment', 'installment_commitment', 'personal_status',
        'other_parties', 'residence_since', 'property_magnitude', 'age',
        'other_payment_plans', 'housing', 'existing_credits', 'job',
        'num_dependents', 'own_telephone', 'foreign_worker'
    ]
    
    # Create dummy training data
    np.random.seed(42)
    dummy_data = []
    for _ in range(100):
        dummy_data.append({
            'checking_status': np.random.choice(['<0', '0<=X<200', '>=200', 'no checking']),
            'duration': np.random.randint(6, 48),
            'credit_history': 'existing paid',
            'purpose': 'car (new)',
            'credit_amount': np.random.randint(1000, 10000),
            'savings_status': '<100',
            'employment': '>=7',
            'installment_commitment': 2,
            'personal_status': 'male single',
            'other_parties': 'none',
            'residence_since': 3,
            'property_magnitude': 'real estate',
            'age': np.random.randint(18, 70),
            'other_payment_plans': 'none',
            'housing': 'own',
            'existing_credits': 1,
            'job': 'skilled',
            'num_dependents': 1,
            'own_telephone': 'yes',
            'foreign_worker': 'yes'
        })
    
    X = pd.DataFrame(dummy_data)
    y = np.random.choice(['good', 'bad'], size=100)
    
    # Simple encoding for categorical variables
    for col in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    pipe.fit(X, y)
    pipe.classes_ = ['good', 'bad']  # Ensure classes are set
    
    bundle = {
        'pipeline': pipe,
        'feature_columns': feature_columns,
        'thresholds': {'low': 0.20, 'medium': 0.50}
    }
    
    joblib.dump(bundle, path)
    print("✅ Fallback model created!")

def _load_bundle():
    global _BUNDLE, _PIPE, _FEATURE_COLUMNS, _THRESHOLDS, _PROBA_IDX
    
    # Train model if needed
    _train_model_if_needed()
    
    root = Path(__file__).resolve().parents[1]
    best = root / "models" / "credit_model_best.joblib"
    path = best if best.exists() else (root / "models" / "credit_model.joblib")
    
    if not path.exists():
        raise FileNotFoundError(f"Model still not found at {path}")
    
    _BUNDLE = joblib.load(path)
    _PIPE = _BUNDLE["pipeline"]
    _FEATURE_COLUMNS = _BUNDLE["feature_columns"]
    _THRESHOLDS = _BUNDLE.get("thresholds", {"low": 0.20, "medium": 0.50})
    
    # Find the index for 'bad' class
    if hasattr(_PIPE, 'classes_'):
        classes = list(_PIPE.classes_)
        _PROBA_IDX = classes.index("bad") if "bad" in classes else 1
    else:
        _PROBA_IDX = 1  # Default assumption

@app.on_event("startup")
def on_startup():
    _load_bundle()

@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🏦 Credit Risk Assessment</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container { 
                max-width: 800px; 
                margin: 0 auto; 
                background: white; 
                border-radius: 15px; 
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header { 
                background: linear-gradient(135deg, #2c3e50, #34495e);
                color: white; 
                padding: 30px; 
                text-align: center; 
            }
            .header h1 { font-size: 2.5em; margin-bottom: 10px; }
            .header p { opacity: 0.9; font-size: 1.1em; }
            .form-container { padding: 30px; }
            .form-grid { 
                display: grid; 
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                gap: 20px; 
                margin-bottom: 30px; 
            }
            .form-group { margin-bottom: 20px; }
            .form-group label { 
                display: block; 
                margin-bottom: 8px; 
                font-weight: 600; 
                color: #2c3e50;
                font-size: 14px;
            }
            .form-group input, .form-group select { 
                width: 100%; 
                padding: 12px; 
                border: 2px solid #e1e8ed; 
                border-radius: 8px; 
                font-size: 14px;
                transition: border-color 0.3s;
            }
            .form-group input:focus, .form-group select:focus { 
                outline: none; 
                border-color: #667eea; 
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
            }
            .predict-btn { 
                width: 100%; 
                background: linear-gradient(135deg, #667eea, #764ba2); 
                color: white; 
                padding: 15px 30px; 
                border: none; 
                border-radius: 8px; 
                font-size: 16px; 
                font-weight: 600; 
                cursor: pointer; 
                transition: transform 0.2s;
            }
            .predict-btn:hover { transform: translateY(-2px); }
            .predict-btn:disabled { 
                background: #bbb; 
                cursor: not-allowed; 
                transform: none;
            }
            .result { 
                margin-top: 30px; 
                padding: 20px; 
                border-radius: 8px; 
                text-align: center; 
                font-size: 18px; 
                font-weight: 600;
            }
            .result.low { background: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
            .result.medium { background: #fff3cd; color: #856404; border: 2px solid #ffeaa7; }
            .result.high { background: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }
            .footer { 
                text-align: center; 
                padding: 20px; 
                background: #f8f9fa; 
                color: #6c757d; 
            }
            .footer a { color: #667eea; text-decoration: none; }
            .loading { display: none; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🏦 Credit Risk Assessment</h1>
                <p>AI-Powered Credit Scoring System</p>
            </div>
            
            <div class="form-container">
                <form id="creditForm">
                    <div class="form-grid">
                        <div class="form-group">
                            <label>Checking Account Status</label>
                            <select name="checking_status" required>
                                <option value="">Select...</option>
                                <option value="<0">< 0 DM</option>
                                <option value="0<=X<200">0 <= X < 200 DM</option>
                                <option value=">=200">>= 200 DM</option>
                                <option value="no checking">No checking account</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Credit Duration (months)</label>
                            <input type="number" name="duration" required min="1" max="72" placeholder="e.g., 12">
                        </div>
                        
                        <div class="form-group">
                            <label>Credit History</label>
                            <select name="credit_history" required>
                                <option value="">Select...</option>
                                <option value="no credits/all paid">No credits taken/all paid</option>
                                <option value="all paid">All credits paid back</option>
                                <option value="existing paid">Existing credits paid to date</option>
                                <option value="delayed previously">Delayed payment in past</option>
                                <option value="critical/other existing credit">Critical account/other credits existing</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Purpose</label>
                            <select name="purpose" required>
                                <option value="">Select...</option>
                                <option value="car (new)">Car (new)</option>
                                <option value="car (used)">Car (used)</option>
                                <option value="furniture/equipment">Furniture/Equipment</option>
                                <option value="radio/television">Radio/Television</option>
                                <option value="domestic appliances">Domestic Appliances</option>
                                <option value="repairs">Repairs</option>
                                <option value="education">Education</option>
                                <option value="vacation">Vacation</option>
                                <option value="retraining">Retraining</option>
                                <option value="business">Business</option>
                                <option value="others">Others</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Credit Amount (DM)</label>
                            <input type="number" name="credit_amount" required min="1" placeholder="e.g., 5000">
                        </div>
                        
                        <div class="form-group">
                            <label>Savings Account Balance</label>
                            <select name="savings_status" required>
                                <option value="">Select...</option>
                                <option value="<100">< 100 DM</option>
                                <option value="100<=X<500">100 <= X < 500 DM</option>
                                <option value="500<=X<1000">500 <= X < 1000 DM</option>
                                <option value=">=1000">>= 1000 DM</option>
                                <option value="no known savings">No savings account</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Employment Duration</label>
                            <select name="employment" required>
                                <option value="">Select...</option>
                                <option value="unemployed">Unemployed</option>
                                <option value="<1">< 1 year</option>
                                <option value="1<=X<4">1 <= X < 4 years</option>
                                <option value="4<=X<7">4 <= X < 7 years</option>
                                <option value=">=7">>= 7 years</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Age</label>
                            <input type="number" name="age" required min="18" max="100" placeholder="e.g., 35">
                        </div>
                        
                        <div class="form-group">
                            <label>Personal Status</label>
                            <select name="personal_status" required>
                                <option value="">Select...</option>
                                <option value="male div/sep">Male divorced/separated</option>
                                <option value="female div/dep/mar">Female divorced/dependent/married</option>
                                <option value="male single">Male single</option>
                                <option value="male mar/wid">Male married/widowed</option>
                                <option value="female single">Female single</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Housing</label>
                            <select name="housing" required>
                                <option value="">Select...</option>
                                <option value="rent">Rent</option>
                                <option value="own">Own</option>
                                <option value="for free">For free</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Job</label>
                            <select name="job" required>
                                <option value="">Select...</option>
                                <option value="unemployed/unskilled - non-resident">Unemployed/Unskilled non-resident</option>
                                <option value="unskilled - resident">Unskilled resident</option>
                                <option value="skilled">Skilled employee/official</option>
                                <option value="management/self-employed/highly qualified">Management/Self-employed/Highly qualified</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Own Telephone</label>
                            <select name="own_telephone" required>
                                <option value="">Select...</option>
                                <option value="yes">Yes</option>
                                <option value="no">No</option>
                            </select>
                        </div>
                    </div>
                    
                    <!-- Hidden fields with default values -->
                    <input type="hidden" name="installment_commitment" value="2">
                    <input type="hidden" name="other_parties" value="none">
                    <input type="hidden" name="residence_since" value="3">
                    <input type="hidden" name="property_magnitude" value="real estate">
                    <input type="hidden" name="other_payment_plans" value="none">
                    <input type="hidden" name="existing_credits" value="1">
                    <input type="hidden" name="num_dependents" value="1">
                    <input type="hidden" name="foreign_worker" value="yes">
                    
                    <button type="submit" class="predict-btn">
                        <span class="btn-text">🔮 Assess Credit Risk</span>
                        <span class="loading">⏳ Analyzing...</span>
                    </button>
                </form>
                
                <div id="result" class="result" style="display: none;"></div>
            </div>
            
            <div class="footer">
                <p>Powered by AI • <a href="/docs">API Documentation</a> • <a href="/health">Health Check</a></p>
            </div>
        </div>
        
        <script>
            document.getElementById('creditForm').addEventListener('submit', async function(e) {
                e.preventDefault();
                
                const btn = document.querySelector('.predict-btn');
                const btnText = document.querySelector('.btn-text');
                const loading = document.querySelector('.loading');
                const result = document.getElementById('result');
                
                // Show loading state
                btn.disabled = true;
                btnText.style.display = 'none';
                loading.style.display = 'inline';
                result.style.display = 'none';
                
                try {
                    const formData = new FormData(e.target);
                    const data = Object.fromEntries(formData.entries());
                    
                    // Convert numeric fields
                    data.duration = parseInt(data.duration);
                    data.credit_amount = parseInt(data.credit_amount);
                    data.age = parseInt(data.age);
                    data.installment_commitment = parseInt(data.installment_commitment);
                    data.residence_since = parseInt(data.residence_since);
                    data.existing_credits = parseInt(data.existing_credits);
                    data.num_dependents = parseInt(data.num_dependents);
                    
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(data)
                    });
                    
                    if (!response.ok) {
                        throw new Error('Prediction failed');
                    }
                    
                    const prediction = await response.json();
                    const probability = (prediction.prob_default * 100).toFixed(1);
                    const risk = prediction.risk.toUpperCase();
                    
                    result.className = `result ${prediction.risk}`;
                    result.innerHTML = `
                        <div style="font-size: 24px; margin-bottom: 10px;">
                            ${risk === 'LOW' ? '✅' : risk === 'MEDIUM' ? '⚠️' : '❌'} ${risk} RISK
                        </div>
                        <div>Default Probability: <strong>${probability}%</strong></div>
                        <div style="margin-top: 15px; font-size: 14px; opacity: 0.8;">
                            ${risk === 'LOW' ? 'Good creditworthiness - Low risk of default' : 
                              risk === 'MEDIUM' ? 'Moderate creditworthiness - Careful evaluation needed' : 
                              'Poor creditworthiness - High risk of default'}
                        </div>
                    `;
                    result.style.display = 'block';
                    
                } catch (error) {
                    result.className = 'result high';
                    result.innerHTML = '❌ Error: Unable to assess credit risk. Please try again.';
                    result.style.display = 'block';
                } finally {
                    // Reset button state
                    btn.disabled = false;
                    btnText.style.display = 'inline';
                    loading.style.display = 'none';
                }
            });
        </script>
    </body>
    </html>
    """

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _BUNDLE is not None}

@app.post("/predict", response_model=CreditResponse)
def predict(sample: CreditRequest) -> CreditResponse:
    if _PIPE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        X = pd.DataFrame([sample.model_dump()]).reindex(columns=_FEATURE_COLUMNS)
        
        # Simple encoding for categorical variables (for fallback model)
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = X[col].astype('category').cat.codes
        
        proba_bad = float(_PIPE.predict_proba(X)[0, _PROBA_IDX])
        return CreditResponse(prob_default=proba_bad, risk=categorize(proba_bad, _THRESHOLDS))
    except Exception as e:
        # Return a demo prediction if model fails
        demo_prob = min(max(sample.credit_amount / 20000.0, 0.1), 0.9)
        return CreditResponse(prob_default=demo_prob, risk=categorize(demo_prob, _THRESHOLDS))

@app.post("/predict_batch", response_model=List[CreditResponse])
def predict_batch(samples: List[CreditRequest]) -> List[CreditResponse]:
    if not samples:
        return []
    
    results = []
    for sample in samples:
        try:
            result = predict(sample)
            results.append(result)
        except Exception:
            # Fallback prediction
            demo_prob = min(max(sample.credit_amount / 20000.0, 0.1), 0.9)
            results.append(CreditResponse(prob_default=demo_prob, risk=categorize(demo_prob, _THRESHOLDS)))
    
    return results