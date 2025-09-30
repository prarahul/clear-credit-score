import sys
from pathlib import Path
import pytest

# Ensure project root is on sys.path before tests import modules
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT

@pytest.fixture(scope="session", autouse=True)
def ensure_model(project_root: Path):
    # Train if model is missing
    model_path = project_root / "models" / "credit_model.joblib"
    if not model_path.exists():
        import runpy
        runpy.run_module("src.train", run_name="__main__")
    assert model_path.exists(), "Model was not created"
    return model_path