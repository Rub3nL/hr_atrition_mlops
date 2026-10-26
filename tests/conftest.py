import pytest

from hr_attrition_mlops.config import METRICS_PATH, MODEL_PATH
from hr_attrition_mlops.train import main as train_main


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    """Garantiza que exista un modelo entrenado antes de las pruebas."""
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        train_main()
