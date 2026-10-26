from fastapi.testclient import TestClient

from hr_attrition_mlops import api
from hr_attrition_mlops.api import app
from hr_attrition_mlops.predict import EXAMPLE_EMPLOYEE

client = TestClient(app)


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    assert "metrics" in response.json()


def test_predict_valid_employee():
    response = client.post("/predict", json=EXAMPLE_EMPLOYEE)
    assert response.status_code == 200
    body = response.json()
    assert body["attrition"] in {"Yes", "No"}
    assert 0 <= body["probabilidad_renuncia"] <= 1
    assert body["nivel_riesgo"] in {"Bajo", "Medio", "Alto"}


def test_predict_rejects_out_of_range():
    response = client.post("/predict", json={**EXAMPLE_EMPLOYEE, "Age": 150, "MonthlyIncome": -10})
    assert response.status_code == 422


def test_predict_rejects_unknown_category():
    response = client.post("/predict", json={**EXAMPLE_EMPLOYEE, "OverTime": "Tal vez"})
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    incompleto = {k: v for k, v in EXAMPLE_EMPLOYEE.items() if k != "JobRole"}
    response = client.post("/predict", json=incompleto)
    assert response.status_code == 422


def test_predict_rejects_inconsistent_years():
    response = client.post("/predict", json={**EXAMPLE_EMPLOYEE, "YearsAtCompany": 3, "YearsInCurrentRole": 10})
    assert response.status_code == 422


def test_predict_returns_503_without_model(monkeypatch):
    def sin_modelo(_):
        raise FileNotFoundError

    monkeypatch.setattr(api, "predict", sin_modelo)
    response = client.post("/predict", json=EXAMPLE_EMPLOYEE)
    assert response.status_code == 503
