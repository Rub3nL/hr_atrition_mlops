import json

import pytest

from hr_attrition_mlops.config import (
    METRICS_PATH,
    MIN_PR_AUC,
    MIN_RECALL,
    MIN_ROC_AUC,
)
from hr_attrition_mlops.data import load_data, split_data
from hr_attrition_mlops.predict import EXAMPLE_EMPLOYEE, LOW_RISK_EMPLOYEE, predict, risk_level
from hr_attrition_mlops.train import check_quality_gate, evaluate_model, train_model


@pytest.fixture(scope="module")
def split():
    return split_data(load_data())


@pytest.fixture(scope="module")
def metrics(split):
    X_train, X_test, y_train, y_test = split
    model = train_model(X_train, y_train)
    return evaluate_model(model, X_test, y_test)


def test_model_passes_quality_gate(metrics):
    assert metrics["roc_auc"] >= MIN_ROC_AUC
    assert metrics["pr_auc"] >= MIN_PR_AUC
    assert metrics["recall"] >= MIN_RECALL
    assert check_quality_gate(metrics) == []


def test_model_beats_random_baseline(metrics):
    assert metrics["pr_auc"] > 2 * metrics["baseline_pr_auc"]


def test_quality_gate_blocks_bad_model():
    malo = {"roc_auc": 0.5, "pr_auc": 0.16, "recall": 0.1}
    assert len(check_quality_gate(malo)) == 3


def test_training_is_reproducible(split):
    X_train, X_test, y_train, y_test = split
    a = evaluate_model(train_model(X_train, y_train), X_test, y_test)
    b = evaluate_model(train_model(X_train, y_train), X_test, y_test)
    assert a == b


def test_predict_output_shape():
    r = predict(EXAMPLE_EMPLOYEE)
    assert set(r) == {
        "attrition", "prediccion", "probabilidad_renuncia", "nivel_riesgo",
        "umbral", "probabilidades", "factores_riesgo",
    }
    assert 0.0 <= r["probabilidad_renuncia"] <= 1.0
    assert sum(r["probabilidades"].values()) == pytest.approx(1.0, abs=1e-3)
    assert r["attrition"] == ("Yes" if r["probabilidad_renuncia"] >= r["umbral"] else "No")


def test_high_risk_profile_scores_above_low_risk():
    alto = predict(EXAMPLE_EMPLOYEE)
    bajo = predict(LOW_RISK_EMPLOYEE)
    assert alto["probabilidad_renuncia"] > bajo["probabilidad_renuncia"]
    assert alto["nivel_riesgo"] == "Alto"
    assert bajo["nivel_riesgo"] != "Alto"
    assert len(alto["factores_riesgo"]) > 0


def test_risk_levels():
    assert risk_level(0.05) == "Bajo"
    assert risk_level(0.45) == "Medio"
    assert risk_level(0.95) == "Alto"


def test_model_card_has_required_fields():
    card = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    for key in ("model_version", "trained_at", "params", "metrics", "quality_gate", "features"):
        assert key in card
    assert card["quality_gate"]["passed"] is True
