"""Etapa 3 — Predicción: cargar el modelo (una sola vez) y predecir un empleado.

Uso:
    uv run predecir
"""

import json
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd

from hr_attrition_mlops.config import (
    CATEGORICAL_FEATURES,
    CLASS_LABELS,
    DECISION_THRESHOLD,
    FEATURE_LABELS,
    FEATURES,
    METRICS_PATH,
    MODEL_PATH,
    NEGATIVE_CLASS,
    POSITIVE_CLASS,
    RISK_LEVELS,
    TOP_FACTORS,
)

# Empleado de ejemplo (perfil de riesgo alto) usado en `uv run predecir`,
# en /docs de la API y en la interfaz.
EXAMPLE_EMPLOYEE: dict[str, Any] = {
    "Age": 26,
    "DistanceFromHome": 22,
    "Education": 3,
    "EnvironmentSatisfaction": 1,
    "JobInvolvement": 2,
    "JobLevel": 1,
    "JobSatisfaction": 1,
    "MonthlyIncome": 2200,
    "NumCompaniesWorked": 4,
    "PercentSalaryHike": 11,
    "RelationshipSatisfaction": 2,
    "StockOptionLevel": 0,
    "TotalWorkingYears": 4,
    "TrainingTimesLastYear": 1,
    "WorkLifeBalance": 1,
    "YearsAtCompany": 1,
    "YearsInCurrentRole": 0,
    "YearsSinceLastPromotion": 0,
    "YearsWithCurrManager": 0,
    "BusinessTravel": "Travel_Frequently",
    "Department": "Sales",
    "EducationField": "Marketing",
    "JobRole": "Sales Representative",
    "MaritalStatus": "Single",
    "OverTime": "Yes",
}

# Perfil de riesgo bajo (para pruebas y la interfaz)
LOW_RISK_EMPLOYEE: dict[str, Any] = {
    **EXAMPLE_EMPLOYEE,
    "Age": 42,
    "DistanceFromHome": 3,
    "EnvironmentSatisfaction": 4,
    "JobInvolvement": 3,
    "JobLevel": 3,
    "JobSatisfaction": 4,
    "MonthlyIncome": 11000,
    "NumCompaniesWorked": 1,
    "PercentSalaryHike": 18,
    "RelationshipSatisfaction": 4,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 18,
    "TrainingTimesLastYear": 3,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 12,
    "YearsInCurrentRole": 7,
    "YearsSinceLastPromotion": 2,
    "YearsWithCurrManager": 7,
    "BusinessTravel": "Travel_Rarely",
    "Department": "Research & Development",
    "EducationField": "Medical",
    "JobRole": "Manager",
    "MaritalStatus": "Married",
    "OverTime": "No",
}


@lru_cache(maxsize=1)
def _load_cached(path: str, mtime: float):
    # La caché depende de la fecha del archivo: si se reentrena, se recarga.
    return joblib.load(path)


def load_model():
    """Carga el modelo una sola vez (no en cada petición)."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}. Ejecuta `uv run entrenar`.")
    return _load_cached(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)


def load_model_info() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(f"No se encontró la ficha del modelo: {METRICS_PATH}")
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def risk_level(probability: float) -> str:
    for limite, etiqueta in RISK_LEVELS:
        if probability < limite:
            return etiqueta
    return RISK_LEVELS[-1][1]


def _readable(feature_name: str, employee: dict[str, Any]) -> str:
    """'num__Age' -> 'Edad = 26'; 'cat__OverTime_Yes' -> 'Horas extra = Yes'."""
    tipo, _, resto = feature_name.partition("__")
    if tipo == "num":
        return f"{FEATURE_LABELS.get(resto, resto)} = {employee[resto]}"
    for col in CATEGORICAL_FEATURES:
        if resto.startswith(col + "_"):
            return f"{FEATURE_LABELS.get(col, col)} = {resto[len(col) + 1:]}"
    return resto


def explain(model, sample: pd.DataFrame, employee: dict[str, Any]) -> list[dict[str, Any]]:
    """Factores que más empujan la probabilidad de renuncia de ESTE empleado.

    En una regresión logística la contribución de cada variable al log-odds es
    coeficiente × valor transformado, así que se puede explicar sin librerías
    adicionales.
    """
    preprocess = model.named_steps["preprocess"]
    x = preprocess.transform(sample)
    x = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    contrib = x[0] * model.named_steps["model"].coef_[0]
    names = preprocess.get_feature_names_out()
    orden = np.argsort(contrib)[::-1]
    return [
        {"factor": _readable(names[i], employee), "impacto": round(float(contrib[i]), 4)}
        for i in orden[:TOP_FACTORS]
        if contrib[i] > 0
    ]


def predict(employee: dict[str, Any]) -> dict[str, Any]:
    model = load_model()
    sample = pd.DataFrame([[employee[f] for f in FEATURES]], columns=FEATURES)
    proba_yes = float(model.predict_proba(sample)[0, 1])
    renuncia = proba_yes >= DECISION_THRESHOLD
    clase = POSITIVE_CLASS if renuncia else NEGATIVE_CLASS

    return {
        "attrition": clase,
        "prediccion": CLASS_LABELS[clase],
        "probabilidad_renuncia": round(proba_yes, 4),
        "nivel_riesgo": risk_level(proba_yes),
        "umbral": DECISION_THRESHOLD,
        "probabilidades": {
            CLASS_LABELS[POSITIVE_CLASS]: round(proba_yes, 4),
            CLASS_LABELS[NEGATIVE_CLASS]: round(1 - proba_yes, 4),
        },
        "factores_riesgo": explain(model, sample, employee),
    }


def main() -> None:
    for nombre, emp in (("Riesgo alto", EXAMPLE_EMPLOYEE), ("Riesgo bajo", LOW_RISK_EMPLOYEE)):
        r = predict(emp)
        print(f"\n== Perfil de ejemplo: {nombre} ==")
        print(f"Predicción: {r['prediccion']} | P(renuncia) = {r['probabilidad_renuncia']:.1%} "
              f"| Riesgo {r['nivel_riesgo']}")
        for f in r["factores_riesgo"]:
            print(f"  · {f['factor']} (+{f['impacto']:.2f})")


if __name__ == "__main__":
    main()
