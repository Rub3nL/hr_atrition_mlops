"""Etapa 2 — Entrenamiento: entrenar, evaluar, aplicar la puerta de calidad y guardar.

Uso:
    uv run entrenar
"""

import json
from datetime import datetime
from typing import Any

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hr_attrition_mlops import __version__
from hr_attrition_mlops.config import (
    CATEGORICAL_FEATURES,
    CATEGORICAL_VALUES,
    CV_FOLDS,
    DATA_URLS,
    DECISION_THRESHOLD,
    FEATURES,
    METRICS_PATH,
    MIN_PR_AUC,
    MIN_RECALL,
    MIN_ROC_AUC,
    MODEL_PARAMS,
    MODEL_PATH,
    MODEL_VERSION,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
)
from hr_attrition_mlops.data import load_data, split_data


def build_pipeline() -> Pipeline:
    """Preprocesamiento + modelo en un solo objeto (evita fuga de datos)."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    categories=[CATEGORICAL_VALUES[c] for c in CATEGORICAL_FEATURES],
                    handle_unknown="ignore",
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", LogisticRegression(**MODEL_PARAMS)),
        ]
    )


def train_model(X_train, y_train) -> Pipeline:
    model = build_pipeline()
    model.fit(X_train, y_train)
    return model


def cross_validate_model(X_train, y_train) -> dict[str, float]:
    """Validación cruzada estratificada sobre train: ¿el resultado es estable?"""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_validate(
        build_pipeline(), X_train, y_train, cv=cv,
        scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision", "recall": "recall"},
    )
    return {
        f"cv_{name}_{stat}": round(float(fn(scores[f"test_{name}"])), 4)
        for name in ("roc_auc", "pr_auc", "recall")
        for stat, fn in (("mean", np.mean), ("std", np.std))
    }


def evaluate_model(model, X_test, y_test) -> dict[str, Any]:
    """Métricas en el conjunto de prueba (datos que el modelo nunca vio)."""
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= DECISION_THRESHOLD).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_test, proba)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "baseline_pr_auc": round(float(np.mean(y_test)), 4),  # modelo al azar
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def check_quality_gate(metrics: dict[str, Any]) -> list[str]:
    """Devuelve la lista de mínimos incumplidos (vacía = el modelo aprueba)."""
    fallas = []
    if metrics["roc_auc"] < MIN_ROC_AUC:
        fallas.append(f"ROC-AUC {metrics['roc_auc']:.3f} < {MIN_ROC_AUC}")
    if metrics["pr_auc"] < MIN_PR_AUC:
        fallas.append(f"PR-AUC {metrics['pr_auc']:.3f} < {MIN_PR_AUC}")
    if metrics["recall"] < MIN_RECALL:
        fallas.append(f"Recall {metrics['recall']:.3f} < {MIN_RECALL}")
    return fallas


def feature_importance(model: Pipeline, top: int = 10) -> dict[str, list[dict]]:
    """Coeficientes de la regresión logística: qué aumenta o reduce el riesgo."""
    names = model.named_steps["preprocess"].get_feature_names_out()
    coefs = model.named_steps["model"].coef_[0]
    pares = sorted(zip(names, coefs), key=lambda p: p[1])
    fmt = lambda ps: [{"variable": n, "coeficiente": round(float(c), 4)} for n, c in ps]
    return {
        "aumentan_riesgo": fmt(reversed(pares[-top:])),
        "reducen_riesgo": fmt(pares[:top]),
    }


def save_model(model, metrics: dict[str, Any], extra: dict[str, Any]) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    model_card = {
        "model_name": "hr-attrition-logreg",
        "model_version": MODEL_VERSION,
        "package_version": __version__,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "problem": "Clasificación binaria: probabilidad de que un empleado renuncie (Attrition = Yes)",
        "intended_users": "Equipos de RR.HH. / gestión del talento para priorizar acciones de retención",
        "model_type": type(model.named_steps["model"]).__name__,
        "params": MODEL_PARAMS,
        "decision_threshold": DECISION_THRESHOLD,
        "features": FEATURES,
        "classes": {"0": "No (permanece)", "1": "Yes (renuncia)"},
        "data": {"source": DATA_URLS[0], **extra["data"]},
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "stratified": True},
        "metrics": metrics,
        "cross_validation": extra["cv"],
        "quality_gate": {
            "min_roc_auc": MIN_ROC_AUC,
            "min_pr_auc": MIN_PR_AUC,
            "min_recall": MIN_RECALL,
            "passed": True,
        },
        "feature_importance": extra["importance"],
        "limitations": [
            "Dataset ficticio creado por científicos de datos de IBM; validar con datos propios antes de usar en producción.",
            "La predicción es un apoyo a la decisión, no un criterio para despedir o sancionar.",
            "Gender se excluye de las variables para reducir sesgo; revisar equidad por grupo periódicamente.",
        ],
    }
    METRICS_PATH.write_text(json.dumps(model_card, indent=2, ensure_ascii=False))


def main() -> None:
    print("1) Cargando datos...")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"    Entrenamiento: {len(X_train)} filas | Test: {len(X_test)} filas")
    print(f"    Tasa de renuncia en train: {y_train.mean():.1%} | test: {y_test.mean():.1%}")

    print(f"2) Validación cruzada ({CV_FOLDS} folds) en train...")
    cv = cross_validate_model(X_train, y_train)
    print(f"    ROC-AUC CV: {cv['cv_roc_auc_mean']:.3f} ± {cv['cv_roc_auc_std']:.3f}")

    print("3) Entrenando modelo final...")
    model = train_model(X_train, y_train)

    print("4) Evaluando en test...")
    metrics = evaluate_model(model, X_test, y_test)
    for k in ("roc_auc", "pr_auc", "recall", "precision", "f1", "accuracy"):
        print(f"    {k:<10}: {metrics[k]:.3f}")
    print(f"    (PR-AUC de un modelo al azar: {metrics['baseline_pr_auc']:.3f})")
    print(f"    Matriz de confusión: {metrics['confusion_matrix']}")

    print("5) Puerta de calidad...")
    fallas = check_quality_gate(metrics)
    if fallas:
        raise ValueError(
            "El modelo NO alcanzó la calidad mínima y no se guardará: " + "; ".join(fallas)
        )
    print(f"    Aprobada ✔ (ROC-AUC ≥ {MIN_ROC_AUC}, PR-AUC ≥ {MIN_PR_AUC}, Recall ≥ {MIN_RECALL})")

    print("6) Guardando modelo y ficha...")
    extra = {
        "data": {
            "rows": int(len(df)),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "attrition_rate": round(float((y_train.sum() + y_test.sum()) / len(df)), 4),
        },
        "cv": cv,
        "importance": feature_importance(model),
    }
    save_model(model, metrics, extra)
    print(f"    Modelo:  {MODEL_PATH}")
    print(f"    Métricas: {METRICS_PATH}")
    print("Listo ✔")


if __name__ == "__main__":
    main()
