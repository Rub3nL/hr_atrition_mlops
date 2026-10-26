import pytest

from hr_attrition_mlops.config import (
    EXPECTED_COLUMNS,
    EXPECTED_ROWS,
    FEATURES,
    TARGET,
    TEST_SIZE,
)
from hr_attrition_mlops.data import load_data, split_data, validate_data


@pytest.fixture(scope="module")
def df():
    return load_data()


def test_data_shape(df):
    assert df.shape == (EXPECTED_ROWS, EXPECTED_COLUMNS)
    assert set(FEATURES + [TARGET]).issubset(df.columns)


def test_target_two_classes_and_imbalanced(df):
    assert set(df[TARGET]) == {"Yes", "No"}
    tasa = (df[TARGET] == "Yes").mean()
    assert 0.10 < tasa < 0.25  # ~16 % de rotación en el dataset de IBM


def test_validate_rejects_missing_column(df):
    with pytest.raises(ValueError, match="Faltan columnas"):
        validate_data(df.drop(columns=["MonthlyIncome"]))


def test_validate_rejects_nulls(df):
    malo = df.copy()
    malo.loc[0, "Age"] = None
    with pytest.raises(ValueError, match="nulos"):
        validate_data(malo)


def test_validate_rejects_out_of_range(df):
    malo = df.copy()
    malo.loc[0, "Age"] = 150
    with pytest.raises(ValueError, match="fuera del rango"):
        validate_data(malo)


def test_validate_rejects_unknown_category(df):
    malo = df.copy()
    malo.loc[0, "OverTime"] = "Tal vez"
    with pytest.raises(ValueError, match="no permitidos"):
        validate_data(malo)


def test_split_is_stratified_and_disjoint(df):
    X_train, X_test, y_train, y_test = split_data(df)
    assert len(X_test) == pytest.approx(len(df) * TEST_SIZE, abs=1)
    assert set(X_train.index).isdisjoint(X_test.index)  # sin fuga de datos
    assert abs(y_train.mean() - y_test.mean()) < 0.01   # misma proporción de renuncias
    assert list(X_train.columns) == FEATURES
