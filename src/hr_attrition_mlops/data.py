"""Etapa 1 — Datos: descargar, validar y separar.

Uso:
    uv run preparar-datos            # descarga solo si no existe el CSV
    uv run preparar-datos --forzar   # vuelve a descargar
"""

import argparse
import io
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from hr_attrition_mlops.config import (
    CATEGORICAL_FEATURES,
    CATEGORICAL_VALUES,
    DATA_DIR,
    DATA_PATH,
    DATA_URLS,
    DOWNLOAD_TIMEOUT,
    FEATURES,
    ID_COLUMN,
    KAGGLE_FILENAME,
    MIN_ROWS,
    NEGATIVE_CLASS,
    NUMERIC_FEATURES,
    NUMERIC_RANGES,
    POSITIVE_CLASS,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
)


def download_data() -> pd.DataFrame:
    """Obtiene el CSV y lo guarda en data/raw.

    1) Si en data/raw está el archivo original de Kaggle
       (WA_Fn-UseC_-HR-Employee-Attrition.csv), lo usa.
    2) Si no, lo descarga desde la primera URL que responda.
    """
    kaggle = DATA_DIR / KAGGLE_FILENAME
    if kaggle.exists():
        print(f"Usando el archivo de Kaggle: {kaggle}")
        df = clean_data(pd.read_csv(kaggle))
        validate_data(df)
        df.to_csv(DATA_PATH, index=False)
        return df

    errores = []
    for url in DATA_URLS:
        try:
            print(f"Descargando datos desde: {url}")
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as resp:
                contenido = resp.read()
            df = clean_data(pd.read_csv(io.BytesIO(contenido)))
            validate_data(df)
        except Exception as exc:  # red caída, 404, archivo corrupto, etc.
            errores.append(f"{url} -> {exc}")
            continue

        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        return df

    raise RuntimeError(
        "No se pudo descargar el dataset desde ninguna fuente.\n"
        + "\n".join(errores)
        + f"\nDescárgalo manualmente de Kaggle y guárdalo como: {DATA_PATH}"
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza mínima: quita espacios/BOM en encabezados y textos."""
    df = df.copy()
    df.columns = df.columns.str.replace("﻿", "", regex=False).str.strip()
    for col in CATEGORICAL_FEATURES + [TARGET]:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.strip()
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Valida columnas, nulos, tipos, rangos y categorías. Lanza ValueError."""
    columnas_perdidas = set(FEATURES + [TARGET]) - set(df.columns)
    if columnas_perdidas:
        raise ValueError(f"Faltan columnas: {sorted(columnas_perdidas)}")

    if len(df) < MIN_ROWS:
        raise ValueError(f"Muy pocas filas: {len(df)} (mínimo {MIN_ROWS})")

    nulos = df[FEATURES + [TARGET]].isnull().sum()
    nulos = nulos[nulos > 0]
    if not nulos.empty:
        raise ValueError(f"Hay valores nulos en: {nulos.to_dict()}")

    clases = set(df[TARGET].unique())
    if clases != {POSITIVE_CLASS, NEGATIVE_CLASS}:
        raise ValueError(
            f"{TARGET} debe tener exactamente {{'{POSITIVE_CLASS}', "
            f"'{NEGATIVE_CLASS}'}} y tiene {clases}"
        )

    for col in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"La columna {col} debe ser numérica")
        minimo, maximo = NUMERIC_RANGES[col]
        fuera = df[(df[col] < minimo) | (df[col] > maximo)]
        if not fuera.empty:
            raise ValueError(
                f"{col}: {len(fuera)} valores fuera del rango [{minimo}, {maximo}]"
            )

    for col, permitidos in CATEGORICAL_VALUES.items():
        desconocidos = set(df[col].unique()) - set(permitidos)
        if desconocidos:
            raise ValueError(f"{col}: valores no permitidos {sorted(desconocidos)}")

    if ID_COLUMN in df.columns and df[ID_COLUMN].duplicated().any():
        raise ValueError(f"Hay empleados duplicados ({ID_COLUMN} repetido)")


def load_data() -> pd.DataFrame:
    """Lee el CSV local (lo descarga si no existe), lo limpia y lo valida."""
    if not DATA_PATH.exists():
        download_data()
    df = clean_data(pd.read_csv(DATA_PATH))
    validate_data(df)
    return df


def get_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """X con las variables del modelo; y = 1 si renunció, 0 si permanece."""
    X = df[FEATURES]
    y = (df[TARGET] == POSITIVE_CLASS).astype(int)
    return X, y


def split_data(df: pd.DataFrame):
    """Separación train/test estratificada (mantiene el % de renuncias)."""
    X, y = get_xy(df)
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga y valida el dataset")
    parser.add_argument("--forzar", action="store_true", help="volver a descargar")
    args = parser.parse_args()

    if args.forzar or not DATA_PATH.exists():
        download_data()
    else:
        print(f"Usando el archivo local: {DATA_PATH}")

    df = load_data()
    tasa = (df[TARGET] == POSITIVE_CLASS).mean()
    print("Datos validados correctamente ✔")
    print(f"Archivo: {DATA_PATH}")
    print(f"Filas: {len(df)} | Columnas: {df.shape[1]} | Variables del modelo: {len(FEATURES)}")
    print(f"Distribución de {TARGET}:")
    print(df[TARGET].value_counts().to_string())
    print(f"Tasa de rotación: {tasa:.1%}")


if __name__ == "__main__":
    main()
