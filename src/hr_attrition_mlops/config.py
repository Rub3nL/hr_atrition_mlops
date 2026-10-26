"""Configuración central del proyecto.

Todo parámetro (rutas, columnas, rangos válidos, hiperparámetros, umbrales y
puertas de calidad) vive aquí para que el resto del código no tenga
"números mágicos". Si cambias algo aquí: `uv run entrenar` y revisa
models/metrics.json.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data" / "raw"
DATA_PATH = DATA_DIR / "hr_employee_attrition.csv"
# Nombre del archivo tal como se descarga de Kaggle (si lo copias a data/raw se usa)
KAGGLE_FILENAME = "WA_Fn-UseC_-HR-Employee-Attrition.csv"

MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "attrition_model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# --------------------------------------------------------------------------
# Fuente de datos: IBM HR Analytics Employee Attrition & Performance
# (1 470 empleados, 35 columnas). Original publicado por IBM en Kaggle:
# https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
# Se usan espejos públicos en GitHub para descargarlo sin credenciales.
# Se prueban en orden: si uno falla se intenta el siguiente.
# --------------------------------------------------------------------------
DATA_URLS = [
    "https://raw.githubusercontent.com/pplonski/datasets-for-start/master/"
    "employee_attrition/HR-Employee-Attrition-All.csv",
    "https://raw.githubusercontent.com/IBM/employee-attrition-aif360/master/"
    "data/emp_attrition.csv",
]
DOWNLOAD_TIMEOUT = 60  # segundos

EXPECTED_ROWS = 1470
EXPECTED_COLUMNS = 35
MIN_ROWS = 1000  # menos filas que esto indica un archivo truncado

# --------------------------------------------------------------------------
# Variable objetivo
# --------------------------------------------------------------------------
TARGET = "Attrition"
POSITIVE_CLASS = "Yes"   # el empleado dejó la empresa
NEGATIVE_CLASS = "No"    # el empleado permanece
CLASS_LABELS = {
    POSITIVE_CLASS: "Renuncia probable",
    NEGATIVE_CLASS: "Permanencia probable",
}

# Columnas que se descartan:
#  - EmployeeCount, Over18, StandardHours: constantes (no aportan información)
#  - EmployeeNumber: identificador
#  - DailyRate, HourlyRate, MonthlyRate: tarifas sin relación con el sueldo real
#  - PerformanceRating: casi constante (solo toma 3 o 4)
#  - Gender: se excluye a propósito por equidad (no discriminar por género)
DROPPED_COLUMNS = [
    "EmployeeCount", "Over18", "StandardHours", "EmployeeNumber",
    "DailyRate", "HourlyRate", "MonthlyRate", "PerformanceRating", "Gender",
]
ID_COLUMN = "EmployeeNumber"

# --------------------------------------------------------------------------
# Variables de entrada del modelo
# --------------------------------------------------------------------------
# Valores permitidos de cada variable categórica (los del dataset original)
CATEGORICAL_VALUES: dict[str, list[str]] = {
    "BusinessTravel": ["Non-Travel", "Travel_Rarely", "Travel_Frequently"],
    "Department": ["Human Resources", "Research & Development", "Sales"],
    "EducationField": [
        "Human Resources", "Life Sciences", "Marketing",
        "Medical", "Other", "Technical Degree",
    ],
    "JobRole": [
        "Healthcare Representative", "Human Resources", "Laboratory Technician",
        "Manager", "Manufacturing Director", "Research Director",
        "Research Scientist", "Sales Executive", "Sales Representative",
    ],
    "MaritalStatus": ["Divorced", "Married", "Single"],
    "OverTime": ["No", "Yes"],
}
CATEGORICAL_FEATURES = list(CATEGORICAL_VALUES)

# Rangos válidos (mínimo, máximo) de cada variable numérica. Son algo más
# amplios que los observados en el dataset para aceptar casos reales
# plausibles, pero rechazan errores de digitación (edad 200, sueldo negativo…).
NUMERIC_RANGES: dict[str, tuple[int, int]] = {
    "Age": (18, 70),
    "DistanceFromHome": (0, 100),          # km
    "Education": (1, 5),                   # 1 Sin estudios sup. … 5 Doctorado
    "EnvironmentSatisfaction": (1, 4),     # 1 Baja … 4 Muy alta
    "JobInvolvement": (1, 4),
    "JobLevel": (1, 5),
    "JobSatisfaction": (1, 4),
    "MonthlyIncome": (500, 50_000),        # USD mensuales
    "NumCompaniesWorked": (0, 20),
    "PercentSalaryHike": (0, 50),          # % último aumento
    "RelationshipSatisfaction": (1, 4),
    "StockOptionLevel": (0, 3),
    "TotalWorkingYears": (0, 50),
    "TrainingTimesLastYear": (0, 10),
    "WorkLifeBalance": (1, 4),
    "YearsAtCompany": (0, 50),
    "YearsInCurrentRole": (0, 50),
    "YearsSinceLastPromotion": (0, 50),
    "YearsWithCurrManager": (0, 50),
}
NUMERIC_FEATURES = list(NUMERIC_RANGES)

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Etiquetas en español para mostrar los factores de riesgo
FEATURE_LABELS: dict[str, str] = {
    "Age": "Edad",
    "DistanceFromHome": "Distancia a casa",
    "Education": "Nivel educativo",
    "EnvironmentSatisfaction": "Satisfacción con el entorno",
    "JobInvolvement": "Compromiso con el trabajo",
    "JobLevel": "Nivel del puesto",
    "JobSatisfaction": "Satisfacción con el puesto",
    "MonthlyIncome": "Sueldo mensual",
    "NumCompaniesWorked": "Empresas anteriores",
    "PercentSalaryHike": "Último aumento (%)",
    "RelationshipSatisfaction": "Satisfacción con relaciones laborales",
    "StockOptionLevel": "Nivel de acciones",
    "TotalWorkingYears": "Años de experiencia",
    "TrainingTimesLastYear": "Capacitaciones el último año",
    "WorkLifeBalance": "Equilibrio vida-trabajo",
    "YearsAtCompany": "Años en la empresa",
    "YearsInCurrentRole": "Años en el puesto actual",
    "YearsSinceLastPromotion": "Años desde el último ascenso",
    "YearsWithCurrManager": "Años con el jefe actual",
    "BusinessTravel": "Viajes de trabajo",
    "Department": "Departamento",
    "EducationField": "Área de estudios",
    "JobRole": "Puesto",
    "MaritalStatus": "Estado civil",
    "OverTime": "Horas extra",
}

# --------------------------------------------------------------------------
# Entrenamiento
# --------------------------------------------------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Regresión logística: interpretable (coeficientes = factores de riesgo) y
# class_weight="balanced" compensa el desbalance (~16 % renuncias).
MODEL_PARAMS = {
    "C": 0.5,
    "class_weight": "balanced",
    "max_iter": 2000,
    "random_state": RANDOM_STATE,
}

# Probabilidad a partir de la cual se predice "renuncia"
DECISION_THRESHOLD = 0.5

# Niveles de riesgo según la probabilidad de renuncia
RISK_LEVELS = [  # (límite superior exclusivo, etiqueta)
    (0.30, "Bajo"),
    (0.60, "Medio"),
    (1.01, "Alto"),
]

# Número de factores de riesgo que se devuelven en cada predicción
TOP_FACTORS = 3

# --------------------------------------------------------------------------
# Puerta de calidad: si el modelo no alcanza TODOS estos mínimos en el
# conjunto de prueba, train.py NO lo guarda.
#  - ROC-AUC: capacidad de ordenar empleados por riesgo.
#  - PR-AUC: precisión promedio en la clase minoritaria; la línea base de un
#    modelo al azar es la tasa de renuncia (~0.16).
#  - Recall: de los que realmente renuncian, cuántos detectamos. Para RR.HH.
#    es peor no detectar a quien se va (perder talento) que revisar un caso
#    de más, por eso es la métrica principal.
# --------------------------------------------------------------------------
MIN_ROC_AUC = 0.75
MIN_PR_AUC = 0.40
MIN_RECALL = 0.60

MODEL_VERSION = "0.1.0"
