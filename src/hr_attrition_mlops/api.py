"""Etapa 4 — API REST con FastAPI.

Local:
    uv run fastapi dev src/hr_attrition_mlops/api.py
    -> http://127.0.0.1:8000/docs
"""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator

from hr_attrition_mlops import __version__
from hr_attrition_mlops.config import CATEGORICAL_VALUES as CV
from hr_attrition_mlops.config import NUMERIC_RANGES as NR
from hr_attrition_mlops.predict import (
    EXAMPLE_EMPLOYEE,
    load_model,
    load_model_info,
    predict,
)


def rango(col: str, descripcion: str):
    """Field con el rango definido en config.py (una sola fuente de verdad)."""
    minimo, maximo = NR[col]
    return Field(ge=minimo, le=maximo, description=f"{descripcion} [{minimo}–{maximo}]")


class EmployeeInput(BaseModel):
    """Datos de un empleado. Los nombres coinciden con las columnas del dataset."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [EXAMPLE_EMPLOYEE]},
    )

    # Datos personales y formación
    Age: int = rango("Age", "Edad en años")
    MaritalStatus: Literal[tuple(CV["MaritalStatus"])] = Field(description="Estado civil")
    DistanceFromHome: int = rango("DistanceFromHome", "Distancia de casa al trabajo (km)")
    Education: int = rango("Education", "1 Sin estudios sup., 2 Técnico, 3 Bachiller, 4 Maestría, 5 Doctorado")
    EducationField: Literal[tuple(CV["EducationField"])] = Field(description="Área de estudios")

    # Puesto
    Department: Literal[tuple(CV["Department"])] = Field(description="Departamento")
    JobRole: Literal[tuple(CV["JobRole"])] = Field(description="Puesto")
    JobLevel: int = rango("JobLevel", "Nivel jerárquico del puesto")
    BusinessTravel: Literal[tuple(CV["BusinessTravel"])] = Field(description="Frecuencia de viajes")
    OverTime: Literal[tuple(CV["OverTime"])] = Field(description="¿Hace horas extra?")

    # Compensación
    MonthlyIncome: int = rango("MonthlyIncome", "Sueldo mensual (USD)")
    PercentSalaryHike: int = rango("PercentSalaryHike", "Último aumento salarial (%)")
    StockOptionLevel: int = rango("StockOptionLevel", "Nivel de opciones sobre acciones")

    # Trayectoria
    TotalWorkingYears: int = rango("TotalWorkingYears", "Años totales de experiencia")
    NumCompaniesWorked: int = rango("NumCompaniesWorked", "Empresas en las que trabajó antes")
    YearsAtCompany: int = rango("YearsAtCompany", "Años en la empresa")
    YearsInCurrentRole: int = rango("YearsInCurrentRole", "Años en el puesto actual")
    YearsSinceLastPromotion: int = rango("YearsSinceLastPromotion", "Años desde el último ascenso")
    YearsWithCurrManager: int = rango("YearsWithCurrManager", "Años con el jefe actual")
    TrainingTimesLastYear: int = rango("TrainingTimesLastYear", "Capacitaciones el último año")

    # Encuestas de clima (1 = Baja … 4 = Muy alta)
    EnvironmentSatisfaction: int = rango("EnvironmentSatisfaction", "Satisfacción con el entorno")
    JobSatisfaction: int = rango("JobSatisfaction", "Satisfacción con el puesto")
    RelationshipSatisfaction: int = rango("RelationshipSatisfaction", "Satisfacción con relaciones laborales")
    WorkLifeBalance: int = rango("WorkLifeBalance", "Equilibrio vida-trabajo")
    JobInvolvement: int = rango("JobInvolvement", "Compromiso con el trabajo")

    @model_validator(mode="after")
    def coherencia_de_anios(self):
        """Reglas de negocio entre campos (si no se cumplen -> error 422)."""
        if self.TotalWorkingYears > self.Age - 14:
            raise ValueError("TotalWorkingYears no puede ser mayor que Age - 14")
        if self.YearsAtCompany > self.TotalWorkingYears:
            raise ValueError("YearsAtCompany no puede ser mayor que TotalWorkingYears")
        for campo in ("YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"):
            if getattr(self, campo) > self.YearsAtCompany:
                raise ValueError(f"{campo} no puede ser mayor que YearsAtCompany")
        return self


class RiskFactor(BaseModel):
    factor: str
    impacto: float


class PredictionOutput(BaseModel):
    attrition: Literal["Yes", "No"]
    prediccion: str
    probabilidad_renuncia: float = Field(ge=0, le=1)
    nivel_riesgo: Literal["Bajo", "Medio", "Alto"]
    umbral: float
    probabilidades: dict[str, float]
    factores_riesgo: list[RiskFactor]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Precarga el modelo al iniciar (si existe) para que la 1.ª petición sea rápida
    try:
        load_model()
    except FileNotFoundError:
        pass
    yield


app = FastAPI(
    title="HR Attrition API",
    description=(
        "Predice la probabilidad de que un empleado deje la empresa (rotación) "
        "a partir de su perfil laboral. Dataset: IBM HR Analytics Employee Attrition."
    ),
    version=__version__,
    lifespan=lifespan,
)


@app.get("/")
def health():
    try:
        load_model()
        cargado = True
    except FileNotFoundError:
        cargado = False
    return {
        "status": "ok",
        "message": "HR Attrition API operativa",
        "model_loaded": cargado,
        "version": __version__,
    }


@app.get("/model-info")
def model_info():
    try:
        return load_model_info()
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Modelo no encontrado o no entrenado")


@app.post("/predict", response_model=PredictionOutput)
def predict_attrition(employee: EmployeeInput):
    try:
        return predict(employee.model_dump())
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Modelo no encontrado o no entrenado")
