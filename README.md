# 👥 HR Attrition MLOps — Predicción de rotación de personal

Proyecto de Machine Learning de punta a punta para **Recursos Humanos y Gestión del Capital Humano**:
predice la **probabilidad de que un empleado deje la empresa** (rotación / *attrition*), su
**nivel de riesgo** y los **factores** que más lo explican.

`datos → entrenamiento → API (FastAPI) → interfaz web (Gradio) → contenedor (Docker) → CI/CD (GitHub Actions) → nube (Render)`

Sigue la misma estructura que el proyecto de clase `iris_mlops`. Cambian el problema, los datos y el modelo.

---

## 1. Problema

| | |
|---|---|
| **¿Qué se predice?** | Si un colaborador renunciará (`Attrition = Yes/No`) y con qué probabilidad. |
| **¿Para quién?** | Equipos de RR.HH., *HR Business Partners* y jefaturas que necesitan **priorizar acciones de retención** (entrevistas de permanencia, revisión salarial, plan de carrera). |
| **¿Por qué importa?** | Reemplazar a un colaborador cuesta entre el 50 % y el 200 % de su sueldo anual (selección, inducción, curva de aprendizaje). Detectar el riesgo a tiempo reduce ese costo. |
| **Tipo de problema** | Clasificación binaria **desbalanceada** (~16 % de renuncias). |

## 2. Datos

**IBM HR Analytics Employee Attrition & Performance**: 1 470 empleados y 35 columnas (edad, puesto,
sueldo, horas extra, satisfacción, antigüedad, viajes, etc.). Lo crearon científicos de datos de IBM
y es el dataset de referencia en *People Analytics*.

- Fuente original (Kaggle): <https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset>
- El CSV ya viene incluido en `data/raw/hr_employee_attrition.csv` (igual que `iris.csv` en iris_mlops),
  así que el proyecto funciona sin internet y Render no depende de ninguna descarga.
- Si borras el archivo, `uv run preparar-datos` lo vuelve a descargar desde espejos públicos en GitHub
  (ver `DATA_URLS` en `config.py`). También acepta el CSV original de Kaggle
  (`WA_Fn-UseC_-HR-Employee-Attrition.csv`) si lo copias en `data/raw/`.

**Validación en código** (`validate_data` en `data.py`): columnas obligatorias, número mínimo de filas,
nulos, tipos numéricos, rangos válidos por variable, categorías permitidas, clases del objetivo y
empleados duplicados.

**Variables usadas (25):** se descartan las constantes (`EmployeeCount`, `Over18`, `StandardHours`),
el identificador (`EmployeeNumber`), las tarifas sin relación con el sueldo (`DailyRate`, `HourlyRate`,
`MonthlyRate`) y la casi constante `PerformanceRating`. **`Gender` se excluye a propósito por equidad.**

## 3. Modelo y métricas

- **Modelo:** `Pipeline(StandardScaler + OneHotEncoder → LogisticRegression(class_weight="balanced"))`.
  - Es interpretable: sus coeficientes son "factores de riesgo" explicables a RR.HH.
  - `class_weight="balanced"` compensa el desbalance de clases.
  - El preprocesamiento va dentro del pipeline, así no hay fuga de datos del test.
- **Separación:** train/test 80/20 **estratificada** con semilla fija (`RANDOM_STATE = 42`) + validación cruzada de 5 folds.
- **Métricas** (no se usa accuracy como métrica principal porque con 84 % de "No" un modelo que siempre dice "No" tiene 84 % de accuracy):
  - **Recall de la clase "renuncia"** (principal): de los que realmente se van, ¿a cuántos detectamos? Para RR.HH. cuesta más no detectar a quien se va que revisar un caso de más.
  - **PR-AUC**: calidad en la clase minoritaria (al azar ≈ 0.16).
  - **ROC-AUC**: capacidad de ordenar a los empleados por riesgo.
- **Puerta de calidad** (`config.py`): el modelo **solo se guarda** si `ROC-AUC ≥ 0.75`, `PR-AUC ≥ 0.40` y `Recall ≥ 0.60`. Si no, `uv run entrenar` falla, y con él fallan el CI y el build de Docker.
- **Ficha del modelo:** `models/metrics.json` guarda versión, fecha, parámetros, métricas, validación cruzada, puerta de calidad, importancia de variables y limitaciones.

**Resultados** (test de 294 empleados, semilla 42):

| Métrica | Valor | Mínimo exigido |
|---|---|---|
| ROC-AUC | 0.811 (CV 5 folds: 0.834 ± 0.026) | 0.75 |
| PR-AUC | 0.580 (al azar: 0.160) | 0.40 |
| Recall (renuncia) | 0.660 → detecta 31 de 47 renuncias | 0.60 |
| Precisión | 0.365 | — |
| F1 | 0.470 | — |

Los factores que más elevan el riesgo son los puestos de técnico de laboratorio y representante de
ventas, viajar con frecuencia, hacer horas extra, llevar años sin ascenso y ser soltero(a)
(ver `feature_importance` en `models/metrics.json`).

## 4. Estructura del repositorio

```
hr_attrition_mlops/
├── src/hr_attrition_mlops/
│   ├── config.py      # Configuración central: rutas, columnas, rangos, hiperparámetros, umbrales
│   ├── data.py        # Descarga, limpieza, validación y split estratificado
│   ├── train.py       # Entrenamiento, CV, evaluación, puerta de calidad, ficha del modelo
│   ├── predict.py     # Carga del modelo (una vez, en caché) + predicción + factores de riesgo
│   └── api.py         # FastAPI: GET /, GET /model-info, POST /predict (validación Pydantic)
├── app/gradio_app.py  # Interfaz web que consume la API por HTTP
├── tests/             # pytest: datos, modelo y API
├── data/raw/          # hr_employee_attrition.csv (dataset IBM, incluido)
├── models/            # Modelo .joblib + metrics.json (se generan con `uv run entrenar`)
├── .github/workflows/ # ci.yml (pruebas) y cd.yml (despliegue en Render)
├── Dockerfile, compose.yaml, .dockerignore
└── pyproject.toml, uv.lock, .python-version
```

## 5. Ejecutar desde cero (local, con `uv`)

Requisitos: [uv](https://docs.astral.sh/uv/getting-started/installation/) y Git. uv instala Python 3.14 por ti.

```bash
# 0) Abrir la carpeta en VS Code
code .

# 1) Instalar dependencias (la primera vez genera uv.lock: súbelo a GitHub)
uv sync

# 2) Descargar y validar los datos
uv run preparar-datos

# 3) Entrenar (con puerta de calidad) -> models/attrition_model.joblib + models/metrics.json
uv run entrenar

# 4) Probar una predicción desde la terminal
uv run predecir

# 5) Pruebas automáticas
uv run pytest -v
```

**Levantar la API y la interfaz** (dos terminales):

```bash
# Terminal 1 – API  -> http://127.0.0.1:8000/docs
uv run fastapi dev src/hr_attrition_mlops/api.py

# Terminal 2 – Interfaz  -> http://127.0.0.1:7860
uv run python app/gradio_app.py
```

### Ejemplo de llamada a la API

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{
  "Age": 26, "MaritalStatus": "Single", "DistanceFromHome": 22, "Education": 3,
  "EducationField": "Marketing", "Department": "Sales", "JobRole": "Sales Representative",
  "JobLevel": 1, "BusinessTravel": "Travel_Frequently", "OverTime": "Yes",
  "MonthlyIncome": 2200, "PercentSalaryHike": 11, "StockOptionLevel": 0,
  "TotalWorkingYears": 4, "NumCompaniesWorked": 4, "YearsAtCompany": 1,
  "YearsInCurrentRole": 0, "YearsSinceLastPromotion": 0, "YearsWithCurrManager": 0,
  "TrainingTimesLastYear": 1, "EnvironmentSatisfaction": 1, "JobSatisfaction": 1,
  "RelationshipSatisfaction": 2, "WorkLifeBalance": 1, "JobInvolvement": 2
}'
```

Respuesta (los valores dependen del entrenamiento):

```json
{
  "attrition": "Yes",
  "prediccion": "Renuncia probable",
  "probabilidad_renuncia": 0.93,
  "nivel_riesgo": "Alto",
  "umbral": 0.5,
  "probabilidades": {"Renuncia probable": 0.93, "Permanencia probable": 0.07},
  "factores_riesgo": [{"factor": "Horas extra = Yes", "impacto": 0.84}, "..."]
}
```

| Caso | Código |
|---|---|
| Datos válidos | `200` |
| Fuera de rango (`Age: 150`), categoría inválida (`OverTime: "Tal vez"`), campo faltante o extra, años incoherentes (`YearsInCurrentRole > YearsAtCompany`) | `422` |
| Modelo no entrenado | `503` |

## 6. Docker

```bash
docker compose up --build
# API       -> http://localhost:8000/docs
# Interfaz  -> http://localhost:7860
```

El `Dockerfile` instala las dependencias por capas con `uv sync --locked`, descarga los datos y **entrena
dentro del build** respetando la puerta de calidad, y corre como usuario sin privilegios (`appuser`).
La misma imagen sirve para la API y para la interfaz.

## 7. Subir a GitHub

```bash
git init
git add .
git commit -m "Proyecto HR Attrition MLOps"
git branch -M main
git remote add origin git@github.com:Rub3nL/hr_atrition_mlops.git
git push -u origin main
```

> ⚠️ Antes del primer push ejecuta `uv sync` para que exista **`uv.lock`** y súbelo. El CI y el
> Dockerfile usan `--locked` y fallan sin ese archivo.

Cada push a `main` ejecuta **CI** (`.github/workflows/ci.yml`): `uv sync --locked` → `preparar-datos` →
`entrenar` → `pytest`, y guarda el modelo como artefacto. Si el CI pasa, el **CD** (`cd.yml`) llama a los
*Deploy Hooks* de Render.

## 8. Desplegar en Render (URL pública)

1. Entra a <https://render.com> e inicia sesión con GitHub.
2. **New → Web Service** → elige tu repositorio → *Language/Runtime*: **Docker** → plan **Free**.
   - Nombre: `hr-attrition-api`. No hace falta comando: usa el `CMD` del Dockerfile.
   - *Health Check Path*: `/`.
3. Crea un **segundo Web Service** con el mismo repositorio (Docker, Free):
   - Nombre: `hr-attrition-ui`.
   - *Docker Command*: `python app/gradio_app.py`
   - Variables de entorno: `API_URL = https://hr-attrition-api.onrender.com` (la URL de tu API) y `GRADIO_SERVER_NAME = 0.0.0.0`.
4. En cada servicio: **Settings → Auto-Deploy: Off** y copia su **Deploy Hook** (URL secreta).
5. En GitHub: **Settings → Secrets and variables → Actions → New repository secret**:
   - `RENDER_DEPLOY_API` = deploy hook de la API
   - `RENDER_DEPLOY_UI` = deploy hook de la interfaz
6. Haz un push a `main`. Si el CI queda en verde, el CD despliega los dos servicios.

> Los deploy hooks **nunca** se escriben en el código, solo en *Secrets* de GitHub.
> Render Free "duerme" tras 15 min sin uso: la primera petición puede tardar ~1 min.
> Para la demo ten listo `docker compose up` como respaldo.

## 9. Limitaciones y mejoras

- El dataset es **ficticio** (IBM). Antes de usarlo en una empresa real hay que reentrenar con datos propios.
- La predicción es **apoyo a la decisión**, no un criterio para despedir ni sancionar. Requiere supervisión humana y cumplir la Ley N.º 29733 de Protección de Datos Personales (Perú).
- Mejoras posibles: registro de experimentos con **MLflow**, comparación con *gradient boosting*,
  calibración del umbral según el costo de RR.HH., monitoreo de **data drift**, auditoría de **equidad**
  por grupo y predicción por lotes (CSV de toda la planilla).

## 10. Guion breve para la presentación

1. **Problema** (1 min): rotación de personal, costo, usuario = RR.HH.
2. **Datos** (1 min): IBM HR, 1 470 filas, 16 % de renuncias, validación en `data.py`.
3. **Modelo** (2 min): regresión logística balanceada. Por qué Recall/PR-AUC y no accuracy. Puerta de calidad. `metrics.json`.
4. **Arquitectura** (1 min): datos → train → API → Gradio → Docker → GitHub Actions → Render.
5. **Demo** (3 min): URL pública. Perfil de riesgo alto vs. bajo, error 422 en `/docs`, pestaña *Actions* en verde.
6. **Cierre**: limitaciones éticas y mejoras (MLflow, drift, equidad).
