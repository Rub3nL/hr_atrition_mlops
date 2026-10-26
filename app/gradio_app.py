"""Etapa 5 — Interfaz web (Gradio) que consume la API por HTTP.

La interfaz NO carga el modelo: solo envía los datos a la API.

Local (con la API levantada en otra terminal):
    uv run python app/gradio_app.py   -> http://127.0.0.1:7860
"""

import os

import gradio as gr
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 60  # Render Free puede tardar ~1 min en despertar

# Etiquetas en español -> valor que espera la API
SATISFACCION = [("1 - Baja", 1), ("2 - Media", 2), ("3 - Alta", 3), ("4 - Muy alta", 4)]
EDUCACION = [("1 - Sin estudios superiores", 1), ("2 - Técnico", 2), ("3 - Bachiller", 3),
             ("4 - Maestría", 4), ("5 - Doctorado", 5)]
VIAJES = [("No viaja", "Non-Travel"), ("Viaja rara vez", "Travel_Rarely"),
          ("Viaja frecuentemente", "Travel_Frequently")]
DEPARTAMENTOS = [("Recursos Humanos", "Human Resources"),
                 ("Investigación y Desarrollo", "Research & Development"), ("Ventas", "Sales")]
AREAS = [("Recursos Humanos", "Human Resources"), ("Ciencias de la vida", "Life Sciences"),
         ("Marketing", "Marketing"), ("Medicina", "Medical"), ("Otra", "Other"),
         ("Carrera técnica", "Technical Degree")]
PUESTOS = [("Representante de salud", "Healthcare Representative"),
           ("Recursos Humanos", "Human Resources"),
           ("Técnico de laboratorio", "Laboratory Technician"), ("Gerente", "Manager"),
           ("Director de manufactura", "Manufacturing Director"),
           ("Director de investigación", "Research Director"),
           ("Científico investigador", "Research Scientist"),
           ("Ejecutivo de ventas", "Sales Executive"),
           ("Representante de ventas", "Sales Representative")]
ESTADO_CIVIL = [("Soltero(a)", "Single"), ("Casado(a)", "Married"), ("Divorciado(a)", "Divorced")]
SI_NO = [("No", "No"), ("Sí", "Yes")]

# Orden de los campos = orden de los componentes de entrada
CAMPOS = [
    "Age", "MaritalStatus", "DistanceFromHome", "Education", "EducationField",
    "Department", "JobRole", "JobLevel", "BusinessTravel", "OverTime",
    "MonthlyIncome", "PercentSalaryHike", "StockOptionLevel",
    "TotalWorkingYears", "NumCompaniesWorked", "YearsAtCompany", "YearsInCurrentRole",
    "YearsSinceLastPromotion", "YearsWithCurrManager", "TrainingTimesLastYear",
    "EnvironmentSatisfaction", "JobSatisfaction", "RelationshipSatisfaction",
    "WorkLifeBalance", "JobInvolvement",
]

EJEMPLO_ALTO = [26, "Single", 22, 3, "Marketing", "Sales", "Sales Representative", 1,
                "Travel_Frequently", "Yes", 2200, 11, 0, 4, 4, 1, 0, 0, 0, 1, 1, 1, 2, 1, 2]
EJEMPLO_BAJO = [42, "Married", 3, 3, "Medical", "Research & Development", "Manager", 3,
                "Travel_Rarely", "No", 11000, 18, 1, 18, 1, 12, 7, 2, 7, 3, 4, 4, 4, 3, 3]

COLOR_RIESGO = {"Bajo": "🟢", "Medio": "🟡", "Alto": "🔴"}
RECOMENDACION = {
    "Bajo": "Mantener el seguimiento habitual y el plan de desarrollo.",
    "Medio": "Agendar una conversación 1:1 y revisar carga laboral, crecimiento y clima.",
    "Alto": "Prioridad de retención: entrevista de permanencia, revisar compensación, "
            "horas extra y plan de carrera en las próximas semanas.",
}


def construir_payload(*valores) -> dict:
    payload = dict(zip(CAMPOS, valores))
    for k, v in payload.items():
        if isinstance(v, float) and v.is_integer():
            payload[k] = int(v)
    return payload


def _mensaje_error(response: requests.Response) -> str:
    try:
        detalle = response.json().get("detail")
    except ValueError:
        return response.text[:300]
    if isinstance(detalle, list):  # errores de validación 422 de Pydantic
        return "; ".join(f"{'.'.join(map(str, e.get('loc', [])[1:]))}: {e.get('msg')}" for e in detalle)
    return str(detalle)


def evaluar(*valores):
    payload = construir_payload(*valores)
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=TIMEOUT)
    except requests.RequestException:
        raise gr.Error(f"No se pudo conectar a la API ({API_URL}). ¿Está levantada?")

    if not response.ok:
        raise gr.Error(f"La API respondió {response.status_code}: {_mensaje_error(response)}")

    r = response.json()
    nivel = r["nivel_riesgo"]
    factores = "\n".join(f"- {f['factor']}" for f in r["factores_riesgo"]) or "- Ninguno destacado"
    resumen = (
        f"## {COLOR_RIESGO[nivel]} Riesgo {nivel} — {r['prediccion']}\n"
        f"**Probabilidad de renuncia:** {r['probabilidad_renuncia']:.1%} "
        f"(umbral de decisión {r['umbral']:.0%})\n\n"
        f"**Factores que más aumentan el riesgo:**\n{factores}\n\n"
        f"**Acción sugerida:** {RECOMENDACION[nivel]}"
    )
    return r["probabilidades"], resumen


def info_modelo() -> str:
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=TIMEOUT)
        response.raise_for_status()
    except requests.RequestException:
        return "⚠️ No se pudo obtener la información del modelo desde la API."
    m = response.json()
    met = m["metrics"]
    return (
        f"**Modelo:** {m['model_type']} v{m['model_version']} — entrenado el {m['trained_at']}\n\n"
        f"**Datos:** {m['data']['rows']} empleados, tasa de rotación {m['data']['attrition_rate']:.1%}\n\n"
        f"| Métrica (test) | Valor |\n|---|---|\n"
        f"| ROC-AUC | {met['roc_auc']:.3f} |\n| PR-AUC | {met['pr_auc']:.3f} |\n"
        f"| Recall | {met['recall']:.3f} |\n| Precisión | {met['precision']:.3f} |\n"
        f"| F1 | {met['f1']:.3f} |\n"
    )


with gr.Blocks(title="Predicción de rotación de personal") as demo:
    gr.Markdown(
        "# 👥 Predicción de rotación de personal (HR Attrition)\n"
        "Completa el perfil del colaborador y obtén la probabilidad de que deje la "
        "empresa, su nivel de riesgo y los factores que más influyen. "
        "Herramienta de **apoyo** a la gestión del talento."
    )
    with gr.Row():
        with gr.Column(scale=3):
            with gr.Tab("Personal"):
                age = gr.Slider(18, 70, value=35, step=1, label="Edad")
                marital = gr.Dropdown(ESTADO_CIVIL, value="Married", label="Estado civil")
                distance = gr.Slider(0, 100, value=7, step=1, label="Distancia de casa al trabajo (km)")
                education = gr.Dropdown(EDUCACION, value=3, label="Nivel educativo")
                edu_field = gr.Dropdown(AREAS, value="Life Sciences", label="Área de estudios")
            with gr.Tab("Puesto"):
                department = gr.Dropdown(DEPARTAMENTOS, value="Research & Development", label="Departamento")
                job_role = gr.Dropdown(PUESTOS, value="Research Scientist", label="Puesto")
                job_level = gr.Slider(1, 5, value=2, step=1, label="Nivel del puesto (1 junior – 5 directivo)")
                travel = gr.Dropdown(VIAJES, value="Travel_Rarely", label="Viajes de trabajo")
                overtime = gr.Radio(SI_NO, value="No", label="¿Hace horas extra?")
            with gr.Tab("Compensación"):
                income = gr.Number(value=5000, minimum=500, maximum=50000, precision=0, label="Sueldo mensual (USD)")
                hike = gr.Slider(0, 50, value=14, step=1, label="Último aumento salarial (%)")
                stock = gr.Slider(0, 3, value=1, step=1, label="Nivel de opciones sobre acciones (0–3)")
            with gr.Tab("Trayectoria"):
                total_years = gr.Slider(0, 50, value=10, step=1, label="Años totales de experiencia")
                num_companies = gr.Slider(0, 20, value=2, step=1, label="Empresas anteriores")
                years_company = gr.Slider(0, 50, value=5, step=1, label="Años en la empresa")
                years_role = gr.Slider(0, 50, value=3, step=1, label="Años en el puesto actual")
                years_promo = gr.Slider(0, 50, value=1, step=1, label="Años desde el último ascenso")
                years_manager = gr.Slider(0, 50, value=3, step=1, label="Años con el jefe actual")
                trainings = gr.Slider(0, 10, value=3, step=1, label="Capacitaciones el último año")
            with gr.Tab("Clima laboral"):
                env_sat = gr.Dropdown(SATISFACCION, value=3, label="Satisfacción con el entorno")
                job_sat = gr.Dropdown(SATISFACCION, value=3, label="Satisfacción con el puesto")
                rel_sat = gr.Dropdown(SATISFACCION, value=3, label="Satisfacción con relaciones laborales")
                wlb = gr.Dropdown(SATISFACCION, value=3, label="Equilibrio vida-trabajo")
                involvement = gr.Dropdown(SATISFACCION, value=3, label="Compromiso con el trabajo")

            boton = gr.Button("Evaluar riesgo de rotación", variant="primary")

        with gr.Column(scale=2):
            probabilidades = gr.Label(num_top_classes=2, label="Probabilidades")
            resumen = gr.Markdown()
            with gr.Accordion("Acerca del modelo", open=False):
                info = gr.Markdown("Pulsa el botón para consultar la ficha del modelo.")
                boton_info = gr.Button("Ver ficha del modelo")

    entradas = [
        age, marital, distance, education, edu_field,
        department, job_role, job_level, travel, overtime,
        income, hike, stock,
        total_years, num_companies, years_company, years_role,
        years_promo, years_manager, trainings,
        env_sat, job_sat, rel_sat, wlb, involvement,
    ]
    gr.Examples(
        examples=[EJEMPLO_ALTO, EJEMPLO_BAJO],
        inputs=entradas,
        label="Perfiles de ejemplo (1: riesgo alto · 2: riesgo bajo)",
    )
    boton.click(evaluar, inputs=entradas, outputs=[probabilidades, resumen])
    boton_info.click(info_modelo, outputs=info)


if __name__ == "__main__":
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        # Render asigna el puerto en la variable PORT
        server_port=int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860"))),
    )
