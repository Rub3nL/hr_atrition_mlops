# Imagen única para la API y la interfaz (cambia solo el comando).
FROM python:3.14-slim

COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /uvx /bin/

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# 1) Dependencias (capa cacheada mientras no cambien pyproject.toml / uv.lock)
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# 2) Código del proyecto
COPY . .
RUN uv sync --locked --no-dev

ENV PATH="/app/.venv/bin:$PATH"

# 3) Datos + entrenamiento. Si la puerta de calidad falla, el build falla.
RUN preparar-datos && entrenar

# 4) Usuario sin privilegios
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000 7860

CMD ["sh", "-c", "exec fastapi run src/hr_attrition_mlops/api.py --host 0.0.0.0 --port ${PORT:-8000}"]
