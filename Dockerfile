# Imagen de trabajo de los laboratorios: Python 3.11 con todas las dependencias.
# La usan el contenedor 'lab' y el servidor de MLflow.
FROM python:3.11-slim

# git: MLflow guarda en cada run el commit del código (reproducibilidad).
# safe.directory: la carpeta montada pertenece a otro usuario del sistema anfitrión.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && git config --system --add safe.directory '*'

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ENV PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1
