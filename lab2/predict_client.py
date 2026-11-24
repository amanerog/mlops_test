"""Lab 2 · Cliente que llama al modelo servido como API REST.

Antes, en otra terminal:
  mlflow models serve -m "models:/churn-nova-stream@champion" -p 5001 --env-manager local
(con MLFLOW_TRACKING_URI apuntando a la base de datos del repo; ver lab2/README.md)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from churn import cargar, separar

X, _ = separar(cargar("produccion_mes1.csv"))
muestra = X.head(5)

resp = requests.post(
    os.environ.get("MODEL_URL", "http://127.0.0.1:5001") + "/invocations",
    json={"dataframe_split": muestra.to_dict(orient="split")},
    timeout=30,
)
resp.raise_for_status()
for (_, fila), pred in zip(muestra.iterrows(), resp.json()["predictions"]):
    estado = "RIESGO DE BAJA" if pred == 1 else "ok"
    print(f"plan={fila.plan:<9} antigüedad={fila.antiguedad_meses:>4.0f}m  horas/sem={fila.horas_semana:>5.1f}  -> {estado}")
