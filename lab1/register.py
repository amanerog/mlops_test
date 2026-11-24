"""Lab 1 · Busca el mejor run del experimento y lo registra como 'champion' en el Model Registry.

Uso: python lab1/register.py --metrica recall
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from mlflow import MlflowClient

from churn import MODEL_NAME, setup_mlflow

p = argparse.ArgumentParser()
p.add_argument("--metrica", default="recall", help="métrica para elegir el mejor run")
args = p.parse_args()

setup_mlflow("lab1-churn")
runs = mlflow.search_runs(order_by=[f"metrics.{args.metrica} DESC"], max_results=1)
if runs.empty:
    sys.exit("No hay runs en el experimento: ejecuta antes lab1/train.py")

mejor = runs.iloc[0]
print(f"Mejor run por {args.metrica}: {mejor['tags.mlflow.runName']} ({mejor[f'metrics.{args.metrica}']:.3f})")

# En MLflow 3 el modelo es una entidad propia (LoggedModel) vinculada al run que lo creó
modelo = mlflow.search_logged_models(filter_string=f"source_run_id = '{mejor.run_id}'",
                                     output_format="list")[0]
version = mlflow.register_model(modelo.model_uri, MODEL_NAME)
MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", version.version)
print(f"Registrado {MODEL_NAME} v{version.version} con el alias 'champion'")
