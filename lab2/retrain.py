"""Lab 2 · Reentrena con datos recientes y compara con el champion antes de promocionar.

El candidato solo sustituye al champion si, en el mes más reciente, mejora el AUC sin empeorar el recall.
Uso: python lab2/retrain.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

from churn import MODEL_NAME, cargar, crear_modelo, metricas, separar, setup_mlflow

setup_mlflow("lab2-monitorizacion")
reciente = cargar("produccion_mes2.csv")
# Reservamos parte del mes reciente para comparar los dos modelos con el mismo dato
reciente_train, reciente_test = train_test_split(reciente, test_size=0.5, stratify=reciente.churn, random_state=42)
X_test, y_test = separar(reciente_test)

champion = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
m_champion = metricas(champion, X_test, y_test)

with mlflow.start_run(run_name="reentreno-con-mes2") as run:
    X_train, y_train = separar(pd.concat([cargar(), reciente_train]))
    candidato = crear_modelo("logreg", class_weight="balanced").fit(X_train, y_train)
    m_candidato = metricas(candidato, X_test, y_test)
    mlflow.log_metrics(m_candidato)
    info = mlflow.sklearn.log_model(candidato, name="modelo", input_example=X_test.head(3),
                             signature=infer_signature(X_test, candidato.predict(X_test)))

print(f"{'métrica':<10}{'champion':>10}{'candidato':>11}")
for k in m_champion:
    print(f"{k:<10}{m_champion[k]:>10.3f}{m_candidato[k]:>11.3f}")

if m_candidato["roc_auc"] > m_champion["roc_auc"] and m_candidato["recall"] >= m_champion["recall"]:
    v = mlflow.register_model(info.model_uri, MODEL_NAME)
    MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", v.version)
    print(f"\nEl candidato mejora: promocionado como champion (v{v.version}).")
else:
    print("\nEl candidato no mejora al champion: no se promociona.")
