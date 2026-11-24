"""Código compartido por los laboratorios: carga de datos, pipeline del modelo y métricas."""
import os
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
# Con Docker se usa el servidor de MLflow del compose; sin Docker, una base de datos local
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")
MODEL_NAME = "churn-nova-stream"

NUMERICAS = ["cuota_mensual", "antiguedad_meses", "horas_semana", "dispositivos",
             "incidencias_90d", "dias_sin_acceso", "descuento_activo"]
CATEGORICAS = ["plan", "metodo_pago"]
OBJETIVO = "churn"


def setup_mlflow(experimento):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experimento)


def cargar(nombre="clientes_historico.csv"):
    return pd.read_csv(DATA / nombre)


def separar(df):
    # Las numéricas como float: así el esquema del modelo admite valores nulos en producción
    X = df[NUMERICAS + CATEGORICAS].astype({c: float for c in NUMERICAS})
    return X, df[OBJETIVO]


def crear_modelo(tipo="logreg", **params):
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERICAS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
    ])
    if tipo == "logreg":
        clf = LogisticRegression(max_iter=1000, C=params.get("C", 1.0),
                                 class_weight=params.get("class_weight"))
    elif tipo == "rf":
        clf = RandomForestClassifier(n_estimators=params.get("n_estimators", 200),
                                     max_depth=params.get("max_depth"),
                                     class_weight=params.get("class_weight"), random_state=42)
    else:
        raise ValueError(f"Tipo de modelo no soportado: {tipo}")
    return Pipeline([("prep", prep), ("clf", clf)])


def metricas(modelo, X, y):
    pred = modelo.predict(X)
    prob = modelo.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, prob),
    }
