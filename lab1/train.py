"""Lab 1 · Entrena un modelo de baja de clientes y registra el experimento en MLflow.

Ejemplos:
  python lab1/train.py
  python lab1/train.py --tipo logreg --C 0.01 --class-weight balanced
  python lab1/train.py --tipo rf --n-estimators 300 --max-depth 8
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

from churn import cargar, crear_modelo, metricas, separar, setup_mlflow

p = argparse.ArgumentParser()
p.add_argument("--tipo", choices=["logreg", "rf"], default="logreg")
p.add_argument("--C", type=float, default=1.0)
p.add_argument("--n-estimators", type=int, default=200)
p.add_argument("--max-depth", type=int, default=None)
p.add_argument("--class-weight", choices=["balanced"], default=None)
args = p.parse_args()

setup_mlflow("lab1-churn")
X, y = separar(cargar())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

params = {"tipo": args.tipo, "class_weight": args.class_weight}
params.update({"C": args.C} if args.tipo == "logreg" else
              {"n_estimators": args.n_estimators, "max_depth": args.max_depth})

# El nombre incluye los hiperparámetros para distinguir los runs en la vista Compare
hiper = f"C{args.C}" if args.tipo == "logreg" else f"n{args.n_estimators}-depth{args.max_depth or 'max'}"
with mlflow.start_run(run_name=f"{args.tipo}-{args.class_weight or 'sin-pesos'}-{hiper}") as run:
    modelo = crear_modelo(**params)
    modelo.fit(X_train, y_train)
    resultado = metricas(modelo, X_test, y_test)

    mlflow.log_params(params)
    mlflow.log_param("filas_entrenamiento", len(X_train))
    mlflow.log_metrics(resultado)
    # Los árboles se serializan con tipos que MLflow pide declarar como confiables (el modelo es nuestro)
    confiables = ["sklearn.tree._tree.Tree"] if args.tipo == "rf" else None
    mlflow.sklearn.log_model(modelo, name="modelo", input_example=X_test.head(3),
                             signature=infer_signature(X_test, modelo.predict(X_test)),
                             skops_trusted_types=confiables)

    print(f"Run {run.info.run_id}")
    for k, v in resultado.items():
        print(f"  {k:<10} {v:.3f}")
