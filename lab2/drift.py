"""Lab 2 · Detecta drift comparando los datos de producción con los de entrenamiento.

Para cada variable numérica calcula:
  - PSI (Population Stability Index): < 0.1 estable, 0.1-0.25 vigilar, > 0.25 drift
  - Test de Kolmogorov-Smirnov: p-valor < 0.05 indica que las distribuciones difieren
Y evalúa el modelo 'champion' con las etiquetas reales del mes.

Uso: python lab2/drift.py produccion_mes1.csv
     python lab2/drift.py produccion_mes2.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import numpy as np
from scipy.stats import ks_2samp

from churn import MODEL_NAME, NUMERICAS, cargar, metricas, separar, setup_mlflow


def psi(referencia, actual, bins=10):
    # Mismos cortes para las dos muestras, sobre el rango conjunto
    cortes = np.histogram_bin_edges(np.concatenate([referencia, actual]), bins=bins)
    ref = np.histogram(referencia, cortes)[0] / len(referencia)
    act = np.histogram(actual, cortes)[0] / len(actual)
    ref, act = np.clip(ref, 1e-4, None), np.clip(act, 1e-4, None)
    return float(np.sum((act - ref) * np.log(act / ref)))


fichero = sys.argv[1] if len(sys.argv) > 1 else "produccion_mes2.csv"
setup_mlflow("lab2-monitorizacion")
ref, prod = cargar(), cargar(fichero)

with mlflow.start_run(run_name=f"drift-{fichero.removesuffix('.csv')}"):
    print(f"\nDrift de datos: entrenamiento vs {fichero}\n")
    print(f"{'variable':<18}{'PSI':>8}{'KS p-valor':>13}  estado")
    alertas = 0
    for col in NUMERICAS:
        v_psi = psi(ref[col], prod[col])
        p_ks = ks_2samp(ref[col], prod[col]).pvalue
        estado = "DRIFT" if v_psi > 0.25 else "vigilar" if v_psi > 0.1 else "estable"
        alertas += estado == "DRIFT"
        mlflow.log_metric(f"psi_{col}", v_psi)
        print(f"{col:<18}{v_psi:>8.3f}{p_ks:>13.4f}  {estado}")

    print(f"\nTasa de baja real: entrenamiento {ref.churn.mean():.1%} -> {fichero} {prod.churn.mean():.1%}")

    modelo = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    X, y = separar(prod)
    m = metricas(modelo, X, y)
    mlflow.log_metrics({f"prod_{k}": v for k, v in m.items()})
    mlflow.log_metric("variables_con_drift", alertas)
    print("\nRendimiento del modelo champion con las etiquetas reales del mes:")
    for k, v in m.items():
        print(f"  {k:<10} {v:.3f}")
    if alertas:
        print(f"\n⚠ {alertas} variable(s) con drift: revisar y valorar reentrenar el modelo.")
