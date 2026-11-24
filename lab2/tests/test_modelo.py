"""Lab 2 · Tests de modelo: un modelo solo se promociona si supera los umbrales acordados con negocio."""
import pytest
from sklearn.model_selection import train_test_split

from churn import cargar, crear_modelo, metricas, separar

# Umbrales acordados con negocio: preferimos detectar bajas (recall) aunque haya falsos positivos
RECALL_MINIMO = 0.60
AUC_MINIMO = 0.75


@pytest.fixture(scope="module")
def resultado():
    X, y = separar(cargar())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    modelo = crear_modelo("logreg", class_weight="balanced").fit(X_train, y_train)
    return modelo, metricas(modelo, X_test, y_test), X_test


def test_recall_minimo(resultado):
    _, m, _ = resultado
    assert m["recall"] >= RECALL_MINIMO, f"Recall {m['recall']:.3f} < {RECALL_MINIMO}"


def test_auc_minimo(resultado):
    _, m, _ = resultado
    assert m["roc_auc"] >= AUC_MINIMO, f"AUC {m['roc_auc']:.3f} < {AUC_MINIMO}"


def test_predicciones_validas(resultado):
    modelo, _, X_test = resultado
    prob = modelo.predict_proba(X_test)[:, 1]
    assert ((prob >= 0) & (prob <= 1)).all()


def test_invariancia_cliente_id(resultado):
    # El identificador no debe influir: el modelo ni siquiera debería recibirlo
    modelo, _, X_test = resultado
    assert "cliente_id" not in X_test.columns
