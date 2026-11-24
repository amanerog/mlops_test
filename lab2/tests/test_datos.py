"""Lab 2 · Tests de datos: si el dato de entrada está mal, no se entrena."""
import pytest

from churn import CATEGORICAS, NUMERICAS, OBJETIVO, cargar


@pytest.fixture(scope="module")
def df():
    return cargar()


def test_columnas_esperadas(df):
    faltan = set(NUMERICAS + CATEGORICAS + [OBJETIVO]) - set(df.columns)
    assert not faltan, f"Faltan columnas: {faltan}"


def test_sin_nulos(df):
    nulos = df[NUMERICAS + CATEGORICAS].isna().sum()
    assert nulos.sum() == 0, f"Columnas con nulos:\n{nulos[nulos > 0]}"


def test_rangos_validos(df):
    assert df.cuota_mensual.between(0, 50).all(), "Cuota fuera de rango"
    assert (df.antiguedad_meses >= 0).all(), "Antigüedad negativa"
    assert (df.horas_semana >= 0).all(), "Horas de uso negativas"


def test_categorias_conocidas(df):
    assert set(df.plan) <= {"basico", "estandar", "premium"}
    assert set(df.metodo_pago) <= {"tarjeta", "paypal", "domiciliacion"}


def test_tasa_de_baja_razonable(df):
    # Si la tasa se sale de este rango, probablemente el dato viene mal etiquetado
    assert 0.05 <= df[OBJETIVO].mean() <= 0.40
