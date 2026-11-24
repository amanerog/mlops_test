"""Genera los datos sintéticos de Nova Stream, una plataforma de streaming ficticia.

- clientes_historico.csv: datos de entrenamiento (clientes de los últimos 12 meses).
- produccion_mes1.csv: clientes nuevos, mismo comportamiento que el histórico.
- produccion_mes2.csv: clientes tras una subida de precios (hay drift).

Uso: python data/generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
PLANES = {"basico": 7.99, "estandar": 12.99, "premium": 17.99}


def generar(n, seed, subida_precio=0.0, cambio_habitos=False):
    rng = np.random.default_rng(seed)
    plan = rng.choice(list(PLANES), size=n, p=[0.40, 0.40, 0.20])
    cuota = np.array([PLANES[p] for p in plan]) + subida_precio
    antiguedad = rng.integers(1, 72, size=n)
    horas = np.clip(rng.gamma(2.2, 4.0, size=n) - (3.0 if cambio_habitos else 0.0), 0, None).round(1)
    dispositivos = rng.integers(1, 6, size=n)
    incidencias = rng.poisson(0.6 + (0.5 if cambio_habitos else 0.0), size=n)
    dias_sin_acceso = rng.exponential(6 + (6 if cambio_habitos else 0), size=n).round().astype(int)
    descuento = rng.random(n) < 0.25
    metodo_pago = rng.choice(["tarjeta", "paypal", "domiciliacion"], size=n, p=[0.55, 0.25, 0.20])

    # Probabilidad de baja: más baja con antigüedad y uso; más alta con precio, incidencias e inactividad
    sens_precio, efecto_uso = (0.10, -0.12) if not cambio_habitos else (0.45, 0.0)
    z = (-0.5 - 0.035 * antiguedad + efecto_uso * horas + sens_precio * (cuota - 12) + 0.45 * incidencias
         + 0.06 * dias_sin_acceso - 0.5 * descuento + 0.3 * (metodo_pago == "tarjeta"))
    if cambio_habitos:
        # Tras la subida de precio cambia la relación entre variables y baja (concept drift):
        # se van también clientes que usan mucho el servicio, sobre todo en los planes caros
        z -= 1.2
    churn = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)

    return pd.DataFrame({
        "cliente_id": [f"C{seed}{i:05d}" for i in range(n)],
        "plan": plan,
        "cuota_mensual": cuota.round(2),
        "antiguedad_meses": antiguedad,
        "horas_semana": horas,
        "dispositivos": dispositivos,
        "incidencias_90d": incidencias,
        "dias_sin_acceso": dias_sin_acceso,
        "descuento_activo": descuento.astype(int),
        "metodo_pago": metodo_pago,
        "churn": churn,
    })


if __name__ == "__main__":
    generar(6000, seed=1).to_csv(OUT / "clientes_historico.csv", index=False)
    generar(2000, seed=2).to_csv(OUT / "produccion_mes1.csv", index=False)
    generar(2000, seed=3, subida_precio=2.0, cambio_habitos=True).to_csv(OUT / "produccion_mes2.csv", index=False)
    for f in sorted(OUT.glob("*.csv")):
        df = pd.read_csv(f)
        print(f"{f.name}: {len(df)} filas, tasa de baja {df.churn.mean():.1%}")
