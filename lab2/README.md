# Laboratorio 2 · Tests, CI, API del modelo y drift

**Objetivo:** automatizar las comprobaciones de calidad, servir el modelo `champion` como API, detectar drift tras una subida de precio y decidir si se reentrena.

**Duración:** 35 minutos. Necesitas el modelo registrado en el laboratorio 1. Si no lo tienes, ejecuta:
```bash
python data/generate_data.py && python lab1/train.py --class-weight balanced && python lab1/register.py
```

## Pasos
1. **Pasar los tests de datos y de modelo**
   ```bash
   pytest -v
   ```
   Revisa `lab2/tests/test_datos.py` y `lab2/tests/test_modelo.py`: son las «puertas de calidad» del pipeline.
2. **Romper un test a propósito.** Cambia `RECALL_MINIMO = 0.60` por `0.90` en `lab2/tests/test_modelo.py` y repite `pytest`. Un test que falla bloquea el despliegue. Después, vuelve a dejarlo en `0.60`.
3. **Servir el modelo champion como API**
   ```bash
   mlflow models serve -m "models:/churn-nova-stream@champion" -h 0.0.0.0 -p 5001 --env-manager local
   ```
   El servidor ocupa la terminal: déjalo abierto.
4. **Pedir predicciones.** En **otra** terminal (`docker compose exec lab bash`):
   ```bash
   python lab2/predict_client.py
   ```
   También desde tu máquina: `curl http://localhost:5001/ping`. Para el servidor con `Ctrl+C`.
5. **Detectar drift.** `produccion_mes1.csv` son clientes normales; `produccion_mes2.csv` son clientes del mes posterior a una subida de 2 € en todos los planes.
   ```bash
   python lab2/drift.py produccion_mes1.csv
   python lab2/drift.py produccion_mes2.csv
   ```
   Compara el PSI de cada variable y el rendimiento del modelo en los dos meses. Los resultados quedan en el experimento `lab2-monitorizacion` de MLflow.
6. **Reentrenar y decidir**
   ```bash
   python lab2/retrain.py
   ```
   El candidato solo se promociona si mejora el AUC sin empeorar el recall en el mes reciente.
7. **CI en GitHub (opcional).** Sube el repositorio a tu cuenta de GitHub y haz un push a `main`: el workflow `.github/workflows/ci.yml` genera los datos, pasa los tests y entrena. Míralo en la pestaña *Actions*.

## Preguntas para pensar
- En el mes 2, el recall del modelo **sube** pero el AUC y el accuracy **bajan**. ¿Por qué?
- ¿Qué variables tienen drift? ¿Cuáles esperabas?
- ¿Qué pasaría si el reentreno se promocionara automáticamente sin comparar con el champion?

## Reto
Cambia en `lab2/retrain.py` la regla de promoción (por ejemplo, exigir que el recall mejore al menos 2 puntos) y comprueba si el candidato se sigue promocionando.
