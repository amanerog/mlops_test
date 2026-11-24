# Laboratorio 1 · Entrenar, comparar y registrar modelos

**Objetivo:** entrenar un modelo de baja de clientes de Nova Stream, registrar cada experimento en MLflow, compararlos y registrar el mejor como `champion`.

**Duración:** 40 minutos. Todos los comandos se ejecutan dentro del contenedor (`docker compose exec lab bash`).

## Los datos
`data/generate_data.py` genera datos sintéticos de clientes con estas variables: `plan`, `cuota_mensual`, `antiguedad_meses`, `horas_semana`, `dispositivos`, `incidencias_90d`, `dias_sin_acceso`, `descuento_activo` y `metodo_pago`. La variable objetivo es `churn`: 1 si el cliente se da de baja. Solo un 15 % de los clientes se da de baja.

## Pasos
1. **Generar los datos**
   ```bash
   python data/generate_data.py
   ```
2. **Entrenar el modelo base** (regresión logística)
   ```bash
   python lab1/train.py
   ```
   Mira las métricas que imprime. ¿El accuracy es bueno? ¿Y el recall?
3. **Probar variantes**
   ```bash
   python lab1/train.py --class-weight balanced
   python lab1/train.py --tipo rf --max-depth 8 --class-weight balanced
   python lab1/train.py --tipo logreg --C 0.01 --class-weight balanced
   python lab1/train.py --tipo rf --class-weight balanced
   ```
   El último es un random forest sin límite de profundidad: ¿qué le pasa al recall?
4. **Comparar en MLflow.** En http://localhost:5050, abre el experimento `lab1-churn`, selecciona los runs y pulsa *Compare*. Compara parámetros y métricas.
5. **Registrar el mejor modelo**
   ```bash
   python lab1/register.py --metrica recall
   ```
   En la pestaña *Models* aparece `churn-nova-stream` con el alias `champion`.
6. **Guardar el trabajo en Git** (fuera del contenedor, en tu terminal):
   ```bash
   git checkout -b lab1
   git add -A && git commit -m "Lab 1: experimentos"
   ```

## Preguntas para pensar
- ¿Por qué el modelo con mejor accuracy es el peor para el negocio?
- ¿Qué métrica elegiríais con el equipo de fidelización? ¿Por qué?
- ¿Qué información necesitarías para reproducir exactamente uno de tus runs dentro de seis meses?

## Reto
Añade a `lab1/train.py` un tercer tipo de modelo (por ejemplo, `GradientBoostingClassifier` en `churn.py`) y compáralo con los anteriores.
