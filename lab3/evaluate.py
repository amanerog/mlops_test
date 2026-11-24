"""Lab 3 · Evalúa el asistente con el golden set y registra el resultado en MLflow.

Para cada pregunta del golden set se genera la respuesta y se calculan métricas deterministas:
baratas, reproducibles y fáciles de explicar. El resultado queda en un run de MLflow
(métricas agregadas + tabla con cada respuesta) y cada llamada queda como traza.

Uso: python lab3/evaluate.py --prompt 1
     python lab3/evaluate.py --prompt 2
     python lab3/evaluate.py --prompt 2 --modelo qwen2.5:0.5b
"""
import argparse
import json
import unicodedata

import mlflow
import pandas as pd

from config import LAB, MODELO_CHAT, setup_mlflow
from rag import responder


def normalizar(texto):
    sin_tildes = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in sin_tildes if unicodedata.category(c) != "Mn")


def respuesta_correcta(respuesta, caso):
    """Cada grupo de 'debe_contener' tiene que aparecer (vale cualquiera de sus alternativas)."""
    texto = normalizar(respuesta)
    return all(any(normalizar(alt) in texto for alt in grupo) for grupo in caso["debe_contener"])


def recupera_fuente(fuentes, caso):
    """¿El buscador ha traído el documento que contiene la respuesta?"""
    return caso["fuente"] is None or caso["fuente"] in fuentes


def cita_fuente(respuesta, caso):
    """¿La respuesta indica de qué documento sale?"""
    return caso["fuente"] is None or caso["fuente"] in respuesta


p = argparse.ArgumentParser()
p.add_argument("--modelo", default=MODELO_CHAT)
p.add_argument("--prompt", default="1", help="versión del prompt a evaluar")
args = p.parse_args()

setup_mlflow("lab3-llmops")
mlflow.openai.autolog()
casos = json.loads((LAB / "golden_set.json").read_text(encoding="utf-8"))

filas = []
with mlflow.start_run(run_name=f"eval-{args.modelo}-prompt-v{args.prompt}"):
    mlflow.log_params({"modelo": args.modelo, "prompt_version": args.prompt, "casos": len(casos)})
    for i, caso in enumerate(casos, 1):
        r = responder(caso["pregunta"], args.modelo, args.prompt)
        fila = {
            "pregunta": caso["pregunta"],
            "respuesta": r["respuesta"],
            "correcta": respuesta_correcta(r["respuesta"], caso),
            "recupera_fuente": recupera_fuente(r["fuentes"], caso),
            "cita_fuente": cita_fuente(r["respuesta"], caso),
            "latencia_s": r["latencia_s"],
            "tokens": r["tokens"],
        }
        filas.append(fila)
        print(f"[{i:>2}/{len(casos)}] {'OK ' if fila['correcta'] else 'MAL'} {caso['pregunta']}")

    df = pd.DataFrame(filas)
    metricas = {
        "respuestas_correctas": df.correcta.mean(),
        "recuperacion_correcta": df.recupera_fuente.mean(),
        "cita_la_fuente": df.cita_fuente.mean(),
        "latencia_media_s": df.latencia_s.mean(),
        "tokens_medios": df.tokens.mean(),
    }
    mlflow.log_metrics(metricas)
    mlflow.log_table(df, artifact_file="resultados_evaluacion.json")

print(f"\nModelo {args.modelo} · prompt v{args.prompt}")
for nombre, valor in metricas.items():
    print(f"  {nombre:<24} {valor:.2f}")
print("\nFallos:")
for f in filas:
    if not f["correcta"]:
        print(f"  - {f['pregunta']}\n    -> {f['respuesta'][:150]}")
