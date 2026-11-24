"""Lab 3 · Asistente RAG de Nova Stream: busca en los documentos y responde con el modelo local.

Uso: python lab3/rag.py "¿Cuánto cuesta el plan Premium?"
     python lab3/rag.py "..." --prompt 2 --modelo qwen2.5:0.5b
"""
import argparse
import time

import mlflow
import numpy as np

from config import DOCS, LAB, MODELO_CHAT, MODELO_EMBEDDINGS, PROMPT, cliente, setup_mlflow
from guardrails import revisar_entrada

INDICE = LAB / "indice.npz"


def trocear():
    """Divide cada documento en fragmentos por sección (##), conservando el título del documento."""
    fragmentos = []
    for fichero in sorted(DOCS.glob("*.md")):
        texto = fichero.read_text(encoding="utf-8")
        titulo = texto.splitlines()[0].lstrip("# ")
        for seccion in texto.split("\n## "):
            fragmentos.append({"fuente": fichero.name, "texto": f"{titulo}. {seccion.strip()}"})
    return fragmentos


def embeber(textos):
    resp = cliente.embeddings.create(model=MODELO_EMBEDDINGS, input=textos)
    v = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def cargar_indice():
    """Indexación (offline): se calcula una vez y se guarda en disco."""
    if INDICE.exists():
        d = np.load(INDICE, allow_pickle=True)
        return list(d["fragmentos"]), d["vectores"]
    fragmentos = trocear()
    vectores = embeber([f["texto"] for f in fragmentos])
    np.savez(INDICE, fragmentos=np.array(fragmentos, dtype=object), vectores=vectores)
    return fragmentos, vectores


@mlflow.trace(span_type="RETRIEVER")
def buscar(pregunta, k=3):
    fragmentos, vectores = cargar_indice()
    similitud = vectores @ embeber([pregunta])[0]
    return [dict(fragmentos[i], score=float(similitud[i])) for i in np.argsort(-similitud)[:k]]


@mlflow.trace(name="asistente_nova_stream")
def responder(pregunta, modelo=MODELO_CHAT, prompt_version="produccion", k=3):
    inicio = time.perf_counter()
    permitido, motivo = revisar_entrada(pregunta)
    if not permitido:
        return {"respuesta": motivo, "fuentes": [], "bloqueada": True,
                "latencia_s": round(time.perf_counter() - inicio, 2), "tokens": 0}

    docs = buscar(pregunta, k)
    contexto = "\n\n".join(f"[{d['fuente']}]\n{d['texto']}" for d in docs)
    uri = f"prompts:/{PROMPT}@{prompt_version}" if not str(prompt_version).isdigit() else f"prompts:/{PROMPT}/{prompt_version}"
    prompt = mlflow.genai.load_prompt(uri)

    resp = cliente.chat.completions.create(
        model=modelo,
        messages=[{"role": "user", "content": prompt.format(contexto=contexto, pregunta=pregunta)}],
        temperature=0,
        max_tokens=200,  # límite de salida: evita respuestas en bucle y acota coste y latencia
    )
    return {
        "respuesta": resp.choices[0].message.content.strip(),
        "fuentes": [d["fuente"] for d in docs],
        "bloqueada": False,
        "latencia_s": round(time.perf_counter() - inicio, 2),
        "tokens": resp.usage.total_tokens,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pregunta")
    p.add_argument("--modelo", default=MODELO_CHAT)
    p.add_argument("--prompt", default="produccion", help="versión (1, 2...) o alias del prompt")
    args = p.parse_args()

    setup_mlflow("lab3-llmops")
    mlflow.openai.autolog()
    r = responder(args.pregunta, args.modelo, args.prompt)
    print(r["respuesta"])
    print(f"\nfuentes recuperadas: {r['fuentes']} · {r['latencia_s']} s · {r['tokens']} tokens")
