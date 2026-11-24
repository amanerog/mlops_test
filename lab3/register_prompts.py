"""Lab 3 · Paso 2: registrar las versiones del prompt en MLflow, igual que versionamos modelos.

Uso: python lab3/register_prompts.py
"""
import mlflow

from config import PROMPT, setup_mlflow

V1 = """Responde a la pregunta del cliente de Nova Stream.

Información:
{{contexto}}

Pregunta: {{pregunta}}"""

V2 = """Eres el asistente de atención al cliente de Nova Stream.

Reglas:
1. Responde SOLO con la información del contexto. No inventes datos.
2. Si la respuesta no está en el contexto, responde exactamente: "No tengo esa información."
3. Responde en español, en una o dos frases, con los datos concretos (precios, plazos, cifras).
4. Termina indicando la fuente así: [fuente: nombre_del_fichero.md]

Contexto:
{{contexto}}

Pregunta: {{pregunta}}"""

setup_mlflow("lab3-llmops")
v1 = mlflow.genai.register_prompt(PROMPT, V1, commit_message="v1: prompt mínimo")
v2 = mlflow.genai.register_prompt(PROMPT, V2, commit_message="v2: reglas, formato y cita de la fuente")
mlflow.genai.set_prompt_alias(PROMPT, alias="produccion", version=v1.version)
print(f"Registradas las versiones {v1.version} y {v2.version} de '{PROMPT}'.")
print(f"El alias 'produccion' apunta a la versión {v1.version}.")
