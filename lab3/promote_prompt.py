"""Lab 3 · Promociona una versión del prompt moviendo el alias 'produccion'.

La aplicación siempre carga prompts:/asistente-nova-stream@produccion: cambiar de versión
no requiere tocar el código ni redesplegar, y volver atrás es igual de sencillo.

Uso: python lab3/promote_prompt.py 2
"""
import sys

import mlflow

from config import PROMPT, setup_mlflow

version = int(sys.argv[1])
setup_mlflow("lab3-llmops")
mlflow.genai.set_prompt_alias(PROMPT, alias="produccion", version=version)
print(f"El alias 'produccion' de '{PROMPT}' apunta ahora a la versión {version}.")
