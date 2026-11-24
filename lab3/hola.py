"""Lab 3 · Paso 1: comprobar que el modelo local responde y ver la traza en MLflow.

Uso: python lab3/hola.py
"""
import mlflow

from config import MODELO_CHAT, cliente, setup_mlflow

setup_mlflow("lab3-llmops")
mlflow.openai.autolog()  # registra automáticamente cada llamada al modelo como una traza

resp = cliente.chat.completions.create(
    model=MODELO_CHAT,
    messages=[{"role": "user", "content": "Explica qué es MLOps en una sola frase."}],
    max_tokens=100,
)
print(resp.choices[0].message.content)
print(f"\nTokens: {resp.usage.prompt_tokens} de entrada, {resp.usage.completion_tokens} de salida")
print("Abre MLflow (pestaña Traces del experimento lab3-llmops) para ver la traza.")
