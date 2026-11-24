"""Configuración del lab 3: modelos locales servidos con Ollama (API compatible con OpenAI)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

from churn import setup_mlflow  # noqa: F401  (reutilizamos la configuración de MLflow del repo)

LAB = Path(__file__).resolve().parent
DOCS = LAB / "docs"

MODELO_CHAT = "qwen2.5:1.5b"          # modelo principal (~1 GB)
MODELO_LIGERO = "qwen2.5:0.5b"        # alternativa más pequeña (~400 MB)
MODELO_EMBEDDINGS = "embeddinggemma"  # embeddings multilingües (~600 MB)
PROMPT = "asistente-nova-stream"      # nombre del prompt en el registro de MLflow

# Ollama expone una API compatible con OpenAI: el mismo código serviría con un modelo en la nube
cliente = OpenAI(base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"), api_key="ollama",
                 timeout=120, max_retries=0)
