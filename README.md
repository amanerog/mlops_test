# Laboratorios · MLOps y LLMOps (IFFE 2026)

Repositorio de prácticas del curso de MLOps y LLMOps. Las tres sesiones usan el mismo caso:
**Nova Stream**, una plataforma de streaming ficticia.

| Sesión | Laboratorio | Qué se practica |
|---|---|---|
| 1 · MLOps I | [lab1](lab1/README.md) | Entrenar, comparar y registrar modelos con MLflow |
| 2 · MLOps II | [lab2](lab2/README.md) | Tests, CI con GitHub Actions, servir el modelo como API, drift y reentrenamiento |
| 3 · LLMOps | [lab3](lab3/README.md) | Asistente RAG con un modelo local: trazas, prompts versionados, evaluación y guardrails |

## Antes de la primera sesión

**Haz esta instalación antes del curso.** En remoto, resolver problemas de instalación en directo consume mucho tiempo.

### Requisitos
- **Docker Desktop** (Windows, macOS) o Docker Engine (Linux), con al menos **8 GB de RAM** asignados.
- **Git** y una cuenta de **GitHub**.
- Unos **8 GB libres** en disco (imágenes y modelos).
- Un editor de código (VS Code o el que prefieras).

### Instalación
```bash
git clone <url-del-repositorio>
cd iffe-mlops-labs
docker compose up -d lab
```
Esto basta para las **sesiones 1 y 2** (MLflow y el contenedor de trabajo, unos 1,5 GB la primera vez).

**Antes de la sesión 3**, levanta el entorno completo, que añade el servidor de modelos de lenguaje y descarga los modelos (unos 5 GB más; puede tardar entre 10 y 30 minutos):
```bash
docker compose up -d
```

### Comprobación
1. Abre **http://localhost:5050**: debe aparecer la web de MLflow.
2. Solo antes de la sesión 3: comprueba que los modelos se han descargado:
   ```bash
   docker compose exec ollama ollama list
   ```
   Deben aparecer `qwen2.5:1.5b`, `qwen2.5:0.5b` y `embeddinggemma`.
   Si no aparecen, espera a que termine la descarga (`docker compose logs -f ollama-modelos`).
3. Entra en el contenedor de trabajo y genera los datos:
   ```bash
   docker compose exec lab bash
   python data/generate_data.py
   ```

Si los tres pasos funcionan, estás listo.

## Cómo se trabaja
- `docker compose up -d lab` (o `docker compose up -d` desde la sesión 3) arranca el entorno; `docker compose stop` lo para sin perder nada.
- `docker compose exec lab bash` abre una terminal **dentro** del contenedor de trabajo. Todos los comandos `python ...` de los laboratorios se ejecutan ahí.
- El código está en una carpeta compartida: edita los ficheros con tu editor habitual y ejecútalos en el contenedor.
- Los enlaces que imprime MLflow (`http://mlflow:5000/...`) son internos del contenedor: desde tu navegador usa **http://localhost:5050**.

## Arquitectura del entorno
| Servicio | Qué es | Puerto |
|---|---|---|
| `lab` | Python 3.11 con las dependencias; aquí se ejecuta el código | 5001 (API del modelo, lab 2) |
| `mlflow` | Servidor de MLflow: experimentos, modelos, prompts y trazas | 5050 |
| `ollama` | Servidor de modelos de lenguaje locales | 11434 |
| `ollama-modelos` | Descarga los modelos la primera vez y termina | — |

## Problemas frecuentes
- **El puerto 5050 o 5001 está ocupado**: cambia el puerto de la izquierda en `docker-compose.yml` (por ejemplo, `"5051:5000"`).
- **El modelo de lenguaje va muy lento o se queda sin memoria**: asigna más RAM a Docker Desktop (Settings > Resources) o usa el modelo pequeño con `--modelo qwen2.5:0.5b`.
- **Quiero empezar de cero**: `docker compose down -v` borra los contenedores y los volúmenes (experimentos y modelos descargados).

## Sin Docker (alternativa)
Con Python 3.11 y [Ollama](https://ollama.com) instalados en tu máquina:
```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen2.5:1.5b && ollama pull qwen2.5:0.5b && ollama pull embeddinggemma
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5050   # en otra terminal
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db                   # en Windows: set MLFLOW_TRACKING_URI=...
```
Sin la variable `MLFLOW_TRACKING_URI`, los scripts usan igualmente `mlflow.db` en la raíz del repositorio.
