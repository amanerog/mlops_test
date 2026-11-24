"""Lab 3 · Guardrails de entrada: reglas sencillas que se ejecutan ANTES de llamar al modelo."""
import re

import mlflow

PATRONES_PII = {
    "DNI/NIE": r"\b[XYZ]?\d{7,8}[A-Z]\b",
    "email": r"\b[\w.+-]+@[\w-]+\.[\w.]+\b",
    "tarjeta": r"\b(?:\d[ -]?){13,16}\b",
}
PATRONES_INYECCION = [
    r"ignora\w* (todas )?(tus|las) (instrucciones|reglas)",
    r"olvida\w* (todo|tus instrucciones)",
    r"(prompt|instrucciones) del sistema",
    r"act[uú]a como",
]


@mlflow.trace(name="guardrail_entrada", span_type="GUARDRAIL")
def revisar_entrada(texto):
    """Devuelve (permitido, motivo)."""
    for nombre, patron in PATRONES_PII.items():
        if re.search(patron, texto, re.IGNORECASE):
            return False, f"La pregunta contiene datos personales ({nombre}). Por seguridad, no los compartas en el chat."
    for patron in PATRONES_INYECCION:
        if re.search(patron, texto, re.IGNORECASE):
            return False, "No puedo atender esa petición."
    return True, ""
