import subprocess
import json
import os
import sys

from openai import OpenAI

sys.path.insert(0, os.path.dirname(__file__))
from config import OPENAI_API_KEY

# ==============================
# CONFIG
# ==============================

PYTHON_PATH = r"C:/Users/plata/Desktop/graph_rag/rag_grafos/.venv/Scripts/python.exe"
SCRIPT_PATH = "03_consultar.py"

PREGUNTAS_FILE = "preguntas.txt"
RESULTADOS_FILE = "evaluaciones.jsonl"
RESUMEN_FILE = "resumen.txt"
QA_FILE = "preguntas_respuestas.txt"

client = OpenAI(api_key=OPENAI_API_KEY)

# ==============================
# FUNCIONES
# ==============================

def extraer_respuesta_final(stdout: str) -> str:
    """
    Extrae solo el bloque de texto posterior a 'RESPUESTA:'.
    Si no existe, devuelve stdout completo.
    """
    marker = "RESPUESTA:"
    if marker in stdout:
        return stdout.split(marker, 1)[1].strip()
    return stdout.strip()


def ejecutar_query(query: str):
    cmd = [
        PYTHON_PATH,
        SCRIPT_PATH,
        "--modo",
        "vector",
        "--query",
        query,
        "--verbose"
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8"
    )

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    respuesta_final = extraer_respuesta_final(stdout)

    return {
        "stdout_completo": stdout,
        "stderr": stderr,
        "respuesta_final": respuesta_final,
        "returncode": result.returncode,
    }


def evaluar_respuesta(query: str, respuesta: str):
    """
    Evalúa con un criterio tolerante para RAG jurídico en desarrollo:
    - correcta
    - parcialmente_correcta
    - incorrecta
    """

    prompt = f"""
Eres un evaluador de un sistema RAG jurídico EN DESARROLLO.

IMPORTANTE:
Tu tarea NO es responder la pregunta jurídica.
Tu tarea es evaluar la calidad de la respuesta generada por el sistema.

Debes evaluar si la respuesta es útil y sustancialmente correcta para un RAG inicial.

Pregunta:
{query}

Respuesta generada:
{respuesta}

CRITERIOS DE EVALUACIÓN

1. "correcta"
- La respuesta responde correctamente al núcleo de la pregunta.
- Puede ser breve o no totalmente exhaustiva.
- No contiene errores jurídicos relevantes.
- Puede omitir detalles secundarios.

2. "parcialmente_correcta"
- La respuesta apunta en la dirección correcta.
- Puede contener omisiones relevantes o simplificaciones.
- Puede mezclar conceptos de forma leve.
- Sigue siendo utilizable para entender la cuestión.

3. "incorrecta"
- No responde realmente a la pregunta.
- Contiene errores jurídicos importantes.
- Confunde conceptos fundamentales.
- Inventar artículos, requisitos o categorías jurídicas se considera incorrecto.

CÓMO PUNTUAR

score:
- 8-10 → correcta
- 5-7 → parcialmente_correcta
- 0-4 → incorrecta

IMPORTANTE PARA RAG EN DESARROLLO

- No penalices la falta de exhaustividad.
- No penalices respuestas breves si el núcleo es correcto.
- Penaliza solo errores materiales o confusiones jurídicas claras.
- Si la respuesta es básicamente correcta pero incompleta, usa "parcialmente_correcta".

FORMATO DE RESPUESTA

Devuelve SOLO JSON válido.
No escribas nada antes ni después del JSON.

Formato exacto:

{{
  "clasificacion": "correcta" | "parcialmente_correcta" | "incorrecta",
  "score": 0-10,
  "explicacion": "explicación breve y concreta",
  "fallos": ["fallo 1", "fallo 2"]
}}
"""

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Eres un evaluador experto de sistemas RAG jurídicos en desarrollo. "
                    "Tu tarea es evaluar la calidad de las respuestas generadas, NO responder las preguntas. "
                    "Sé estricto con errores materiales y confusiones jurídicas, pero tolerante con "
                    "respuestas breves o incompletas si el núcleo es correcto. "
                    "Responde SOLO con JSON válido."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    texto = completion.choices[0].message.content

    try:
        data = json.loads(texto)
        if "fallos" not in data or not isinstance(data["fallos"], list):
            data["fallos"] = []
        return data
    except Exception:
        return {
            "clasificacion": "incorrecta",
            "score": 0,
            "explicacion": "No se pudo parsear la respuesta del evaluador",
            "fallos": ["JSON inválido del evaluador"],
            "raw": texto,
        }


def valor_clasificacion(clasificacion: str) -> float:
    mapping = {
        "correcta": 1.0,
        "parcialmente_correcta": 0.5,
        "incorrecta": 0.0,
    }
    return mapping.get(clasificacion, 0.0)


# ==============================
# MAIN
# ==============================

def main():
    with open(PREGUNTAS_FILE, "r", encoding="utf-8") as f:
        preguntas = [p.strip() for p in f if p.strip()]

    correctas = 0
    parciales = 0
    incorrectas = 0
    total = 0
    suma_scores = 0.0
    suma_precision_ponderada = 0.0

    resultados_qa = []

    with open(RESULTADOS_FILE, "w", encoding="utf-8") as out:
        for pregunta in preguntas:
            print(f"\nPregunta: {pregunta}")

            ejecucion = ejecutar_query(pregunta)
            respuesta = ejecucion["respuesta_final"]

            evaluacion = evaluar_respuesta(pregunta, respuesta)

            clasificacion = evaluacion.get("clasificacion", "incorrecta")
            score = float(evaluacion.get("score", 0))

            registro = {
                "pregunta": pregunta,
                "respuesta": respuesta,
                "stdout_completo": ejecucion["stdout_completo"],
                "stderr": ejecucion["stderr"],
                "returncode": ejecucion["returncode"],
                "evaluacion": evaluacion,
            }

            out.write(json.dumps(registro, ensure_ascii=False) + "\n")

            resultados_qa.append((pregunta, respuesta))

            total += 1
            suma_scores += score
            suma_precision_ponderada += valor_clasificacion(clasificacion)

            if clasificacion == "correcta":
                correctas += 1
            elif clasificacion == "parcialmente_correcta":
                parciales += 1
            else:
                incorrectas += 1

            print("Clasificación:", clasificacion)
            print("Score:", score)
            print("Explicación:", evaluacion.get("explicacion", ""))

    precision_binaria = correctas / total if total else 0.0
    precision_ponderada = suma_precision_ponderada / total if total else 0.0
    media_score = suma_scores / total if total else 0.0

    with open(RESUMEN_FILE, "w", encoding="utf-8") as f:
        f.write(f"Total preguntas: {total}\n")
        f.write(f"Correctas: {correctas}\n")
        f.write(f"Parcialmente correctas: {parciales}\n")
        f.write(f"Incorrectas: {incorrectas}\n")
        f.write(f"Precisión binaria: {precision_binaria:.2f}\n")
        f.write(f"Precisión ponderada: {precision_ponderada:.2f}\n")
        f.write(f"Score medio: {media_score:.2f}\n")

    # Generar fichero legible pregunta/respuesta
    with open(QA_FILE, "w", encoding="utf-8") as qa:
        for i, (pregunta, respuesta) in enumerate(resultados_qa):
            if i > 0:
                qa.write("\n")
            qa.write(f"Pregunta: {pregunta}\n")
            qa.write(f"Respuesta: {respuesta}\n")

    print("\n==== RESULTADOS ====")
    print("Total:", total)
    print("Correctas:", correctas)
    print("Parcialmente correctas:", parciales)
    print("Incorrectas:", incorrectas)
    print("Precisión binaria:", round(precision_binaria, 2))
    print("Precisión ponderada:", round(precision_ponderada, 2))
    print("Score medio:", round(media_score, 2))
    print(f"\nFichero pregunta/respuesta guardado en: {QA_FILE}")


if __name__ == "__main__":
    main()