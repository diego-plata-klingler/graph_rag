import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import OPENAI_API_KEY


BASE_DIR = Path(__file__).resolve().parent
PYTHON_PATH = sys.executable
SCRIPT_PATH = BASE_DIR / "03_consultar.py"

PREGUNTAS_FILE = BASE_DIR / "preguntas.txt"
RESULTADOS_FILE = BASE_DIR / "evaluaciones.jsonl"
RESUMEN_FILE = BASE_DIR / "resumen.txt"
QA_FILE = BASE_DIR / "preguntas_respuestas.txt"

DEFAULT_MODO = "grafo"
DEFAULT_MAX_PREGUNTAS = 30

_client = None


def extraer_respuesta_final(stdout: str) -> str:
    """Extrae solo el bloque de texto posterior a 'RESPUESTA:'."""
    marker = "RESPUESTA:"
    if marker in stdout:
        return stdout.split(marker, 1)[1].strip()
    return stdout.strip()


def cargar_preguntas(path: Path, max_preguntas: int | None = DEFAULT_MAX_PREGUNTAS) -> list[str]:
    """Carga preguntas y recompone líneas partidas del fichero."""
    raw_lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    preguntas: list[str] = []

    for line in raw_lines:
        if line.startswith("¿"):
            preguntas.append(line)
            continue

        idx_pendiente = next(
            (idx for idx in range(len(preguntas) - 1, -1, -1) if not preguntas[idx].endswith("?")),
            None,
        )

        if idx_pendiente is None:
            if preguntas:
                preguntas[-1] = f"{preguntas[-1]} {line}".strip()
            else:
                preguntas.append(line)
            continue

        joiner = "" if line and line[0].islower() else " "
        preguntas[idx_pendiente] = f"{preguntas[idx_pendiente]}{joiner}{line}".strip()

    if max_preguntas is not None:
        return preguntas[:max_preguntas]
    return preguntas


def ejecutar_query(query: str, modo: str, verbose_consulta: bool = False) -> dict:
    cmd = [
        PYTHON_PATH,
        str(SCRIPT_PATH),
        "--modo",
        modo,
        "--query",
        query,
    ]
    if verbose_consulta:
        cmd.append("--verbose")

    result = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()
    respuesta_final = extraer_respuesta_final(stdout)

    return {
        "cmd": cmd,
        "stdout_completo": stdout,
        "stderr": stderr,
        "respuesta_final": respuesta_final,
        "returncode": result.returncode,
    }


def get_client():
    global _client
    if _client is not None:
        return _client

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("No se pudo importar 'openai' para evaluar respuestas.") from exc

    _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def evaluar_respuesta(query: str, respuesta: str) -> dict:
    """Evalúa con un criterio tolerante para RAG jurídico en desarrollo."""
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

    completion = get_client().chat.completions.create(
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
        "no_evaluada": 0.0,
    }
    return mapping.get(clasificacion, 0.0)


def main():
    parser = argparse.ArgumentParser(description="Ejecuta una batería de preguntas contra 03_consultar.py")
    parser.add_argument("--modo", choices=["cypher", "grafo", "vector"], default=DEFAULT_MODO)
    parser.add_argument("--max-preguntas", type=int, default=DEFAULT_MAX_PREGUNTAS)
    parser.add_argument("--verbose-consulta", action="store_true", help="Pasa --verbose a 03_consultar.py")
    parser.add_argument("--sin-evaluacion", action="store_true", help="Ejecuta las preguntas sin llamar al evaluador")
    args = parser.parse_args()

    preguntas = cargar_preguntas(PREGUNTAS_FILE, max_preguntas=args.max_preguntas)
    total_preguntas = len(preguntas)

    correctas = 0
    parciales = 0
    incorrectas = 0
    no_evaluadas = 0
    total = 0
    suma_scores = 0.0
    suma_precision_ponderada = 0.0
    resultados_qa: list[tuple[str, str]] = []

    print(f"Modo de consulta: {args.modo}")
    print(f"Python usado: {PYTHON_PATH}")
    print(f"Preguntas a ejecutar: {total_preguntas}")

    with RESULTADOS_FILE.open("w", encoding="utf-8") as out:
        for idx, pregunta in enumerate(preguntas, start=1):
            print(f"\n[{idx}/{total_preguntas}] Pregunta: {pregunta}")

            ejecucion = ejecutar_query(pregunta, modo=args.modo, verbose_consulta=args.verbose_consulta)
            respuesta = ejecucion["respuesta_final"]
            if args.sin_evaluacion:
                evaluacion = {
                    "clasificacion": "no_evaluada",
                    "score": 0,
                    "explicacion": "Evaluación omitida por parámetro",
                    "fallos": [],
                }
            else:
                try:
                    evaluacion = evaluar_respuesta(pregunta, respuesta)
                except Exception as exc:
                    evaluacion = {
                        "clasificacion": "no_evaluada",
                        "score": 0,
                        "explicacion": f"No se pudo evaluar: {exc}",
                        "fallos": ["evaluacion_no_disponible"],
                    }

            clasificacion = evaluacion.get("clasificacion", "incorrecta")
            score = float(evaluacion.get("score", 0))

            registro = {
                "pregunta": pregunta,
                "respuesta": respuesta,
                "stdout_completo": ejecucion["stdout_completo"],
                "stderr": ejecucion["stderr"],
                "returncode": ejecucion["returncode"],
                "cmd": ejecucion["cmd"],
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
            elif clasificacion == "no_evaluada":
                no_evaluadas += 1
            else:
                incorrectas += 1

            if ejecucion["returncode"] != 0:
                print(f"Return code: {ejecucion['returncode']}")
            print("Clasificación:", clasificacion)
            print("Score:", score)
            print("Explicación:", evaluacion.get("explicacion", ""))

    precision_binaria = correctas / total if total else 0.0
    precision_ponderada = suma_precision_ponderada / total if total else 0.0
    media_score = suma_scores / total if total else 0.0

    with RESUMEN_FILE.open("w", encoding="utf-8") as f:
        f.write(f"Total preguntas: {total}\n")
        f.write(f"Correctas: {correctas}\n")
        f.write(f"Parcialmente correctas: {parciales}\n")
        f.write(f"Incorrectas: {incorrectas}\n")
        f.write(f"No evaluadas: {no_evaluadas}\n")
        f.write(f"Precisión binaria: {precision_binaria:.2f}\n")
        f.write(f"Precisión ponderada: {precision_ponderada:.2f}\n")
        f.write(f"Score medio: {media_score:.2f}\n")

    with QA_FILE.open("w", encoding="utf-8") as qa:
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
    print("No evaluadas:", no_evaluadas)
    print("Precisión binaria:", round(precision_binaria, 2))
    print("Precisión ponderada:", round(precision_ponderada, 2))
    print("Score medio:", round(media_score, 2))
    print(f"\nFichero pregunta/respuesta guardado en: {QA_FILE.name}")


if __name__ == "__main__":
    main()
