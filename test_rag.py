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

MODOS_DISPONIBLES = ("cypher", "grafo", "vector")
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


def inicializar_metricas() -> dict[str, float]:
    return {
        "correctas": 0,
        "parciales": 0,
        "incorrectas": 0,
        "no_evaluadas": 0,
        "total": 0,
        "suma_scores": 0.0,
        "suma_precision_ponderada": 0.0,
    }


def actualizar_metricas(metricas: dict[str, float], clasificacion: str, score: float) -> None:
    metricas["total"] += 1
    metricas["suma_scores"] += score
    metricas["suma_precision_ponderada"] += valor_clasificacion(clasificacion)

    if clasificacion == "correcta":
        metricas["correctas"] += 1
    elif clasificacion == "parcialmente_correcta":
        metricas["parciales"] += 1
    elif clasificacion == "no_evaluada":
        metricas["no_evaluadas"] += 1
    else:
        metricas["incorrectas"] += 1


def calcular_resumen(metricas: dict[str, float]) -> dict[str, float]:
    total = int(metricas["total"])
    precision_binaria = metricas["correctas"] / total if total else 0.0
    precision_ponderada = metricas["suma_precision_ponderada"] / total if total else 0.0
    media_score = metricas["suma_scores"] / total if total else 0.0
    return {
        "total": total,
        "correctas": int(metricas["correctas"]),
        "parciales": int(metricas["parciales"]),
        "incorrectas": int(metricas["incorrectas"]),
        "no_evaluadas": int(metricas["no_evaluadas"]),
        "precision_binaria": precision_binaria,
        "precision_ponderada": precision_ponderada,
        "media_score": media_score,
    }


def escribir_resumen(f, titulo: str, resumen: dict[str, float], include_header: bool = True) -> None:
    if include_header:
        f.write(f"[{titulo}]\n")
    f.write(f"Total preguntas: {resumen['total']}\n")
    f.write(f"Correctas: {resumen['correctas']}\n")
    f.write(f"Parcialmente correctas: {resumen['parciales']}\n")
    f.write(f"Incorrectas: {resumen['incorrectas']}\n")
    f.write(f"No evaluadas: {resumen['no_evaluadas']}\n")
    f.write(f"Precisión binaria: {resumen['precision_binaria']:.2f}\n")
    f.write(f"Precisión ponderada: {resumen['precision_ponderada']:.2f}\n")
    f.write(f"Score medio: {resumen['media_score']:.2f}\n")


def main():
    parser = argparse.ArgumentParser(description="Ejecuta una batería de preguntas contra 03_consultar.py")
    parser.add_argument("--modo", choices=[*MODOS_DISPONIBLES, "todos"], default=DEFAULT_MODO)
    parser.add_argument("--max-preguntas", type=int, default=DEFAULT_MAX_PREGUNTAS)
    parser.add_argument("--verbose-consulta", action="store_true", help="Pasa --verbose a 03_consultar.py")
    parser.add_argument("--sin-evaluacion", action="store_true", help="Ejecuta las preguntas sin llamar al evaluador")
    args = parser.parse_args()

    preguntas = cargar_preguntas(PREGUNTAS_FILE, max_preguntas=args.max_preguntas)
    total_preguntas = len(preguntas)

    modos = list(MODOS_DISPONIBLES) if args.modo == "todos" else [args.modo]
    metricas_por_modo = {modo: inicializar_metricas() for modo in modos}
    resultados_qa: list[dict[str, object]] = []

    print(f"Modos de consulta: {', '.join(modos)}")
    print(f"Python usado: {PYTHON_PATH}")
    print(f"Preguntas a ejecutar: {total_preguntas}")

    with RESULTADOS_FILE.open("w", encoding="utf-8") as out:
        for idx, pregunta in enumerate(preguntas, start=1):
            print(f"\n[{idx}/{total_preguntas}] Pregunta: {pregunta}")

            respuestas_por_modo: dict[str, str] = {}
            for modo in modos:
                print(f"  -> Ejecutando modo {modo}")
                ejecucion = ejecutar_query(pregunta, modo=modo, verbose_consulta=args.verbose_consulta)
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
                    "modo": modo,
                    "pregunta": pregunta,
                    "respuesta": respuesta,
                    "stdout_completo": ejecucion["stdout_completo"],
                    "stderr": ejecucion["stderr"],
                    "returncode": ejecucion["returncode"],
                    "cmd": ejecucion["cmd"],
                    "evaluacion": evaluacion,
                }
                out.write(json.dumps(registro, ensure_ascii=False) + "\n")

                respuestas_por_modo[modo] = respuesta
                actualizar_metricas(metricas_por_modo[modo], clasificacion, score)

                if ejecucion["returncode"] != 0:
                    print(f"     Return code ({modo}): {ejecucion['returncode']}")
                print(f"     {modo}: {clasificacion} | score {score}")
                print(f"     {modo} explicación: {evaluacion.get('explicacion', '')}")

            resultados_qa.append(
                {
                    "pregunta": pregunta,
                    "respuestas": respuestas_por_modo,
                }
            )

    with RESUMEN_FILE.open("w", encoding="utf-8") as f:
        if len(modos) == 1:
            resumen = calcular_resumen(metricas_por_modo[modos[0]])
            escribir_resumen(f, modos[0], resumen, include_header=False)
        else:
            for idx, modo in enumerate(modos):
                if idx > 0:
                    f.write("\n")
                resumen = calcular_resumen(metricas_por_modo[modo])
                escribir_resumen(f, modo, resumen)

    with QA_FILE.open("w", encoding="utf-8") as qa:
        for i, resultado in enumerate(resultados_qa):
            if i > 0:
                qa.write("\n")
            qa.write(f"Pregunta: {resultado['pregunta']}\n")
            respuestas = resultado["respuestas"]
            if len(modos) == 1:
                qa.write(f"Respuesta: {respuestas[modos[0]]}\n")
            else:
                for modo in modos:
                    qa.write(f"Respuesta [{modo}]: {respuestas.get(modo, '')}\n")

    print("\n==== RESULTADOS ====")
    for idx, modo in enumerate(modos):
        if idx > 0:
            print("")
        resumen = calcular_resumen(metricas_por_modo[modo])
        print(f"[{modo}]")
        print("Total:", resumen["total"])
        print("Correctas:", resumen["correctas"])
        print("Parcialmente correctas:", resumen["parciales"])
        print("Incorrectas:", resumen["incorrectas"])
        print("No evaluadas:", resumen["no_evaluadas"])
        print("Precisión binaria:", round(resumen["precision_binaria"], 2))
        print("Precisión ponderada:", round(resumen["precision_ponderada"], 2))
        print("Score medio:", round(resumen["media_score"], 2))
    print(f"\nFichero pregunta/respuesta guardado en: {QA_FILE.name}")


if __name__ == "__main__":
    main()
