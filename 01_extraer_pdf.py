"""
Script 1: Extrae, parsea y chunkifica el texto del PDF.

Genera data/articulos.json con la estructura jerárquica de la ley:
  - Títulos, Capítulos, Secciones
  - Artículos (con texto completo)
  - Chunks estructurados por artículo y disposición
  - Referencias cruzadas entre artículos ("conforme al artículo X")
  - Disposiciones adicionales / finales / transitorias

Uso:
    python 01_extraer_pdf.py
"""

import re
import json
import os
import sys

import fitz  # pymupdf

sys.path.insert(0, os.path.dirname(__file__))
from config import PDF_PATH, DATA_DIR, ARTICULOS_JSON
from chunking_utils import (
    MarkdownChunker,
    build_article_markdown,
    build_disposition_markdown,
    clean_legal_text,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Extraer texto bruto del PDF
# ──────────────────────────────────────────────────────────────────────────────

def extraer_texto_pdf(pdf_path: str) -> str:
    """Extrae el texto de todas las páginas del PDF."""
    doc = fitz.open(pdf_path)
    paginas = []
    for i, pagina in enumerate(doc):
        texto = pagina.get_text("text")
        paginas.append(f"\n[PAGINA_{i+1}]\n{texto}")
    print(f"  {len(paginas)} páginas extraídas")
    return "\n".join(paginas)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Limpiar texto
# ──────────────────────────────────────────────────────────────────────────────

def limpiar_texto(texto: str) -> str:
    """Elimina artefactos de PDF (saltos de línea internos en párrafos, etc.)."""
    # Unir líneas rotas dentro de un párrafo (línea que no termina en punto/dos puntos)
    texto = re.sub(r'(?<![.\n:;])\n(?![A-ZÁÉÍÓÚÑ\d\[])', ' ', texto)
    # Espacios múltiples
    texto = re.sub(r'  +', ' ', texto)
    return texto


# ──────────────────────────────────────────────────────────────────────────────
# 3. Parsear estructura jerárquica
# ──────────────────────────────────────────────────────────────────────────────

# Patrones de cabeceras de sección — orden de mayor a menor nivel
PATRON_TITULO    = re.compile(r'\bT[ÍI]TULO\s+(I{1,3}V?|VI{0,3}|IX|X{1,3}(?:I{0,3})?)\b', re.IGNORECASE)
PATRON_CAPITULO  = re.compile(r'\bCAP[ÍI]TULO\s+(I{1,3}V?|VI{0,3}|IX|X{1,3}(?:I{0,3})?)\b', re.IGNORECASE)
PATRON_SECCION   = re.compile(r'\bSecci[oó]n\s+(\d+[.ª]?)\b', re.IGNORECASE)
PATRON_ARTICULO  = re.compile(
    r'Art[íi]culo\s+(\d+(?:\s+(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies))?)\.'
    r'\s*([^\n]{3,120})',
    re.IGNORECASE
)
PATRON_DISPOSICION = re.compile(
    r'(DISPOSICI[ÓO]N\s+(?:ADICIONAL|FINAL|TRANSITORIA|DEROGATORIA)\s+(?:PRIMERA|SEGUNDA|TERCERA|CUARTA|QUINTA|SEXTA|SÉPTIMA|ÚNICA|\d+))',
    re.IGNORECASE
)
PATRON_REFERENCIA = re.compile(
    r'(?:art[íi]culo|arts?\.)\s+(\d+(?:\s+(?:bis|ter|quater|quinquies|sexies|septies))?)',
    re.IGNORECASE
)


def _normalizar_numero_articulo(num_str: str) -> str:
    """Normaliza 'art_42 bis' → '42_bis', '42' → '42'."""
    return num_str.strip().lower().replace(' ', '_')


def parsear_estructura(texto: str) -> dict:
    """
    Parsea el texto completo y devuelve:
    {
        "titulos": [...],
        "capitulos": [...],
        "secciones": [...],
        "articulos": [
            {
                "id": "art_42",
                "numero": "42",
                "titulo": "Requisitos de acceso a la actividad",
                "texto": "...",
                "titulo_padre": "TÍTULO IV",
                "capitulo_padre": "CAPÍTULO I",
                "seccion_padre": null,
                "referencias": ["38", "41", "45"],
                "tipo": "articulo"
            },
            ...
        ],
        "disposiciones": [...]
    }
    """
    lineas = texto.split('\n')

    contexto = {
        "titulo_actual": None,
        "capitulo_actual": None,
        "seccion_actual": None,
    }

    # Para acumular texto de cada artículo
    articulo_actual = None
    articulos = []
    disposicion_actual = None
    disposiciones = []
    titulos_vistos = set()
    capitulos_vistos = set()
    secciones_vistas = set()

    def guardar_articulo_actual():
        nonlocal articulo_actual
        if articulo_actual:
            art = articulo_actual
            texto_art = art["_lineas_texto"]
            titulo_art = art["titulo"]

            # ── Filtrar entradas del índice (título con puntos de tabulación '. . .')
            if '. .' in titulo_art:
                articulo_actual = None
                return

            # ── Filtrar referencias a párrafos de leyes modificadoras ('1.d)', '2.a)', etc.)
            if re.match(r'^\d+\.\s*[a-z]\)', titulo_art.strip()):
                articulo_actual = None
                return

            # ── Filtrar fragmentos de cierre de modificación ('1.d).' sin texto relevante)
            if re.match(r'^\d+\.\w*\)\.\s*$', titulo_art.strip()):
                articulo_actual = None
                return

            # ── Filtrar citas a artículos de otras leyes ('3 de la Ley 30/1992', '149 de la Constitución')
            if re.match(r'^\d[\d.]*\s+de\s+(la|el|los|las)\b', titulo_art.strip(), re.IGNORECASE):
                articulo_actual = None
                return
            if re.search(r'\bde la Constituci[oó]n\b|\bde la Ley\b|\bdel\s+Texto\s+Refundido\b', titulo_art, re.IGNORECASE):
                articulo_actual = None
                return
            # ── Filtrar títulos truncados que acaban en "de la/el/los" (texto cortado por el regex)
            if re.search(r'\bde\s+(la|el|los|las)\s*$', titulo_art.strip(), re.IGNORECASE):
                articulo_actual = None
                return
            # ── Filtrar listas de ordinales ('1.6.ª, 11.ª y 13.ª ...')
            if re.match(r'^\d+\.?\d*[ªº°]', titulo_art.strip()):
                articulo_actual = None
                return

            # ── Filtrar párrafos letrados de otras leyes ('d) del Texto Refundido')
            if re.match(r'^[a-z]\)\s+del?\s+', titulo_art.strip(), re.IGNORECASE):
                articulo_actual = None
                return

            texto_completo = clean_legal_text(f"Artículo {art['numero']}. {titulo_art}\n{texto_art}")
            art["texto"] = texto_completo
            art["referencias"] = extraer_referencias(texto_completo)
            del art["_lineas_texto"]
            articulos.append(art)
            articulo_actual = None

    def guardar_disposicion_actual():
        nonlocal disposicion_actual
        if disposicion_actual:
            disp = disposicion_actual
            disp["texto"] = clean_legal_text(disp["_lineas_texto"].strip())
            del disp["_lineas_texto"]
            disposiciones.append(disp)
            disposicion_actual = None

    for linea in lineas:
        linea_strip = linea.strip()
        if not linea_strip:
            if articulo_actual:
                articulo_actual["_lineas_texto"] += "\n"
            continue

        # ── Detectar TÍTULO
        m_titulo = PATRON_TITULO.search(linea_strip)
        if m_titulo and len(linea_strip) < 120:
            guardar_articulo_actual()
            guardar_disposicion_actual()
            titulo_str = linea_strip
            if titulo_str not in titulos_vistos:
                titulos_vistos.add(titulo_str)
            contexto["titulo_actual"] = titulo_str
            contexto["capitulo_actual"] = None
            contexto["seccion_actual"] = None
            continue

        # ── Detectar CAPÍTULO
        m_cap = PATRON_CAPITULO.search(linea_strip)
        if m_cap and len(linea_strip) < 120:
            guardar_articulo_actual()
            guardar_disposicion_actual()
            cap_str = linea_strip
            if cap_str not in capitulos_vistos:
                capitulos_vistos.add(cap_str)
            contexto["capitulo_actual"] = cap_str
            contexto["seccion_actual"] = None
            continue

        # ── Detectar Sección
        m_sec = PATRON_SECCION.search(linea_strip)
        if m_sec and len(linea_strip) < 120:
            guardar_articulo_actual()
            guardar_disposicion_actual()
            sec_str = linea_strip
            if sec_str not in secciones_vistas:
                secciones_vistas.add(sec_str)
            contexto["seccion_actual"] = sec_str
            continue

        # ── Detectar DISPOSICIÓN
        m_disp = PATRON_DISPOSICION.search(linea_strip)
        if m_disp:
            guardar_articulo_actual()
            guardar_disposicion_actual()
            disposicion_actual = {
                "id": f"disp_{len(disposiciones)}",
                "titulo": m_disp.group(1).strip(),
                "tipo": "disposicion",
                "_lineas_texto": linea_strip + "\n",
            }
            continue

        # ── Detectar Artículo
        m_art = PATRON_ARTICULO.search(linea_strip)
        if m_art:
            guardar_articulo_actual()
            guardar_disposicion_actual()
            num_raw = m_art.group(1).strip()
            num_id  = _normalizar_numero_articulo(num_raw)
            titulo_art = m_art.group(2).strip().rstrip('.')
            articulo_actual = {
                "id": f"art_{num_id}",
                "numero": num_raw,
                "titulo": titulo_art,
                "tipo": "articulo",
                "titulo_padre": contexto["titulo_actual"],
                "capitulo_padre": contexto["capitulo_actual"],
                "seccion_padre": contexto["seccion_actual"],
                "_lineas_texto": "",
            }
            continue

        # ── Acumular texto en el artículo / disposición abiertos
        if articulo_actual is not None:
            articulo_actual["_lineas_texto"] += linea_strip + " "
        elif disposicion_actual is not None:
            disposicion_actual["_lineas_texto"] += linea_strip + " "

    # Guardar último
    guardar_articulo_actual()
    guardar_disposicion_actual()

    # ── Deduplicar: para cada número de artículo, conservar el que tenga más texto
    vistos: dict[str, dict] = {}
    for art in articulos:
        num = art["numero"]
        if num not in vistos or len(art["texto"]) > len(vistos[num]["texto"]):
            vistos[num] = art
    articulos_dedup = list(vistos.values())

    return {
        "articulos": articulos_dedup,
        "disposiciones": disposiciones,
        "titulos": list(titulos_vistos),
        "capitulos": list(capitulos_vistos),
        "secciones": list(secciones_vistas),
    }


def extraer_referencias(texto: str) -> list[str]:
    """Extrae números de artículo referenciados en un texto."""
    matches = PATRON_REFERENCIA.findall(texto)
    # Normalizar y deduplicar
    refs = list({_normalizar_numero_articulo(m) for m in matches})
    return refs


# ──────────────────────────────────────────────────────────────────────────────
# 4. Enriquecer: entidades y conceptos clave por regex
# ──────────────────────────────────────────────────────────────────────────────

# Entidades jurídicas relevantes en la ley
ENTIDADES = [
    "SGEIC", "ECR", "SICC", "FICC", "CNMV",
    "sociedad gestora", "entidad de capital-riesgo",
    "fondo de capital-riesgo", "sociedad de capital-riesgo",
    "entidad de inversión colectiva de tipo cerrado",
    "AIFMD", "GFIA",
]

# Conceptos clave
CONCEPTOS = [
    "autorización", "capital mínimo", "requisitos", "registro",
    "supervisión", "solvencia", "honorabilidad", "conflictos de interés",
    "política de inversión", "depositario", "apalancamiento",
    "partícipes", "socios", "inversores profesionales",
    "capital riesgo", "private equity", "venture capital",
    "patrimonio", "patrimonio mínimo", "capital social",
    "presupuesto", "gastos", "comisiones", "remuneración",
    "liquidación", "disolución", "fusión", "escisión",
    "obligaciones", "derechos", "infracciones", "sanciones",
    "publicidad", "transparencia", "información", "folleto",
    "inversión", "desinversión", "cartera", "activos",
    "préstamos", "financiación", "endeudamiento",
    "gestor", "gestora", "consejo de administración",
]


def enriquecer_con_entidades(articulos: list[dict]) -> list[dict]:
    """Añade listas de entidades y conceptos mencionados en cada artículo."""
    for art in articulos:
        texto_lower = art.get("texto", "").lower()
        art["entidades_mencionadas"] = [
            e for e in ENTIDADES if e.lower() in texto_lower
        ]
        art["conceptos_mencionados"] = [
            c for c in CONCEPTOS if c.lower() in texto_lower
        ]
    return articulos


def generar_chunks_estructura(estructura: dict) -> list[dict]:
    """Genera chunks estructurados usando la estrategia de rag_engine."""
    chunker = MarkdownChunker()
    chunks: list[dict] = []

    for art in estructura["articulos"]:
        markdown = build_article_markdown(art)
        art_chunks = chunker.chunk(markdown)
        art["chunk_ids"] = []
        art["num_chunks"] = len(art_chunks)

        for idx, chunk in enumerate(art_chunks):
            chunk_id = f"{art['id']}_chunk_{idx:03d}"
            art["chunk_ids"].append(chunk_id)
            chunks.append({
                "id": chunk_id,
                "parent_id": art["id"],
                "parent_tipo": "articulo",
                "articulo_id": art["id"],
                "articulo_numero": art["numero"],
                "articulo_titulo": art["titulo"],
                "titulo_padre": art.get("titulo_padre"),
                "capitulo_padre": art.get("capitulo_padre"),
                "seccion_padre": art.get("seccion_padre"),
                "orden": chunk.order,
                "texto": chunk.text,
                "token_count": chunk.token_count,
            })

    for disp in estructura["disposiciones"]:
        markdown = build_disposition_markdown(disp)
        disp_chunks = chunker.chunk(markdown)
        disp["chunk_ids"] = []
        disp["num_chunks"] = len(disp_chunks)

        for idx, chunk in enumerate(disp_chunks):
            chunk_id = f"{disp['id']}_chunk_{idx:03d}"
            disp["chunk_ids"].append(chunk_id)
            chunks.append({
                "id": chunk_id,
                "parent_id": disp["id"],
                "parent_tipo": "disposicion",
                "disposicion_id": disp["id"],
                "disposicion_titulo": disp["titulo"],
                "orden": chunk.order,
                "texto": chunk.text,
                "token_count": chunk.token_count,
            })

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print(f"📄  Extrayendo texto de: {PDF_PATH}")
    texto_raw = extraer_texto_pdf(PDF_PATH)

    print("🧹  Limpiando texto...")
    texto = limpiar_texto(texto_raw)

    print("🔍  Parseando estructura (Títulos → Capítulos → Artículos)...")
    estructura = parsear_estructura(texto)

    print("🔗  Enriqueciendo con entidades y conceptos...")
    estructura["articulos"] = enriquecer_con_entidades(estructura["articulos"])

    print("✂️  Generando chunks estructurados...")
    estructura["chunks"] = generar_chunks_estructura(estructura)

    # Estadísticas
    print(f"\n📊  Resultado:")
    print(f"   Títulos:      {len(estructura['titulos'])}")
    print(f"   Capítulos:    {len(estructura['capitulos'])}")
    print(f"   Secciones:    {len(estructura['secciones'])}")
    print(f"   Artículos:    {len(estructura['articulos'])}")
    print(f"   Disposiciones:{len(estructura['disposiciones'])}")
    print(f"   Chunks:       {len(estructura['chunks'])}")

    total_refs = sum(len(a["referencias"]) for a in estructura["articulos"])
    print(f"   Referencias:  {total_refs} (cruces entre artículos)")

    # Guardar
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(ARTICULOS_JSON, "w", encoding="utf-8") as f:
        json.dump(estructura, f, ensure_ascii=False, indent=2)

    print(f"\n✅  Guardado en: {ARTICULOS_JSON}")
    print("    Siguiente paso: python 02_construir_grafo.py")


if __name__ == "__main__":
    main()
