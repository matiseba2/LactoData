"""Anonimiza el nombre de archivo de los informes de leche cruda (columna `archivo`).

El nombre del informe trae la fecha ("INF L11484 171222.xls") y a veces también el nombre del
cliente ("<cliente> 180225.xls"). Esta función reemplaza cada término de cliente por su código
IND-nn y deja todo lo demás igual, así `fecha_del_nombre` (ingesta_leche_cruda.py) sigue dando la
misma fecha y el nombre sigue sirviendo para agrupar las mediciones de un mismo informe.

Es la única implementación: la usan `codificar_clientes.py` (el paso que codifica `cliente`) y
`anonimizar.py`. No tiene efectos al importarla, así que cualquiera de los dos puede traerla.

Los términos son los mismos que audita `auditar_nombres_mq.py`: cada `nombre_original` y cada
`canonico` de la clave de clientes con 4 o más caracteres (más los alias confirmados), sin
acentos ni mayúsculas, omitiendo los canónicos de palabra común. Se reemplaza TODA aparición como
subcadena, no solo la palabra completa: el nombre del cliente aparece pegado a otras cosas
("<cliente>2", "<cliente>lac"), y tapar de más un nombre de archivo no cuesta nada.
"""

import re
import unicodedata

import pandas as pd

# Mismo criterio que auditar_nombres_mq.py. Si se agrega uno acá, agregarlo allá.
CANONICOS_PALABRA_COMUN = {"marca", "prote"}
MIN_LARGO = 4


def _norm(t):
    t = unicodedata.normalize("NFKD", str(t))
    return "".join(c for c in t if not unicodedata.combining(c)).lower()


def terminos_cliente(clave, alias=None):
    """{término normalizado: IND-nn} a partir de la clave (y los alias confirmados).

    `clave` tiene nombre_original, canonico y codigo; `alias`, canonico y codigo. Si un término
    apareciera con dos códigos distintos falla: sería ambiguo y no hay forma limpia de elegir.
    """
    pares = []
    for col in ["nombre_original", "canonico"]:
        for v, cod in zip(clave[col], clave["codigo"]):
            if pd.isna(v) or len(str(v)) < MIN_LARGO:
                continue
            if col == "canonico" and _norm(v) in CANONICOS_PALABRA_COMUN:
                continue
            pares.append((_norm(v), cod))
    if alias is not None:
        for v, cod in zip(alias["canonico"], alias["codigo"]):
            if pd.isna(v) or len(str(v)) < MIN_LARGO or _norm(v) in CANONICOS_PALABRA_COMUN:
                continue
            pares.append((_norm(v), cod))
    terminos = {}
    for t, cod in pares:
        if terminos.setdefault(t, cod) != cod:
            raise ValueError(f"un término de la clave tiene dos códigos ({terminos[t]} y {cod})")
    return terminos


def cargar_terminos(ruta_clave, ruta_alias=None):
    """Lee la clave privada (y los alias si existen) y devuelve los términos."""
    from pathlib import Path
    clave = pd.read_csv(ruta_clave, dtype=str, keep_default_na=False)
    alias = None
    if ruta_alias and Path(ruta_alias).exists():
        alias = pd.read_csv(ruta_alias, dtype=str, keep_default_na=False)
    return terminos_cliente(clave, alias)


def _normalizar_con_indices(texto):
    """Texto en minúsculas y sin acentos, y para cada carácter su posición en el original."""
    salida, idx = [], []
    for i, ch in enumerate(texto):
        for c in unicodedata.normalize("NFKD", ch):
            if unicodedata.combining(c):
                continue
            for c2 in c.lower():
                salida.append(c2)
                idx.append(i)
    return "".join(salida), idx


def anonimizar_nombre(nombre, terminos):
    """Reemplaza cada término de cliente del nombre por su IND-nn. Idempotente.

    Si el código queda pegado a un dígito se le pone un `_`, para que "IND-68" + "2 140225" no
    forme "IND-682" ni los dígitos del código se mezclen con los de la fecha.
    """
    if not isinstance(nombre, str) or not terminos:
        return nombre
    norm, idx = _normalizar_con_indices(nombre)
    if not norm:
        return nombre
    patron = _patron(terminos)
    tramos = [(m.start(), m.end(), terminos[m.group(0)]) for m in patron.finditer(norm)]
    if not tramos:
        return nombre
    partes, cursor = [], 0
    for ini, fin, cod in tramos:
        a, b = idx[ini], idx[fin - 1] + 1          # tramo en el texto original
        partes.append(nombre[cursor:a])
        pegado_antes = bool(partes[-1][-1:].isdigit()) if partes[-1] else (a > 0 and nombre[a - 1].isdigit())
        pegado_desp = b < len(nombre) and nombre[b].isdigit()
        partes.append(("_" if pegado_antes else "") + cod + ("_" if pegado_desp else ""))
        cursor = b
    partes.append(nombre[cursor:])
    return "".join(partes)


_CACHE = {}


def _patron(terminos):
    clave = id(terminos)
    hit = _CACHE.get(clave)
    if hit is None or hit[0] is not terminos:
        # Alternancia de las más largas a las más cortas: si no, el término corto se comería
        # parte del largo que lo contiene.
        orden = sorted(terminos, key=len, reverse=True)
        hit = (terminos, re.compile("|".join(re.escape(t) for t in orden)))
        _CACHE[clave] = hit
    return hit[1]


def anonimizar_columna_archivo(serie, terminos):
    """Aplica `anonimizar_nombre` a una columna de nombres de archivo (una vez por valor distinto)."""
    mapa = {v: anonimizar_nombre(v, terminos) for v in serie.dropna().unique()}
    return serie.map(mapa).where(serie.notna(), serie)


_CACHE_PALABRAS = {}


def _patron_palabras(terminos):
    clave = id(terminos)
    hit = _CACHE_PALABRAS.get(clave)
    if hit is None or hit[0] is not terminos:
        orden = sorted(terminos, key=len, reverse=True)
        hit = (terminos, re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(t) for t in orden) + r")(?![a-z0-9])"))
        _CACHE_PALABRAS[clave] = hit
    return hit[1]


def anonimizar_palabras(texto, terminos):
    """Como `anonimizar_nombre`, pero solo palabras completas. Idempotente.

    Es para texto libre (columnas del catálogo), donde reemplazar subcadenas rompería palabras
    comunes que contienen un canónico corto (un cliente de 4 letras caería dentro de una palabra
    de azúcares). El nombre de
    un cliente que se cuela en una columna de códigos ("CUÑETE CARTON <cliente>") es una palabra
    completa, así que alcanza.
    """
    if not isinstance(texto, str) or not terminos:
        return texto
    norm, idx = _normalizar_con_indices(texto)
    partes, cursor = [], 0
    for m in _patron_palabras(terminos).finditer(norm):
        a, b = idx[m.start()], idx[m.end() - 1] + 1
        partes.append(texto[cursor:a])
        partes.append(terminos[m.group(1)])
        cursor = b
    if not partes:
        return texto
    partes.append(texto[cursor:])
    return "".join(partes)
