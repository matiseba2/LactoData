"""Familia por descripción clara, para las muestras de MQ que el modelo dejó «sin clasificar».

Criterio (decisión del usuario, 2026-11-09): una muestra sin clasificar se clasifica solo si la
descripción nombra el producto sin ambigüedad («CREMA CRUDA», «MANTECA», «ARROZ CON LECHE»). Si la
descripción no alcanza para decidir sin mirar la composición («LECHE ENTERA» sin estado, «ALIMENTO»,
«FORMULA», «FAT FILLED»), queda sin clasificar. Lo mismo vale para el subtipo: sale del texto
(catalogo.SUBTIPOS) y, si el texto no lo dice, queda vacío.

Cada regla sigue la convención de la clasificación manual de Senasa donde hay ejemplos;
`validar()` muestra, regla por regla, a qué familia fueron a mano las descripciones que la cumplen.

No se aplica a muestras que ya tienen familia (manual o del modelo): solo llena «sin clasificar».
"""

import re
import unicodedata

import pandas as pd

SIN = "sin clasificar"
INI = r"^\W*"          # la descripción empieza con el producto (después de «·», espacios o signos)

# (nombre, familia, patrón que debe estar, patrón que la descarta)
REGLAS = [
    ("crema", "manteca / crema", INI + r"CREMA\b",
     r"SUERO|AZUL|PASTELERA|HELAD|QUESO|DULCE DE LECHE|SABOR|MEZCLA"),
    ("manteca", "manteca / crema", INI + r"MANTECA\b", r"SUERO|ACEITE|MANI\b|CACAO|SABOR"),
    ("arroz con leche / postre", "arroz con leche / postre", r"ARROZ CON LECHE|\bPOSTRE\b|\bFLAN\b", r"POLVO|PREMEZCLA"),
    ("carnicos", "carnicos",
     r"\b(SALAME|SALAMIN|CHORIZO|BONDIOLA|SALCHICHA|SPIANATT?A|JAMON|MORTADELA)\b|HARINA DE CARNE|" + INI + r"CARNE\b",
     r"QUESO|SABOR"),
    ("ricota", "queso", r"\bRICOTT?A\b", r"SUERO|SABOR|RELLENO"),
    ("wpc", "concentrado proteico", r"\bWPC\b", r"SUERO|PERMEADO|MEZCLA|CREMA|FORMULA"),
    ("soja", "base nutricional", r"\bSOJA\b", r"DULCE DE LECHE|AGUA|ACEITE|HARINA"),
    ("harina / cereal / lactosa", "panificados / otros",
     r"\bHARINA\b|" + INI + r"(LACTOSA|GLUTEN|MAIZ|AVENA|ARROZ)\b", r"CARNE|ARROZ CON LECHE|SUERO|LECHE"),
    ("yogur / bebida lactea", "yogur / bebida lactea", r"\bYOGUR|\bYOGHUR|BEBIDA LACTEA", r"POLVO|SABOR YOGUR|HELAD"),
    ("leche en polvo", "leche en polvo",
     INI + r"LECHE\b.*\b(POLVO|INSTANTANEA)\b", r"FAT FILLED|COMPOSTO|FORMULA|MODIFICADA|SUERO|CRUDA|SABOR|PREMEZCLA|CACAO|CHOCOLAT"),
    ("suero", "suero / permeado", INI + r"(SUERO|PERMEADO)\b", r"CREMA|MEZCLA|MANTECA|WPC|CONCENTRADO|QUESO EN POLVO"),
    ("dulce de leche", "dulce de leche", INI + r"DULCE DE LECHE\b", r"SABOR|HELAD|RELLENO|ALFAJOR"),
]
_REGLAS = [(n, f, re.compile(si), re.compile(no)) for n, f, si, no in REGLAS]


def normalizar(t):
    """Mayúsculas sin acentos; «·» (palabra tapada por la anonimización) queda como separador."""
    if not isinstance(t, str):
        return ""
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().upper()
    return re.sub(r"\s+", " ", t).strip()


def familia_clara(texto):
    """(familia, nombre de la regla) si exactamente una familia sale de las reglas; (None, None) si no."""
    t = normalizar(texto)
    if not t:
        return None, None
    hallado = [(f, n) for n, f, si, no in _REGLAS if si.search(t) and not no.search(t)]
    if len({f for f, _ in hallado}) != 1:      # ninguna, o dos familias distintas: no es una descripción clara
        return None, None
    return hallado[0]


def aplicar(familia, texto):
    """familia y texto: Series con el mismo índice. Devuelve (familia nueva, regla usada o NA).
    Solo cambia las filas que están «sin clasificar»."""
    familia = familia.copy()
    regla = pd.Series(pd.NA, index=familia.index, dtype="object")
    for i in familia.index[familia.eq(SIN)]:
        f, n = familia_clara(texto[i])
        if f:
            familia[i] = f
            regla[i] = n
    return familia, regla


def validar(descripcion, familia_manual):
    """Tabla por regla: cuántas descripciones con familia manual la cumplen y a qué familia fueron a mano."""
    r = pd.DataFrame([familia_clara(t) for t in descripcion], index=descripcion.index, columns=["familia_regla", "regla"])
    r["manual"] = familia_manual
    r = r[r["regla"].notna() & r["manual"].ne(SIN)]
    filas = []
    for (n, f), g in r.groupby(["regla", "familia_regla"]):
        vc = g["manual"].value_counts()
        filas.append(dict(regla=n, familia_regla=f, n_manual=len(g), coincide=round(float((g["manual"] == f).mean()), 3),
                          otras=", ".join(f"{k} ({v})" for k, v in vc.items() if k != f)))
    return pd.DataFrame(filas)
