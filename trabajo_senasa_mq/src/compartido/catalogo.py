"""Normalización del catálogo encolumnado y consistencia de los quesos.

El catálogo (`Senasa encolumnado.xlsx`) se armó aparte del notebook, así que arrastra dos cosas
que en `proyecto.ipynb` ya estaban resueltas: el vocabulario en portugués de los rótulos de
exportación (QUEIJO PARMESÃO) y los errores de tipeo repetidos (MOZARELLA, POR SALUT). Acá se
aplican las mismas tablas a las columnas derivadas del catálogo.

Además agrega la consistencia del queso. El catálogo clasifica por lo que dice el rótulo, y el
rótulo unas veces nombra la variedad (QUESO PRATO) y otras la categoría del CAA (QUESO DE
MEDIANA HUMEDAD). Son el mismo queso y no se juntarían en un filtro. La columna
`desc_consistencia` los lleva a la escala del CAA, que clasifica por humedad:

    baja humedad    (< 36 %)   = duro
    mediana humedad (36-46 %)  = semiduro
    alta humedad    (>= 46 %)  = blando

y aparte: untable, procesado/fundido y mezcla (hebras de varias variedades). Lo que el rótulo
no permite decidir queda "sin especificar" — no se asume.

`anonimizar.py` llama a `normalizar_catalogo` antes de codificar y redactar, así el archivo
compartido sale normalizado y con la columna nueva.
"""

import re
import unicodedata

import pandas as pd

# ---------------------------------------------------------------------------------------------
# Vocabulario: las mismas tablas que usa proyecto.ipynb (sección 5), con los casos que aparecen
# en el catálogo y no en el consolidado.
# ---------------------------------------------------------------------------------------------

PT_ES = {
    "QUEIJO": "QUESO", "QUEIJOS": "QUESOS", "LEITE": "LECHE", "MANTEIGA": "MANTECA",
    "MANTEIGUEIRA": "MANTECA", "REQUEIJAO": "QUESO CREMA", "IOGURTE": "YOGUR",
    "PARMESAO": "PARMESANO", "MUSSARELA": "MOZZARELLA", "MUCARELA": "MOZZARELLA",
    "DOCE": "DULCE", "SORO": "SUERO", "PO": "POLVO", "INTEGRAL": "ENTERA",
    "DESNATADO": "DESCREMADO", "RALADO": "RALLADO", "CREME": "CREMA",
    "GORDURA": "GRASA", "LACTOSE": "LACTOSA", "RESTRICAO": "RESTRICCION",
    "COM": "CON", "SEM": "SIN", "COMPOSTO": "COMPUESTO", "OITO": "OCHO",
}

ERRORES = {
    "MOZARELLA": "MOZZARELLA", "MUZZARELLA": "MOZZARELLA", "MUSARELLA": "MOZZARELLA",
    "MOZZRELLA": "MOZZARELLA", "MOZZA": "MOZZARELLA",
    "POR SALUT": "PORT SALUT", "PATEGAS": "PATEGRAS", "SANWICH": "SANDWICH",
    "PAT": "PATEGRAS",
    "PRATTO": "PRATO", "GAUDA": "GOUDA", "DAMBO": "DANBO", "PARMESADO": "PARMESANO",
    "REGGIANITTO": "REGGIANITO", "REGGINIATO": "REGGIANITO", "REGIANITO": "REGGIANITO",
    "PROVOLETAS": "PROVOLETA", "EMMENTHAL": "EMMENTAL", "CHEEDAR": "CHEDDAR",
    "REPOSERO": "REPOSTERO", "REPSOTERO": "REPOSTERO", "RESPOTERO": "REPOSTERO",
    "FAMIIAR": "FAMILIAR", "FAMILEAR": "FAMILIAR", "FAMILILAR": "FAMILIAR", "FLIAR": "FAMILIAR",
    "HELADOF": "HELADERO", "HELDAERO": "HELADERO", "HELDERO": "HELADERO",
    "PREMIUN": "PREMIUM", "SEMIDESCREMDA": "SEMIDESCREMADA", "INSNTANTANEA": "INSTANTANEA",
}


def sin_acentos(texto):
    return "".join(c for c in unicodedata.normalize("NFKD", str(texto))
                   if not unicodedata.combining(c))


def normalizar_texto(s):
    """Mayúsculas, sin acentos, portugués traducido y errores de tipeo corregidos."""
    if not isinstance(s, str):
        return s
    s = sin_acentos(s.upper())
    # "LECHE L.V. RED. EN LACTOSA" = leche larga vida reducida en lactosa: es una UAT.
    s = re.sub(r"\bL\.?\s?V\.?(?=\s|$)", "LARGA VIDA", s)
    s = re.sub(r"\bRED\.", "REDUCIDA", s)
    for pt, es in PT_ES.items():
        s = re.sub(rf"\b{pt}\b", es, s)
    for mal, bien in ERRORES.items():
        s = re.sub(rf"\b{mal}\b", bien, s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------------------------
# Consistencia del queso
# ---------------------------------------------------------------------------------------------

# Variedad -> categoría del CAA. Es la tabla VARIEDADES del notebook, contrastada allí contra la
# humedad medida, con las variedades que sólo aparecen en el catálogo.
VARIEDADES = {
    "baja humedad": [
        "PARMESANO", "PARMIGIANO", "REGGIANITO", "REGGIANO", "SARDO", "ROMANO", "ROMANITO",
        "SBRINZ", "GRANA", "GOYA", "PROVOLONE", "PEPATO",
    ],
    "mediana humedad": [
        "GOUDA", "DANBO", "PATEGRAS", "TYBO", "PRATO", "HOLANDA", "GRUYERE", "GRUYERITO",
        "EDAM", "COLONIA", "CHEDDAR", "MAR DEL PLATA", "SAINT PAULIN", "PROVOLETA",
        "CASCARA COLORADA", "CRIOLLO", "TILSIT", "SANDWICH", "AZUL", "FONTINA", "EMMENTAL",
    ],
    "alta humedad": [
        "CREMOSO", "CUARTIROLO", "PORT SALUT", "RICOTA", "MASCARPONE", "BRIE", "CAMEMBERT",
        "PETIT SUISSE", "MOZZARELLA", "PASTA HILADA",
    ],
}
# AZUL y FONTINA no estaban en la tabla del notebook porque allí midieron humedades dispersas
# (38-70). Era por mezclar las versiones fundidas: acá el procesado se resuelve antes, y lo que
# queda mide 40-46 % (azul, 504 muestras, mediana 43,3) y 37-39 % (fontina, 16). Son de mediana
# humedad. EMMENTAL es semiduro (criterio del laboratorio; dos muestras, sin humedad medida).

# Lo que el rótulo dice de forma explícita manda sobre la variedad.
EXPLICITA = {
    "baja humedad": [r"\bDURO\b", r"\bBAJA HUMEDAD\b", r"\bPASTA DURA\b"],
    "mediana humedad": [r"\bSEMIDURO\b", r"\bMEDIANA HUMEDAD\b", r"\bPASTA SEMIDURA\b"],
    "alta humedad": [r"\bBLANDO\b", r"\bALTA HUMEDAD\b", r"\bPASTA BLANDA\b"],
}

PROCESADO = r"\bPROCESADO\b|\bFUNDIDO\b|\bADEREZO\b|\bSALSA\b|\bPARA FUNDIR\b"
# Queso blanco es untable (criterio del laboratorio).
UNTABLE = r"\bUNTABLE\b|\bPARA UNTAR\b|\bQUESO (DE )?CREMA\b|\bDOBLE CREMA\b|\bQUESO BLANCO\b"
# Hebras con nombre de fantasía: son mezclas de varias variedades.
MEZCLA = r"\bCUATRO QUESOS\b|\bITALIANO\b|\bNAPOLITANO\b|\bMEDITERRANEO\b|\bMEZCLA\b"


def consistencia_queso(texto):
    """(consistencia, origen) a partir del rótulo normalizado de un queso."""
    t = texto if isinstance(texto, str) else ""
    if re.search(PROCESADO, t):
        return ("procesado", "rotulo")
    if re.search(UNTABLE, t):
        return ("untable", "rotulo")
    for nivel, patrones in EXPLICITA.items():
        if any(re.search(p, t) for p in patrones):
            return (nivel, "rotulo")
    if re.search(MEZCLA, t):
        return ("mezcla", "rotulo")
    niveles = {nivel for nivel, vs in VARIEDADES.items()
               if any(re.search(rf"\b{v}\b", t) for v in vs)}
    if len(niveles) == 1:
        return (niveles.pop(), "variedad")
    if len(niveles) > 1:
        return ("mezcla", "variedad")
    # Sin variedad nombrada. Dos formatos la implican:
    #   - rallado: se hace de reggianito o parmesano (criterio del laboratorio);
    #   - barra: gouda, tybo, pategrás o edam, todos de mediana humedad.
    if re.search(r"\bRALLADO\b|\bPARA RALLAR\b", t):
        return ("baja humedad", "formato")
    if re.search(r"\bBARRA\b", t):
        return ("mediana humedad", "formato")
    return ("sin especificar", "sin dato")


# Sinónimos en desc_formato: el catálogo los tenía como valores distintos.
FORMATOS = {
    "PARA UNTAR": "UNTABLE", "EN HEBRAS": "HEBRAS", "HEBRA": "HEBRAS", "BLOCK": "BLOQUE",
    "TROZADA": "TROZADO", "FETEADA": "FETEADO",
}


# ---------------------------------------------------------------------------------------------
# Ejes: un valor por columna
# ---------------------------------------------------------------------------------------------
#
# El catálogo traía en `desc_subtipo` todas las palabras clave que reconoció, unidas con " + ":
# "CONCENTRADO DE PROTEINAS DE SUERO + WPC + SUERO DE QUESERIA + POLVO". Mezcla qué es (WPC),
# el estado (polvo), el origen (suero de quesería) y sinónimos (WPC = concentrado de proteínas
# de suero). Acá se separa en un valor por eje. Cada tabla es una lista ordenada
# (regex, canónico): el primero que coincide gana.

def _b(patron):
    return rf"\b(?:{patron})\b"


ESTADO = [
    ("EN POLVO|POLVO|PWD", "POLVO"),
    ("LIQUID[AO]|FLUID[AO]", "LIQUIDO"),
    ("CONGELAD[AO]", "CONGELADO"),
    ("REFRIGERAD[AO]", "REFRIGERADO"),
]
# Familias cuyo nombre ya fija el estado.
ESTADO_POR_FAMILIA = {"leche en polvo": "POLVO", "leche fluida": "LIQUIDO"}

GRASA = [
    ("PARCIALMENTE DESCREMAD[AO]|SEMIDESCREMAD[AO]", "PARCIALMENTE DESCREMADA"),
    ("DESCREMAD[AO]|DESNATAD[AO]", "DESCREMADA"),
    ("ENTER[AO]", "ENTERA"),
    ("CONTENIDO GRASO REDUCIDO|REDUCID[AO] EN GRASAS?", "REDUCIDO EN GRASA"),
    ("SEMIGRAS[AO]", "SEMIGRASO"),
    ("GRAS[AO]", "GRASO"),
    ("MAGR[AO]", "MAGRO"),
    ("LIGHT|LIGTH", "LIGHT"),
    ("FAT FILLED", "FAT FILLED"),
]

LACTOSA = [
    ("SIN DESLACTOSAR", "SIN DESLACTOSAR"),
    ("DESLACTOSAD[AO]", "DESLACTOSADA"),
    ("REDUCID[AO] EN LACTOSA|BAJ[AO] EN LACTOSA", "REDUCIDA EN LACTOSA"),
    ("SIN LACTOSA|LACTOSE FREE|RESTRICCION DE LACTOSA", "SIN LACTOSA"),
    ("CON LACTASA", "CON LACTASA"),
    ("SIN LACTASA", "SIN LACTASA"),
]

# Formatos que el extractor había puesto como subtipo.
FORMATO_DESDE_SUBTIPO = [("BLOCK|BLOQUE", "BLOQUE"), ("SACHET", "SACHET"), ("MINI", "MINI")]

# Subtipo canónico por familia. El orden es la prioridad cuando el rótulo dice dos cosas.
_QUESO = (
    [("SANDWICH", "PATEGRAS SANDWICH"), ("QUESO (DE )?CREMA|DOBLE CREMA", "QUESO CREMA"),
     ("CUATRO QUESOS", "CUATRO QUESOS"), ("ITALIANO", "ITALIANO"), ("NAPOLITANO", "NAPOLITANO"),
     ("MEDITERRANEO", "MEDITERRANEO")]
    + [(v, v) for lista in VARIEDADES.values() for v in lista if v != "PASTA HILADA"]
    + [("PASTA HILADA", "PASTA HILADA"), ("MASA", "MASA"),
       ("PREPARACION ALIMENTICIA", "PREPARACION ALIMENTICIA")]
)

SUBTIPOS = {
    "queso": _QUESO,
    # Criterio del laboratorio: clásico, tradicional y común son el familiar; alfajorero,
    # panadero, pastelero y conito son el repostero. El nombre original queda en atributos.
    "dulce de leche": [
        ("COLONIAL", "COLONIAL"), ("HELADERO", "HELADERO"),
        ("REPOSTERO|ALFAJORERO|PANADERO|PASTELERO|CONITO", "REPOSTERO"),
        ("FAMILIAR|CLASICO|TRADICIONAL|COMUN", "FAMILIAR"),
    ],
    "leche fluida": [
        ("PREMEZCLA", "PREMEZCLA"), ("MEZCLA", "MEZCLA"), ("UAT|UHT|LARGA VIDA", "UAT"),
        ("CRUDA", "CRUDA"), ("PASTEURIZADA", "PASTEURIZADA"), ("CONCENTRADA", "CONCENTRADA"),
        ("CALOSTRAL|CALOSTRO", "CALOSTRAL"), ("CHOCOLATADA", "CHOCOLATADA"),
    ],
    "leche en polvo": [
        ("FORMULA INFANTIL", "FORMULA INFANTIL"), ("COMPUESTO LACTEO", "COMPUESTO LACTEO"),
        ("MODIFICADA", "MODIFICADA"),
    ],
    # Prioridad del laboratorio: lo más procesado primero; el desmineralizado es el estándar y
    # va al final, con el suero simple. "Líquido" no compite: es estado.
    "suero / permeado": [
        ("PERMEADO DE SUERO", "PERMEADO DE SUERO"), ("PERMEADO DE LECHE", "PERMEADO DE LECHE"),
        ("PERMEADO DE QUESO", "PERMEADO DE QUESO"), ("PERMEADO", "PERMEADO"),
        ("WPC|CONCENTRADO DE PROTEINAS? DE SUERO", "WPC"), ("CONCENTRADO", "CONCENTRADO"),
        ("REENGRASADO", "REENGRASADO"), ("SIN DESNATAR", "SIN DESNATAR"),
        ("DESMINERALIZADO", "DESMINERALIZADO"), ("SUERO", "SUERO"),
    ],
    "concentrado proteico": [
        ("WPC|CONCENTRADO DE PROTEINAS? DE SUERO", "WPC"), ("WPI", "WPI"),
        ("MPC|CONCENTRADO DE PROTEINAS? DE LECHE", "MPC"),
        ("CASEINATO DE SODIO", "CASEINATO DE SODIO"), ("CASEINATO DE CALCIO", "CASEINATO DE CALCIO"),
        ("CASEINATOS?", "CASEINATO"), ("CASEINA", "CASEINA"),
        ("PROTEINAS LACTEAS", "PROTEINAS LACTEAS"),
    ],
    "manteca / crema": [
        ("MANTECA", "MANTECA"), ("CREMA DE LECHE", "CREMA DE LECHE"),
        ("DOBLE CREMA", "DOBLE CREMA"), ("CREMA", "CREMA"),
    ],
    "yogur / bebida lactea": [
        ("POLVO PARA PREPARAR", "POLVO PARA PREPARAR"), ("YOGUR|YOGURT", "YOGUR"),
        ("BEBIDA LACTEA", "BEBIDA LACTEA"),
    ],
    "base nutricional": [
        ("CASEINATOS?", "CASEINATOS"), ("SUSTITUTO LACTEO", "SUSTITUTO LACTEO"),
        ("COMPUESTO LACTEO", "COMPUESTO LACTEO"), ("FORMULA LACTEA", "FORMULA LACTEA"),
    ],
    "arroz con leche / postre": [
        ("ARROZ CON LECHE", "ARROZ CON LECHE"), ("PAN DULCE", "PAN DULCE"),
        ("SAMBAYON|ZAMBAYON|SABAYON", "SAMBAYON"), ("POSTRE", "POSTRE"),
    ],
    "agua / hisopado": [
        ("AGUA (DE )?ENJUAGUE", "AGUA DE ENJUAGUE"), ("HISOPADO|HISOPO", "HISOPADO"),
        ("PERMEADO", "PERMEADO"), ("CONDENSADO", "CONDENSADO"), ("VAHOS", "VAHOS"),
    ],
}

# La palabra que nombra a la familia no es un atributo ("SUERO LIQUIDO" -> estado LIQUIDO y
# nada más).
GENERICO_FAMILIA = {
    "queso": "QUESOS?", "suero / permeado": "SUERO", "leche fluida": "LECHE",
    "leche en polvo": "LECHE", "manteca / crema": "MANTECA", "dulce de leche": "DULCE DE LECHE",
}

# Sinónimos de atributos, para que no queden dos grafías del mismo.
ATRIBUTOS = [
    ("SOYA|SOJA|SOY", "BASE DE SOJA"), ("L ?K DIABETES", "DIABETES"),
    ("OLEO MARGARINA", "OLEOMARGARINA"), ("AGUA ENJUAGUE", "AGUA DE ENJUAGUE"),
    ("LINEA ACEITE", "LINEA DE ACEITE"), ("NANO", "NANOFILTRADO"), ("HOMO", "HOMOGENEIZADOR"),
    ("PROT|ALTA PT", "ALTA PROTEINA"),
]

CONSERVAR_ORIGINAL = {"dulce de leche"}

CONECTORES = {"Y", "DE", "DEL", "CON", "A", "AL", "EN", "POR", "PARA", "TIPO", "O", "SABOR"}


def primer_match(texto, tabla):
    """(canónico, regex) del primer patrón de la tabla que aparece en el texto."""
    for patron, canonico in tabla:
        if re.search(_b(patron), texto):
            return canonico, patron
    return None, None


def limpiar_lista(v, conservar_inicio=()):
    """Ordena una columna desc_* que es lista.

    Saca el elemento vacío o tapado ("+ GRASO" -> "GRASO"), el que quedó reducido a un conector
    ("REPOSTERO Y" -> "REPOSTERO") y el que está contenido en otro del mismo valor
    ("DE QUESO + PERMEADO DE QUESO" -> "PERMEADO DE QUESO"). `conservar_inicio` deja el conector
    inicial cuando significa algo: "DE MANTECA" es el origen del suero.
    """
    if not isinstance(v, str):
        return v
    partes = []
    for x in v.split("+"):
        x = x.strip(" ·")
        palabras = x.split()
        while palabras and palabras[-1] in CONECTORES:
            palabras.pop()
        while (palabras and palabras[0] in CONECTORES - {"SABOR", "TIPO"}
               and not (palabras[0] in conservar_inicio and len(palabras) > 1)):
            palabras.pop(0)
        x = " ".join(palabras)
        if x and x not in partes:
            partes.append(x)
    partes = [x for x in partes if not any(x != y and x in y for y in partes)]
    return " + ".join(partes) if partes else None


def _elementos(v):
    return [x.strip() for x in v.split("+") if x.strip()] if isinstance(v, str) else []


def desglosar(fila):
    """Un valor por eje a partir del rótulo desglosado. Devuelve un dict de columnas nuevas.

    Regla de conservación: cada elemento de la lista original de `desc_subtipo` termina en un
    eje o en `desc_atributos`. Lo único que se descarta es lo que ya está dicho por el eje (el
    sinónimo canonizado) o la palabra que nombra a la familia.
    """
    fam = fila.desc_familia if isinstance(fila.desc_familia, str) else ""
    texto = " ".join(str(fila[c]) for c in ("desc_producto", "desc_subtipo", "desc_claims",
                                             "desc_formato") if isinstance(fila[c], str))
    texto = normalizar_texto(texto)

    subtipo, patron_sub = primer_match(texto, SUBTIPOS.get(fam, []))
    estado, patron_est = primer_match(texto, ESTADO)
    if estado is None:
        estado = ESTADO_POR_FAMILIA.get(fam)
    grasa, patron_gra = primer_match(texto, GRASA)
    lactosa, patron_lac = primer_match(texto, LACTOSA)

    # Lo que consumen los ejes se quita de cada elemento; lo que sobra es atributo.
    formatos = _elementos(fila.desc_formato)
    atributos = []
    generico = GENERICO_FAMILIA.get(fam)
    for elem in _elementos(fila.desc_subtipo):
        if subtipo and (elem == subtipo or elem in subtipo):
            continue
        e = elem
        for patron, canonico in FORMATO_DESDE_SUBTIPO:
            if re.search(_b(patron), e):
                if canonico not in formatos:
                    formatos.append(canonico)
                e = re.sub(_b(patron), " ", e)
        # Cualquier palabra de un eje sale del elemento, no sólo la que ganó. Si era otro valor
        # del mismo eje ("DESLACTOSADA + SIN LACTOSA"), se conserva como atributo; salvo que el
        # ganador ya lo contenga (PARCIALMENTE DESCREMADA dice DESCREMADA).
        for tabla, ganador in ((ESTADO, estado), (GRASA, grasa), (LACTOSA, lactosa)):
            for patron, canonico in tabla:
                if re.search(_b(patron), e):
                    e = re.sub(_b(patron), " ", e)
                    if canonico != ganador and not (ganador and canonico in ganador):
                        atributos.append(canonico)
        if generico:
            e = re.sub(_b(generico), " ", e)
        e = limpiar_lista(e, conservar_inicio=("DE", "CON")) or ""
        if not e or e == subtipo:
            continue
        # Un sinónimo del subtipo ("CONCENTRADO DE PROTEINAS DE SUERO" -> WPC) ya está dicho.
        # La excepción es el dulce de leche: ahí la unificación es un criterio del laboratorio
        # y el nombre original ("ALFAJORERO") se conserva para que sea reversible.
        if patron_sub and re.fullmatch(_b(patron_sub), e) and fam not in CONSERVAR_ORIGINAL:
            continue
        for patron, canonico in ATRIBUTOS:
            e = re.sub(_b(patron), canonico, e)
        if e != subtipo:
            atributos.append(e)

    # Claims: grasa y lactosa salen; el resto (gluten, TACC, uso industrial...) queda.
    claims = []
    for elem in _elementos(fila.desc_claims):
        e = elem
        for p in (patron_gra, patron_lac):
            if p:
                e = re.sub(_b(p), " ", e)
        e = limpiar_lista(e)
        if e:
            claims.append(e)

    return {
        "desc_subtipo": subtipo,
        "desc_estado": estado,
        "desc_grasa": grasa,
        "desc_lactosa": lactosa,
        "desc_atributos": limpiar_lista(" + ".join(atributos), conservar_inicio=("DE", "CON")),
        "desc_claims": limpiar_lista(" + ".join(claims)),
        "desc_formato": limpiar_lista(" + ".join(formatos)),
    }


ORDEN_EJES = ["desc_subtipo", "desc_estado", "desc_grasa", "desc_lactosa", "desc_atributos",
              "desc_consistencia", "desc_consistencia_origen"]


def normalizar_catalogo(df):
    """Columnas derivadas normalizadas, un valor por eje y consistencia de los quesos.

    No toca `Descripción`. Conserva `desc_subtipo_original` sólo en memoria para la auditoría
    de `python catalogo.py`; no se escribe.
    """
    df = df.copy()
    for col in ("desc_producto", "desc_subtipo", "desc_formato", "desc_claims", "desc_sabor"):
        if col in df.columns:
            df[col] = df[col].map(normalizar_texto)
    # Sinónimos de formato que el catálogo separaba.
    if "desc_formato" in df.columns:
        for a, b in FORMATOS.items():
            df["desc_formato"] = df.desc_formato.str.replace(rf"\b{a}\b", b, regex=True)

    # Consistencia de los quesos, sobre el texto completo (antes de desglosar).
    es_queso = df.desc_familia.eq("queso")
    texto = (df.desc_producto.fillna("") + " " + df.desc_subtipo.fillna("")
             + " " + df.desc_formato.fillna("")).map(normalizar_texto)
    res = texto.where(es_queso).map(lambda t: consistencia_queso(t) if isinstance(t, str) else (None, None))
    df["desc_consistencia"] = res.map(lambda r: r[0])
    df["desc_consistencia_origen"] = res.map(lambda r: r[1])

    # Un valor por eje.
    df.attrs["desc_subtipo_original"] = df.desc_subtipo.copy()
    ejes = pd.DataFrame([desglosar(f) for _, f in df.iterrows()], index=df.index)
    for c in ejes.columns:
        df[c] = ejes[c]

    # Los ejes van juntos, después del producto.
    cols = [c for c in df.columns if c not in ORDEN_EJES]
    i = cols.index("desc_producto") + 1
    return df[cols[:i] + ORDEN_EJES + cols[i:]]


def auditar_conservacion(df):
    """Palabras de la lista original que no aparecen en ningún eje ni en atributos.

    Se compara palabra por palabra, sin conectores ni la palabra de la familia, y aceptando el
    canónico de cualquier tabla como representación del sinónimo.
    """
    perdidos = {}
    original = df.attrs["desc_subtipo_original"]
    for idx, v in original.items():
        fila = df.loc[idx]
        fam = fila.desc_familia
        destino = " ".join(str(fila[c]) for c in ("desc_subtipo", "desc_estado", "desc_grasa",
                                                  "desc_lactosa", "desc_atributos",
                                                  "desc_formato", "desc_claims")
                           if isinstance(fila[c], str))
        tablas = [SUBTIPOS.get(fam, []), ESTADO, GRASA, LACTOSA, FORMATO_DESDE_SUBTIPO, ATRIBUTOS]
        for elem in _elementos(v):
            if elem in destino:
                continue
            if any(re.search(_b(p), elem) and c in destino for t in tablas for p, c in t):
                continue
            generico = GENERICO_FAMILIA.get(fam)
            resto = re.sub(_b(generico), " ", elem) if generico else elem
            faltan = [w for w in resto.split() if w not in CONECTORES and w not in destino.split()]
            if faltan:
                perdidos[elem] = perdidos.get(elem, 0) + 1
    return perdidos


if __name__ == "__main__":
    import sys
    origen = sys.argv[1] if len(sys.argv) > 1 else "datos/Senasa encolumnado_con_nombres.xlsx"
    df = normalizar_catalogo(pd.read_excel(origen))
    pd.set_option("display.width", 200)
    q = df[df.desc_familia.eq("queso")]
    print(pd.crosstab(q.desc_consistencia, q.desc_consistencia_origen, margins=True))
    for fam, g in df.groupby("desc_familia"):
        antes = df.attrs["desc_subtipo_original"][g.index].notna().sum()
        print(f"\n=== {fam}: {len(g)} filas | con subtipo antes {antes} -> ahora {g.desc_subtipo.notna().sum()}")
        print("   subtipo:", g.desc_subtipo.fillna("(vacio)").value_counts().to_dict())
        print("   estado:", g.desc_estado.fillna("(vacio)").value_counts().to_dict())
        print("   grasa:", g.desc_grasa.fillna("(vacio)").value_counts().to_dict())
        print("   lactosa:", g.desc_lactosa.fillna("(vacio)").value_counts().to_dict())
        print("   atributos:", g.desc_atributos.value_counts().head(12).to_dict())
    perdidos = auditar_conservacion(df)
    print(f"\nelementos originales sin destino: {sum(perdidos.values())} ({len(perdidos)} distintos)")
    print(dict(sorted(perdidos.items(), key=lambda kv: -kv[1])[:30]))
