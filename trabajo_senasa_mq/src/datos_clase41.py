"""Lectura, corrección y unión de los datos del agrupamiento de la clase 41 (un solo lugar).

Lo usan las secciones 1 a 4 y 6 de trabajo_senasa_mq.ipynb (vía celdas_clase41.py, revision_excel.py y
categorizar.py). Las rutas salen de rutas.py.

Reglas de lectura (Senasa encolumnado.xlsx y muestras_ancha.csv no se modifican):
  - correcciones puntuales de correcciones_clase41.csv (Identificador, campo, antes, despues, motivo;
    despues vacío = valor anulado por error de carga);
  - coma decimal, se toma el valor y no la incertidumbre («+/-»), «< LD» (o «&lt;») = 0;
  - tipeos: «,.» → «,»; «NO DETECTADO» vale 0 por defecto (como «< LD»);
  - quedan fuera del análisis las muestras de Senasa con humedad + sólidos totales fuera de 100 ± 1
    (si falta alguno de los dos, no se excluyen por esto).

Fuentes de MQ (leer_mq). MQ tiene que verse como Senasa: los resultados y la familia de las muestras MQ salen de
«MQ encolumnado.xlsx» (resultados/excel/, lo arma mq_encolumnado.py; mismas 177 columnas y mismo formato que «Senasa
encolumnado.xlsx»: texto con coma decimal, y la humedad en `humedad_derivada_100_menos_solidos` porque MQ mide sólidos
totales). Las solo MQ llevan ahí su desc_familia, desc_producto y desc_tipo_muestra, así que `unir` les aplica las mismas
reglas que a Senasa (FAMILIAS_NO_LACTEAS, queso confirmado por el catálogo). Las muestras de tipo MO (oficiales) de
`muestras_ancha.csv` NO entran por MQ (decisión de Matías): MQ encolumnado las deja afuera a propósito porque ya forman
parte de Senasa encolumnado, y las que tienen código en Senasa entran por Senasa como cualquier muestra de Senasa (de
las 1839 MO, 1511 están en Senasa y 328 no; esas 328 no se agregan). Los valores de grasa, proteína, cenizas y humedad de MQ encolumnado coinciden con los de la tabla ancha
(donde MQ encolumnado tiene valor; lo que la tabla ancha completaba desde el catálogo queda vacío; leer_mq lo verifica). Una familia «sin clasificar» (desc_familia de las que el modelo de mq_tipo_muestra no
asigna) NO es SIN_FAMILIA: no se sabe que no sea láctea, y entra como entra en Senasa.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

import rutas as R

RUTA_SENASA = R.SENASA
RUTA_MQ = R.MQ_ANCHA
RUTA_CORR = R.CORRECCIONES
RUTA_MQ_ENC = R.EXCEL / "MQ encolumnado.xlsx"
DESC_MQ = ["desc_familia", "desc_producto", "desc_tipo_muestra", "desc_subtipo", "desc_estado"]

SEMILLA = 42                     # la cátedra pide semilla 42 en todo el trabajo
N_INIT_KMEDIAS = 50              # reinicios de k-medias. Con 10, el k elegido dependía de la semilla (mínimos
                                 # locales del k-medias preliminar); con 50, todas las semillas probadas dan lo mismo
SIN_FAMILIA = "sin familia (solo MQ)"
ORIGEN = {"humedad": "Resultado Humedad g/ 100g (*)",
          "grasa": "Resultado Materia Grasa g/ 100g (*)_x",
          "proteinas": "Resultado Proteínas g/ 100g (*)",
          "cenizas": "Resultado Cenizas g/ 100g"}
COL_SOLIDOS = "Resultado Sólidos totales g/ 100g_x"
COLUMNAS = list(ORIGEN)
TOL_HUM_SOL = 1.0
SUMA_MAX = 102
# Familias que no son productos lácteos: no entran al agrupamiento (secciones 2, 3 y 4). Con los cuatro análisis, los
# cárnicos (embutidos, queso de cerdo) tienen la composición de un queso de muy alta humedad y caían en el tag QUESO.
# «panificados / otros» incluye la lactosa alimenticia; «base nutricional», fórmulas (de soja, de caseinatos, sustitutos
# lácteos). Las muestras solo MQ no tienen familia: entran (no se sabe que no sean lácteas).
FAMILIAS_NO_LACTEAS = ("carnicos", "panificados / otros", "agua / hisopado", "base nutricional")


def a_numero(serie):
    """'28,45' · '29,35 +/- 0,09' · '&lt; 0,03' · '54,.34' · 'NO DETECTADO' → número; «< LD» y NO DETECTADO valen 0."""
    t = serie.astype("string").str.replace("&lt;", "<", regex=False).str.replace(",.", ",", regex=False).str.strip()
    bajo_ld = (t.str.startswith("<") | (t.str.upper() == "NO DETECTADO")).fillna(False).astype(bool)
    v = pd.to_numeric(t.str.replace("<", "", regex=False).str.split("+/-", regex=False).str[0]
                      .str.replace(",", ".", regex=False).str.strip(), errors="coerce")
    return v.where(~bajo_ld, 0.0)


def aplicar_correcciones(crudo, correcciones):
    """Aplica la tabla de correcciones sobre una copia; falla si el valor actual no es el 'antes' de la tabla."""
    crudo = crudo.copy()
    for _, c in correcciones.iterrows():
        sel = crudo["Identificador"] == c["Identificador"]
        assert sel.sum() == 1 and str(crudo.loc[sel, c["campo"]].iloc[0]).strip() == c["antes"], f"corrección no aplicable: {c.to_dict()}"
        crudo[c["campo"]] = crudo[c["campo"]].astype(object)
        crudo.loc[sel, c["campo"]] = c["despues"] if pd.notna(c["despues"]) else np.nan
    return crudo


@dataclass
class Senasa:
    crudo: pd.DataFrame            # corregido, con humedad/grasa/proteinas/cenizas/solidos_senasa parseados
    crudo_original: pd.DataFrame   # tal cual el xlsx
    correcciones: pd.DataFrame
    excluidas: set                 # Identificadores con humedad + sólidos fuera de 100 ± 1
    celdas_tipeo: dict             # columna → (celdas con «,.», celdas NO DETECTADO)


def leer_senasa(ruta=RUTA_SENASA, ruta_corr=RUTA_CORR):
    crudo_original = pd.read_excel(ruta)
    correcciones = pd.read_csv(ruta_corr, dtype=str, encoding="utf-8-sig")
    crudo = aplicar_correcciones(crudo_original, correcciones)
    for nombre, col in ORIGEN.items():
        crudo[nombre] = a_numero(crudo[col])
    crudo["solidos_senasa"] = a_numero(crudo[COL_SOLIDOS])
    excluidas = set(crudo.loc[(crudo["humedad"] + crudo["solidos_senasa"] - 100).abs() > TOL_HUM_SOL, "Identificador"])
    celdas = {}
    for col in crudo.columns:
        if col.startswith("Resultado "):
            t = crudo[col].astype("string").str.strip()
            celdas[col] = (int(t.str.contains(",.", regex=False).fillna(False).sum()),
                           int((t.str.upper() == "NO DETECTADO").fillna(False).sum()))
    return Senasa(crudo, crudo_original, correcciones, excluidas, celdas)


@lru_cache(maxsize=1)
def _mq_encolumnado(ruta=RUTA_MQ_ENC):
    cols = ["Identificador"] + DESC_MQ + ["humedad_derivada_100_menos_solidos"] + \
           [ORIGEN[c] for c in ("grasa", "proteinas", "cenizas")]
    return pd.read_excel(ruta, usecols=cols)


def leer_mq(ruta=RUTA_MQ):
    """Devuelve (tabla ancha cruda, tabla de MQ con los cuatro análisis en g/100 g y la descripción).

    La segunda tabla (Identificador, humedad, grasa, proteinas, cenizas, desc_familia, desc_producto, desc_tipo_muestra) sale
    de MQ encolumnado y solo trae los códigos MQ (humedad = 100 − sólidos, proteína con piso 0). Los MO de la tabla
    ancha no entran por acá: ya están en Senasa encolumnado (las que tienen código en Senasa entran por Senasa). Se
    verifica que los valores de MQ encolumnado coincidan con los de la tabla ancha."""
    mq_crudo = pd.read_csv(ruta, low_memory=False)
    mq = pd.DataFrame({"Identificador": mq_crudo["codigo"], "humedad": 100 - mq_crudo["solidos"], "grasa": mq_crudo["grasa"],
                       "proteinas": mq_crudo["proteina"].clip(lower=0), "cenizas": mq_crudo["ceniza"]})
    enc = _mq_encolumnado().drop_duplicates("Identificador").set_index("Identificador")
    es_mq = mq["Identificador"].isin(enc.index)
    assert enc.index.isin(mq["Identificador"]).all(), "MQ encolumnado tiene códigos que no están en muestras_ancha.csv"
    assert not mq.loc[~es_mq, "Identificador"].str.startswith("MQ").any(), "hay códigos MQ que MQ encolumnado no tiene"
    e = enc.reindex(mq.loc[es_mq, "Identificador"])
    valores = pd.DataFrame({"humedad": a_numero(e["humedad_derivada_100_menos_solidos"]), "grasa": a_numero(e[ORIGEN["grasa"]]),
                            "proteinas": a_numero(e[ORIGEN["proteinas"]]).clip(lower=0), "cenizas": a_numero(e[ORIGEN["cenizas"]])})
    for c in COLUMNAS:
        a_, b_ = mq.loc[es_mq, c].to_numpy(dtype=float), valores[c].to_numpy(dtype=float)
        ambos = ~np.isnan(a_) & ~np.isnan(b_)
        # donde las dos fuentes tienen valor, es el mismo; MQ encolumnado puede tener huecos que la tabla ancha no
        assert np.allclose(a_[ambos], b_[ambos], atol=0.006), f"MQ encolumnado y muestras_ancha.csv no coinciden en {c}"
        assert not (np.isnan(a_) & ~np.isnan(b_)).any(), f"MQ encolumnado tiene un valor de {c} que la tabla ancha no tiene"
    mq = mq[es_mq].reset_index(drop=True)
    mq[COLUMNAS] = valores[COLUMNAS].to_numpy()          # solo lo medido en MQ (Senasa manda: el catálogo ya lo tiene)
    for c in DESC_MQ:
        mq[c] = e[c].to_numpy()
    return mq_crudo, mq


def unir(senasa, mq, extra=()):
    """Regla del proyecto: Senasa manda; MQ solo completa huecos; las muestras solo MQ se agregan con su familia, producto
    y tipo de muestra de MQ encolumnado. extra: otras columnas desc_* de MQ que se conservan en las solo MQ
    (p. ej. desc_subtipo y desc_estado, para elegir los cortes por estado).
    Devuelve (todas, completado por análisis)."""
    unidas = senasa.merge(mq[["Identificador"] + COLUMNAS], on="Identificador", how="left", suffixes=("", "_mq"))
    completado = {}
    for col in COLUMNAS:
        completado[col] = int((unidas[col].isna() & unidas[f"{col}_mq"].notna()).sum())
        unidas[col] = unidas[col].fillna(unidas[f"{col}_mq"])
    completa_antes = senasa[COLUMNAS].notna().all(axis=1).values   # el merge conserva el orden de Senasa (código único)
    unidas["origen"] = np.where(~completa_antes & unidas[COLUMNAS].notna().all(axis=1), "Senasa completada con MQ", "Senasa")
    unidas = unidas.drop(columns=[f"{c}_mq" for c in COLUMNAS])
    solo_mq = mq[~mq["Identificador"].isin(senasa["Identificador"])].copy()
    solo_mq["desc_familia"] = solo_mq["desc_familia"].fillna(SIN_FAMILIA) if "desc_familia" in solo_mq else SIN_FAMILIA
    for c in ("desc_producto", "desc_tipo_muestra"):
        if c not in solo_mq:
            solo_mq[c] = np.nan
    solo_mq["origen"] = "solo MQ"
    solo_mq = solo_mq[["Identificador", "desc_familia", "desc_producto", "desc_tipo_muestra", "origen"] + COLUMNAS + list(extra)]
    return pd.concat([unidas, solo_mq], ignore_index=True), completado


# ---------------------------------------------------------------- nombres y tags de los grupos
# Cada regla de nombrar tiene su tag: corto, estable (sale del centroide, no del número de cluster) y legible.
TAGS = {"líquido (leche fluida / suero)": "LIQUIDO",
        "alta humedad": "ALTA_HUMEDAD",
        "semiduro graso (tipo queso)": "QUESO",
        "pastoso tipo dulce de leche": "DDL",
        "polvo de proteína muy alta": "PROT_ALTA",
        "polvo graso (tipo leche entera)": "POLVO_GRASO",
        "polvo magro (suero / concentrado)": "POLVO_MAGRO"}


def nombrar(cen):
    # la regla de queso va antes que la de alta humedad: un queso graso con humedad cerca de 50 es QUESO
    if cen["humedad"] > 70: return "líquido (leche fluida / suero)"
    if 35 < cen["humedad"] <= 65 and cen["grasa"] > 15: return "semiduro graso (tipo queso)"
    if cen["humedad"] > 50: return "alta humedad"
    if cen["humedad"] > 15: return "pastoso tipo dulce de leche"
    if cen["proteinas"] > 70: return "polvo de proteína muy alta"
    if cen["grasa"] > 15: return "polvo graso (tipo leche entera)"
    return "polvo magro (suero / concentrado)"


def etiquetar(centros):
    """centros: DataFrame (una fila por cluster, en orden 0..K-1) con las cuatro columnas en g/100 g.
    Devuelve (nombres, tags, repetidos): dicts cluster → nombre y cluster → tag. Si dos clusters caen en la misma
    regla, se desambiguan con sufijo _2, _3... ordenados por humedad y grasa del centroide (no por número de cluster)."""
    base = {j: nombrar(centros.iloc[j]) for j in range(len(centros))}
    nombres, tags, repetidos = {}, {}, []
    for n in dict.fromkeys(base.values()):
        js = sorted([j for j in base if base[j] == n], key=lambda j: (centros.iloc[j]["humedad"], centros.iloc[j]["grasa"]))
        if len(js) > 1:
            repetidos.append(TAGS[n])
        for i, j in enumerate(js):
            nombres[j] = n if i == 0 else f"{n} {i + 1}"
            tags[j] = TAGS[n] if i == 0 else f"{TAGS[n]}_{i + 1}"
    return nombres, tags, repetidos
