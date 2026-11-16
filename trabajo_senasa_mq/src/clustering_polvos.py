"""Agrupamiento propio de los PRODUCTOS EN POLVO (prueba, aparte del notebook completo).

Lo usa prueba_clustering_polvos.ipynb (lo arma build_prueba_polvos.py). No modifica ningún archivo de entrada ni de
resultados existentes; escribe solo lo que el notebook le pida en resultados/prueba_polvos/.

Mismo método que la prueba de quesos (clustering_quesos.py; de ahí salen los algoritmos y la evaluación):
  - Polvos = familias leche en polvo, suero / permeado y concentrado proteico (Senasa; las solo MQ, con la familia de
    mq_tipo_muestra.csv) con los cuatro análisis (humedad, grasa, proteínas, cenizas) y humedad ≤ HUMEDAD_MAX: los sueros,
    permeados y concentrados líquidos (humedad 67 a 93) no son polvos y quedan fuera.
  - En un polvo la humedad es de proceso (2 a 6 g/100 g) y lo que define el tipo es la composición de los sólidos. El
    espacio base es grasa, proteína y cenizas en BASE SECA (g/100 g de extracto seco). La lactosa (≈ 100 − las otras
    tres en base seca) y la humedad son funciones de los cuatro análisis: no se suman como variables (sería contar dos
    veces la misma información). La parametrización medida (humedad, grasa, proteínas, cenizas) va de control.
  - La lactosa medida (Senasa) va en un nivel extendido, solo para las muestras que la tienen.

    preparar_polvos(S)            una fila por polvo (Senasa + MQ, Senasa manda), con base seca, resto por diferencia,
                                  lactosa medida, tipo de referencia, familia y cliente
    tipo_referencia(...)          el tipo de polvo del rótulo (leche entera / descremada, suero, permeado, WPC, caseína...)
    ordenar(et, proteina_bs)      clusters numerados por proteína en base seca creciente
    describir(df, et, cols)       centroides y tipo / familia / clase CAA mayoritarios con su pureza
    proponer_subtags(desc)        POLVO_<tipo> solo donde el cluster se corresponde con un tipo real
    subtag_por_muestra(et, prop)  el subtag de cada muestra (sin subtag o ruido → POLVO)
    clase_caa_leche(grasa)        descremada / parcialmente descremada / entera (CAA art. 567), solo leche en polvo
    tags_generales()              el tag del agrupamiento general (resultados/tablas/clusters_*.csv) de cada muestra
"""
import re

import numpy as np
import pandas as pd

import rutas as R
from clustering_quesos import (PALETA, elegir_hdbscan, escalar, evaluar, hdbscan_sensibilidad, kmedias,  # noqa: F401
                               matriz, mezcla, quitar_outliers)
from compartido.catalogo import normalizar_texto
from datos_clase41 import COLUMNAS, SIN_FAMILIA, SUMA_MAX, a_numero, leer_mq, leer_senasa, unir

FAMILIAS = ["leche en polvo", "suero / permeado", "concentrado proteico"]
HUMEDAD_MAX = 15.0                       # hueco natural: no hay polvos entre 12 y 25 g/100 g
MQ_NUEVAS = R.EXCEL / "MQ encolumnado - solo nuevas.xlsx"
CORTES = R.TABLAS / "cortes_categorias.csv"
BASE = ["grasa_bs", "proteina_bs", "cenizas_bs"]      # g/100 g de extracto seco
BASE_ALT = COLUMNAS                                   # humedad, grasa, proteinas, cenizas (lo medido)
NOMBRES = {"grasa_bs": "grasa (base seca)", "proteina_bs": "proteína (base seca)", "cenizas_bs": "cenizas (base seca)",
           "resto_bs": "resto por diferencia (base seca)", "lactosa_bs": "lactosa medida (base seca)",
           "humedad": "humedad", "grasa": "grasa", "proteinas": "proteínas", "cenizas": "cenizas"}
SIN_DATO = {"sin especificar", "nan", "None", "", "<NA>"}
COL_LACTOSA = "Resultado Lactosa g /100g (*)"

# subtipo del rótulo → tipo de referencia. La leche en polvo casi no trae subtipo: su tipo sale de desc_grasa.
TIPO_SUBTIPO = {
    "suero / permeado": {"SUERO": "SUERO", "SIN DESNATAR": "SUERO SIN DESNATAR", "DESMINERALIZADO": "SUERO DESMINERALIZADO",
                         "PERMEADO": "PERMEADO", "PERMEADO DE SUERO": "PERMEADO", "PERMEADO DE LECHE": "PERMEADO",
                         "PERMEADO DE QUESO": "PERMEADO", "CONCENTRADO": "SUERO CONCENTRADO", "WPC": "WPC",
                         "REENGRASADO": "SUERO REENGRASADO"},
    "concentrado proteico": {"WPC": "WPC", "CASEINA": "CASEINA", "CASEINATO DE SODIO": "CASEINATO",
                             "CASEINATO DE CALCIO": "CASEINATO", "MPC": "MPC", "PROTEINAS LACTEAS": "PROTEINAS LACTEAS"},
    "leche en polvo": {"MODIFICADA": "LECHE MODIFICADA", "FORMULA INFANTIL": "FORMULA INFANTIL",
                       "COMPUESTO LACTEO": "COMPUESTO LACTEO"},
}
TIPO_GRASA = {"ENTERA": "LECHE ENTERA", "DESCREMADA": "LECHE DESCREMADA",
              "PARCIALMENTE DESCREMADA": "LECHE PARCIALMENTE DESCREMADA", "FAT FILLED": "LECHE FAT FILLED"}
CORTO_FAMILIA = {"leche en polvo": "LECHE", "suero / permeado": "SUERO_PERMEADO", "concentrado proteico": "CONCENTRADO"}


# ------------------------------------------------------------------ datos
def _limpio(s):
    s = pd.Series(s, dtype=object)
    return s.where(~s.astype(str).isin(SIN_DATO) & s.notna(), np.nan)


def tipo_referencia(familia, subtipo, desc_grasa):
    """Tipo de polvo del rótulo, independiente de lo medido: el subtipo (TIPO_SUBTIPO) y, en leche en polvo sin subtipo,
    desc_grasa (TIPO_GRASA). Lo que no se puede asignar queda NaN («sin especificar» no se adivina)."""
    out = []
    for f, s, g in zip(familia, _limpio(subtipo), _limpio(desc_grasa)):
        t = TIPO_SUBTIPO.get(f, {}).get(str(s).upper()) if isinstance(s, str) else None
        if t is None and f == "leche en polvo" and isinstance(g, str):
            t = TIPO_GRASA.get(g.upper())
        out.append(t if t is not None else np.nan)
    return pd.Series(out, dtype=object)


def _cortes_leche():
    import cortes_caa
    c = pd.read_csv(cortes_caa.asegurar())
    c = c[(c["analisis"] == "materia_grasa") & (c["familia"] == "leche en polvo") & c["fuente"].str.startswith("CAA")]
    return c.sort_values("desde")[["categoria", "desde", "hasta"]].reset_index(drop=True)


def clase_caa_leche(grasa, familia):
    """Clase de grasa del CAA art. 567 (descremada ≤ 1,5; parcialmente descremada; entera ≥ 26), solo en leche en
    polvo; NaN en las demás familias. Los cortes salen de resultados/tablas/cortes_categorias.csv."""
    c = _cortes_leche()
    cat = pd.cut(pd.Series(grasa, dtype=float), bins=list(c["desde"]) + [np.inf], right=False,
                 labels=list(c["categoria"])).astype(object)
    return cat.where(pd.Series(familia).eq("leche en polvo").to_numpy(), np.nan)


def preparar_polvos(S=None):
    """Polvos con los cuatro análisis, de Senasa y de MQ, una fila por Identificador.

    Senasa manda y MQ completa huecos (datos_clase41.unir, la misma regla del agrupamiento general). Las muestras solo
    MQ entran si mq_tipo_muestra.csv les da una familia de polvo (es un desglose no revisado a mano: queda marcado en
    `fuente_familia`). Exclusiones, en orden: humedad + ST fuera de 100 ± 1 (S.excluidas), suma de los cuatro > SUMA_MAX
    y humedad > HUMEDAD_MAX (no es polvo). Devuelve (polvos, conteo de exclusiones)."""
    S = S if S is not None else leer_senasa()
    _, mq = leer_mq()
    tipo_mq = pd.read_csv(R.MQ_TIPO).drop_duplicates("codigo").set_index("codigo")
    cols_sen = ["Identificador", "Cliente", "desc_familia", "desc_producto", "desc_subtipo", "desc_grasa", "desc_lactosa",
                "Fecha de Recepción", COL_LACTOSA]
    sen = S.crudo[cols_sen + COLUMNAS].copy()
    todas, _ = unir(sen, mq)
    solo = todas["origen"].eq("solo MQ")          # las solo MQ: su familia sale del modelo de mq_tipo_muestra.csv
    todas["fuente_familia"] = np.where(solo, "mq_tipo_muestra (no revisado)", "Senasa")
    todas.loc[solo, "desc_familia"] = todas.loc[solo, "Identificador"].map(tipo_mq["desc_familia"])
    p = todas[todas["desc_familia"].isin(FAMILIAS)].copy()

    # rótulo de las solo MQ: catálogo normalizado (MQ encolumnado - solo nuevas) y, si falta, mq_tipo_muestra.csv
    nuevas = pd.read_excel(MQ_NUEVAS, usecols=["Identificador", "Cliente", "desc_subtipo", "desc_grasa", "desc_lactosa",
                                               "desc_producto"]).drop_duplicates("Identificador").set_index("Identificador")
    s = p["origen"].eq("solo MQ")
    for col in ["Cliente", "desc_subtipo", "desc_grasa", "desc_lactosa", "desc_producto"]:
        p.loc[s, col] = p.loc[s, "Identificador"].map(nuevas[col])
    p.loc[s, "desc_subtipo"] = _limpio(p.loc[s, "desc_subtipo"]).fillna(
        _limpio(p.loc[s, "Identificador"].map(tipo_mq["desc_subtipo"]))).to_numpy()

    conteo = {"polvos leídos (las tres familias)": len(p)}
    p = p.dropna(subset=COLUMNAS)
    conteo["con los cuatro análisis"] = len(p)
    fuera = p["Identificador"].isin(S.excluidas)
    suma = p[COLUMNAS].sum(axis=1) > SUMA_MAX
    liquido = p["humedad"] > HUMEDAD_MAX
    conteo["excluidas: humedad + ST fuera de 100 ± 1"] = int(fuera.sum())
    conteo[f"excluidas: suma de los cuatro > {SUMA_MAX}"] = int((suma & ~fuera).sum())
    conteo[f"excluidas: humedad > {HUMEDAD_MAX:g} (líquidos, no polvo)"] = int((liquido & ~fuera & ~suma).sum())
    p = p[~fuera & ~suma & ~liquido].reset_index(drop=True)
    assert p["Identificador"].is_unique

    st = 100 - p["humedad"]
    for v, bs in [("grasa", "grasa_bs"), ("proteinas", "proteina_bs"), ("cenizas", "cenizas_bs")]:
        p[bs] = 100 * p[v] / st
    p["resto_bs"] = 100 - p[BASE].sum(axis=1)              # lactosa + lo no medido, por diferencia
    p["lactosa"] = a_numero(p[COL_LACTOSA]) if COL_LACTOSA in p else np.nan
    p["lactosa_bs"] = 100 * p["lactosa"] / st
    p["tipo"] = tipo_referencia(p["desc_familia"], p["desc_subtipo"], p["desc_grasa"]).to_numpy()
    p["clase_caa"] = clase_caa_leche(p["grasa"], p["desc_familia"]).to_numpy()
    p["subtipo"] = _limpio(p["desc_subtipo"]).map(lambda x: normalizar_texto(x) if isinstance(x, str) else x)
    p["anio"] = pd.to_datetime(p["Fecha de Recepción"], dayfirst=True, errors="coerce").dt.year
    conteo["quedan"] = len(p)
    cols = ["Identificador", "Cliente", "origen", "fuente_familia", "desc_familia", "tipo", "subtipo", "desc_grasa",
            "desc_lactosa", "clase_caa", "anio"] + COLUMNAS + BASE + ["resto_bs", "lactosa", "lactosa_bs", "desc_producto"]
    return p[cols].rename(columns={"desc_familia": "familia"}), conteo


# ------------------------------------------------------------------ algoritmos y descripción
def ordenar(et, proteina_bs):
    """Renumera los clusters por proteína en base seca media creciente (0 = el de menos proteína: permeados), así el
    número no depende de la semilla ni del algoritmo. El ruido (-1) queda -1."""
    et = np.asarray(et); p = pd.Series(np.asarray(proteina_bs, dtype=float))
    medias = p[et != -1].groupby(et[et != -1]).mean().sort_values()
    mapa = {g: i for i, g in enumerate(medias.index)}
    return np.array([mapa.get(g, -1) for g in et])


def _mayoritario(s):
    s = s.dropna()
    if s.empty:
        return "-", np.nan, 0
    m = s.value_counts().index[0]
    return m, 100 * (s == m).mean(), len(s)


def describir(df, et, cols=("grasa_bs", "proteina_bs", "cenizas_bs", "resto_bs", "humedad")):
    """Por cluster: muestras, centroide (base seca, g/100 g ES; humedad en g/100 g) y, para tipo, familia y clase CAA de
    la leche: el valor mayoritario, su % (pureza, sobre las muestras con dato) y cuántas tienen dato; para el tipo,
    además la cobertura (% de ese tipo que cae en el cluster) y cuántos clientes hay. El ruido (-1) aparece «ruido»."""
    d = df.reset_index(drop=True).copy()
    d["cluster"] = np.where(np.asarray(et) == -1, "ruido", np.asarray(et).astype(str))
    t = d.groupby("cluster")[list(cols)].mean().round(1)
    t.insert(0, "muestras", d.groupby("cluster").size())
    t.insert(1, "clientes", d.groupby("cluster")["Cliente"].nunique())
    for ref, corto in [("tipo", "tipo"), ("familia", "familia"), ("clase_caa", "CAA leche")]:
        res = d.groupby("cluster")[ref].apply(_mayoritario)
        t[f"{corto} mayoritario"] = res.map(lambda x: x[0])
        t[f"% {corto}"] = res.map(lambda x: x[1]).round(0)
        if ref == "tipo":
            t["con tipo"] = res.map(lambda x: x[2])
            tot = d["tipo"].value_counts()
            t["% cobertura tipo"] = [round(100 * ((d["cluster"] == g) & (d["tipo"] == m)).sum() / tot[m]) if m in tot else np.nan
                                     for g, m in t["tipo mayoritario"].items()]
    orden = sorted(t.index, key=lambda g: (g == "ruido", t.loc[g, "proteina_bs"] if "proteina_bs" in t else 0))
    return t.loc[orden]


def _tag(s):
    return "POLVO_" + re.sub(r"[^A-Z0-9]+", "_", str(s).upper()).strip("_")


def proponer_subtags(desc, pureza_min=70, conocidas_min=20, frac_conocida_min=0.5, cobertura_min=50):
    """Un subtag por cluster solo si corresponde a un tipo real (mismo criterio que los quesos, más la cobertura del DDL):
      1) tipo del rótulo con pureza >= pureza_min y cobertura >= cobertura_min (sobre >= conocidas_min muestras con tipo,
         que sean al menos frac_conocida_min del cluster) → POLVO_<TIPO>;
      2) si no, familia con pureza >= pureza_min → POLVO_<LECHE | SUERO_PERMEADO | CONCENTRADO>.
    Un tipo puede ocupar dos regiones (el WPC de dos clientes, por ejemplo): la cobertura se suma sobre todos los
    clusters puros de ese tipo, y todos llevan el mismo subtag. El ruido nunca lleva subtag.
    Devuelve DataFrame cluster → subtag, criterio, pureza, cobertura (vacío = no hay estructura)."""
    d = desc.drop(index="ruido", errors="ignore")
    puro = (d["con tipo"] >= conocidas_min) & (d["con tipo"] >= frac_conocida_min * d["muestras"]) & (d["% tipo"] >= pureza_min)
    cobertura = d["% cobertura tipo"].where(puro, 0).groupby(d["tipo mayoritario"]).transform("sum")
    filas = []
    for g, f in d.iterrows():
        if puro[g] and cobertura[g] >= cobertura_min:
            filas.append((g, _tag(f["tipo mayoritario"]), "tipo", f["% tipo"], cobertura[g]))
        elif f["% familia"] >= pureza_min and f["familia mayoritario"] in CORTO_FAMILIA:
            filas.append((g, _tag(CORTO_FAMILIA[f["familia mayoritario"]]), "familia", f["% familia"], np.nan))
    return pd.DataFrame(filas, columns=["cluster", "subtag", "criterio", "% pureza", "% cobertura"]).set_index("cluster")


def subtag_por_muestra(et, propuesta):
    """Subtag de cada muestra a partir de las etiquetas y de proponer_subtags (sin subtag o ruido → POLVO)."""
    m = propuesta["subtag"].to_dict()
    return np.array([m.get(str(g), "POLVO") if g != -1 else "POLVO" for g in np.asarray(et)], dtype=object)


def tags_generales():
    """Identificador → tag y subtag del agrupamiento general, de resultados/tablas/clusters_senasa.csv y
    clusters_senasa_mq.csv (las columnas llevan el sufijo _senasa / _senasa_mq)."""
    out = []
    for nombre in ["senasa", "senasa_mq"]:
        d = pd.read_csv(R.TABLAS / f"clusters_{nombre}.csv", usecols=["Identificador", "tag", "outlier"])
        d = d.assign(tag=d["tag"].where(~d["outlier"], "outlier")).drop(columns="outlier")
        out.append(d.set_index("Identificador").add_suffix(f"_{nombre}"))
    return out[0].join(out[1], how="outer")


# ------------------------------------------------------------------ gráficos
def dispersion(ax, df, color_por, titulo, x="proteina_bs", y="grasa_bs", max_cat=10, orden=None, s=9):
    """Dispersión (por defecto proteína × grasa en base seca) coloreada por una columna categórica (las max_cat más
    frecuentes; el resto «otros»; sin dato gris claro)."""
    v = pd.Series(df[color_por].astype(object).values).where(pd.notna(df[color_por].values), "sin dato").astype(str)
    v = v.replace({"-1": "ruido"})
    cats = orden or [c for c in v.value_counts().index if c not in ("sin dato", "ruido")][:max_cat]
    v = v.where(v.isin(cats + ["sin dato", "ruido"]), "otros")
    xs, ys = df[x].to_numpy(), df[y].to_numpy()
    for c, col in [("sin dato", "#d1d5db"), ("ruido", "#9ca3af"), ("otros", "#6b7280")]:
        sel = (v == c).to_numpy()
        if sel.any():
            ax.scatter(xs[sel], ys[sel], s=s, c=col, alpha=0.5, lw=0, label=f"{c} ({sel.sum()})")
    for i, c in enumerate(cats):
        sel = (v == c).to_numpy()
        ax.scatter(xs[sel], ys[sel], s=s, c=PALETA[i % len(PALETA)], alpha=0.8, lw=0, label=f"{c} ({sel.sum()})")
    ax.set_xlabel(f"{NOMBRES.get(x, x)} (g/100 g ES)" if x.endswith("_bs") else f"{NOMBRES.get(x, x)} (g/100 g)")
    ax.set_ylabel(f"{NOMBRES.get(y, y)} (g/100 g ES)" if y.endswith("_bs") else f"{NOMBRES.get(y, y)} (g/100 g)")
    ax.set_title(titulo, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.5, markerscale=2, loc="best", frameon=True, framealpha=0.85)
