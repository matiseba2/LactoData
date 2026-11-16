"""Agrupamiento propio de los QUESOS (prueba, aparte del notebook completo).

Lo usa prueba_clustering_quesos.ipynb (lo arma build_prueba_quesos.py). No modifica ningún archivo de entrada ni de
resultados existentes; escribe solo lo que el notebook le pida en resultados/prueba_quesos/.

Criterio del usuario (manda):
  - En queso Senasa pide materia grasa (MG) y sólidos totales (ST). Humedad = 100 − ST y grasa en extracto seco
    MGES = MG / ST × 100 son FUNCIONES de esos dos: el espacio base tiene 2 dimensiones. No se mezclan MG y ST con
    humedad y MGES (sería contar dos veces la misma información).
  - Cloruros (NaCl), proteína y cenizas solo están en MQ: van en un nivel extendido, para las muestras que los tengan.

    preparar_quesos(S)              una fila por queso con MG y ST (Senasa + MQ, sin duplicados), con humedad, MGES,
                                    NaCl, proteína, cenizas, subtipo, consistencia del rótulo y clases del CAA
    rama_queso(S, con_mq)           los quesos del notebook único (secciones 3 y 4): solo los que confirma el catálogo
    no_supervisado(rama)            k-medias y mezcla sobre humedad × MGES sin tag previo, comparadas con el CAA y el rótulo
    clases_caa(humedad, mges)       las clases del CAA art. 605 (cortes de resultados/tablas/cortes_categorias.csv)
    matriz(df, variables)           la matriz en g/100 g de las variables pedidas
    kmedias / mezcla / hdbscan_*    los tres algoritmos (k por silueta, k por BIC, sensibilidad a min_cluster_size)
    evaluar(Z, et, refs)            métricas internas (mejoras_clustering.metricas) + ARI y NMI contra cada referencia
    describir(df, et, cols)         centroides en g/100 g, subtipo, consistencia y clase CAA mayoritarios con su pureza
    proponer_subtags(desc)          QUESO_<algo> solo donde el cluster se corresponde con un tipo real
    subtag_por_muestra(et, prop)    el subtag de cada muestra; subtag_regla_caa: la alternativa sin agrupar (corte del CAA)
    dispersion_caa(ax, ...)         humedad × MGES con las líneas de corte del CAA
    no_quesos_con_nacl(S)           muestras MQ que no son queso y tienen NaCl (para ver si se confundirían)
"""
import re

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import rutas as R
from compartido.catalogo import consistencia_queso, normalizar_texto
from datos_clase41 import N_INIT_KMEDIAS, SEMILLA, TOL_HUM_SOL, leer_mq, leer_senasa
from mejoras_clustering import metricas, outliers_consenso

MQ_NUEVAS = R.EXCEL / "MQ encolumnado - solo nuevas.xlsx"
CORTES = R.TABLAS / "cortes_categorias.csv"
BASE = ["humedad", "mges"]                 # parametrización principal (la del CAA)
BASE_ALT = ["mg", "st"]                    # la medida, para mostrar que el resultado no depende de la elección
EXTRA = ["nacl", "proteina", "cenizas"]
NOMBRES = {"humedad": "humedad", "mges": "MG en extracto seco", "mg": "materia grasa", "st": "sólidos totales",
           "nacl": "NaCl", "proteina": "proteína", "cenizas": "cenizas"}
SIN_DATO = {"sin especificar", "nan", "None", ""}


# ------------------------------------------------------------------ datos
def _cortes(analisis):
    import cortes_caa
    c = pd.read_csv(cortes_caa.asegurar())
    c = c[(c["analisis"] == analisis) & (c["familia"] == "queso") & c["fuente"].str.startswith("CAA")]
    return c.sort_values("desde")[["categoria", "desde", "hasta"]].reset_index(drop=True)


def cortes_caa():
    """dict análisis → DataFrame (categoria, desde, hasta) del CAA art. 605 para humedad y MGES."""
    return {"humedad": _cortes("humedad"), "mges": _cortes("mg_extracto_seco")}


def _clasificar(v, cortes):
    bordes = list(cortes["desde"]) + [np.inf]
    cat = pd.cut(v, bins=bordes, right=False, labels=list(cortes["categoria"]))
    return cat.astype(object).where(pd.notna(v), np.nan)


def clases_caa(humedad, mges):
    """(clase de humedad, clase de MGES) del CAA, intervalos cerrados a la izquierda (36 ya es mediana humedad)."""
    c = cortes_caa()
    return _clasificar(pd.Series(humedad), c["humedad"]), _clasificar(pd.Series(mges), c["mges"])


def _limpio(s):
    s = pd.Series(s, dtype=object)
    return s.where(~s.astype(str).isin(SIN_DATO) & s.notna(), np.nan)


def _quesos_leidos(S, con_mq=True, sin_mo=False):
    """Los quesos (desc_familia = queso en Senasa; familia queso de mq_tipo_muestra.csv en las solo MQ) con mg, st y los
    descriptores, sin ningún filtro todavía. con_mq=False: solo Senasa, sin completar con MQ y sin las solo MQ. sin_mo=True:
    la parte MQ sale solo de MQ encolumnado (sin las MO), como en el notebook único."""
    mq_crudo, _ = leer_mq()
    tipo = pd.read_csv(R.MQ_TIPO).drop_duplicates("codigo").set_index("codigo")
    mq = pd.DataFrame({"Identificador": mq_crudo["codigo"], "mg_mq": mq_crudo["grasa"], "st_mq": mq_crudo["solidos"],
                       "nacl": mq_crudo["cloruro_nacl_g100g"].clip(lower=0), "proteina": mq_crudo["proteina"].clip(lower=0),
                       "cenizas": mq_crudo["ceniza"]}).drop_duplicates("Identificador")
    fam_mq = tipo["desc_familia"]
    if sin_mo:
        # como el notebook único: la parte MQ sale solo de MQ encolumnado (sin las MO de la tabla ancha, que ya están en
        # Senasa); MG, ST, proteína, cenizas y la familia son los de ese Excel (con sus huecos)
        _, e = leer_mq()
        e = e.drop_duplicates("Identificador").set_index("Identificador")
        mq = mq[mq["Identificador"].isin(e.index)].copy()
        for col, c in (("mg_mq", "grasa"), ("proteina", "proteinas"), ("cenizas", "cenizas")):
            mq[col] = mq["Identificador"].map(e[c])
        mq["st_mq"] = 100 - mq["Identificador"].map(e["humedad"])
        fam_mq = e["desc_familia"]
    if not con_mq:
        mq = mq.iloc[0:0]

    sen = S.crudo[S.crudo["desc_familia"] == "queso"]
    sen = pd.DataFrame({"Identificador": sen["Identificador"], "mg": sen["grasa"], "st": sen["solidos_senasa"],
                        "humedad_medida": sen["humedad"], "subtipo": sen["desc_subtipo"],
                        "consistencia": sen["desc_consistencia"], "producto": sen["desc_producto"]})
    sen = sen.merge(mq, on="Identificador", how="left")
    completa = sen["mg"].isna() & sen["mg_mq"].notna() | sen["st"].isna() & sen["st_mq"].notna()
    sen["origen"] = np.where(completa, "Senasa completada con MQ", "Senasa")
    sen["mg"] = sen["mg"].fillna(sen["mg_mq"]); sen["st"] = sen["st"].fillna(sen["st_mq"])

    nuevas = pd.read_excel(MQ_NUEVAS, usecols=["Identificador", "desc_subtipo", "desc_consistencia", "desc_producto"])
    nuevas = nuevas.drop_duplicates("Identificador").set_index("Identificador")
    solo = mq[~mq["Identificador"].isin(S.crudo["Identificador"])
              & mq["Identificador"].map(fam_mq).eq("queso")].copy()
    solo = solo.rename(columns={"mg_mq": "mg", "st_mq": "st"})
    solo["humedad_medida"] = np.nan
    solo["subtipo"] = solo["Identificador"].map(nuevas["desc_subtipo"])
    solo["consistencia"] = solo["Identificador"].map(nuevas["desc_consistencia"])
    solo["producto"] = solo["Identificador"].map(nuevas["desc_producto"])
    solo["origen"] = "solo MQ"

    q = pd.concat([sen.drop(columns=["mg_mq", "st_mq"]), solo], ignore_index=True)
    assert q["Identificador"].is_unique
    # subtipo: el del rótulo (Senasa o catálogo normalizado de MQ); si falta, el de mq_tipo_muestra.csv
    q["subtipo"] = _limpio(q["subtipo"])
    de_tipo = _limpio(q["Identificador"].map(tipo["desc_subtipo"]))
    q["fuente_subtipo"] = np.where(q["subtipo"].notna(), "rótulo", np.where(de_tipo.notna(), "mq_tipo_muestra", None))
    q["subtipo"] = q["subtipo"].fillna(de_tipo)
    q["subtipo"] = q["subtipo"].map(lambda s: normalizar_texto(s) if isinstance(s, str) else s)
    # consistencia: la del rótulo; si falta, la que implica el subtipo (misma tabla del proyecto, catalogo.py)
    cons = _limpio(q["consistencia"])
    deriv = q["subtipo"].map(lambda s: consistencia_queso(f"QUESO {s}")[0] if isinstance(s, str) else np.nan)
    q["consistencia"] = _limpio(cons.fillna(deriv))
    return q


def _validar(q, S):
    """Se queda con los que tienen MG y ST y quita los imposibles; agrega humedad, MGES y las clases del CAA.
    Devuelve (quesos, conteo de exclusiones)."""
    conteo = {"quesos leídos": len(q), "con MG y ST": int(q[["mg", "st"]].notna().all(axis=1).sum())}
    q = q.dropna(subset=["mg", "st"])
    fuera = (q["humedad_medida"] + q["st"] - 100).abs() > TOL_HUM_SOL
    fuera |= q["Identificador"].isin(S.excluidas)
    imposible = (q["mg"] > q["st"]) | (q["st"] <= 0) | (q["st"] > 100) | (q["mg"] < 0)
    conteo["excluidas: humedad + ST fuera de 100 ± 1"] = int(fuera.sum())
    conteo["excluidas: MG > ST o fuera de rango"] = int((imposible & ~fuera).sum())
    q = q[~fuera & ~imposible].reset_index(drop=True)
    q["humedad"] = 100 - q["st"]
    q["mges"] = 100 * q["mg"] / q["st"]
    q["caa_humedad"], q["caa_mges"] = clases_caa(q["humedad"], q["mges"])
    q["caa"] = q["caa_humedad"].astype(str) + " · " + q["caa_mges"].astype(str)
    conteo["quedan"] = len(q)
    cols = ["Identificador", "origen", "mg", "st", "humedad", "mges", "nacl", "proteina", "cenizas", "subtipo",
            "fuente_subtipo", "consistencia", "caa_humedad", "caa_mges", "caa", "producto"]
    return q[cols], conteo


def preparar_quesos(S=None):
    """Quesos (desc_familia = queso) con MG y ST, de Senasa y de MQ, una fila por Identificador.

    Senasa manda: MG y ST de Senasa (solidos_senasa); si faltan y la muestra está en MQ, los completa MQ. NaCl, proteína
    y cenizas vienen de MQ (Senasa no los pide en queso), con piso 0 (como la proteína en datos_clase41.leer_mq). Las muestras solo MQ son las de familia queso en
    mq_tipo_muestra.csv que no están en Senasa. Exclusiones: humedad + ST fuera de 100 ± 1 donde Senasa midió las dos
    (la regla del proyecto) y MG > ST (imposible: MGES > 100).
    Devuelve (quesos, conteo de exclusiones). Es lo que usa la prueba de quesos: incluye TODAS las solo MQ de familia
    queso, confirmadas o no (el notebook único usa rama_queso, que pide confirmación)."""
    S = S if S is not None else leer_senasa()
    return _validar(_quesos_leidos(S), S)


def rama_queso(S, con_mq):
    """Lo que el notebook único manda a la rama de queso (secciones 3 y 4), fuera del k-medias de cuatro análisis.

    Solo entra como queso lo que el catálogo confirma:
      - Senasa: desc_familia = queso (la clasificación del catálogo, revisada).
      - solo MQ (solo si con_mq; solo códigos MQ, los MO no entran por MQ): la familia queso de mq_tipo_muestra.csv
        (la misma desc_familia de MQ encolumnado) sale de un modelo sin revisar
        (familia_fuente = desglose_no_revisado), así que no alcanza: hace falta además que el rótulo (la descripción
        normalizada con catalogo.py, `fuente_subtipo` = rótulo) nombre una variedad de queso. Sin eso, la muestra no entra
        a la rama y sigue en el k-medias si tiene los cuatro análisis (desconfiar por defecto).
    En la rama van las que tienen MG y ST (humedad = 100 − ST) y pasan las exclusiones de _validar; el subtag sale de la
    humedad del CAA. `confirmados` son TODOS los Identificadores confirmados (con o sin MG y ST): salen del k-medias.
    Devuelve (rama, confirmados, sin_confirmar, conteo): rama es la tabla de preparar_quesos; sin_confirmar, las solo MQ de
    familia queso (modelo) sin variedad en el rótulo."""
    q = _quesos_leidos(S, con_mq, sin_mo=True)
    ok = (q["origen"] != "solo MQ") | (q["fuente_subtipo"] == "rótulo")
    rama, conteo = _validar(q[ok], S)
    sin = q[~ok]
    conteo["solo MQ de familia queso sin confirmar por el rótulo (no entran)"] = len(sin)
    conteo["de ellas, con MG y ST"] = int(sin[["mg", "st"]].notna().all(axis=1).sum())
    return rama, set(q.loc[ok, "Identificador"]), sin, conteo


def matriz(df, variables):
    return df[list(variables)].to_numpy(dtype=float)


def escalar(X):
    esc = StandardScaler().fit(X)
    return esc.transform(X), esc


def quitar_outliers(X, pct=3):
    """outliers_consenso (LOF + IsolationForest + EllipticEnvelope, >= 2 de 3, corte pct %) con todos los quesos como
    un solo grupo (el tag QUESO). Devuelve (es_outlier, votos)."""
    return outliers_consenso(X, np.zeros(len(X), dtype=int), pct=pct)


# ------------------------------------------------------------------ algoritmos
def kmedias(Z, ks=range(2, 11), semilla=SEMILLA):
    """k-medias con k por silueta máxima. Devuelve (etiquetas, k, siluetas por k, modelo)."""
    mods = {k: KMeans(n_clusters=k, n_init=N_INIT_KMEDIAS, random_state=semilla).fit(Z) for k in ks}
    sil = {k: silhouette_score(Z, m.labels_, sample_size=min(len(Z), 4000), random_state=semilla) for k, m in mods.items()}
    k = max(sil, key=sil.get)
    return mods[k].labels_, k, sil, mods[k]


def mezcla(Z, ks=range(1, 9), semilla=SEMILLA):
    """GaussianMixture (covarianza completa, n_init 3) con k por BIC mínimo. Devuelve (etiquetas, k, BIC por k, modelo)."""
    mods = {k: GaussianMixture(n_components=k, n_init=3, random_state=semilla).fit(Z) for k in ks}
    bic = {k: m.bic(Z) for k, m in mods.items()}
    k = min(bic, key=bic.get)
    return mods[k].predict(Z), k, bic, mods[k]


def hdbscan_sensibilidad(Z, tamanos=(20, 30, 40, 50, 60)):
    """HDBSCAN con cada min_cluster_size. Devuelve dict tamaño → etiquetas (-1 = ruido)."""
    return {m: HDBSCAN(min_cluster_size=m, copy=True).fit_predict(Z) for m in tamanos if m < len(Z)}


def elegir_hdbscan(hd):
    """min_cluster_size sin mirar las referencias: el número de grupos más frecuente en la grilla (el más estable) y,
    entre los que lo dan, el menor tamaño. Devuelve (tamaño, etiquetas)."""
    n = {m: len(set(e) - {-1}) for m, e in hd.items()}
    moda = pd.Series(n).mode().max()
    m = min(t for t, g in n.items() if g == moda)
    return m, hd[m]


def ordenar(et, humedad):
    """Renumera los clusters por humedad media creciente (0 = el más seco), así el número no depende de la semilla ni
    del algoritmo. El ruido (-1) queda -1."""
    et = np.asarray(et); h = pd.Series(np.asarray(humedad, dtype=float))
    medias = h[et != -1].groupby(et[et != -1]).mean().sort_values()
    mapa = {g: i for i, g in enumerate(medias.index)}
    return np.array([mapa.get(g, -1) for g in et])


# ------------------------------------------------------------------ evaluación
def evaluar(Z, et, refs):
    """metricas() de mejoras_clustering (sin el ruido) + ARI y NMI contra cada referencia (dict nombre → serie). Para
    cada referencia se usan solo las muestras con dato; el ruido de HDBSCAN cuenta como un grupo más (no premia mandar lo
    difícil al ruido)."""
    et = np.asarray(et)
    m = metricas(Z, et)
    fila = {"grupos": m["grupos"], "ruido": m["ruido"], "silueta": m["silueta global"], "silueta macro": m["silueta macro"],
            "peor grupo": m["peor grupo"], "Davies-Bouldin": m["Davies-Bouldin"], "Calinski-Harabasz": m["Calinski-Harabasz"]}
    for nombre, r in refs.items():
        r = pd.Series(np.asarray(r, dtype=object))
        ok = r.notna().to_numpy()
        fila[f"ARI {nombre}"] = adjusted_rand_score(r[ok].astype(str), et[ok]) if ok.sum() > 1 else np.nan
        fila[f"NMI {nombre}"] = normalized_mutual_info_score(r[ok].astype(str), et[ok]) if ok.sum() > 1 else np.nan
    return fila


def _mayoritario(s):
    s = s.dropna()
    if s.empty:
        return "-", np.nan, 0
    m = s.value_counts().index[0]
    return m, 100 * (s == m).mean(), len(s)


def describir(df, et, cols=("mg", "st", "humedad", "mges")):
    """Por cluster: muestras, centroide (medias en g/100 g; MGES en g/100 g de extracto seco), y para subtipo,
    consistencia del rótulo y clase CAA de humedad: el valor mayoritario, su % (pureza, sobre las muestras con dato) y
    cuántas tienen dato. El ruido (-1) aparece rotulado «ruido». Clusters ordenados por humedad media."""
    d = df.reset_index(drop=True).copy()
    d["cluster"] = np.where(np.asarray(et) == -1, "ruido", np.asarray(et).astype(str))
    t = d.groupby("cluster")[list(cols)].mean().round(1)
    t.insert(0, "muestras", d.groupby("cluster").size())
    for ref, corto in [("subtipo", "subtipo"), ("consistencia", "consistencia"), ("caa_humedad", "CAA humedad"),
                       ("caa_mges", "CAA MGES")]:
        res = d.groupby("cluster")[ref].apply(_mayoritario)
        t[f"{corto} mayoritario"] = res.map(lambda x: x[0])
        t[f"% {corto}"] = res.map(lambda x: x[1]).round(0)
        if ref in ("subtipo", "consistencia"):
            t[f"con {corto}"] = res.map(lambda x: x[2])
    orden = sorted(t.index, key=lambda g: (g == "ruido", t.loc[g, "humedad"] if "humedad" in t else 0))
    return t.loc[orden]


# ------------------------------------------------------------------ subtags
NOMBRE_CONSISTENCIA = {"baja humedad": "DURO", "mediana humedad": "SEMIDURO", "alta humedad": "PASTA_BLANDA",
                       "procesado": "PROCESADO", "untable": "UNTABLE", "mezcla": "MEZCLA"}
# consistencia del rótulo compatible con cada clase de humedad del CAA (muy alta humedad: untables, crema, ricota)
COMPATIBLE = {"baja humedad": ("baja humedad",), "mediana humedad": ("mediana humedad",),
              "alta humedad": ("alta humedad",), "muy alta humedad": ("untable", "alta humedad")}
NOMBRE_CAA = {"baja humedad": "DURO", "mediana humedad": "SEMIDURO", "alta humedad": "PASTA_BLANDA",
              "muy alta humedad": "MUY_ALTA_HUMEDAD"}


def _tag(s):
    s = re.sub(r"^QUESO\s+", "", str(s).upper())          # «QUESO CREMA» → QUESO_CREMA, no QUESO_QUESO_CREMA
    return "QUESO_" + re.sub(r"[^A-Z0-9]+", "_", s).strip("_")


def proponer_subtags(desc, pureza_min=70, conocidas_min=20, frac_conocida_min=0.5):
    """Un subtag por cluster solo si corresponde a un tipo real, en este orden:
      1) subtipo del rótulo con pureza >= pureza_min (sobre >= conocidas_min muestras con subtipo, que sean al menos
         frac_conocida_min del cluster) → QUESO_<SUBTIPO>;
      2) si no, consistencia del rótulo con esa pureza → QUESO_<DURO | SEMIDURO | PASTA_BLANDA | PROCESADO | UNTABLE>;
      3) si no, clase de humedad del CAA con esa pureza Y la consistencia mayoritaria del rótulo compatible con esa clase
         (COMPATIBLE) en al menos el 50 % → QUESO_<DURO | SEMIDURO | PASTA_BLANDA | MUY_ALTA_HUMEDAD>, marcado
         «CAA + consistencia» (el corte legal solo no alcanza: es función de las mismas dos variables).
    El ruido nunca lleva subtag. Devuelve DataFrame cluster → subtag, criterio, pureza (vacío = no hay estructura)."""
    filas = []
    for g, f in desc.iterrows():
        if g == "ruido":
            continue
        n = f["muestras"]
        ok = lambda c: f[f"con {c}"] >= conocidas_min and f[f"con {c}"] >= frac_conocida_min * n
        if ok("subtipo") and f["% subtipo"] >= pureza_min:
            filas.append((g, _tag(f["subtipo mayoritario"]), "subtipo", f["% subtipo"]))
        elif ok("consistencia") and f["% consistencia"] >= pureza_min and f["consistencia mayoritario"] in NOMBRE_CONSISTENCIA:
            filas.append((g, _tag(NOMBRE_CONSISTENCIA[f["consistencia mayoritario"]]), "consistencia", f["% consistencia"]))
        elif (f["% CAA humedad"] >= pureza_min and ok("consistencia") and f["% consistencia"] >= 50
              and f["consistencia mayoritario"] in COMPATIBLE.get(f["CAA humedad mayoritario"], ())):
            filas.append((g, _tag(NOMBRE_CAA[f["CAA humedad mayoritario"]]), "CAA + consistencia", f["% CAA humedad"]))
    return pd.DataFrame(filas, columns=["cluster", "subtag", "criterio", "% pureza"]).set_index("cluster")


def subtag_por_muestra(et, propuesta):
    """Subtag de cada muestra a partir de las etiquetas y de proponer_subtags (sin subtag o ruido → QUESO)."""
    m = propuesta["subtag"].to_dict()
    return np.array([m.get(str(g), "QUESO") if g != -1 else "QUESO" for g in np.asarray(et)], dtype=object)


def subtag_regla_caa(caa_humedad):
    """La alternativa sin agrupar: el subtag sale directo de la clase de humedad del CAA (art. 605)."""
    return np.array([_tag(NOMBRE_CAA[c]) if c in NOMBRE_CAA else "QUESO" for c in caa_humedad], dtype=object)


# ------------------------------------------------------------------ agrupar sin tag previo
def no_supervisado(rama, semillas=range(10), pct=3, ks_km=range(2, 11), ks_gmm=range(1, 9)):
    """¿Se pueden diferenciar los quesos en clusters sin aprender de un tag previo? Agrupa SOLO con humedad y MGES
    (estandarizadas, sin outliers por consenso): k-medias con k por silueta y mezcla de gaussianas con k por BIC. Ni el
    rótulo ni el CAA entran para ajustar nada; recién después se comparan (ARI y NMI, tablas cruzadas) con la clase de
    humedad del CAA, la clase de MGES del CAA y la consistencia y el tipo (subtipo) del rótulo.
    Estabilidad: el mismo k elegido con otras semillas (ARI de cada partición contra la de la semilla base) y el k que
    elige cada criterio con cada semilla.
    Devuelve un dict: base (quesos sin outliers con las etiquetas), n_outliers, k_km, k_gm, silueta, bic, tabla (métricas
    y ARI/NMI por algoritmo), cruzadas (dict nombre → tabla cruzada), estabilidad (DataFrame por algoritmo y semilla)."""
    X = matriz(rama, BASE)
    es_out, _ = quitar_outliers(X, pct=pct)
    base = rama[~es_out].reset_index(drop=True)
    Z, _ = escalar(matriz(base, BASE))
    h = base["humedad"].to_numpy()
    km, k_km, sil, _ = kmedias(Z, ks_km)
    gm, k_gm, bic, _ = mezcla(Z, ks_gmm)
    et = {f"k-medias (k = {k_km})": ordenar(km, h), f"mezcla de gaussianas (k = {k_gm}, BIC)": ordenar(gm, h)}
    refs = {"CAA humedad": base["caa_humedad"], "CAA MGES": base["caa_mges"], "consistencia": base["consistencia"],
            "tipo": base["subtipo"]}
    filas = {n: evaluar(Z, e, refs) for n, e in et.items()}
    filas["regla CAA de humedad (sin agrupar)"] = evaluar(Z, pd.factorize(base["caa_humedad"])[0], refs)
    tabla = pd.DataFrame(filas).T
    cruzadas = {}
    for n, e in et.items():
        for ref, r in refs.items():
            if ref == "tipo":
                continue
            cruzadas[(n, ref)] = pd.crosstab(pd.Series(e, name="cluster"), pd.Series(r.to_numpy(), name=ref).fillna("sin dato"))
    base_km = et[f"k-medias (k = {k_km})"]; base_gm = et[f"mezcla de gaussianas (k = {k_gm}, BIC)"]
    filas_e = []
    for sd in semillas:
        a = KMeans(n_clusters=k_km, n_init=N_INIT_KMEDIAS, random_state=sd).fit_predict(Z)
        b = GaussianMixture(n_components=k_gm, n_init=3, random_state=sd).fit(Z).predict(Z)
        _, kk, _, _ = kmedias(Z, ks_km, semilla=sd)
        _, kg, _, _ = mezcla(Z, ks_gmm, semilla=sd)
        filas_e.append({"semilla": sd, "ARI k-medias (k fijo)": adjusted_rand_score(base_km, a),
                        "ARI mezcla (k fijo)": adjusted_rand_score(base_gm, b), "k elegido k-medias": kk, "k elegido mezcla": kg})
    return dict(base=base, n_outliers=int(es_out.sum()), k_km=k_km, k_gm=k_gm, silueta=sil, bic=bic, et=et, Z=Z,
                tabla=tabla, cruzadas=cruzadas, estabilidad=pd.DataFrame(filas_e).set_index("semilla"))


# ------------------------------------------------------------------ gráficos
PALETA = ["#0055a5", "#f0a800", "#2e8b57", "#c0392b", "#7c3aed", "#0891b2", "#db2777", "#65a30d", "#92400e", "#475569", "#e11d48",
          "#14b8a6"]


def lineas_caa(ax, x="humedad", y="mges", etiquetas=True):
    """Dibuja los cortes del CAA (humedad 36 / 46 / 55; MGES 10 / 25 / 45 / 60) en una dispersión humedad × MGES."""
    c = cortes_caa()
    for v in c["humedad"]["desde"].iloc[1:]:
        ax.axvline(v, color="#6b7280", lw=0.8, ls="--", zorder=0)
    for v in c["mges"]["desde"].iloc[1:]:
        ax.axhline(v, color="#6b7280", lw=0.8, ls="--", zorder=0)
    if etiquetas:
        tr = ax.get_xaxis_transform()
        for _, r in c["humedad"].iterrows():
            centro = (r["desde"] + min(r["hasta"], 75)) / 2 if r["desde"] > 0 else (20 + r["hasta"]) / 2
            ax.text(centro, 1.01, r["categoria"].replace(" humedad", ""), transform=tr, ha="center", va="bottom",
                    fontsize=7, color="#6b7280")
        tr = ax.get_yaxis_transform()
        for _, r in c["mges"].iterrows():
            centro = (r["desde"] + min(r["hasta"], 75)) / 2
            ax.text(1.01, centro, r["categoria"], transform=tr, ha="left", va="center", fontsize=7, color="#6b7280")


def dispersion_caa(ax, df, color_por, titulo, max_cat=9, orden=None, s=7):
    """humedad × MGES coloreado por una columna categórica (las max_cat más frecuentes; el resto «otros»; sin dato gris
    claro), con los cortes del CAA."""
    v = pd.Series(df[color_por].astype(object).values).where(pd.notna(df[color_por].values), "sin dato").astype(str)
    v = v.replace({"-1": "ruido"})
    cats = orden or [c for c in v.value_counts().index if c not in ("sin dato", "ruido")][:max_cat]
    v = v.where(v.isin(cats + ["sin dato", "ruido"]), "otros")
    x, y = df["humedad"].to_numpy(), df["mges"].to_numpy()
    for c, col in [("sin dato", "#d1d5db"), ("ruido", "#9ca3af"), ("otros", "#6b7280")]:
        sel = (v == c).to_numpy()
        if sel.any():
            ax.scatter(x[sel], y[sel], s=s, c=col, alpha=0.5, lw=0, label=f"{c} ({sel.sum()})")
    for i, c in enumerate(cats):
        sel = (v == c).to_numpy()
        ax.scatter(x[sel], y[sel], s=s, c=PALETA[i % len(PALETA)], alpha=0.75, lw=0, label=f"{c} ({sel.sum()})")
    lineas_caa(ax)
    ax.set_xlabel("humedad (g/100 g) = 100 − ST"); ax.set_ylabel("MG en extracto seco (g/100 g ES)")
    ax.set_title(titulo, fontsize=10, pad=14)
    ax.legend(fontsize=6.5, markerscale=2, loc="lower left", frameon=True, framealpha=0.85)


# ------------------------------------------------------------------ no quesos con NaCl
def no_quesos_con_nacl(S=None):
    """Muestras MQ con NaCl que no son queso (familia de Senasa si está, si no la de mq_tipo_muestra.csv), con MG, ST,
    humedad, MGES, NaCl y proteína. Sirve para ver si hay sueros en polvo y dónde caerían."""
    S = S if S is not None else leer_senasa()
    mq_crudo, _ = leer_mq()
    tipo = pd.read_csv(R.MQ_TIPO).drop_duplicates("codigo").set_index("codigo")["desc_familia"]
    fam = mq_crudo["codigo"].map(S.crudo.set_index("Identificador")["desc_familia"]).fillna(mq_crudo["codigo"].map(tipo))
    d = pd.DataFrame({"Identificador": mq_crudo["codigo"], "familia": fam, "mg": mq_crudo["grasa"],
                      "st": mq_crudo["solidos"], "nacl": mq_crudo["cloruro_nacl_g100g"],
                      "proteina": mq_crudo["proteina"].clip(lower=0), "cenizas": mq_crudo["ceniza"]})
    d = d[d["nacl"].notna() & (d["familia"] != "queso")].reset_index(drop=True)
    d["humedad"] = 100 - d["st"]; d["mges"] = 100 * d["mg"] / d["st"]
    return d
