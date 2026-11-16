"""Mejoras de prueba al agrupamiento de la clase 41 (clases 43 y 46): outliers por consenso, segundo nivel y métricas.

Lo usan prueba_mejoras_clustering.ipynb (lo arma build_prueba_mejoras.py), donde se probaron, y desde que se aprobaron
el notebook completo (secciones 3 y 4, celdas de celdas_clase41.py) y revision_excel.py. No lee archivos por su cuenta
salvo `preparar` y `agregar_subtipo` (que reusan datos_clase41.py y mq_tipo_muestra.csv) y no escribe nada.

    preparar(con_mq)                  las mismas muestras que entran al agrupamiento de celdas_clase41.py (+ columnas de
                                      descripción para comparar, que NO entran)
    escalar(X)                        imputar (mediana) y escalar, como el pipeline
    outliers_iqr(Z)                   la regla actual: k-medias preliminar (k por silueta) + Q3 + 1,5·IQR por grupo
    outliers_consenso(X, grupos, pct) LOF, IsolationForest y EllipticEnvelope dentro de cada grupo, corte en el percentil
                                      pct de cada puntaje; outlier si lo marcan al menos 2 de 3 (clase 46)
    agrupar(X)                        k-medias final con k por silueta (2..10) y tags de datos_clase41.etiquetar
    metricas(Z, etiquetas)            silueta global, silueta macro, Davies-Bouldin y Calinski-Harabasz (clase 43)
    segundo_nivel(X, referencia)      dentro de un tag: GaussianMixture (k por BIC 1..6) y HDBSCAN (sensibilidad a
                                      min_cluster_size), con ARI contra el subtipo conocido y métricas sin el ruido
    proponer_subtag(...)              un subtag solo si el subgrupo se corresponde con un subtipo real
    subtags(X, tag_de, referencia)    lo aprobado para el notebook completo: GMM por BIC dentro de DDL y proponer_subtag
                                      (DDL_HELADERO, DDL_FAMILIAR; el resto queda con su tag)
    predecir_subtag(modelos, X, tag)  el subtag de muestras nuevas con los modelos que devuelve subtags (se guardan en
                                      el joblib)
"""
import re

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

import rutas as R
from datos_clase41 import COLUMNAS, FAMILIAS_NO_LACTEAS, N_INIT_KMEDIAS, SEMILLA, SIN_FAMILIA, SUMA_MAX, etiquetar, leer_mq, leer_senasa, unir

DESCRIPTIVAS = ["desc_subtipo", "desc_grasa", "desc_atributos", "Descripción"]
ESPECIALES_DDL = {"repostero": r"REPOSTER|RESPOSTER",
                  "heladero": r"HELADER",
                  "reducido en grasa": r"REDUCID\w*\s+EN\s+GRAS|DESCREMAD|\bLIGHT\b|BAJO\s+EN\s+GRAS"}


# ------------------------------------------------------------------ datos
def preparar(con_mq, S=None):
    """Las muestras del agrupamiento, con la misma lectura, correcciones, unión y filtros que celdas_clase41.py.
    Devuelve (muestras, S). muestras trae además desc_subtipo, desc_grasa, desc_atributos y Descripción (solo para
    comparar; no entran en el agrupamiento). Quedan afuera los quesos que van a la rama propia (clustering_quesos.rama_queso:
    familia queso de Senasa y solo MQ con familia queso confirmada por el rótulo), que no pasan por el k-medias."""
    import clustering_quesos as Q          # acá adentro: clustering_quesos importa este módulo
    S = S if S is not None else leer_senasa()
    senasa = S.crudo[["Identificador", "desc_familia", "desc_producto", "desc_tipo_muestra"] + COLUMNAS].copy()
    if con_mq:
        _, mq = leer_mq()
        todas, _ = unir(senasa, mq)
    else:
        todas = senasa.assign(origen="Senasa")
    todas = todas[~todas["Identificador"].isin(S.excluidas) & ~todas["desc_familia"].isin(FAMILIAS_NO_LACTEAS)]
    _, confirmados, _, _ = Q.rama_queso(S, con_mq)
    todas = todas[~todas["Identificador"].isin(confirmados)]
    completas = todas.dropna(subset=COLUMNAS)
    muestras = completas[completas[COLUMNAS].sum(axis=1) <= SUMA_MAX].reset_index(drop=True)
    return agregar_subtipo(muestras, S), S


def agregar_subtipo(muestras, S):
    """Copia de `muestras` (con Identificador, desc_familia y desc_producto) con las columnas descriptivas de Senasa que
    le falten (DESCRIPTIVAS), desc_familia sin vacíos (SIN_FAMILIA), y subtipo_ref y fuente_subtipo (subtipo_referencia).
    Mismo orden y largo que `muestras`. Nada de esto entra en el agrupamiento: es para comparar y nombrar subtags."""
    faltan = [c for c in DESCRIPTIVAS if c not in muestras]
    m = muestras.merge(S.crudo[["Identificador"] + faltan], on="Identificador", how="left", validate="many_to_one") if faltan else muestras.copy()
    assert len(m) == len(muestras)
    m["desc_familia"] = m["desc_familia"].fillna(SIN_FAMILIA)
    m["subtipo_ref"], m["fuente_subtipo"] = subtipo_referencia(m)
    return m


def subtipo_referencia(muestras, ruta_tipo=R.MQ_TIPO):
    """Subtipo conocido para comparar (no entra en el agrupamiento): desc_subtipo de Senasa; en dulce de leche sin
    subtipo, el que diga desc_producto (REPOSTERO, HELADERO, FAMILIAR/CLÁSICO/TRADICIONAL → FAMILIAR); si sigue sin
    subtipo, el de mq_tipo_muestra.csv (salvo «sin especificar»). Devuelve (subtipo, fuente)."""
    st = muestras["desc_subtipo"].astype("string")
    fuente = pd.Series(np.where(st.notna(), "Senasa", None), index=muestras.index, dtype=object)
    prod = muestras["desc_producto"].astype("string").str.upper().fillna("")
    ddl = (muestras["desc_familia"] == "dulce de leche") & st.isna()
    for patron, valor in [(r"REPOSTER|RESPOSTER", "REPOSTERO"), (r"HELADER", "HELADERO"), (r"FAMILIAR|CLASICO|TRADICIONAL", "FAMILIAR")]:
        sel = ddl & st.isna() & prod.str.contains(patron, regex=True)
        st[sel] = valor; fuente[sel] = "Senasa (desc_producto)"
    tipo = pd.read_csv(ruta_tipo, usecols=["codigo", "desc_subtipo"]).drop_duplicates("codigo").set_index("codigo")["desc_subtipo"]
    tipo = tipo[tipo != "sin especificar"]
    de_mq = muestras["Identificador"].map(tipo).astype("string")
    sel = st.isna() & de_mq.notna()
    st[sel] = de_mq[sel]; fuente[sel] = "mq_tipo_muestra"
    return st.astype(object).where(st.notna(), np.nan), fuente


def especiales_ddl(muestras):
    """DataFrame booleano (una columna por tipo especial) para las muestras de familia dulce de leche: busca en
    desc_producto, desc_subtipo, desc_grasa, desc_atributos y Descripción."""
    texto = muestras[["desc_producto"] + DESCRIPTIVAS].astype("string").fillna("").agg(" ".join, axis=1).str.upper()
    es_ddl = (muestras["desc_familia"] == "dulce de leche").values
    return pd.DataFrame({n: es_ddl & texto.str.contains(p, regex=True).values for n, p in ESPECIALES_DDL.items()},
                        index=muestras.index)


def escalar(X):
    """Imputar con la mediana y escalar (media 0, desvío 1), como el pipeline. Devuelve (Z, imputador, escalador)."""
    imp = SimpleImputer(strategy="median").fit(X)
    esc = StandardScaler().fit(imp.transform(X))
    return esc.transform(imp.transform(X)), imp, esc


# ------------------------------------------------------------------ outliers
def outliers_iqr(Z, semilla=SEMILLA, ks=range(2, 11)):
    """La regla actual (celdas_clase41.py, «Quitar los outliers»). Devuelve (es_outlier, grupos preliminares, K_PRE)."""
    pre = {k: KMeans(n_clusters=k, n_init=N_INIT_KMEDIAS, random_state=semilla).fit(Z) for k in ks}
    k_pre = max(ks, key=lambda k: silhouette_score(Z, pre[k].labels_))
    km = pre[k_pre]
    dist = km.transform(Z)[np.arange(len(Z)), km.labels_]
    es = np.zeros(len(Z), dtype=bool)
    for j in range(k_pre):
        sel = km.labels_ == j
        q1, q3 = np.percentile(dist[sel], [25, 75])
        es[sel] = dist[sel] > q3 + 1.5 * (q3 - q1)
    return es, km.labels_, k_pre


def puntajes_grupo(X, n_vecinos=50, semilla=SEMILLA):
    """Los tres puntajes de la clase 46 para las muestras de un grupo, después de reescalar dentro del grupo.
    Más bajo = más raro en los tres (negative_outlier_factor_ y score_samples)."""
    Zg = StandardScaler().fit_transform(X)
    n = len(Zg)
    lof = LocalOutlierFactor(n_neighbors=min(n_vecinos, n - 1)).fit(Zg).negative_outlier_factor_
    bosque = IsolationForest(n_estimators=200, random_state=semilla, n_jobs=-1).fit(Zg).score_samples(Zg)
    elipse = EllipticEnvelope(random_state=semilla, support_fraction=None).fit(Zg).score_samples(Zg)
    return pd.DataFrame({"factor local": lof, "bosque": bosque, "elipse": elipse})


def outliers_consenso(X, grupos, pct=3, minimo=2, n_vecinos=50, semilla=SEMILLA, tam_min=30):
    """Outliers por consenso dentro de cada grupo (preliminar o tag).

    X: matriz (n × 4) en g/100 g (o ya imputada); grupos: etiqueta de grupo por muestra.
    En cada grupo con al menos tam_min muestras: reescalar, calcular LOF (n_vecinos), IsolationForest (200 árboles) y
    EllipticEnvelope; cada uno marca el pct % de menor puntaje del grupo; outlier si lo marcan >= minimo de 3.
    Los grupos con menos de tam_min muestras no se filtran (no alcanzan para estimar una elipse ni 50 vecinos).
    Devuelve (es_outlier, votos: DataFrame con una columna booleana por método y la columna «votos»).
    """
    X = np.asarray(X, dtype=float); grupos = np.asarray(grupos)
    votos = pd.DataFrame(False, index=range(len(X)), columns=["factor local", "bosque", "elipse"])
    for g in pd.unique(grupos):
        idx = np.flatnonzero(grupos == g)
        if len(idx) < tam_min:
            continue
        p = puntajes_grupo(X[idx], n_vecinos=n_vecinos, semilla=semilla)
        for col in votos.columns:
            votos.loc[idx, col] = (p[col] < np.percentile(p[col], pct)).values
    votos["votos"] = votos[["factor local", "bosque", "elipse"]].sum(axis=1)
    return (votos["votos"] >= minimo).to_numpy(), votos


# ------------------------------------------------------------------ agrupamiento final y métricas
def agrupar(X, semilla=SEMILLA, ks=range(2, 11)):
    """k-medias final como el pipeline: reimputar y reescalar con las muestras que quedan, k por silueta máxima.
    Devuelve dict con Z, km, K, grupos, tags (cluster → tag), tag_de (por muestra), centros (g/100 g), siluetas por k."""
    Z, imp, esc = escalar(X)
    modelos = {k: KMeans(n_clusters=k, n_init=N_INIT_KMEDIAS, random_state=semilla).fit(Z) for k in ks}
    sil = {k: silhouette_score(Z, modelos[k].labels_) for k in ks}
    K = max(sil, key=sil.get)
    km = modelos[K]
    centros = pd.DataFrame(esc.inverse_transform(km.cluster_centers_), columns=COLUMNAS, index=range(K)).round(2)
    nombres, tags, repetidos = etiquetar(centros)
    tag_de = np.array([tags[g] for g in km.labels_])
    centros.insert(0, "muestras", np.bincount(km.labels_)); centros.insert(0, "tag", [tags[j] for j in range(K)])
    return dict(Z=Z, km=km, K=K, grupos=km.labels_, tags=tags, tag_de=tag_de, centros=centros, siluetas=sil,
                repetidos=repetidos, imputador=imp, escalador=esc)


def metricas(Z, etiquetas):
    """Índices internos sin el sesgo del grupo más grande (clase 43). Las etiquetas -1 (ruido) se dejan fuera.
    silueta global: promedio sobre muestras (la domina el grupo más grande) · silueta macro: promedio de las medias por
    grupo (cada grupo pesa igual) · peor grupo: la menor media · Davies-Bouldin (menor es mejor) · Calinski-Harabasz
    (mayor es mejor)."""
    etiquetas = np.asarray(etiquetas); con = etiquetas != -1
    Zc, ec = Z[con], etiquetas[con]
    if len(np.unique(ec)) < 2:
        return {"grupos": int(len(np.unique(ec))), "ruido": int((~con).sum()), "silueta global": np.nan,
                "silueta macro": np.nan, "peor grupo": np.nan, "Davies-Bouldin": np.nan, "Calinski-Harabasz": np.nan}
    s = silhouette_samples(Zc, ec)
    por_grupo = pd.Series(s).groupby(ec).mean()
    return {"grupos": int(len(por_grupo)), "ruido": int((~con).sum()), "silueta global": float(s.mean()),
            "silueta macro": float(por_grupo.mean()), "peor grupo": float(por_grupo.min()),
            "Davies-Bouldin": float(davies_bouldin_score(Zc, ec)), "Calinski-Harabasz": float(calinski_harabasz_score(Zc, ec))}


def silueta_por_grupo(Z, etiquetas):
    s = silhouette_samples(Z, etiquetas)
    return pd.Series(s).groupby(np.asarray(etiquetas)).agg(["size", "mean"]).rename(columns={"size": "muestras", "mean": "silueta media"})


# ------------------------------------------------------------------ segundo nivel
def segundo_nivel(X, referencia=None, semilla=SEMILLA, ks=range(1, 7), tamanos=(20, 30, 40, 50)):
    """Subgrupos dentro de un tag. X: (n × 4) en g/100 g; referencia: subtipo conocido por muestra (NaN = desconocido).

    Reescala dentro del tag; GaussianMixture con k por BIC (n_init 3, como la clase 46) y HDBSCAN con cada
    min_cluster_size de `tamanos` (clase 43; se saltean los mayores que el tag). Para cada partición: grupos, ruido, ARI contra la referencia (solo con
    referencia conocida y, en HDBSCAN, sin el ruido), silueta y Davies-Bouldin sin el ruido.
    Devuelve dict con Z, bic (k → BIC), k_bic, gmm (etiquetas), hdbscan (tamaño → etiquetas), sensibilidad (DataFrame),
    escalador y modelo_gmm (para ubicar muestras nuevas). tamanos=() saltea HDBSCAN.
    """
    X = np.asarray(X, dtype=float)
    escalador = StandardScaler().fit(X)
    Z = escalador.transform(X)
    bic = {k: GaussianMixture(n_components=k, n_init=3, random_state=semilla).fit(Z).bic(Z) for k in ks}
    k_bic = min(bic, key=bic.get)
    modelo_gmm = GaussianMixture(n_components=k_bic, n_init=3, random_state=semilla).fit(Z)
    gmm = modelo_gmm.predict(Z)
    hd = {m: HDBSCAN(min_cluster_size=m, copy=True).fit_predict(Z) for m in tamanos if m < len(Z)}
    ref = None if referencia is None else pd.Series(referencia).reset_index(drop=True)

    def fila(nombre, et):
        m = metricas(Z, et)
        r = {"partición": nombre, "grupos": m["grupos"], "ruido": m["ruido"], "silueta (sin ruido)": m["silueta global"],
             "silueta macro": m["silueta macro"], "Davies-Bouldin": m["Davies-Bouldin"]}
        if ref is not None:
            ok = ref.notna().to_numpy() & (et != -1)
            r["ARI contra subtipo"] = adjusted_rand_score(ref[ok].astype(str), et[ok]) if ok.sum() > 1 else np.nan
            ok2 = ref.notna().to_numpy()      # el ruido como un grupo más: no premia mandar lo difícil al ruido
            r["ARI con el ruido"] = adjusted_rand_score(ref[ok2].astype(str), et[ok2]) if ok2.sum() > 1 else np.nan
        r["tamaños"] = str(np.bincount(et[et >= 0]).tolist()) if (et >= 0).any() else "[]"
        return r

    filas = [fila(f"GMM, k = {k_bic} (BIC)", gmm)] + [fila(f"HDBSCAN min_cluster_size {m}", e) for m, e in hd.items()]
    return dict(Z=Z, bic=bic, k_bic=k_bic, gmm=gmm, hdbscan=hd, sensibilidad=pd.DataFrame(filas).set_index("partición"),
                escalador=escalador, modelo_gmm=modelo_gmm)


def describir_subgrupos(X, etiquetas, referencia=None):
    """Centroide en g/100 g, tamaño, subtipo mayoritario, % de ese subtipo en el subgrupo (pureza) y % de ese subtipo
    que cae en el subgrupo (cobertura). El ruido (-1) aparece como un subgrupo más, rotulado «ruido»."""
    d = pd.DataFrame(np.asarray(X, dtype=float), columns=COLUMNAS)
    d["subgrupo"] = np.where(np.asarray(etiquetas) == -1, "ruido", np.asarray(etiquetas).astype(str))
    t = d.groupby("subgrupo")[COLUMNAS].mean().round(2)
    t.insert(0, "muestras", d.groupby("subgrupo").size())
    if referencia is not None:
        ref = pd.Series(np.asarray(referencia, dtype=object))
        d["ref"] = ref.where(ref.notna(), None)
        may, pur, cob, conocidas = [], [], [], []
        for g in t.index:
            r = d.loc[d["subgrupo"] == g, "ref"].dropna()
            conocidas.append(len(r))
            if r.empty:
                may.append("-"); pur.append(np.nan); cob.append(np.nan); continue
            m = r.mode()[0]; may.append(m); pur.append(100 * (r == m).mean())
            cob.append(100 * ((d["subgrupo"] == g) & (d["ref"] == m)).sum() / (d["ref"] == m).sum())
        t["con subtipo"] = conocidas; t["subtipo mayoritario"] = may
        t["% pureza"] = np.round(pur, 0); t["% cobertura"] = np.round(cob, 0)
    return t


def proponer_subtag(tag, descripcion, ari, ari_min=0.2, pureza_min=60, cobertura_min=50, conocidas_min=20, frac_conocida_min=0.5):
    """Subtag por subgrupo solo si el subgrupo se corresponde con un subtipo real. Hace falta todo esto:
    ARI de la partición >= ari_min; en el subgrupo, al menos conocidas_min muestras con subtipo conocido y que sean al
    menos frac_conocida_min del subgrupo (si no, la pureza se mide sobre una minoría); pureza >= pureza_min % y
    cobertura >= cobertura_min % de su subtipo mayoritario. Devuelve dict subgrupo → subtag (vacío: no hay estructura)."""
    if not np.isfinite(ari) or ari < ari_min or "% pureza" not in descripcion:
        return {}
    out = {}
    for g, f in descripcion.iterrows():
        if g == "ruido" or f["subtipo mayoritario"] == "-":
            continue
        if (f["con subtipo"] >= conocidas_min and f["con subtipo"] >= frac_conocida_min * f["muestras"]
                and f["% pureza"] >= pureza_min and f["% cobertura"] >= cobertura_min):
            out[g] = f"{tag}_{re.sub(r'[^A-Z0-9]+', '_', str(f['subtipo mayoritario']).upper()).strip('_')}"
    return out


# ------------------------------------------------------------------ subtags aprobados (notebook completo)
TAGS_CON_SUBTAG = ("DDL",)      # QUESO y POLVO_GRASO no tienen estructura; POLVO_MAGRO_WPC / _DESMINERALIZADO, inestables


def subtags(X, tag_de, referencia, tags=TAGS_CON_SUBTAG, semilla=SEMILLA, tam_min=30):
    """El segundo nivel aprobado: dentro de cada tag de `tags` (con al menos tam_min muestras), GaussianMixture con k por
    BIC (segundo_nivel, sin HDBSCAN); cada componente se describe contra el subtipo conocido y lleva subtag solo si
    cumple proponer_subtag. Las muestras de los demás componentes, y las de los demás tags, quedan con su tag.

    X: (n × 4) en g/100 g de las muestras agrupadas; tag_de: tag por muestra; referencia: subtipo conocido (NaN si no).
    Devuelve (subtag por muestra, modelos): modelos es tag → dict(escalador, gmm, mapa componente → subtag, k_bic, bic,
    ari, descripcion), lo necesario para predecir_subtag.
    """
    X = np.asarray(X, dtype=float); tag_de = np.asarray(tag_de, dtype=object)
    ref = pd.Series(np.asarray(referencia, dtype=object))
    sub = tag_de.copy(); modelos = {}
    for tag in tags:
        sel = tag_de == tag
        if sel.sum() < tam_min:
            continue
        r = ref[sel].reset_index(drop=True)
        sn = segundo_nivel(X[sel], r, semilla=semilla, tamanos=())
        ari = float(sn["sensibilidad"]["ARI con el ruido"].iloc[0])
        desc = describir_subgrupos(X[sel], sn["gmm"], r)
        mapa = {int(g): st for g, st in proponer_subtag(tag, desc, ari).items()}
        sub[np.flatnonzero(sel)] = [mapa.get(int(e), tag) for e in sn["gmm"]]
        desc.insert(0, "subtag", [mapa.get(int(g), tag) for g in desc.index])
        modelos[tag] = dict(escalador=sn["escalador"], gmm=sn["modelo_gmm"], mapa=mapa, k_bic=sn["k_bic"], bic=sn["bic"],
                            ari=ari, descripcion=desc)
    return sub, modelos


def predecir_subtag(modelos, X, tag_de):
    """Subtag de muestras (nuevas o no) con los modelos de `subtags`: reescala con el escalador del tag y asigna el
    componente de la mezcla; sin subtag para ese componente (o tag sin modelo), queda el tag."""
    X = np.asarray(X, dtype=float); tag_de = np.asarray(tag_de, dtype=object)
    sub = tag_de.copy()
    for tag, m in modelos.items():
        sel = tag_de == tag
        if sel.any():
            comp = m["gmm"].predict(m["escalador"].transform(X[sel]))
            sub[np.flatnonzero(sel)] = [m["mapa"].get(int(e), tag) for e in comp]
    return sub


# ------------------------------------------------------------------ armar y comparar variantes
def variante(X, es_outlier, Z_todas, comunes=None, semilla=SEMILLA):
    """Quita es_outlier, rehace el k-medias final (agrupar) y lo mide tres veces:
    - sobre sus propias muestras (lo que reporta el pipeline; no es comparable entre variantes que quitan distinto),
    - sobre todas las muestras (cada modelo asigna con predict; escala común Z_todas),
    - sobre las comunes (las que ninguna variante quita), en la misma escala común.
    Devuelve (resultado de agrupar, fila de métricas, etiquetas predichas para todas)."""
    X = np.asarray(X, dtype=float)
    r = agrupar(X[~es_outlier], semilla=semilla)
    et_todas = r["km"].predict(r["escalador"].transform(r["imputador"].transform(X)))
    propia = metricas(r["Z"], r["grupos"])
    todas = metricas(Z_todas, et_todas)
    fila = {"% quitado": 100 * es_outlier.mean(), "muestras": int((~es_outlier).sum()), "k": r["K"],
            "tags": " ".join(sorted(r["tags"].values())),
            "silueta global": propia["silueta global"], "silueta macro": propia["silueta macro"],
            "peor grupo": propia["peor grupo"], "Davies-Bouldin": propia["Davies-Bouldin"],
            "Calinski-Harabasz": propia["Calinski-Harabasz"],
            "silueta global (todas)": todas["silueta global"], "silueta macro (todas)": todas["silueta macro"]}
    if comunes is not None:
        c = metricas(Z_todas[comunes], et_todas[comunes])
        fila["silueta global (comunes)"] = c["silueta global"]; fila["silueta macro (comunes)"] = c["silueta macro"]
        fila["Davies-Bouldin (comunes)"] = c["Davies-Bouldin"]
    return r, fila, et_todas


def elegir_particion(sn):
    """La partición del segundo nivel que más se parece al subtipo, contando el ruido como un grupo más (así HDBSCAN no
    gana por mandar lo difícil al ruido). Empate → GMM. Devuelve (nombre, etiquetas)."""
    t = sn["sensibilidad"]
    col = "ARI con el ruido" if "ARI con el ruido" in t else None
    nombre = t.index[0] if col is None else t[col].fillna(-1).idxmax()
    if nombre.startswith("GMM"):
        return nombre, sn["gmm"]
    return nombre, sn["hdbscan"][int(nombre.split()[-1])]
