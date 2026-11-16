"""Segundo nivel por familia del agrupamiento: el subtag de las secciones 3 y 4 del notebook único.

El primer nivel (k-medias sobre humedad, grasa, proteínas y cenizas) da el tag. Dentro de tres ramas, el subtag sale del
método que aprobó cada prueba por familia (README, «Pruebas por familia»):

  rama   muestras                          método                                              subtags
  DDL    tag DDL                           regresión logística multinomial sobre los cuatro    DDL_FAMILIAR, DDL_REPOSTERO,
                                           análisis, entrenada con el subtipo conocido;        DDL_HELADERO; «DDL» si no decide
                                           decide con probabilidad >= 0,70 y si no es novedad
                                           (prueba_subtipo_ddl)
  polvo  tags POLVO_GRASO, POLVO_MAGRO y   dos reglas de composición en base seca, primero:    POLVO_SUERO_REENGRASADO,
         PROT_ALTA, con humedad <= 15      suero reengrasado y leche descremada (REGLAS_POLVO); POLVO_LECHE_DESCREMADA;
                                           el resto, outliers por consenso dentro de la rama y  POLVO_LECHE_ENTERA, POLVO_WPC,
                                           k-medias (k por silueta) en grasa, proteína y        POLVO_CASEINATO, POLVO_SUERO_...;
                                           cenizas en base seca; un cluster lleva subtag solo   el tag si el cluster no
                                           si coincide con un tipo real (prueba_clustering_     coincide con ningún tipo
                                           polvos), contado sin las muestras de las reglas
  queso  tag QUESO                         clase de humedad del CAA (art. 605; prueba_         QUESO_DURO, QUESO_SEMIDURO,
                                           clustering_quesos)                                  QUESO_PASTA_BLANDA, QUESO_MUY_ALTA_HUMEDAD

El queso del notebook único NO pasa por el k-medias: los quesos que confirma el catálogo van a su propia rama
(clustering_quesos.rama_queso, con humedad = 100 − ST y MG) y reciben acá solo `subtag_queso`. La rama «queso» de abajo
(tag QUESO) queda para un cluster del k-medias que caiga en esa regla (hoy ninguno, porque los quesos confirmados salen
antes); no se le asume nada más que la clase de humedad.

Las ramas van por el TAG (el primer nivel), no por desc_familia: así el subtag está siempre dentro de su tag y se puede
predecir para una muestra nueva solo con los cuatro análisis. El subtipo o tipo del rótulo se usa para entrenar (DDL) y
para nombrar (polvo), nunca como entrada. Las demás muestras quedan con subtag = tag. La columna `criterio` dice de dónde
sale el subtag de cada muestra.

    ajustar(muestras, tag_de, S)     → (subtag, criterio, modelos); muestras con Identificador, COLUMNAS y subtipo_ref
    predecir(modelos, X, tag_de)     → (subtag, criterio) de muestras nuevas (o de las mismas, para verificar)
    probabilidades_ddl(modelos, X)   las tres probabilidades de la logística del DDL
    para_guardar(modelos)            lo que va al joblib (sin las tablas descriptivas)
"""
import re

import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import clustering_polvos as P
import cortes_caa
from datos_clase41 import COLUMNAS, SEMILLA
from mejoras_clustering import outliers_consenso

# ------------------------------------------------------------------ parámetros (los de cada prueba)
CLASES_DDL = ["FAMILIAR", "REPOSTERO", "HELADERO"]
UMBRAL_DDL = 0.70            # probabilidad mínima para decidir
PCT_NOVEDAD = 99             # novedad: más allá del percentil 99 de Mahalanobis robusta en las tres clases, o LOF
TAGS_POLVO = ("POLVO_GRASO", "POLVO_MAGRO", "PROT_ALTA")
HUMEDAD_MAX_POLVO = P.HUMEDAD_MAX        # 15: arriba son líquidos (hueco natural entre 12 y 25)
KS_POLVO = range(2, 11)
PCT_OUTLIER_POLVO = 3
# Reglas de composición en base seca (g/100 g de extracto seco), antes del k-medias. El k-medias no las separa: escala
# la proteína con todos los polvos (de 3 a 91), y a esa escala el suero reengrasado queda con la leche entera o con los
# sueros magros, y la leche descremada (proteína 35, cenizas 8) con un concentrado de suero de 46 % de proteína y las
# mismas cenizas; ningún k de 6 a 14 da un cluster de leche descremada con pureza >= 70 % en Senasa + MQ. Los umbrales
# salen de los rótulos (pureza 100 % entre las que tienen tipo, en las dos versiones):
#   suero reengrasado (suero con oleomargarina): grasa >= 15 (el suero magro llega a 8,5) y proteína <= 12 (la leche
#     entera tiene ~26; los polvos para bebida láctea con oleomargarina, 15 a 17).
#   leche descremada: grasa <= 2,5 (el WPC tiene 2,7 a 4,7), cenizas >= 7 (el WPC, 5,0 a 6,6) y proteína 30 a 40 (el
#     permeado tiene ~3 con cenizas altas; el concentrado de 46 % queda afuera). Por proteína sola la leche descremada
#     (34 a 38,5) y el WPC 35 (35 a 39) se pisan: los separan las cenizas y la grasa.
REGLAS_POLVO = [
    ("POLVO_SUERO_REENGRASADO", "polvo: regla suero reengrasado (grasa ≥ 15, proteína ≤ 12)",
     lambda B: (B[:, 0] >= 15) & (B[:, 1] <= 12)),
    ("POLVO_LECHE_DESCREMADA", "polvo: regla leche descremada (grasa ≤ 2,5, cenizas ≥ 7, proteína 30 a 40)",
     lambda B: (B[:, 0] <= 2.5) & (B[:, 2] >= 7) & (B[:, 1] >= 30) & (B[:, 1] <= 40)),
]
# CAA art. 605, intervalos cerrados a la izquierda (36 ya es mediana humedad): los mismos de cortes_categorias.csv
CORTES_QUESO = [(36.0, "QUESO_DURO"), (46.0, "QUESO_SEMIDURO"), (55.0, "QUESO_PASTA_BLANDA"), (np.inf, "QUESO_MUY_ALTA_HUMEDAD")]

C_DDL = "DDL: logística (prob ≥ 0,70)"
C_DDL_PROB = "DDL: sin decidir (prob < 0,70)"
C_DDL_NOV = "DDL: sin decidir (novedad)"
C_POLVO = "polvo: k-medias en base seca"
C_POLVO_SIN = "polvo: cluster sin tipo"
C_POLVO_OUT = "polvo: outlier de la rama"
C_POLVO_LIQ = "polvo: humedad > 15 (no es polvo)"
C_QUESO = "queso: humedad del CAA"
C_TAG = "sin segundo nivel"


# ------------------------------------------------------------------ DDL
def _ddl_ajustar(X, y):
    lab = pd.Series(y).isin(CLASES_DDL).to_numpy()
    Xl, yl = X[lab], np.asarray(y, dtype=object)[lab]
    clf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)).fit(Xl, yl)
    mcd, umbral = {}, {}
    for k in CLASES_DDL:
        mcd[k] = MinCovDet(random_state=SEMILLA).fit(Xl[yl == k])
        umbral[k] = float(np.percentile(mcd[k].mahalanobis(Xl[yl == k]), PCT_NOVEDAD))
    esc = StandardScaler().fit(Xl)
    lof = LocalOutlierFactor(n_neighbors=30, novelty=True).fit(esc.transform(Xl))
    oof = np.zeros(len(Xl))      # el corte de LOF, con el puntaje de las etiquetadas fuera de pliegue (~1 % de falsas alarmas)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEMILLA).split(Xl, yl):
        e_ = StandardScaler().fit(Xl[tr])
        oof[te] = LocalOutlierFactor(n_neighbors=30, novelty=True).fit(e_.transform(Xl[tr])).score_samples(e_.transform(Xl[te]))
    return dict(clasificador=clf, mcd=mcd, umbral_mahalanobis=umbral, escalador_lof=esc, lof=lof,
                corte_lof=float(np.percentile(oof, 100 - PCT_NOVEDAD)), umbral=UMBRAL_DDL, etiquetadas=int(lab.sum()),
                por_clase={k: int((yl == k).sum()) for k in CLASES_DDL})


def _ddl_predecir(m, X):
    Pr = m["clasificador"].predict_proba(X)
    clases = m["clasificador"].classes_
    prob, pred = Pr.max(axis=1), clases[Pr.argmax(axis=1)]
    nov = np.all([m["mcd"][k].mahalanobis(X) > m["umbral_mahalanobis"][k] for k in CLASES_DDL], axis=0)
    nov |= m["lof"].score_samples(m["escalador_lof"].transform(X)) < m["corte_lof"]
    dec = (prob >= m["umbral"]) & ~nov
    sub = np.where(dec, np.char.add("DDL_", pred.astype(str)), "DDL").astype(object)
    crit = np.where(dec, C_DDL, np.where(nov, C_DDL_NOV, C_DDL_PROB)).astype(object)
    return sub, crit, pd.DataFrame(Pr, columns=[f"p_{c.lower()}" for c in clases])


# ------------------------------------------------------------------ polvo
def _base_seca(X):
    X = np.asarray(X, dtype=float)
    st = 100 - X[:, 0]
    return np.column_stack([100 * X[:, 1] / st, 100 * X[:, 2] / st, 100 * X[:, 3] / st])


def _referencias_polvo(ids, S):
    """Cliente, familia, tipo del rótulo y clase CAA de la leche de cada Identificador (de clustering_polvos), solo para
    nombrar los clusters. Lo que no es un polvo de las tres familias queda sin tipo ni familia."""
    if not P.CORTES.exists():                  # la clase CAA de la leche sale de cortes_categorias.csv (sección 5)
        cortes_caa.armar()
    pol, _ = P.preparar_polvos(S)
    ref = pol.drop_duplicates("Identificador").set_index("Identificador")[["Cliente", "familia", "tipo", "clase_caa"]]
    return ref.reindex(list(ids)).reset_index(drop=True)


def _reglas_polvo(B):
    """(subtag, criterio) de las reglas de composición; None donde no aplica ninguna."""
    sub, crit = np.full(len(B), None, dtype=object), np.full(len(B), None, dtype=object)
    for nombre, criterio, regla in REGLAS_POLVO:
        r = regla(B) & pd.isna(sub)
        sub[r], crit[r] = nombre, criterio
    return sub, crit


def _polvo_ajustar(X, ids, S):
    """El k-medias se ajusta con todas las muestras de la rama que no son outliers (la partición no depende de las
    reglas), pero cada cluster se nombra solo con las que no toma ninguna regla: así un cluster de sueros reengrasados y
    bebidas lácteas no le pasa el nombre del suero a las bebidas."""
    B = _base_seca(X)
    es_out, _ = outliers_consenso(B, np.zeros(len(B), dtype=int), pct=PCT_OUTLIER_POLVO)
    esc = StandardScaler().fit(B[~es_out])
    et, k, sil, km = P.kmedias(esc.transform(B[~es_out]), KS_POLVO)
    df = pd.DataFrame(B[~es_out], columns=P.BASE)
    df["resto_bs"] = 100 - df[P.BASE].sum(axis=1)
    df["humedad"] = np.asarray(X, dtype=float)[~es_out, 0]
    df = pd.concat([df, _referencias_polvo(np.asarray(ids)[~es_out], S)], axis=1)
    orden = np.asarray(P.ordenar(et, df["proteina_bs"]))     # clusters numerados por proteína en base seca
    libre = pd.isna(_reglas_polvo(B[~es_out])[0])
    desc = P.describir(df[libre], orden[libre])
    prop = P.proponer_subtags(desc)
    renumera = {int(a): int(b) for a, b in zip(et, orden)}
    mapa = {g: prop["subtag"].get(str(renumera[g])) for g in range(k)}
    desc.insert(0, "subtag", [prop["subtag"].get(g, "(el tag)") for g in desc.index])
    return dict(escalador=esc, kmedias=km, k=k, siluetas=sil, renumera=renumera, mapa=mapa, descripcion=desc,
                reglas=[(n, c) for n, c, _ in REGLAS_POLVO], n_outliers=int(es_out.sum()), n=len(B)), es_out


def _polvo_predecir(m, X, tag):
    """Subtag por k-medias, sin las reglas (se aplican después, también sobre los outliers de la rama)."""
    km = m["kmedias"].predict(m["escalador"].transform(_base_seca(X)))
    sub = np.array([m["mapa"].get(int(g)) or t for g, t in zip(km, tag)], dtype=object)
    crit = np.array([C_POLVO if m["mapa"].get(int(g)) else C_POLVO_SIN for g in km], dtype=object)
    return sub, crit


def _aplicar_reglas_polvo(X, mascara, sub, crit):
    idx = np.flatnonzero(mascara)
    s_, c_ = _reglas_polvo(_base_seca(X[idx]))
    ok = pd.notna(s_)
    sub[idx[ok]], crit[idx[ok]] = s_[ok], c_[ok]


# ------------------------------------------------------------------ queso
def subtag_queso(humedad):
    h = np.asarray(humedad, dtype=float)
    out = np.full(len(h), "QUESO", dtype=object)
    lim_inf = -np.inf
    for lim, nombre in CORTES_QUESO:
        out[(h >= lim_inf) & (h < lim)] = nombre
        lim_inf = lim
    return out


# ------------------------------------------------------------------ las tres ramas juntas
def tag_base(tag_de):
    """El tag sin el sufijo de desambiguación de etiquetar (POLVO_GRASO_2 → POLVO_GRASO): cuando dos clusters caen en
    la misma regla, los dos van a la misma rama. Sin tag (NaN o None) queda None."""
    return np.array([re.sub(r"_\d+$", "", t) if isinstance(t, str) else None for t in tag_de], dtype=object)


def es_polvo(tag_de):
    return np.isin(tag_base(tag_de), TAGS_POLVO)


def _ramas(X, tag_de):
    base = tag_base(tag_de)
    polvo = np.isin(base, TAGS_POLVO)
    return {"DDL": base == "DDL", "polvo": polvo & (X[:, 0] <= HUMEDAD_MAX_POLVO), "liquido": polvo & (X[:, 0] > HUMEDAD_MAX_POLVO),
            "queso": base == "QUESO"}


def ajustar(muestras, tag_de, S):
    """El segundo nivel de las muestras agrupadas. muestras: DataFrame con Identificador, COLUMNAS y subtipo_ref (de
    mejoras_clustering.preparar), en el mismo orden que tag_de (NaN o None = sin tag: queda sin subtag).
    Devuelve (subtag, criterio, modelos); modelos tiene lo que necesita predecir y las tablas para describir."""
    X = muestras[COLUMNAS].to_numpy(dtype=float)
    tag_de = np.asarray(tag_de, dtype=object)
    sub, crit = tag_de.copy(), np.full(len(X), C_TAG, dtype=object)
    r = _ramas(X, tag_de)
    modelos = {}
    if r["DDL"].sum():
        modelos["DDL"] = _ddl_ajustar(X[r["DDL"]], muestras.loc[r["DDL"], "subtipo_ref"].to_numpy())
        s_, c_, _ = _ddl_predecir(modelos["DDL"], X[r["DDL"]])
        sub[r["DDL"]], crit[r["DDL"]] = s_, c_
    if r["polvo"].sum():
        idx = np.flatnonzero(r["polvo"])
        modelos["polvo"], es_out = _polvo_ajustar(X[idx], muestras["Identificador"].to_numpy()[idx], S)
        s_, c_ = _polvo_predecir(modelos["polvo"], X[idx], tag_de[idx])
        s_[es_out], c_[es_out] = tag_de[idx][es_out], C_POLVO_OUT
        sub[idx], crit[idx] = s_, c_
        _aplicar_reglas_polvo(X, r["polvo"], sub, crit)
        modelos["polvo"]["outliers"] = list(muestras["Identificador"].to_numpy()[idx][es_out])
    crit[r["liquido"]] = C_POLVO_LIQ
    if r["queso"].sum():
        sub[r["queso"]], crit[r["queso"]] = subtag_queso(X[r["queso"], 0]), C_QUESO
        modelos["queso"] = {"cortes_humedad": CORTES_QUESO}
    sin_tag = pd.isna(pd.Series(tag_de)).to_numpy()
    sub[sin_tag], crit[sin_tag] = np.nan, np.nan
    return sub, crit, modelos


def predecir(modelos, X, tag_de, outliers_polvo=()):
    """Subtag y criterio de muestras con tag (nuevas o no) a partir de los cuatro análisis y el tag. Una muestra nueva no
    pasa por el filtro de outliers de la rama del polvo; `outliers_polvo` (una máscara booleana, o vacío) marca las que
    ya se sabe que lo son (las del ajuste, para verificar), que quedan con su tag."""
    X = np.asarray(X, dtype=float)
    tag_de = np.asarray(tag_de, dtype=object)
    sub, crit = tag_de.copy(), np.full(len(X), C_TAG, dtype=object)
    r = _ramas(X, tag_de)
    if "DDL" in modelos and r["DDL"].any():
        s_, c_, _ = _ddl_predecir(modelos["DDL"], X[r["DDL"]])
        sub[r["DDL"]], crit[r["DDL"]] = s_, c_
    if "polvo" in modelos and r["polvo"].any():
        s_, c_ = _polvo_predecir(modelos["polvo"], X[r["polvo"]], tag_de[r["polvo"]])
        sub[r["polvo"]], crit[r["polvo"]] = s_, c_
    if len(outliers_polvo):
        out = np.asarray(outliers_polvo, dtype=bool) & r["polvo"]
        sub[out], crit[out] = tag_de[out], C_POLVO_OUT
    if "polvo" in modelos and r["polvo"].any():
        _aplicar_reglas_polvo(X, r["polvo"], sub, crit)
    crit[r["liquido"]] = C_POLVO_LIQ
    if r["queso"].any():
        sub[r["queso"]], crit[r["queso"]] = subtag_queso(X[r["queso"], 0]), C_QUESO
    return sub, crit


def probabilidades_ddl(modelos, X):
    """Las tres probabilidades de la logística del DDL (columnas p_familiar, p_heladero, p_repostero)."""
    return _ddl_predecir(modelos["DDL"], np.asarray(X, dtype=float))[2]


def para_guardar(modelos):
    """Lo que va al joblib: sin las tablas descriptivas."""
    out = {}
    for rama, m in modelos.items():
        out[rama] = {c: v for c, v in m.items() if c not in ("descripcion",)}
    return out
