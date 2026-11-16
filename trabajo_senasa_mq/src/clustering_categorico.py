"""Clustering con categorías (cat_*) y comparación con el k-medias numérico de la clase 41.

Lo usa comparacion_numerico_categorico.ipynb (lo arma build_comparacion_num_cat.py). No modifica nada de
resultados/tablas/: lee clusters_*.csv y categorias_*.csv y escribe en resultados/comparacion_num_cat/.

Dos particiones de las mismas muestras (las del k-medias actual: cuatro análisis completos, sin outliers):
  - numérica: k-medias sobre humedad, grasa, proteínas y cenizas escaladas (se reproduce con la misma receta de
    celdas_clase41.py y se verifica contra el tag de clusters_*.csv);
  - categórica: las mismas cuatro variables pasadas a categorías, en dos variantes:
      (a) «con familia»: cat_* de categorias_*.csv (cortes CAA o terciles DENTRO de cada familia: la categoría ya
          sabe de qué familia es la muestra → fuga si después se compara contra la familia);
      (b) «globales»: los mismos cortes para todas las muestras (los cortes «*» de cortes_categorias.csv, terciles
          globales de Senasa), sin información de familia.
Familia, tag y subtipo nunca entran a ningún algoritmo: solo se usan para evaluar.

`kmodes` no está instalado en el .venv y no se instala: el algoritmo categórico principal es el k-modas de Huang
escrito acá en numpy (unas líneas, sin dependencias nuevas), que además da el costo para el codo, con k fijo en
K_KMODAS. Como controles: AgglomerativeClustering(metric="hamming") (average y complete) y k-medias sobre one-hot
(sklearn). La semilla es SEMILLA (42) y el k-medias usa N_INIT_KMEDIAS (50).
"""
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import categorizar
import rutas as R
from datos_clase41 import COLUMNAS, N_INIT_KMEDIAS, SEMILLA, SIN_FAMILIA, etiquetar

SALIDA = R.RESULTADOS / "comparacion_num_cat"
VERSIONES = {"senasa": "Solo Senasa", "senasa_mq": "Senasa + MQ"}
# análisis de la tabla de categorías ↔ columna numérica del clustering (grasa se llama materia_grasa allá)
ANALISIS = {"humedad": "humedad", "materia_grasa": "grasa", "proteinas": "proteinas", "cenizas": "cenizas"}
CAT_A = [f"cat_{a}" for a in ANALISIS]            # variante (a): con familia
CAT_B = [f"catg_{a}" for a in ANALISIS]           # variante (b): globales
VARIANTES = {"a": ("con familia", CAT_A), "b": ("globales", CAT_B)}
NO_MEDIDO = "no medido"
SUBTIPOS_DDL = ["FAMILIAR", "REPOSTERO", "HELADERO"]
KS = list(range(2, 15))
# k del k-modas: FIJO en 5, en las dos versiones y las dos variantes (decisión de Matías, 2026-11-11). Es el mismo k que el
# k-medias numérico en las dos versiones y el codo del costo de solo Senasa. No sale de la silueta de Hamming: con 4 variables
# categóricas hay solo unas decenas de perfiles distintos (ver `perfiles`) y la silueta sube sola con k (cada cluster se
# vuelve una copia de un perfil); el k de mayor silueta cae en el borde del rango. Antes era 6; con la semilla 42 el
# numérico da 5 y el codo de Senasa + MQ (b) da 6, así que 6 ya no se sostiene. k_optimo() y el barrido quedan para
# mostrar la curva, y k = 6 queda en el cuaderno como sensibilidad.
K_KMODAS = 5
K_SENSIBILIDAD = 6     # el k fijo hasta el 2026-11-11: el cuaderno lo muestra al lado como sensibilidad
N_INIT = 1000          # reinicios del k-modas: el costo no es monótono en k con pocos reinicios; con 1000 todas las semillas
                       # llegan al mismo costo mínimo (≈1 s por corrida en la versión Senasa + MQ)
MIXTO_UMBRAL = 0.6     # si el tag numérico mayoritario cubre menos de esto: MIXTO_<tag>
TAM_MIN = 0.01          # un k con un grupo de menos del 1 % de las muestras no se elige como «óptimo»


# ================================================================ datos
def categorias_globales(tabla):
    """cat_* con los cortes «*» de cortes_categorias.csv para todas las muestras (sin familia). Usa
    categorizar.categorizar con una tabla de cortes que solo tiene la familia «*» y la familia borrada."""
    cortes = categorizar.leer_cortes()
    cortes = cortes[(cortes["familia"] == "*") & cortes["analisis"].isin(list(ANALISIS))]
    t = tabla[list(ANALISIS)].assign(familia="sin clasificar", subtipo=np.nan)
    cat = categorizar.categorizar(t, cortes=cortes)
    return cat[[f"cat_{a}" for a in ANALISIS]].rename(columns=lambda c: "catg_" + c[4:])


def subtipo_eval(t):
    """Subtipo para evaluar: DDL familiar / repostero / heladero, y la clase de humedad del CAA (art. 605) en el
    queso. El resto queda sin subtipo (NaN)."""
    s = pd.Series(np.nan, index=t.index, dtype=object)
    ddl = (t["familia"] == "dulce de leche") & t["subtipo"].isin(SUBTIPOS_DDL)
    s[ddl] = "DDL " + t.loc[ddl, "subtipo"].str.lower()
    q = (t["familia"] == "queso") & t["cat_humedad"].notna()
    s[q] = "queso " + t.loc[q, "cat_humedad"]
    return s


def cargar(version):
    """Las muestras del k-medias actual (con cluster en clusters_<version>.csv; sin la rama de queso), con los cuatro análisis, el tag y el
    cluster numérico, las cat_* de las dos variantes, la familia, el subtipo y el subtipo de evaluación."""
    cl = pd.read_csv(R.TABLAS / f"clusters_{version}.csv", encoding="utf-8-sig")
    cl = cl[cl["cluster"].notna()].rename(columns={"cluster": "cluster_num", "tag": "tag_num", "silueta": "silueta_num_csv"})
    cat = pd.read_csv(R.TABLAS / f"categorias_{version}.csv", encoding="utf-8-sig", low_memory=False,
                      usecols=["Identificador", "subtipo"] + CAT_A)
    t = cl.merge(cat, on="Identificador", how="left", validate="one_to_one").reset_index(drop=True)
    t = pd.concat([t, categorias_globales(t.rename(columns={"grasa": "materia_grasa"}))], axis=1)
    t["subtipo_eval"] = subtipo_eval(t)
    t["familia_conocida"] = t["familia"] != SIN_FAMILIA
    # Unas pocas muestras (yogur / bebida láctea con humedad y cenizas sin corte por estado, p. ej. 5 en Senasa y 7 en
    # Senasa + MQ) no tienen cat_* en la variante (a), que usa cortes por familia y estado. Se quedan en el conjunto (es el
    # del k-medias y hace falta completo para reproducirlo) con el nivel propio «sin corte» en esas columnas.
    t[CAT_A] = t[CAT_A].fillna("sin corte")
    assert t[CAT_B].notna().all().all(), "hay muestras del clustering sin categoría"
    return t


def cargar_ampliado(version, min_medidos=2):
    """El conjunto que puede usar el categórico: además de las del k-medias, las muestras incompletas (al menos
    `min_medidos` de los cuatro análisis categorizados), con «no medido» en lo que falta. Quedan fuera las excluidas
    por humedad + sólidos, las que suman > 102 con los cuatro análisis y los outliers del k-medias (para que el
    conjunto completo sea el mismo). Devuelve la tabla (variante a y b) y el conteo por análisis medidos."""
    cat = pd.read_csv(R.TABLAS / f"categorias_{version}.csv", encoding="utf-8-sig", low_memory=False,
                      usecols=["Identificador", "familia", "subtipo", "origen", "estado_cluster", "excluida", "tag"]
                      + list(ANALISIS) + CAT_A)
    medidos = cat[CAT_A].notna().sum(axis=1)
    no_agrupada = cat["estado_cluster"].str.startswith("no agrupada")
    base = (cat["estado_cluster"] == "con tag") | (no_agrupada & ~cat["excluida"] & medidos.between(1, 3))
    conteo = pd.crosstab(np.where(cat.loc[base, "estado_cluster"] == "con tag", "completa (k-medias)", "incompleta"),
                         medidos[base].rename("análisis medidos (de 4)"), margins=True, margins_name="total")
    t = cat[base & (medidos >= min_medidos)].reset_index(drop=True)
    t = pd.concat([t, categorias_globales(t)], axis=1)
    # catg_ con NaN donde el valor falta; categorizar ya devuelve NaN en esos casos
    t[CAT_A + CAT_B] = t[CAT_A + CAT_B].fillna(NO_MEDIDO)
    t["completa"] = t["estado_cluster"] == "con tag"
    t["subtipo_eval"] = subtipo_eval(t.assign(cat_humedad=t["cat_humedad"].replace(NO_MEDIDO, np.nan)))
    return t, conteo


# ================================================================ numérico
def espacio_numerico(t):
    return StandardScaler().fit_transform(t[COLUMNAS])


def reproducir_numerico(t):
    """k-medias de la clase 41 (escalado, n_init=N_INIT_KMEDIAS, semilla 42, k del CSV) y sus tags con etiquetar().
    Verifica que da el mismo tag que clusters_*.csv. Devuelve (Z, etiquetas, tag por muestra)."""
    Z = espacio_numerico(t)
    k = t["tag_num"].nunique()
    km = KMeans(n_clusters=k, n_init=N_INIT_KMEDIAS, random_state=SEMILLA).fit(Z)
    centros = t[COLUMNAS].groupby(km.labels_).mean().reset_index(drop=True)
    _, tags, _ = etiquetar(centros)
    tag = np.array([tags[g] for g in km.labels_])
    assert (tag == t["tag_num"].values).all(), "el k-medias reproducido no da el tag del CSV"
    return Z, km.labels_, tag


# ================================================================ categórico
def codificar(t, cols):
    """Cada columna categórica a enteros 0..m-1 (para Hamming y k-modas). Devuelve (X, niveles por columna)."""
    X = np.empty((len(t), len(cols)), dtype=np.int16)
    niveles = {}
    for j, c in enumerate(cols):
        codigos, uniques = pd.factorize(t[c], sort=True)
        X[:, j] = codigos; niveles[c] = list(uniques)
    return X, niveles


def kmodas(X, k, n_init=N_INIT, max_iter=100, semilla=SEMILLA):
    """k-modas de Huang (1998) con distancia de Hamming: modas iniciales = k perfiles distintos al azar; asignar a
    la moda más cercana; cada moda = la categoría más frecuente de su grupo, por columna; repetir. Se queda con la
    corrida de menor costo (suma de desacuerdos). Devuelve (etiquetas, modas, costo)."""
    rng = np.random.default_rng(semilla)
    perfiles = np.unique(X, axis=0)
    mejor = None
    for _ in range(n_init):
        modas = perfiles[rng.choice(len(perfiles), size=min(k, len(perfiles)), replace=False)].copy()
        et = None
        for _ in range(max_iter):
            d = (X[:, None, :] != modas[None, :, :]).sum(axis=2)
            nuevas = d.argmin(axis=1)
            if et is not None and (nuevas == et).all():
                break
            et = nuevas
            for j in range(len(modas)):
                sel = X[et == j]
                if len(sel):
                    modas[j] = [np.bincount(sel[:, c]).argmax() for c in range(X.shape[1])]
        costo = int(d[np.arange(len(X)), et].sum())
        if mejor is None or costo < mejor[2]:
            mejor = (et, modas, costo)
    et, modas, costo = mejor
    return renumerar(et), modas, costo


def estabilidad(X, k=None, semillas=(SEMILLA, 1, 2, 3, 4), n_init=N_INIT):
    """k-modas con varias semillas: costo, grupo más chico y ARI entre todos los pares de corridas. Devuelve (tabla, aris)."""
    from itertools import combinations
    k = K_KMODAS if k is None else k
    res = [kmodas(X, k, n_init=n_init, semilla=s) for s in semillas]
    aris = [adjusted_rand_score(a[0], b[0]) for a, b in combinations(res, 2)]
    tabla = pd.DataFrame({"semilla": list(semillas), "costo": [r[2] for r in res],
                          "grupo mínimo": [int(np.bincount(r[0]).min()) for r in res]})
    return tabla, np.array(aris)


def renumerar(et):
    """Clusters numerados por tamaño (0 = el más grande), para que la numeración sea estable."""
    orden = pd.Series(et).value_counts().index
    mapa = {g: i for i, g in enumerate(orden)}
    return np.array([mapa[g] for g in et])


def agrupar(X, k, metodo, onehot=None):
    """metodo: «aglomerativo-average», «aglomerativo-complete» (Hamming, sklearn), «kmodas» (numpy, acá) o
    «kmedias-onehot» (sklearn, referencia)."""
    if metodo.startswith("aglomerativo"):
        et = AgglomerativeClustering(n_clusters=k, metric="hamming", linkage=metodo.split("-")[1]).fit_predict(X)
    elif metodo == "kmodas":
        et = kmodas(X, k)[0]
    elif metodo == "kmedias-onehot":
        et = KMeans(n_clusters=k, n_init=N_INIT_KMEDIAS, random_state=SEMILLA).fit_predict(onehot)
    else:
        raise ValueError(metodo)
    return renumerar(et)


def una_caliente(X):
    return OneHotEncoder(sparse_output=False).fit_transform(X)


METODOS = ["aglomerativo-average", "aglomerativo-complete", "kmodas", "kmedias-onehot"]


def barrido(X, metodos=METODOS, ks=KS):
    """Para cada método y k: silueta con Hamming, tamaño del grupo más chico y (k-modas) el costo."""
    H = una_caliente(X)
    filas, particiones = [], {}
    for m in metodos:
        for k in ks:
            if m == "kmodas":
                et, _, costo = kmodas(X, k)
            else:
                et, costo = agrupar(X, k, m, H), np.nan
            particiones[(m, k)] = et
            n = np.bincount(et)
            filas.append({"metodo": m, "k": k, "silueta_hamming": silhouette_score(X, et, metric="hamming"),
                          "grupo_min": int(n.min()), "grupo_min_%": 100 * n.min() / len(et), "costo_kmodas": costo})
    return pd.DataFrame(filas), particiones


def k_optimo(tabla, metodo):
    """El k de mayor silueta (Hamming) entre los que no dejan un grupo de menos del TAM_MIN de las muestras. YA NO elige el
    k del k-modas (es K_KMODAS = 5): solo muestra dónde cae el máximo de la silueta, que suele ser el borde del rango."""
    t = tabla[(tabla["metodo"] == metodo) & (tabla["grupo_min_%"] >= 100 * TAM_MIN)]
    return int(t.loc[t["silueta_hamming"].idxmax(), "k"])


# ================================================================ comparación
def silueta_macro(X, et, metric):
    """Promedio de las siluetas medias de cada grupo (cada grupo pesa lo mismo, sea grande o chico)."""
    s = silhouette_samples(X, et, metric=metric)
    return float(pd.Series(s).groupby(et).mean().mean())


def contra(et, ref, sel=None):
    """(ARI, NMI) de una partición contra una referencia, solo donde la referencia está definida (y sel)."""
    ref = pd.Series(ref)
    ok = ref.notna().values & (np.ones(len(ref), bool) if sel is None else np.asarray(sel))
    if ok.sum() < 2 or ref[ok].nunique() < 2:
        return np.nan, np.nan
    return adjusted_rand_score(ref[ok], et[ok]), normalized_mutual_info_score(ref[ok], et[ok])


def metricas(t, et, Z, X, propio):
    """Una fila de la tabla comparativa. propio: «euclidea» (espacio numérico Z) o «hamming» (X categórico)."""
    ari_f, nmi_f = contra(et, t["familia"].where(t["familia_conocida"]))
    ari_s, nmi_s = contra(et, t["subtipo_eval"])
    ddl = (t["familia"] == "dulce de leche").values
    ari_d, nmi_d = contra(et, t["subtipo_eval"].where(t["subtipo_eval"].str.startswith("DDL", na=False)), ddl)
    e_prop, x_prop = (Z, "euclidean") if propio == "euclidea" else (X, "hamming")
    e_otro, x_otro = (X, "hamming") if propio == "euclidea" else (Z, "euclidean")
    return {"k": len(np.unique(et)),
            "silueta propia": silhouette_score(e_prop, et, metric=x_prop),
            "silueta macro propia": silueta_macro(e_prop, et, x_prop),
            "silueta en el otro espacio": silhouette_score(e_otro, et, metric=x_otro),
            "silueta macro en el otro espacio": silueta_macro(e_otro, et, x_otro),
            "ARI familia": ari_f, "NMI familia": nmi_f, "ARI subtipo": ari_s, "NMI subtipo": nmi_s,
            "ARI subtipo DDL": ari_d, "NMI subtipo DDL": nmi_d}


def perfil(t, et, cols):
    """Por cluster categórico: muestras, categoría modal (y su %) por análisis, familia mayoritaria (entre las de
    familia conocida), tag numérico mayoritario, centroide en g/100 g y un tag con la convención de etiquetar()
    (si el tag numérico mayoritario cubre menos de MIXTO_UMBRAL del grupo: MIXTO_<tag>). Hasta 2026-11-10 había además
    un tag POLVO para los grupos que mezclaban solo polvos; no apareció en ninguna de las 52 particiones y Matías lo
    quitó. Devuelve (tabla, tags)."""
    centros = t[COLUMNAS].groupby(et).mean()
    _, tags, _ = etiquetar(centros.reset_index(drop=True))
    filas = []
    for i, g in enumerate(centros.index):
        sel = et == g
        f = {"cluster": g, "tag": tags[i], "muestras": int(sel.sum())}
        for c in cols:
            vc = t.loc[sel, c].value_counts(normalize=True)
            f[c.split("_", 1)[1]] = f"{vc.index[0]} ({100 * vc.iloc[0]:.0f} %)"
        fam = t.loc[sel & t["familia_conocida"].values, "familia"].value_counts(normalize=True)
        f["familia mayoritaria"] = f"{fam.index[0]} ({100 * fam.iloc[0]:.0f} %)" if len(fam) else "-"
        f["sin familia"] = int((sel & ~t["familia_conocida"].values).sum())
        tn = t.loc[sel, "tag_num"].value_counts(normalize=True)
        f["tag numérico mayoritario"] = f"{tn.index[0]} ({100 * tn.iloc[0]:.0f} %)"
        if tn.iloc[0] < MIXTO_UMBRAL:             # grupo que mezcla productos: el centroide no es un producto
            f["tag"] = tags[i] = f"MIXTO_{tags[i]}"
        for c in COLUMNAS:
            f[f"{c} (g/100 g)"] = round(float(centros.loc[g, c]), 2)
        filas.append(f)
    return pd.DataFrame(filas).set_index("cluster"), tags


def guardar(t, columnas, version):
    """CSV en resultados/comparacion_num_cat/: Identificador, familia, subtipo, tag numérico y, por cada partición
    categórica, cluster y tag."""
    SALIDA.mkdir(parents=True, exist_ok=True)
    ruta = SALIDA / f"comparacion_{version}.csv"
    t[["Identificador", "familia", "subtipo", "origen", "tag_num"] + columnas].to_csv(ruta, index=False, encoding="utf-8-sig")
    return ruta


def codo(ks, costos):
    """k del codo: el punto de la curva de costo más alejado de la recta entre el primero y el último (kneedle)."""
    x = np.asarray(ks, float); y = np.asarray(costos, float)
    x_ = (x - x[0]) / (x[-1] - x[0]); y_ = (y - y[-1]) / (y[0] - y[-1])
    return int(x[np.argmax(np.abs(1 - x_ - y_))])


PRINCIPAL = "kmodas"


def analizar(version):
    """Todo lo de una versión: muestras, k-medias numérico reproducido y, por variante de categorías, el barrido de
    k (cuatro métodos), el k comparable (el del numérico), el k del k-modas (K_KMODAS = 5, fijo: `k_opt`) y sus
    particiones. `k_sil` es el k de mayor silueta de Hamming (solo informativo) y `k_codo` el codo del costo."""
    t = cargar(version)
    Z, et_num, _ = reproducir_numerico(t)
    out = {"t": t, "Z": Z, "et_num": et_num, "K": int(t["tag_num"].nunique()), "var": {}}
    for clave, (nombre, cols) in VARIANTES.items():
        X, niveles = codificar(t, cols)
        tabla, part = barrido(X)
        k_sil = k_optimo(tabla, PRINCIPAL)
        cos = tabla[tabla["metodo"] == PRINCIPAL].set_index("k")["costo_kmodas"]
        out["var"][clave] = {"nombre": nombre, "cols": cols, "X": X, "niveles": niveles, "barrido": tabla,
                             "part": part, "k_opt": K_KMODAS, "k_sil": k_sil, "k_codo": codo(cos.index, cos.values),
                             "perfiles": int(len(np.unique(X, axis=0)))}
    return out
