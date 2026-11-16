"""Prueba: ¿un agrupamiento sin rótulos con solo humedad y grasa separa las familias? (todas las muestras, una sola rama)

Lo usa prueba_clustering_todas_hg.ipynb (lo arma build_prueba_todas_hg.py). No modifica ningún archivo de entrada ni de
resultados existentes; escribe solo lo que el notebook le pida en resultados/prueba_todas_hg/.

    poblacion(S, con_mq)          todas las muestras (sin FAMILIAS_NO_LACTEAS, sin las excluidas por humedad + sólidos) con su
                                  estado: entra (tiene humedad, o 100 − sólidos, y grasa) o por qué queda afuera; y el flujo
    outliers(X)                   consenso LOF + bosque + elipse (mejoras_clustering), con k-medias preliminar como grupos
    barrido_kmedias / barrido_gmm k por silueta y codo / k por BIC; guardan las etiquetas de cada k
    evaluar(...), por_cluster(...) ARI, AMI, pureza y cobertura contra desc_familia (que NO se usa para ajustar), siluetas
    estabilidad(...)              ARI entre semillas y por bootstrap
    techo_supervisado(...)        cota de lo que H + G (o los cuatro) podrían separar SI se usaran los rótulos (validación cruzada)
    solapamiento(...)             cuánto se pisan dos familias en la partición (suma de mínimos de sus reparticiones)
    grafico_lado_a_lado(...)      humedad × grasa coloreado por cluster y por familia (paleta categórica del skill dataviz)
"""
import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (adjusted_mutual_info_score, adjusted_rand_score, balanced_accuracy_score, silhouette_samples)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

import clustering_quesos as Q
import mejoras_clustering as M
from datos_clase41 import COLUMNAS, FAMILIAS_NO_LACTEAS, N_INIT_KMEDIAS, SEMILLA, SIN_FAMILIA, SUMA_MAX, leer_mq, unir

CORTE_AGUA = 99.0                         # humedad > 99: la grasa en base seca (÷ (100 − humedad)) se dispara
SIN_ROTULO = ("sin clasificar", SIN_FAMILIA)
MUESTRA_SIL = 10000                       # la silueta se calcula sobre 10 000 muestras al azar (fijas), no sobre todas
VARIABLES = {"H + G": ["humedad", "grasa"], "base seca": ["humedad", "mges"]}
CORTA = {"dulce de leche": "DDL", "queso": "queso", "leche fluida": "leche fluida", "leche en polvo": "leche en polvo",
         "suero / permeado": "suero", "helado": "helado", "concentrado proteico": "conc. proteico",
         "manteca / crema": "manteca / crema", "yogur / bebida lactea": "yogur", "arroz con leche / postre": "postre",
         "sin clasificar": "sin clasificar", SIN_FAMILIA: "sin familia"}
# color por familia, fijo (la entidad conserva el color en todos los gráficos); las dos más chicas, gris
COLOR_FAMILIA = {"DDL": "#2a78d6", "queso": "#eb6834", "leche fluida": "#1baf7a", "leche en polvo": "#eda100",
                 "suero": "#e87ba4", "helado": "#008300", "conc. proteico": "#6250d6", "manteca / crema": "#e34948",
                 "yogur": "#898781", "postre": "#898781", "sin clasificar": "#c3c2b7", "sin familia": "#c3c2b7"}
MARCA_FAMILIA = {"yogur": "v", "postre": "s"}
SERIE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MARCAS = ["o", "s", "^", "D", "P"]


# ------------------------------------------------------------------ población
def poblacion(S, con_mq):
    """Devuelve (candidatas, flujo). Misma lectura que el notebook único: Senasa manda, MQ completa huecos y la parte MQ sale de
    MQ encolumnado (sin las MO); se sacan FAMILIAS_NO_LACTEAS, las muestras con humedad + sólidos fuera de 100 ± 1 y, como en
    el notebook único, el queso solo MQ que el rótulo no confirma. La humedad de Senasa sin humedad medida es 100 − sólidos.
    candidatas: una fila por muestra con `estado` ('entra' o el motivo) y mges; flujo: cuántas quedan en cada paso."""
    sen = S.crudo[["Identificador", "desc_familia", "desc_producto", "desc_tipo_muestra"] + COLUMNAS].copy()
    de_solidos = sen["humedad"].isna() & S.crudo["solidos_senasa"].notna()
    sen["humedad"] = sen["humedad"].fillna(100 - S.crudo["solidos_senasa"])
    antes = sen.set_index("Identificador")[["humedad", "grasa"]].notna().all(axis=1)
    if con_mq:
        _, mq = leer_mq()
        todas, _ = unir(sen, mq)
    else:
        todas = sen.assign(origen="Senasa")
    en_senasa = todas["Identificador"].isin(sen["Identificador"])
    ahora = todas[["humedad", "grasa"]].notna().all(axis=1)
    todas["origen"] = np.where(~en_senasa, "solo MQ", np.where(~todas["Identificador"].map(antes).fillna(False).astype(bool) & ahora,
                                                              "Senasa completada con MQ", "Senasa"))
    flujo = {"muestras leídas": len(todas), "humedad de Senasa tomada de 100 − sólidos": int(de_solidos.sum())}
    no_lact = todas["desc_familia"].isin(FAMILIAS_NO_LACTEAS)
    flujo["fuera: familia no láctea"] = int(no_lact.sum())
    todas = todas[~no_lact]
    exc = todas["Identificador"].isin(S.excluidas)
    flujo["fuera: humedad + sólidos fuera de 100 ± 1"] = int(exc.sum())
    todas = todas[~exc]
    if con_mq:
        _, confirmados, _, _ = Q.rama_queso(S, True)
        sin_conf = (todas["origen"] == "solo MQ") & (todas["desc_familia"] == "queso") & ~todas["Identificador"].isin(confirmados)
        flujo["fuera: queso solo MQ sin variedad en el rótulo (familia sin confirmar)"] = int(sin_conf.sum())
        flujo["  de ellas, con humedad y grasa"] = int(todas.loc[sin_conf, ["humedad", "grasa"]].notna().all(axis=1).sum())
        todas = todas[~sin_conf]
    todas = todas.reset_index(drop=True)
    h, g = todas["humedad"], todas["grasa"]
    imposible = h.notna() & g.notna() & ((h < 0) | (h > CORTE_AGUA) | (g < 0) | (h + g > SUMA_MAX))
    todas["estado"] = np.select([h.isna() & g.isna(), h.isna(), g.isna(), imposible],
                                ["sin humedad ni grasa", "sin humedad", "sin grasa", "valores imposibles"], "entra")
    flujo["candidatas (lácteas, sin las excluidas)"] = len(todas)
    flujo["entran (humedad y grasa, valores posibles)"] = int((todas["estado"] == "entra").sum())
    todas["mges"] = 100 * todas["grasa"] / (100 - todas["humedad"])
    s4 = todas[COLUMNAS].notna().all(axis=1) & (todas[COLUMNAS].sum(axis=1) <= SUMA_MAX)
    todas["tiene4"] = s4 & (todas["estado"] == "entra")
    todas["familia"] = todas["desc_familia"].fillna(SIN_FAMILIA).map(lambda f: CORTA.get(f, f))
    todas["fam_eval"] = todas["familia"].where(~todas["desc_familia"].fillna(SIN_FAMILIA).isin(SIN_ROTULO))
    return todas, pd.Series(flujo, name="muestras")


def tabla_por_familia(cand):
    """Una fila por familia: muestras, cuántas entran y por qué quedan afuera las otras."""
    t = pd.crosstab(cand["familia"], cand["estado"])
    for c in ["entra", "sin humedad ni grasa", "sin humedad", "sin grasa", "valores imposibles"]:
        if c not in t:
            t[c] = 0
    t = t[["entra", "sin humedad ni grasa", "sin humedad", "sin grasa", "valores imposibles"]]
    t.insert(0, "muestras", t.sum(axis=1))
    t["% entra"] = (100 * t["entra"] / t["muestras"]).round(1)
    t["con los 4 análisis"] = cand[cand["tiene4"]].groupby("familia").size().reindex(t.index).fillna(0).astype(int)
    t = t.sort_values("muestras", ascending=False)
    t.loc["total"] = t.sum(numeric_only=True)
    t.loc["total", "% entra"] = round(100 * t.loc["total", "entra"] / t.loc["total", "muestras"], 1)
    return t


# ------------------------------------------------------------------ outliers
def estandarizar(X):
    esc = StandardScaler().fit(X)
    return esc.transform(X), esc


def outliers(X, pct=3):
    """Mismo criterio que el notebook único: k-medias preliminar (k por silueta, 2 a 10) como grupos y, dentro de cada grupo,
    factor local + bosque de aislamiento + elipse, cada uno marca el pct % más raro; outlier si lo marcan al menos 2 de 3.
    Sirve con 2 variables (LOF con 50 vecinos, bosque y elipse no dependen del número de columnas)."""
    Z, _ = estandarizar(X)
    out_iqr, grupos, k_pre = M.outliers_iqr(Z)
    es, votos = M.outliers_consenso(X, grupos, pct=pct, minimo=2, n_vecinos=50)
    return es, votos, grupos, k_pre


def indice_silueta(n, tam=MUESTRA_SIL, semilla=SEMILLA):
    return np.sort(np.random.RandomState(semilla).choice(n, min(n, tam), replace=False))


def silueta(Z, et, idx):
    """(global, macro, por cluster) sobre la submuestra idx; el ruido (-1) queda afuera."""
    e = np.asarray(et)[idx]; z = Z[idx]; ok = e != -1
    if len(np.unique(e[ok])) < 2:
        return np.nan, np.nan, pd.Series(dtype=float)
    s = silhouette_samples(z[ok], e[ok])
    por = pd.Series(s).groupby(e[ok]).mean()
    return float(s.mean()), float(por.mean()), por


# ------------------------------------------------------------------ algoritmos
def barrido_kmedias(Z, ks=range(2, 13), semilla=SEMILLA):
    """k-medias para cada k (n_init N_INIT_KMEDIAS). Devuelve dict con etiquetas, inercia, silueta (submuestra), k por silueta y k por codo."""
    idx = indice_silueta(len(Z))
    mods = {k: KMeans(n_clusters=k, n_init=N_INIT_KMEDIAS, random_state=semilla).fit(Z) for k in ks}
    sil = {k: silueta(Z, m.labels_, idx)[0] for k, m in mods.items()}
    ine = {k: m.inertia_ for k, m in mods.items()}
    k_sil = max(sil, key=sil.get)
    return dict(modelos=mods, silueta=sil, inercia=ine, k_sil=k_sil, k_codo=codo(ine), idx=idx)


def codo(inercia):
    """k del codo: el de mayor distancia a la recta que une el primer y el último punto de la curva de inercia (normalizada)."""
    ks = np.array(list(inercia)); y = np.array(list(inercia.values()), dtype=float)
    x = (ks - ks[0]) / (ks[-1] - ks[0]); y = (y - y[-1]) / (y[0] - y[-1])
    d = np.abs(x + y - 1) / np.sqrt(2)
    return int(ks[int(np.argmax(d))])


def barrido_gmm(Z, ks=range(1, 13), semilla=SEMILLA):
    """Mezcla de gaussianas (covarianza completa, n_init 3) para cada k; k por BIC mínimo."""
    mods = {k: GaussianMixture(n_components=k, n_init=3, random_state=semilla).fit(Z) for k in ks}
    bic = {k: m.bic(Z) for k, m in mods.items()}
    return dict(modelos=mods, bic=bic, k_bic=min(bic, key=bic.get))


def barrido_hdbscan(Z, tamanos=(50, 100, 200, 400)):
    hd = {m: HDBSCAN(min_cluster_size=m, copy=True).fit_predict(Z) for m in tamanos if m < len(Z)}
    m, et = Q.elegir_hdbscan(hd)
    return dict(etiquetas=hd, m=m, elegida=et)


def ordenar(et, humedad):
    """Renumera los clusters por humedad media creciente (el ruido queda -1)."""
    return Q.ordenar(et, humedad)


# ------------------------------------------------------------------ evaluación contra la familia
def balanceado(e, f, tope=100, n_sorteos=20, semilla=SEMILLA):
    """ARI y AMI en muestras balanceadas: hasta `tope` muestras por familia (todas si tiene menos), promedio de n_sorteos
    sorteos. El ARI común lo domina el par más grande (DDL y queso son ~85 % de las muestras); este mide cuánto separa
    cada familia por igual."""
    rng = np.random.RandomState(semilla); f = np.asarray(f, dtype=object); e = np.asarray(e)
    por = [np.flatnonzero(f == x) for x in pd.unique(f)]
    ari, ami = [], []
    for _ in range(n_sorteos):
        ix = np.concatenate([rng.choice(i, min(len(i), tope), replace=False) for i in por])
        ari.append(adjusted_rand_score(f[ix], e[ix])); ami.append(adjusted_mutual_info_score(f[ix], e[ix]))
    return float(np.mean(ari)), float(np.mean(ami))


def evaluar(Z, et, familia, idx):
    """ARI, AMI, pureza (global y macro), cobertura macro, silueta global y macro. ARI y AMI también balanceados por familia. La familia solo se usa acá; las muestras sin
    rótulo (sin clasificar / sin familia) no cuentan en las métricas contra la familia. El ruido de HDBSCAN cuenta como un grupo."""
    e = np.asarray(et); f = pd.Series(np.asarray(familia, dtype=object)); ok = f.notna().to_numpy()
    ct = pd.crosstab(e[ok], f[ok].to_numpy())
    sg, sm, _ = silueta(Z, e, idx)
    bal = balanceado(e[ok], f[ok].to_numpy())
    return {"grupos": int(len(set(e)) - (1 if -1 in e else 0)), "ruido": int((e == -1).sum()),
            "ARI": adjusted_rand_score(f[ok], e[ok]), "AMI": adjusted_mutual_info_score(f[ok], e[ok]),
            "ARI balanceado": bal[0], "AMI balanceado": bal[1],
            "pureza global": float(ct.max(axis=1).sum() / ct.values.sum()),
            "pureza macro": float((ct.max(axis=1) / ct.sum(axis=1)).mean()),
            "cobertura macro": float((ct.max(axis=0) / ct.sum(axis=0)).mean()),
            "silueta": sg, "silueta macro": sm}


def por_cluster(df, et, familia, Z, idx):
    """Una fila por cluster: muestras, humedad y grasa medianas, familia mayoritaria con su pureza (% del cluster) y su
    cobertura (% de esa familia que está en este cluster), segunda familia y silueta media."""
    e = np.asarray(et); f = pd.Series(np.asarray(familia, dtype=object))
    ct = pd.crosstab(e[f.notna().to_numpy()], f.dropna().to_numpy())
    tot_f = ct.sum(axis=0)
    _, _, sil = silueta(Z, e, idx)
    filas = {}
    for c in sorted(set(e)):
        sel = e == c
        fila = {"muestras": int(sel.sum()), "humedad": df.loc[sel, "humedad"].median(), "grasa": df.loc[sel, "grasa"].median()}
        if c in ct.index and ct.loc[c].sum() > 0:
            r = ct.loc[c].sort_values(ascending=False)
            fila.update({"familia mayoritaria": r.index[0], "pureza %": 100 * r.iloc[0] / r.sum(),
                         "cobertura %": 100 * r.iloc[0] / tot_f[r.index[0]],
                         "segunda familia": f"{r.index[1]} ({100 * r.iloc[1] / r.sum():.0f} %)" if len(r) > 1 and r.iloc[1] > 0 else "-"})
        fila["silueta"] = sil.get(c, np.nan)
        filas["ruido" if c == -1 else f"C{c}"] = fila
    return pd.DataFrame(filas).T


def cruzada(et, familia):
    """Cluster × familia en % de cada cluster (filas suman 100) y n por cluster."""
    e = np.asarray(et); f = pd.Series(np.asarray(familia, dtype=object)); ok = f.notna().to_numpy()
    ct = pd.crosstab(e[ok], f[ok].to_numpy())
    t = (100 * ct.div(ct.sum(axis=1), axis=0)).round(0)
    t.index = ["ruido" if c == -1 else f"C{c}" for c in t.index]
    t.insert(0, "n", ct.sum(axis=1).to_numpy())
    return t


def reparto_familia(et, familia):
    """Familia × cluster en % de cada familia (cómo se reparte cada familia entre los clusters)."""
    e = np.asarray(et); f = pd.Series(np.asarray(familia, dtype=object)); ok = f.notna().to_numpy()
    ct = pd.crosstab(f[ok].to_numpy(), e[ok])
    ct.columns = ["ruido" if c == -1 else f"C{c}" for c in ct.columns]
    return ct.div(ct.sum(axis=1), axis=0) * 100


def solapamiento(et, familia, minimo=30):
    """Matriz familia × familia: suma sobre clusters del mínimo de las dos reparticiones (0 = nunca comparten cluster,
    100 = reparten sus muestras igual). Solo familias con al menos `minimo` muestras."""
    r = reparto_familia(et, familia)
    n = pd.Series(np.asarray(familia, dtype=object)).dropna().value_counts()
    r = r.loc[[f for f in r.index if n[f] >= minimo]]
    m = pd.DataFrame({b: {a: np.minimum(r.loc[a], r.loc[b]).sum() for a in r.index} for b in r.index})
    return m


# ------------------------------------------------------------------ estabilidad
def estabilidad(Z, k, tipo, ref, semillas=range(10), n_boot=20, semilla=SEMILLA):
    """ARI entre corridas con distintas semillas (una inicialización cada una) y por bootstrap (se ajusta con una muestra
    con reposición y se asignan todas las muestras; ARI contra las etiquetas de referencia `ref`)."""
    def ajustar(X, s, n_init):
        return (KMeans(n_clusters=k, n_init=n_init, random_state=s) if tipo == "km"
                else GaussianMixture(n_components=k, n_init=n_init, random_state=s)).fit(X)
    et = [ajustar(Z, s, 1).predict(Z) for s in semillas]
    pares = [adjusted_rand_score(et[i], et[j]) for i in range(len(et)) for j in range(i + 1, len(et))]
    rng = np.random.RandomState(semilla)
    boot = []
    for b in range(n_boot):
        ix = rng.choice(len(Z), len(Z), replace=True)
        boot.append(adjusted_rand_score(ref, ajustar(Z[ix], b, 3).predict(Z)))
    return {"semillas: ARI medio": float(np.mean(pares)), "semillas: ARI mínimo": float(np.min(pares)),
            "bootstrap: ARI medio": float(np.mean(boot)), "bootstrap: ARI mínimo": float(np.min(boot)),
            "bootstrap: desvío": float(np.std(boot))}


# ------------------------------------------------------------------ techo supervisado
def techo_supervisado(X, familia, cv=5, semilla=SEMILLA):
    """Cuánto podrían separar las familias estas variables si SE USARAN los rótulos: bosque aleatorio (200 árboles, probado contra k vecinos y da lo
    mismo; el bosque de gradiente multiclase divergía) con validación cruzada estratificada. Es una cota superior optimista (muestras casi repetidas entre pliegues, mismo cliente).
    Devuelve (exactitud, exactitud balanceada, recall por familia, matriz de confusión en % de cada familia)."""
    f = pd.Series(np.asarray(familia, dtype=object)); ok = f.notna().to_numpy()
    y = f[ok].to_numpy(); Xo = np.asarray(X, dtype=float)[ok]
    clf = RandomForestClassifier(200, min_samples_leaf=2, random_state=semilla, n_jobs=-1)
    p = cross_val_predict(clf, Xo, y, cv=StratifiedKFold(cv, shuffle=True, random_state=semilla))
    conf = pd.crosstab(pd.Series(y, name="familia"), pd.Series(p, name="predicha"))
    conf = conf.reindex(columns=conf.index, fill_value=0)
    conf_pct = 100 * conf.div(conf.sum(axis=1), axis=0)
    rec = pd.Series(np.diag(conf_pct), index=conf_pct.index)
    return float((y == p).mean()), float(balanced_accuracy_score(y, p)), rec, conf_pct


def separabilidad_pares(X_por_variante, familia, minimo=30, cv=5, semilla=SEMILLA):
    """Para cada par de familias con al menos `minimo` muestras: exactitud balanceada (validación cruzada, con rótulos) de
    distinguir una de la otra con cada conjunto de variables. 0,5 = no se distinguen; 1 = se distinguen del todo."""
    f = pd.Series(np.asarray(familia, dtype=object))
    n = f.value_counts(); fams = [x for x in n.index if n[x] >= minimo]
    filas = []
    for i, a in enumerate(fams):
        for b in fams[i + 1:]:
            sel = f.isin([a, b]).to_numpy(); y = f[sel].to_numpy()
            fila = {"par": f"{a} vs {b}", "n": int(sel.sum())}
            for nombre, X in X_por_variante.items():
                clf = RandomForestClassifier(100, min_samples_leaf=2, random_state=semilla, n_jobs=-1)
                p = cross_val_predict(clf, np.asarray(X, dtype=float)[sel], y, cv=StratifiedKFold(cv, shuffle=True, random_state=semilla))
                fila[nombre] = balanced_accuracy_score(y, p)
            filas.append(fila)
    return pd.DataFrame(filas).set_index("par")


# ------------------------------------------------------------------ gráficos
def grafico_lado_a_lado(df, et, titulo, ax_par, ruido_gris=True):
    """df con humedad, grasa (g/100 g), familia; et: etiquetas. ax_par: (ax_cluster, ax_familia)."""
    a, b = ax_par; e = np.asarray(et)
    for c in sorted(set(e)):
        sel = e == c
        col = "#c3c2b7" if c == -1 else SERIE[c % 8]; mk = "x" if c == -1 else MARCAS[(c // 8) % len(MARCAS)]
        a.scatter(df.loc[sel, "humedad"], df.loc[sel, "grasa"], s=7, alpha=0.4, color=col, marker=mk, linewidths=0.6,
                  label=f"{'ruido' if c == -1 else 'C' + str(c)} ({sel.sum()})")
    for f in df["familia"].value_counts().index:
        sel = (df["familia"] == f).to_numpy()
        b.scatter(df.loc[sel, "humedad"], df.loc[sel, "grasa"], s=7, alpha=0.4, color=COLOR_FAMILIA[f],
                  marker=MARCA_FAMILIA.get(f, "o"), linewidths=0.6, label=f"{f} ({sel.sum()})")
    for ax, t in ((a, "agrupamiento (sin rótulos)"), (b, "familia del catálogo")):
        ax.set_xlabel("humedad (g/100 g)"); ax.set_ylabel("materia grasa (g/100 g)"); ax.set_title(t, fontsize=10, loc="left")
        ax.grid(alpha=0.25); ax.spines[["top", "right"]].set_visible(False)
        leg = ax.legend(markerscale=2.5, fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
        for h in leg.legend_handles:
            h.set_alpha(1)
