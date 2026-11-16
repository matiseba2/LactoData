"""Regresión logística sobre la calidad de la leche cruda: ¿cuál es la probabilidad de que una muestra supere un umbral
de células somáticas (RCS) o de recuento bacteriano total (RBT)?

Lo usa logistica_calidad_leche_cruda.ipynb (armado por build_logistica_calidad_leche.py). Reúne:
  · la carga y el tratamiento del RBT (una industria cambió de unidad y de tope de medición en mayo de 2024: se descarta);
  · la temperatura de Rafaela, su climatología y la anomalía respecto de ella (para separar «calor» de «estación»);
  · los modelos anidados (Pipeline: escalado + LogisticRegression) y las métricas (AUC, log-loss, Brier, calibración);
  · el bootstrap por bloques de semanas, hecho con pesos y arranque en caliente para que sea rápido.
Las rutas salen de rutas.py; ningún nombre de cliente ni de archivo se usa (las industrias ya vienen como IND-nn).
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, confusion_matrix, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import rutas as R

SEMILLA = 42   # la cátedra pide semilla 42
INICIO = pd.Timestamp("2019-01-01")        # antes hay <60 mediciones sueltas de una sola industria
CORTE_PRUEBA = pd.Timestamp("2024-01-01")  # entrenar hasta fin de 2023, probar 2024-2026
CORTE_VALID = pd.Timestamp("2023-01-01")   # para elegir N y C: entrenar hasta 2022, validar en 2023 (sin mirar la prueba)
UMBRAL_RCS = 400.0                         # miles de cél/mL (= 400 000 cél/mL)
UMBRAL_RBT = 100.0                         # miles de UFC/mL (= 100 000 UFC/mL)
CAMBIO_UNIDAD_RBT = {"cliente": "IND-59", "desde": pd.Timestamp("2024-05-01"), "minimo_en_ufc": 5000.0,
                     "descartar_desde": pd.Timestamp("2024-06-01")}
COMPOSICION = ["lactosa", "grasa", "proteina"]
OBJETIVOS = {"RCS": "y_rcs", "RBT": "y_rbt"}


# ----------------------------------------------------------------------------------------------------- carga
def cargar_leche():
    """Lee leche_cruda.csv (sin la columna `archivo`), corrige unidades y arma los dos objetivos.

    Devuelve (DataFrame, dict con los conteos de las correcciones). Los objetivos valen NaN si la medición falta.
    """
    cols = ["cliente", "fecha", "fecha_origen", "formato", "celulas_somaticas", "rbt", "lactosa", "grasa", "proteina"]
    d = pd.read_csv(R.LECHE_CRUDA, encoding="utf-8-sig", low_memory=False, parse_dates=["fecha"], usecols=cols)
    info = {"filas_archivo": len(d)}
    d = d[d["fecha"] >= INICIO].copy()
    info["filas_desde_2019"] = len(d)

    # RBT: la industria CAMBIO_UNIDAD_RBT empezó a informar en UFC/mL (no en miles) desde mayo de 2024, por informe, y con un
    # tope de 100 000 UFC/mL: casi no hay valores sobre 100 (en miles) y no se puede medir si pasan el umbral. Ningún otro valor
    # del archivo supera 5000, así que en esa industria y fecha un valor >= 5000 es del régimen nuevo.
    c = CAMBIO_UNIDAD_RBT
    en_ufc = (d["cliente"] == c["cliente"]) & (d["fecha"] >= c["desde"]) & (d["rbt"] >= c["minimo_en_ufc"])
    tras = (d["cliente"] == c["cliente"]) & (d["fecha"] >= c["descartar_desde"])
    info["rbt_en_ufc_descartados"] = int(en_ufc.sum())
    info["rbt_ind59_desde_junio_2024_descartados"] = int((tras & ~en_ufc & d["rbt"].notna()).sum())
    d["rbt_equiv_miles"] = d["rbt"].where(~en_ufc, d["rbt"] / 1000.0)    # solo para diagnóstico: el mismo valor en miles
    d["rbt_corr"] = d["rbt"].where(~(en_ufc | tras))                      # RBT utilizable (miles UFC/mL)
    # Esa industria dejó de informar RCS en junio de 2024; quedan 6 valores redondos (2000, 3000, 4000, 7000) que parecen
    # un RBT mal ubicado: se descartan.
    raros = ((d["cliente"] == c["cliente"]) & (d["fecha"] >= pd.Timestamp("2024-06-01"))
             & (d["celulas_somaticas"] >= 2000) & (d["celulas_somaticas"] % 1000 == 0))
    info["rcs_descartados_ind59"] = int(raros.sum())
    d.loc[raros, "celulas_somaticas"] = np.nan

    d["y_rcs"] = (d["celulas_somaticas"] > UMBRAL_RCS).where(d["celulas_somaticas"].notna()).astype(float)
    d["y_rbt"] = (d["rbt_corr"] > UMBRAL_RBT).where(d["rbt_corr"].notna()).astype(float)
    d["anio"] = d["fecha"].dt.year
    d["mes"] = d["fecha"].dt.month
    d["semana"] = d["fecha"].dt.to_period("W").dt.start_time
    d["fecha_fina"] = d["fecha_origen"] != "creación"   # fecha escrita en el archivo (día exacto) vs fecha de creación (±3 días)
    doy = d["fecha"].dt.dayofyear.to_numpy()
    d["sen"] = np.sin(2 * np.pi * doy / 365.25)
    d["cos"] = np.cos(2 * np.pi * doy / 365.25)
    return d.reset_index(drop=True), info


def acotar_composicion(d, cuantiles=(0.001, 0.999)):
    """Recorta lactosa, grasa y proteína a los percentiles 0,1-99,9 % de las filas de entrenamiento (valores imposibles
    como grasa = 10 g/100 g). Devuelve una copia y los límites usados."""
    d = d.copy()
    ent = d[d["fecha"] < CORTE_PRUEBA]
    lim = {c: tuple(ent[c].quantile(cuantiles)) for c in COMPOSICION}
    for c, (lo, hi) in lim.items():
        d[c] = d[c].clip(lo, hi)
    return d, lim


def cargar_temperatura():
    t = pd.read_csv(R.TEMPERATURA, encoding="utf-8-sig", parse_dates=["fecha"]).sort_values("fecha")
    t = t.set_index("fecha").asfreq("D")
    n_vacios = int(t["t_media"].isna().sum())
    return t.interpolate(limit_area="inside"), n_vacios


# ------------------------------------------------------------------------------------- temperatura y anomalía
def _armonicos(doy, k=3):
    w = 2 * np.pi * np.asarray(doy) / 365.25
    return np.column_stack([np.ones(len(w))] + [f(j * w) for j in range(1, k + 1) for f in (np.sin, np.cos)])


def serie_ventana(temp, estadistico, n):
    """Temperatura de los n días que terminan el día de la medición: media de t_media o máxima de t_max."""
    if estadistico == "media":
        return temp["t_media"].rolling(n, min_periods=n).mean()
    return temp["t_max"].rolling(n, min_periods=n).max()


def anomalia_ventana(temp, estadistico, n, hasta=CORTE_PRUEBA):
    """DataFrame diario con la ventana (`temp_x`), su climatología por día del año (`clim_x`) y la anomalía (`anom_x`).

    La climatología es una regresión con 3 armónicos del día del año ajustada **solo con fechas anteriores a `hasta`**
    (no usa los años de prueba). La anomalía es qué tan por encima de lo habitual para esa época del año estuvo la ventana.
    """
    s = serie_ventana(temp, estadistico, n).dropna()
    ent = s[s.index < hasta]
    X = _armonicos(ent.index.dayofyear)
    beta = np.linalg.lstsq(X, ent.to_numpy(), rcond=None)[0]
    clim = _armonicos(s.index.dayofyear) @ beta
    out = pd.DataFrame({"temp_x": s.to_numpy(), "clim_x": clim}, index=s.index)
    out["anom_x"] = out["temp_x"] - out["clim_x"]
    out.index.name = "fecha"
    return out


def agregar_temperatura(d, temp, estadistico, n, hasta=CORTE_PRUEBA):
    """Suma a `d` las columnas temp_x, clim_x y anom_x de la ventana elegida (unión por fecha)."""
    a = anomalia_ventana(temp, estadistico, n, hasta).reset_index()
    base = d.drop(columns=[c for c in ("temp_x", "clim_x", "anom_x") if c in d.columns])
    return base.merge(a, on="fecha", how="left", validate="many_to_one")


# ----------------------------------------------------------------------------------------------- modelos
def hacer_pipeline(num, cat, C=1.0, class_weight=None):
    """Escalado de las numéricas + one-hot de las categóricas + LogisticRegression (L2, parámetro C)."""
    trans = []
    if num:
        trans.append(("num", StandardScaler(), list(num)))
    if cat:
        trans.append(("cat", OneHotEncoder(handle_unknown="ignore"), list(cat)))
    pre = ColumnTransformer(trans)
    lr = LogisticRegression(C=C, class_weight=class_weight, max_iter=1000, solver="lbfgs")
    return Pipeline([("pre", pre), ("lr", lr)])


def ajustar_predecir(pipe, ent, prueba, objetivo, num, cat):
    """Ajusta en `ent` y devuelve la probabilidad de la clase positiva en `prueba` (y también el pipeline ajustado)."""
    cols = list(num) + list(cat)
    pipe.fit(ent[cols], ent[objetivo].astype(int))
    return pipe.predict_proba(prueba[cols])[:, 1], pipe


def validar_rodante(x, y, num, cat, C=1.0, class_weight=None, anios=(2021, 2022, 2023)):
    """Validación temporal rodante: para cada año, entrena con los años anteriores y valida en ese año.

    Devuelve el promedio de AUC, log-loss y Brier de los años (un solo año de validación es muy ruidoso).
    """
    filas = []
    for a in anios:
        ini, fin = pd.Timestamp(a, 1, 1), pd.Timestamp(a + 1, 1, 1)
        ent, val = x[x["fecha"] < ini], x[(x["fecha"] >= ini) & (x["fecha"] < fin)]
        p, _ = ajustar_predecir(hacer_pipeline(num, cat, C=C, class_weight=class_weight), ent, val, y, num, cat)
        filas.append(metricas(val[y], p))
    return pd.DataFrame(filas).mean().to_dict()


def metricas(y, p, y_ref_tasa=None):
    """AUC, log-loss y Brier. `y_ref_tasa` (opcional) añade el log-loss y Brier de predecir siempre esa tasa."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    out = {"AUC": roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
           "log-loss": log_loss(y, p), "Brier": brier_score_loss(y, p)}
    return out


def calibracion(y, p, cubos=10):
    """Tabla de calibración por cuantiles de la probabilidad predicha: prob. media predicha, tasa observada y n.

    Devuelve (tabla, ECE): el ECE es el promedio ponderado de |observada − predicha|.
    """
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    cuts = np.unique(np.quantile(p, np.linspace(0, 1, cubos + 1)))
    cuts[0], cuts[-1] = -np.inf, np.inf
    g = pd.cut(p, cuts, labels=False, include_lowest=True)
    t = pd.DataFrame({"g": g, "p": p, "y": y}).groupby("g").agg(predicha=("p", "mean"), observada=("y", "mean"), n=("y", "size"))
    ece = float(np.average((t["observada"] - t["predicha"]).abs(), weights=t["n"]))
    return t.reset_index(drop=True), ece


def pendiente_calibracion(y, p):
    """Pendiente e intercepto de la regresión logística de y sobre logit(p): 1 y 0 = perfectamente calibrado."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=500).fit(z, np.asarray(y, int))
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def matriz_umbral(y, p, umbral):
    tn, fp, fn, tp = confusion_matrix(np.asarray(y, int), (np.asarray(p) >= umbral).astype(int), labels=[0, 1]).ravel()
    return {"umbral": umbral, "VP": int(tp), "FN": int(fn), "FP": int(fp), "VN": int(tn),
            "sensibilidad": tp / max(tp + fn, 1), "especificidad": tn / max(tn + fp, 1),
            "precisión (VPP)": tp / max(tp + fp, 1), "exactitud": (tp + tn) / len(y)}


def umbral_youden(y, p):
    """Umbral que maximiza sensibilidad + especificidad − 1 (índice de Youden)."""
    from sklearn.metrics import roc_curve
    fpr, tpr, th = roc_curve(np.asarray(y, int), p)
    return float(th[np.argmax(tpr - fpr)])


# ------------------------------------------------------------------------------- celdas y bootstrap por bloques
def armar_celdas(d, objetivo, claves, extra=()):
    """Colapsa las filas con las mismas características (`claves`, p. ej. cliente y fecha) en celdas con pesos.

    Una celda con n filas y k positivas equivale a dos filas (y=1 con peso k, y=0 con peso n−k): el ajuste de la
    regresión logística es **idéntico** al de las filas originales cuando los predictores solo dependen de `claves`
    (las columnas `extra` deben ser función de las claves). Devuelve un DataFrame con 2 filas por celda: las claves,
    `extra`, `semana` (el bloque del bootstrap), `y` y el peso `w`.
    """
    claves = list(claves)
    agg = {"suma": (objetivo, "sum"), "n": (objetivo, "count"), "semana": ("semana", "first")}
    agg.update({c: (c, "first") for c in extra})
    g = d.groupby(claves, observed=True).agg(**agg).reset_index()
    g = g[g["n"] > 0]
    pos = g.assign(y=1, w=g["suma"])
    neg = g.assign(y=0, w=g["n"] - g["suma"])
    x = pd.concat([pos, neg], ignore_index=True)
    return x[x["w"] > 0].drop(columns=["suma", "n"]).reset_index(drop=True)


def bloques_de(semanas, ancho=1):
    """Número de bloque de cada fila: semanas consecutivas agrupadas de a `ancho` (bloques móviles no solapados)."""
    s = pd.Series(pd.to_datetime(semanas))
    orden = {w: i for i, w in enumerate(sorted(s.unique()))}
    return (s.map(orden).to_numpy() // ancho).astype(int)


def bootstrap_bloques(pipe, X, y, w_base, bloque, B=300, semilla=SEMILLA):
    """Bootstrap por bloques con arranque en caliente.

    Preprocesa X una sola vez con el `pipe` ya ajustado y reajusta solo la LogisticRegression B veces, con pesos
    = (veces que el bloque salió sorteado) × peso base, partiendo de los coeficientes de la muestra completa.
    Devuelve (matriz B × (p+1) con [intercepto, coeficientes] en el espacio escalado).
    """
    pre = pipe.named_steps["pre"]
    lr0 = pipe.named_steps["lr"]
    Xt = pre.transform(X)
    y = np.asarray(y, int)
    w_base = np.asarray(w_base, float)
    bloque = np.asarray(bloque)
    nb = bloque.max() + 1
    rng = np.random.default_rng(semilla)
    sal = np.empty((B, Xt.shape[1] + 1))
    lr = LogisticRegression(C=lr0.C, max_iter=500, solver="lbfgs", warm_start=True, tol=1e-6)
    lr.coef_, lr.intercept_ = lr0.coef_.copy(), lr0.intercept_.copy()
    for b in range(B):
        mult = rng.multinomial(nb, np.full(nb, 1.0 / nb))
        w = mult[bloque] * w_base
        keep = w > 0
        lr.fit(Xt[keep], y[keep], sample_weight=w[keep])
        sal[b, 0], sal[b, 1:] = lr.intercept_[0], lr.coef_[0]
    return sal


def nombres_columnas(pipe):
    """Nombres de las columnas del preprocesado (num__x, cat__cliente_IND-xx)."""
    return list(pipe.named_steps["pre"].get_feature_names_out())


def coef_crudos(pipe, nombres=None):
    """Coeficientes del pipeline en unidades originales (las numéricas se des-escalan): Series con el intercepto aparte."""
    pre = pipe.named_steps["pre"]
    lr = pipe.named_steps["lr"]
    nombres = nombres or nombres_columnas(pipe)
    esc = pre.named_transformers_["num"].scale_ if "num" in pre.named_transformers_ else np.array([])
    coef = lr.coef_[0].copy()
    coef[: len(esc)] = coef[: len(esc)] / esc
    return pd.Series(coef, index=nombres)


def desescalar_matriz(pipe, sal):
    """Convierte la matriz del bootstrap (espacio escalado) a coeficientes crudos (sin intercepto)."""
    pre = pipe.named_steps["pre"]
    esc = pre.named_transformers_["num"].scale_ if "num" in pre.named_transformers_ else np.array([])
    coef = sal[:, 1:].copy()
    coef[:, : len(esc)] = coef[:, : len(esc)] / esc
    return coef


def resumen_or(coef_punto, coef_boot, nombres, nivel=0.95):
    """Odds ratio (exp del coeficiente) con intervalo percentil del bootstrap. Devuelve DataFrame indexado por nombre."""
    a = (1 - nivel) / 2
    lo, hi = np.quantile(coef_boot, [a, 1 - a], axis=0)
    return pd.DataFrame({"OR": np.exp(np.asarray(coef_punto)), "IC_inf": np.exp(lo), "IC_sup": np.exp(hi)}, index=nombres)


def or_desvio_categoria(coef, nombres, prefijo):
    """Odds ratio de cada nivel de una categórica respecto del promedio de sus niveles (codificación por desvíos).

    `coef` puede ser un vector (p,) o una matriz (B, p). Con la regresión con todos los niveles (sin referencia) y
    ridge, los niveles son simétricos; el desvío evita elegir un nivel de referencia arbitrario.
    Devuelve (niveles, matriz de log-odds por desvío con forma (B, k)).
    """
    coef = np.atleast_2d(coef)
    idx = [i for i, n in enumerate(nombres) if n.startswith(prefijo)]
    niveles = [nombres[i][len(prefijo):] for i in idx]
    sub = coef[:, idx]
    return niveles, sub - sub.mean(axis=1, keepdims=True)
