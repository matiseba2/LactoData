"""Regresión lineal y polinómica (scikit-learn) del ciclo anual de proteína y grasa de la leche cruda.

Lo usa regresion_ciclo_leche_cruda.ipynb (armado por build_regresion_ciclo_leche.py). Todo el modelado va con
Pipeline / ColumnTransformer / PolynomialFeatures / LinearRegression de sklearn; lo único hecho a mano son dos
transformadores chicos (armónicos de Fourier y codificación de efectos de la industria) y el bootstrap por bloques,
que usa los mismos coeficientes de mínimos cuadrados pero remuestrea semanas enteras.

Un modelo se describe con una lista de componentes (tuplas):
    ("mes",)               mes del año como one-hot                      (lineal)
    ("poly", g)            polinomio de grado g del día del año          (x = (día - 183) / 183, entre -1 y 1)
    ("fourier", k)         armónicos sen/cos(2*pi*j*día/365,25), j = 1..k (cíclico)
    ("industria",)         efecto de la industria (codificación de efectos: una industria nueva = promedio)
    ("tendencia",)         años transcurridos desde 2019 (lineal)
    ("temp", N, g)         temperatura media de los últimos N días, polinomio de grado g
    ("anom", N, g)         anomalía de esa temperatura (temperatura menos su climatología), polinomio de grado g
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

import rutas as R

OBJETIVOS = ["proteina", "grasa"]
INICIO = pd.Timestamp("2019-01-01")          # antes de 2019 hay 32 filas sueltas (fechas dudosas), sin cobertura
FIN_ENTRENAMIENTO = pd.Timestamp("2023-12-31")
CORTES = {"grasa": (2.0, 6.5), "proteina": (2.5, 4.5)}   # g/100 g, rango físico plausible de leche de vaca cruda
VENTANAS_T = (7, 14, 30)
PERIODO = 365.25
MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
SEMILLA = 42   # la cátedra pide semilla 42
MIN_CELDA = 20            # una celda industria x semana entra a la métrica semanal si tiene al menos 20 muestras
MIN_MUESTRAS_INDUSTRIA = 1000


# ------------------------------------------------------------------ datos
def cargar_leche():
    """Lee leche_cruda.csv con las columnas que se usan. `archivo` solo sirve para identificar informes (no se imprime)."""
    cols = ["cliente", "archivo", "muestra", "barra", "grasa", "proteina", "lactosa", "solidos_totales",
            "celulas_somaticas", "crioscopia", "fecha", "fecha_origen"]
    return pd.read_csv(R.LECHE_CRUDA, encoding="utf-8-sig", usecols=cols, parse_dates=["fecha"], low_memory=False)


def cargar_temperatura():
    """Temperatura diaria de Rafaela con la media móvil de N días (el día de la muestra y los N-1 anteriores), su climatología
    y la anomalía. La climatología (ciclo medio de la temperatura) se ajusta con 3 armónicos usando solo 2015-2023
    y nunca mira la leche."""
    t = pd.read_csv(R.TEMPERATURA, encoding="utf-8-sig", parse_dates=["fecha"]).sort_values("fecha").reset_index(drop=True)
    t = t.set_index("fecha").asfreq("D").reset_index()
    doy = t["fecha"].dt.dayofyear.to_numpy()
    Xc = Ciclo(3).fit_transform(doy.reshape(-1, 1))
    Xc = np.column_stack([np.ones(len(t)), Xc])
    for n in VENTANAS_T:
        tn = t["t_media"].rolling(n, min_periods=n).mean()
        t[f"t{n}"] = tn
        ok = tn.notna() & (t["fecha"] <= FIN_ENTRENAMIENTO)
        b = np.linalg.lstsq(Xc[ok.to_numpy()], tn[ok].to_numpy(), rcond=None)[0]
        t[f"clim{n}"] = Xc @ b
        t[f"an{n}"] = tn - t[f"clim{n}"]
    return t


def limpiar(L, T):
    """Aplica la limpieza paso a paso. Devuelve (datos limpios, tabla de cuántas filas saca cada paso)."""
    pasos = []

    def paso(nombre, d, mascara_ok):
        pasos.append({"paso": nombre, "saca": int((~mascara_ok).sum()), "quedan": int(mascara_ok.sum())})
        return d[mascara_ok]

    pasos.append({"paso": "filas del archivo leche_cruda.csv", "saca": 0, "quedan": len(L)})
    d = paso("sin fecha, sin grasa o sin proteína", L, L[["fecha", "grasa", "proteina"]].notna().all(axis=1))
    d = paso(f"fecha anterior a {INICIO.year} (sin cobertura)", d, d["fecha"] >= INICIO)
    for v, (lo, hi) in CORTES.items():
        d = paso(f"{v} fuera de {lo}-{hi} g/100 g", d, d[v].between(lo, hi))
    clave = [c for c in d.columns if c != "archivo"]
    d = paso("duplicados (misma industria, fecha, muestra y mismos valores)", d, ~d.duplicated(clave))
    cols_t = ["fecha"] + [f"{p}{n}" for n in VENTANAS_T for p in ("t", "clim", "an")]
    d = d.merge(T[cols_t], on="fecha", how="left", validate="many_to_one")
    d = paso("sin temperatura de las ventanas de 7/14/30 días", d, d[[f"t{n}" for n in VENTANAS_T]].notna().all(axis=1))
    d = d.reset_index(drop=True)

    d["informe"] = pd.factorize(d["cliente"].astype(str) + "|" + d["archivo"].astype(str) + "|" + d["fecha"].astype(str))[0]
    d = d.drop(columns="archivo")
    d["doy"] = d["fecha"].dt.dayofyear
    d["x"] = (d["doy"] - 183) / 183                           # día del año escalado a [-1, 1]
    d["mes"] = d["fecha"].dt.month
    d["anios"] = (d["fecha"] - INICIO).dt.days / PERIODO
    d["anio"] = d["fecha"].dt.year
    d["semana"] = d["fecha"] - pd.to_timedelta(d["fecha"].dt.weekday, unit="D")      # lunes de la semana
    d["bloque"] = ((d["fecha"] - INICIO).dt.days // 28).astype(int)                  # bloques de 28 días
    d["sem_id"] = ((d["fecha"] - INICIO).dt.days // 7).astype(int)
    d["celda"] = pd.factorize(d["cliente"].astype(str) + "|" + d["semana"].astype(str))[0]   # industria x semana
    d["exacta"] = d["fecha_origen"] != "creación"                                     # fecha escrita en el archivo, no la de creación
    return d, pd.DataFrame(pasos)


def sin_industrias_chicas(d, minimo=MIN_MUESTRAS_INDUSTRIA):
    """Saca las industrias con menos de `minimo` muestras. Devuelve (datos, lista de industrias sacadas)."""
    n = d["cliente"].value_counts()
    chicas = sorted(n[n < minimo].index)
    return d[~d["cliente"].isin(chicas)].reset_index(drop=True), chicas


# ------------------------------------------------------------------ transformadores
class Ciclo(BaseEstimator, TransformerMixin):
    """Armónicos de Fourier del día del año: sen y cos de 2*pi*j*día/365,25, para j = 1..k (2k columnas)."""

    def __init__(self, k=2):
        self.k = k

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        th = 2 * np.pi * np.asarray(X, dtype=float).reshape(-1, 1) / PERIODO * np.arange(1, self.k + 1)
        return np.hstack([np.sin(th), np.cos(th)])

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{f}{j}" for j in range(1, self.k + 1) for f in ("sen", "cos")])


class EfectoIndustria(BaseEstimator, TransformerMixin):
    """Codificación de efectos: una columna por industria menos una (la última vale -1). Una industria que no estaba en el
    entrenamiento queda en 0, o sea, se predice con el promedio de las industrias (no con la mitad del nivel, como pasaría
    con un one-hot cuyo coeficiente se reparte con la constante)."""

    def fit(self, X, y=None):
        self.cats_ = sorted(pd.unique(np.asarray(X).ravel()))
        return self

    def transform(self, X):
        v = np.asarray(X).ravel()
        out = np.zeros((len(v), max(len(self.cats_) - 1, 0)))
        for j, c in enumerate(self.cats_[:-1]):
            out[v == c, j] = 1.0
        out[v == self.cats_[-1]] = -1.0
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array([f"ef_{c}" for c in self.cats_[:-1]])


# ------------------------------------------------------------------ modelos
def construir(comps):
    """Pipeline (preprocesamiento por columnas + LinearRegression) a partir de la lista de componentes."""
    trs = []
    for c in comps:
        t = c[0]
        if t == "mes":
            trs.append(("ciclo", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["mes"]))
        elif t == "poly":
            trs.append(("ciclo", PolynomialFeatures(c[1], include_bias=False), ["x"]))
        elif t == "fourier":
            trs.append(("ciclo", Ciclo(c[1]), ["doy"]))
        elif t == "industria":
            trs.append(("industria", EfectoIndustria(), ["cliente"]))
        elif t == "tendencia":
            trs.append(("tendencia", "passthrough", ["anios"]))
        elif t == "temp":
            trs.append(("temp", PolynomialFeatures(c[2], include_bias=False), [f"t{c[1]}"]))
        elif t == "anom":
            trs.append(("anom", PolynomialFeatures(c[2], include_bias=False), [f"an{c[1]}"]))
        else:
            raise ValueError(t)
    return Pipeline([("pre", ColumnTransformer(trs, remainder="drop")), ("reg", LinearRegression())])


def modelo(comps):
    """comps vacío = media global (DummyRegressor); si no, el pipeline lineal."""
    return DummyRegressor(strategy="mean") if not comps else construir(comps)


def ajustar_predecir(comps, d, ent, pru):
    """Ajusta con las filas `ent` (máscara) y predice las filas `pru`. Devuelve (modelo, pred_entrenamiento, pred_prueba), n x 2."""
    m = modelo(comps)
    m.fit(d.loc[ent], d.loc[ent, OBJETIVOS].to_numpy())
    return m, m.predict(d.loc[ent]), m.predict(d.loc[pru])


def metricas(y, p, celda):
    """MAE, RMSE y R² por muestra y por promedio semanal de cada industria (las celdas industria x semana).
    y y p son n x 2 (proteína, grasa). Devuelve un dict por objetivo."""
    _, inv = np.unique(celda, return_inverse=True)
    cnt = np.bincount(inv)
    out = {}
    for j, v in enumerate(OBJETIVOS):
        e = y[:, j] - p[:, j]
        r = {"mae": np.abs(e).mean(), "rmse": np.sqrt((e ** 2).mean()),
             "r2": 1 - (e ** 2).sum() / ((y[:, j] - y[:, j].mean()) ** 2).sum()}
        ym, pm = np.bincount(inv, y[:, j]) / cnt, np.bincount(inv, p[:, j]) / cnt
        ok = cnt >= MIN_CELDA
        es = (ym - pm)[ok]
        r.update({"mae_sem": np.abs(es).mean(), "rmse_sem": np.sqrt((es ** 2).mean()),
                  "r2_sem": 1 - (es ** 2).sum() / ((ym[ok] - ym[ok].mean()) ** 2).sum(), "n_celdas": int(ok.sum())})
        out[v] = r
    return out


def predicciones_rodantes(comps, d, anios):
    """Ventana que crece por año: para cada año Y de `anios` se ajusta con todo lo anterior a Y y se predice Y.
    Devuelve (pred n x 2 con NaN fuera de esos años)."""
    pred = np.full((len(d), 2), np.nan)
    for y in anios:
        ent, pru = (d["anio"] < y).to_numpy(), (d["anio"] == y).to_numpy()
        if pru.any() and ent.any():
            pred[pru] = ajustar_predecir(comps, d, ent, pru)[2]
    return pred


def tabla_modelos(d, modelos, anios_val=(2021, 2022, 2023), anios_test=(2024, 2025, 2026)):
    """Para cada modelo: entrenamiento (ajustado hasta 2023), validación (ventana rodante 2021-23, fuera de tiempo) y prueba
    (ajustado hasta 2023, probado 2024-26). Devuelve (filas largas, dict de predicciones rodantes por modelo)."""
    ent = (d["fecha"] <= FIN_ENTRENAMIENTO).to_numpy()
    pru = ~ent
    val = d["anio"].isin(anios_val).to_numpy()
    Y = d[OBJETIVOS].to_numpy()
    filas, rodantes = [], {}
    for nombre, comps in modelos.items():
        m, p_ent, p_pru = ajustar_predecir(comps, d, ent, pru)
        rod = predicciones_rodantes(comps, d, list(anios_val) + [a for a in range(2024, 2027)])
        rodantes[nombre] = rod
        for etapa, mk, pr in (("entrenamiento", ent, p_ent), ("validación 2021-23", val, rod[val]), ("prueba 2024-26", pru, p_pru)):
            for v, r in metricas(Y[mk], pr, d.loc[mk, "celda"].to_numpy()).items():
                filas.append({"modelo": nombre, "etapa": etapa, "objetivo": v, **r})
    return pd.DataFrame(filas), rodantes


def por_anio(d, rodantes, anios=range(2021, 2027)):
    """RMSE por año de la ventana rodante (por muestra y por promedio semanal) para cada modelo."""
    Y = d[OBJETIVOS].to_numpy()
    filas = []
    for nombre, rod in rodantes.items():
        for a in anios:
            mk = (d["anio"] == a).to_numpy()
            if not mk.any():
                continue
            for v, r in metricas(Y[mk], rod[mk], d.loc[mk, "celda"].to_numpy()).items():
                filas.append({"modelo": nombre, "anio": a, "objetivo": v, "n": int(mk.sum()), **r})
    return pd.DataFrame(filas)


def industria_afuera(d, modelos, solo_pasado=False):
    """GroupKFold por industria: cada industria se deja afuera una vez, el modelo se ajusta con las demás y se prueba en ella.
    Con solo_pasado=True además se ajusta hasta 2023 y se prueba 2024-26 (industria nueva y período futuro a la vez).
    Los modelos no pueden llevar el término de industria. Devuelve una tabla por industria, objetivo y modelo."""
    grupos = d["cliente"].to_numpy()
    n_g = len(np.unique(grupos))
    Y = d[OBJETIVOS].to_numpy()
    filas = []
    for k_ent, k_pru in GroupKFold(n_splits=n_g).split(d, groups=grupos):
        cli = np.unique(grupos[k_pru])
        assert len(cli) == 1
        ent = np.zeros(len(d), bool); ent[k_ent] = True
        pru = np.zeros(len(d), bool); pru[k_pru] = True
        if solo_pasado:
            ent &= (d["fecha"] <= FIN_ENTRENAMIENTO).to_numpy()
            pru &= (d["fecha"] > FIN_ENTRENAMIENTO).to_numpy()
        if pru.sum() < 100:
            continue
        for nombre, comps in modelos.items():
            _, _, p = ajustar_predecir(comps, d, ent, pru)
            for v, r in metricas(Y[pru], p, d.loc[pru, "celda"].to_numpy()).items():
                filas.append({"industria": cli[0], "modelo": nombre, "objetivo": v, "n": int(pru.sum()), **r})
    return pd.DataFrame(filas)


def curva_pocas_muestras(d, grados=range(1, 11), n=150, sorteos=40, semilla=SEMILLA):
    """Ejemplo didáctico de sobreajuste: polinomio del día del año de grado g ajustado con solo `n` muestras al azar del período
    de entrenamiento (promedio de varios sorteos), probado en 2024-26. RMSE por muestra de cada grado."""
    rng = np.random.default_rng(semilla)
    ent_idx = np.flatnonzero((d["fecha"] <= FIN_ENTRENAMIENTO).to_numpy())
    pru = d[d["fecha"] > FIN_ENTRENAMIENTO]
    Yp = pru[OBJETIVOS].to_numpy()
    filas = []
    for s in range(sorteos):
        sub = d.iloc[rng.choice(ent_idx, n, replace=False)]
        Ys = sub[OBJETIVOS].to_numpy()
        for g in grados:
            m = Pipeline([("poly", PolynomialFeatures(g, include_bias=False)), ("reg", LinearRegression())])
            m.fit(sub[["x"]], Ys)
            e_e, e_p = Ys - m.predict(sub[["x"]]), Yp - m.predict(pru[["x"]])
            for j, v in enumerate(OBJETIVOS):
                filas.append({"sorteo": s, "grado": g, "objetivo": v,
                              "rmse_entrenamiento": np.sqrt((e_e[:, j] ** 2).mean()), "rmse_prueba": np.sqrt((e_p[:, j] ** 2).mean())})
    return pd.DataFrame(filas).groupby(["objetivo", "grado"], as_index=False)[["rmse_entrenamiento", "rmse_prueba"]].mean()


# ------------------------------------------------------------------ curva anual, interpretación y bootstrap por bloques
def grilla_anual(d, cliente="IND-NUEVA"):
    """Un año de días (1..365) con las columnas que piden los modelos de ciclo (industria desconocida = promedio)."""
    f = pd.date_range("2023-01-01", "2023-12-31")
    g = pd.DataFrame({"fecha": f, "doy": f.dayofyear, "mes": f.month, "cliente": cliente, "anios": d["anios"].mean()})
    g["x"] = (g["doy"] - 183) / 183
    return g


def curva_anual(m, d):
    """Predicción del modelo (para proteína y grasa) a lo largo del año en una industria promedio (solo modelos de ciclo)."""
    return m.predict(grilla_anual(d))


def diseno(m, d):
    """Matriz de diseño del pipeline ajustado, con la constante adelante, y los nombres de las columnas."""
    X = m.named_steps["pre"].transform(d)
    nombres = ["const"] + list(m.named_steps["pre"].get_feature_names_out())
    return np.column_stack([np.ones(len(X)), X]), nombres


def estadisticos_por_bloque(X, Y, bloque):
    """X'X y X'y de cada bloque (suma por bloque), para remuestrear bloques sin rehacer el ajuste fila por fila."""
    orden = np.argsort(bloque, kind="stable")
    b = np.asarray(bloque)[orden]
    ub, ini = np.unique(b, return_index=True)
    fin = np.append(ini[1:], len(b))
    Xs, Ys = X[orden], Y[orden]
    XtX = np.empty((len(ub), X.shape[1], X.shape[1]))
    XtY = np.empty((len(ub), X.shape[1], Y.shape[1]))
    for i, (a, z) in enumerate(zip(ini, fin)):
        XtX[i] = Xs[a:z].T @ Xs[a:z]
        XtY[i] = Xs[a:z].T @ Ys[a:z]
    return XtX, XtY


def bootstrap_bloques(XtX, XtY, B=1000, semilla=SEMILLA):
    """B remuestreos con reposición de bloques completos (semanas o bloques de 28 días). Devuelve (coef. ajuste completo, coef. de los remuestreos)."""
    rng = np.random.default_rng(semilla)
    nb = len(XtX)
    beta0 = np.linalg.solve(XtX.sum(0), XtY.sum(0))
    betas = np.empty((B,) + beta0.shape)
    for i in range(B):
        w = rng.multinomial(nb, np.full(nb, 1 / nb))
        betas[i] = np.linalg.solve(np.tensordot(w, XtX, 1), np.tensordot(w, XtY, 1))
    return beta0, betas


def ajuste_bloques(comps, d, bloque="bloque", B=1000, mascara=None):
    """Ajuste con todas las filas (o `mascara`) y bootstrap por bloques. Devuelve dict con el modelo ajustado, los nombres de columnas,
    los coeficientes (p x 2: proteína, grasa) y los de los remuestreos (B x p x 2)."""
    dd = d if mascara is None else d[mascara]
    m = construir(comps).fit(dd, dd[OBJETIVOS].to_numpy())
    X, nombres = diseno(m, dd)
    XtX, XtY = estadisticos_por_bloque(X, dd[OBJETIVOS].to_numpy(), dd[bloque].to_numpy())
    beta0, betas = bootstrap_bloques(XtX, XtY, B)
    # el ajuste de sklearn y el de las ecuaciones normales (que usa el bootstrap) deben coincidir
    assert np.allclose(beta0[0], m.named_steps["reg"].intercept_, atol=1e-6), "las ecuaciones normales no coinciden con sklearn"
    return {"modelo": m, "nombres": nombres, "beta": beta0, "boot": betas, "n_bloques": len(XtX), "n": len(dd)}


def amplitud_pico(aj, d):
    """Amplitud (pico a pico, g/100 g y % del promedio anual) y día/mes del pico del ciclo de un modelo con armónicos,
    con su estimación puntual y los remuestreos. Devuelve DataFrame con una fila por objetivo y estadístico."""
    pre = aj["modelo"].named_steps["pre"]
    g = grilla_anual(d)
    _, tr, cols_ciclo = next(t for t in pre.transformers_ if t[0] == "ciclo")
    Bc = tr.transform(g[cols_ciclo])
    cols = [i for i, n in enumerate(aj["nombres"]) if n.startswith("ciclo__")]
    doys = g["doy"].to_numpy()

    def resumen(beta):
        curva = Bc @ beta[cols]                              # (365 x 2)
        return curva.max(0) - curva.min(0), doys[curva.argmax(0)], doys[curva.argmin(0)]
    a0, p0, v0 = resumen(aj["beta"])
    Ab, Pb, Vb = zip(*[resumen(b) for b in aj["boot"]])
    Ab, Pb, Vb = np.array(Ab), np.array(Pb), np.array(Vb)
    nivel = d[OBJETIVOS].mean().to_numpy()

    def ic_circular(x_boot, x0):
        """Intervalo 95 % de un día del año, cuidando que el calendario es un círculo (el mínimo cae cerca del 31-dic / 1-ene)."""
        delta = (x_boot - x0 + 182) % 365 - 182
        lo, hi = np.percentile(delta, [2.5, 97.5])
        return f"{dia_del_anio(x0 + lo)} a {dia_del_anio(x0 + hi)}"
    filas = []
    for j, v in enumerate(OBJETIVOS):
        filas.append({"objetivo": v, "amplitud pico a pico (g/100 g)": a0[j], "ic95 inf": np.percentile(Ab[:, j], 2.5),
                      "ic95 sup": np.percentile(Ab[:, j], 97.5), "% del promedio": 100 * a0[j] / nivel[j],
                      "pico": dia_del_anio(p0[j]), "pico ic95": ic_circular(Pb[:, j], p0[j]),
                      "mínimo": dia_del_anio(v0[j]), "mínimo ic95": ic_circular(Vb[:, j], v0[j])})
    return pd.DataFrame(filas)


def dia_del_anio(doy):
    """Día del año (1..365) escrito como 'día mes'."""
    f = pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(round(doy)) % 365 - 1 if int(round(doy)) % 365 else 364)
    return f"{f.day} {MESES[f.month - 1]}"


def coef(aj, nombre, objetivo, ic=95):
    """Coeficiente `nombre` del ajuste con su intervalo de confianza por bootstrap de bloques."""
    i, j = aj["nombres"].index(nombre), OBJETIVOS.index(objetivo)
    b = aj["boot"][:, i, j]
    a = (100 - ic) / 2
    return aj["beta"][i, j], np.percentile(b, a), np.percentile(b, 100 - a)


def efecto_anomalia(aj, nombre_lin, nombre_cuad, objetivo, a):
    """Efecto sobre `objetivo` de una anomalía `a` (°C): b1*a + b2*a², con banda de bootstrap. Devuelve (central, inf, sup) por cada a."""
    j = OBJETIVOS.index(objetivo)
    i1 = aj["nombres"].index(nombre_lin)
    i2 = aj["nombres"].index(nombre_cuad) if nombre_cuad else None
    a = np.asarray(a, float)

    def f(beta):
        return beta[i1, j] * a + (beta[i2, j] * a ** 2 if i2 is not None else 0)
    c = f(aj["beta"])
    bs = np.array([f(b) for b in aj["boot"]])
    return c, np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)


def se_por_bin(res, bin_, bloque):
    """Error estándar de la media de `res` en cada bin, agrupado por bloque (las filas del mismo bloque no son independientes)."""
    df = pd.DataFrame({"r": res, "bin": bin_, "bl": bloque})
    out = []
    for b, g in df.groupby("bin", observed=True):
        m = g["r"].mean()
        s = g.groupby("bl")["r"].agg(lambda z: (z - m).sum())
        out.append({"bin": b, "media": m, "se": np.sqrt((s ** 2).sum()) / len(g), "n": len(g)})
    return pd.DataFrame(out)


def escalones(d, residuo, meses=6, minimo=30):
    """Busca, por industria y en el promedio de todas, el mes con el mayor salto del residuo: media de los `meses` meses
    siguientes menos la de los `meses` anteriores. Devuelve DataFrame (industria, mes del salto, tamaño, sd del residuo mensual)."""
    t = d[["cliente", "fecha"]].copy()
    t["r"] = residuo
    t["ym"] = t["fecha"].dt.to_period("M")
    filas = []
    for cli, g in list(t.groupby("cliente")) + [("(todas)", t)]:
        s = g.groupby("ym")["r"].agg(["mean", "size"])
        s = s[s["size"] >= minimo]
        idx = pd.period_range(s.index.min(), s.index.max(), freq="M")
        m = s["mean"].reindex(idx)
        mejor = None
        for i in range(meses, len(idx) - meses + 1):
            ant, des = m.iloc[i - meses:i], m.iloc[i:i + meses]
            if ant.notna().sum() >= meses // 2 + 1 and des.notna().sum() >= meses // 2 + 1:
                dif = des.mean() - ant.mean()
                if mejor is None or abs(dif) > abs(mejor[1]):
                    mejor = (idx[i], dif)
        if mejor:
            filas.append({"industria": cli, "mes del mayor salto": str(mejor[0]), "salto": mejor[1],
                          "sd del residuo mensual": m.std(), "salto / sd": mejor[1] / m.std(), "meses con datos": int(m.notna().sum())})
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ selección por validación
def n_parametros(comps, d):
    """Cantidad de columnas (sin contar la constante) que usa un modelo."""
    if not comps:
        return 0
    pre = construir(comps).named_steps["pre"].fit(d)
    return pre.transform(d.head(5)).shape[1]


def puntaje_validacion(tm, nombres, metrica="rmse_sem"):
    """Puntaje de validación (2021-23, fuera de tiempo): para cada objetivo, el error del modelo dividido por el mejor error entre
    `nombres`; se promedian los dos objetivos. 1,000 = el mejor en los dos."""
    v = tm[(tm["etapa"] == "validación 2021-23") & tm["modelo"].isin(nombres)].pivot(index="modelo", columns="objetivo", values=metrica)
    v = v.loc[list(nombres)]
    return (v / v.min()).mean(axis=1)


def elegir(tm, nombres, n_par, tolerancia=0.005, metrica="rmse_sem"):
    """Regla de elección: de los modelos cuyo puntaje está a menos de `tolerancia` (0,5 %) del mejor, el de menos parámetros;
    a igualdad de parámetros, el primero de la lista. Devuelve (nombre elegido, tabla con puntaje y parámetros)."""
    p = puntaje_validacion(tm, nombres, metrica)
    t = pd.DataFrame({"parámetros": [n_par[n] for n in nombres], "puntaje": p.to_numpy()}, index=list(nombres))
    ok = t[t["puntaje"] <= t["puntaje"].min() * (1 + tolerancia)]
    elegido = ok.sort_values("parámetros", kind="stable").index[0]
    t["dentro de la tolerancia"] = t["puntaje"] <= t["puntaje"].min() * (1 + tolerancia)
    return elegido, t


def ciclo_simple(dd, d, comps):
    """Ajusta `comps` con las filas `dd` y devuelve, por objetivo, (amplitud pico a pico, día del pico, día del mínimo) de la curva anual."""
    m = construir(comps).fit(dd, dd[OBJETIVOS].to_numpy())
    cu = curva_anual(m, d)
    dias = np.arange(1, 366)
    return cu.max(0) - cu.min(0), dias[cu.argmax(0)], dias[cu.argmin(0)]
