"""¿Cuánto esperar para que una leche UAT quede deslactosada? Modelo cinético con los remuestreos de Senasa.

Pedido del usuario (2026-11-09): con la lactosa del análisis original y de los remuestreos, la temperatura
media de los 5 días previos a la recepción y los días entre análisis, ver si se puede predecir cuántos días
esperar, según la temperatura media de la semana, para que el producto baje de UMBRAL (0,1 g/100 g).

Datos: solo Senasa (MQ no mide lactosa); leche fluida con desc_subtipo UAT; lactosa medida, sin «cálculo por
diferencia». Lote = desc_lote o el número de la descripción (misma regla que
_archivo/proyecto/build_remuestreo_temperatura.py). Lote remuestreado = más de una fecha de recepción. Se usa
el par (primera fecha, última fecha); si una fecha tiene varias muestras se promedia. «< 0,03» es censurado:
la lactosa real está entre 0 y 0,03.

Modelo (primer orden, la lactasa residual hidroliza a una tasa que depende de la temperatura):
    ln L1 = ln L0 − k(T)·Δdías + ε,   k(T) = exp(a + b·(T − 20)),   ε ~ N(0, σ²)
T = temperatura media entre las dos recepciones (incluye los 5 días previos a la primera). Si L1 es «< LD»
la verosimilitud usa P(ln L1 < ln LD) (Tobit). Días de espera: ln(L0 / UMBRAL) / k(T).
Comparación: el mismo modelo sin temperatura (b = 0) y un modelo «sin tiempo» (ln L1 − ln L0 constante), que
es lo que se vería si la baja fuera regresión a la media y no hidrólisis.
"""

import numpy as np
import pandas as pd
from scipy import optimize, stats

import rutas as R
from datos_clase41 import SEMILLA, leer_senasa

UMBRAL = 0.1                 # g/100 g
VENTANA_PREVIA = 5           # días antes de la recepción
TEMPERATURA = R.ENTRADA / "temperatura_rafaela.csv"
SALIDA_XLSX = R.EXCEL / "prediccion_deslactosado.xlsx"
SALIDA_MODELO = R.MODELOS / "deslactosado_uat.joblib"
T_REF = 20.0


def _parse(col):
    t = col.astype(str).str.replace("&lt;", "<", regex=False)
    v = pd.to_numeric(t.str.extract(r"(\d+(?:[.,]\d+)?)")[0].str.replace(",", "."), errors="coerce")
    return v, t.str.contains("<", regex=False) & v.notna()


def temperatura():
    t = pd.read_csv(TEMPERATURA, parse_dates=["fecha"]).sort_values("fecha").set_index("fecha")["t_media"]
    return t


def muestras_uat():
    """Muestras UAT de Senasa con lactosa medida, con lote anónimo. No conserva la descripción."""
    s = leer_senasa().crudo
    u = s[s["desc_familia"].eq("leche fluida") & s["desc_subtipo"].eq("UAT")].copy()
    u["fecha"] = pd.to_datetime(u["Fecha de Recepción"], format="%d-%m-%Y", errors="coerce")
    u["lac"], u["lac_ld"] = _parse(u["Resultado Lactosa g /100g (*)"])
    dif = u["Técnica Lactosa g /100g (*)"].fillna("").str.contains("diferencia", case=False)
    u = u[u["lac"].notna() & ~dif & u["fecha"].notna()].copy()
    lote = u["desc_lote"].fillna(u["Descripción"].str.extract(r"·\s*:\s*(\d{5,})")[0]).astype("string").str.strip().str.upper()
    orden = lote.dropna().to_frame("l").assign(f=u["fecha"]).groupby("l")["f"].min().sort_values(kind="stable")
    u["lote_id"] = lote.map({l: f"L{i + 1:04d}" for i, l in enumerate(orden.index)})
    t = temperatura()
    previa = t.rolling(f"{VENTANA_PREVIA}D", closed="left").mean()
    u["t5"] = u["fecha"].map(previa)
    return u[["Identificador", "Cliente", "lote_id", "fecha", "lac", "lac_ld", "t5"]].reset_index(drop=True)


def pares(u=None):
    """Una fila por lote remuestreado: primera y última recepción."""
    u = muestras_uat() if u is None else u
    g = (u.dropna(subset=["lote_id"]).groupby(["lote_id", "fecha"])
         .agg(cliente=("Cliente", "first"), lac=("lac", "mean"), ld=("lac_ld", "all"), t5=("t5", "first"), n=("lac", "size"))
         .reset_index().sort_values(["lote_id", "fecha"]))
    g = g[g.groupby("lote_id")["fecha"].transform("nunique") > 1]
    p = g.groupby("lote_id").agg(cliente=("cliente", "first"), f0=("fecha", "first"), f1=("fecha", "last"),
                                 l0=("lac", "first"), ld0=("ld", "first"), l1=("lac", "last"), ld1=("ld", "last"),
                                 t5_0=("t5", "first"), t5_1=("t5", "last"), n_fechas=("fecha", "size"))
    p["dias"] = (p["f1"] - p["f0"]).dt.days
    t = temperatura()
    # exposición: de 5 días antes de la primera recepción hasta la última
    p["t_exp"] = [t.loc[a - pd.Timedelta(days=VENTANA_PREVIA): b - pd.Timedelta(days=1)].mean() for a, b in zip(p["f0"], p["f1"])]
    p["anio"] = p["f0"].dt.year
    p = p[~p["ld0"]]                         # sin L0 medido no hay punto de partida
    return p.reset_index()


# ---------------- ajuste ----------------

def _nll(theta, y0, y1, cens, d, T, modelo):
    if modelo == "sin_tiempo":
        c, ls = theta
        mu = y0 + c
    elif modelo == "desfase":
        a, b, c, ls = theta
        mu = y0 + c - np.exp(a + b * (T - T_REF)) * d
    else:
        a, b, ls = (theta if modelo == "temperatura" else (theta[0], 0.0, theta[1]))
        mu = y0 - np.exp(a + b * (T - T_REF)) * d
    s = np.exp(ls)
    ll = np.where(cens, stats.norm.logcdf((y1 - mu) / s), stats.norm.logpdf((y1 - mu) / s) - ls)
    return -ll.sum()


def ajustar(p, modelo="temperatura"):
    y0, y1 = np.log(p["l0"].to_numpy()), np.log(p["l1"].to_numpy())
    cens, d, T = p["ld1"].to_numpy(), p["dias"].to_numpy(float), p["t_exp"].to_numpy(float)
    x0 = {"temperatura": [np.log(0.2), 0.0, np.log(0.5)], "constante": [np.log(0.2), np.log(0.5)],
          "sin_tiempo": [-1.0, np.log(0.5)], "desfase": [np.log(0.2), 0.0, -0.3, np.log(0.5)]}[modelo]
    r = optimize.minimize(_nll, x0, args=(y0, y1, cens, d, T, modelo), method="Nelder-Mead",
                          options={"maxiter": 20000, "xatol": 1e-8, "fatol": 1e-8})
    k = len(x0)
    return {"modelo": modelo, "theta": r.x, "nll": r.fun, "aic": 2 * k + 2 * r.fun, "n": len(p)}


def k_de(theta, T):
    a, b = theta[0], theta[1]
    return np.exp(a + b * (np.asarray(T, float) - T_REF))


def dias_hasta(l0, T, theta, umbral=UMBRAL):
    """Días para pasar de l0 a umbral a temperatura media T (°C)."""
    return np.maximum(np.log(np.asarray(l0, float) / umbral), 0) / k_de(theta, T)


def bootstrap(p, n=1000, semilla=SEMILLA, modelo="temperatura"):
    rng = np.random.default_rng(semilla)
    th = []
    for _ in range(n):
        m = p.iloc[rng.integers(0, len(p), len(p))]
        th.append(ajustar(m, modelo)["theta"])
    return np.array(th)


def validar_loo(p, theta_completo=None):
    """Deja un lote afuera, predice ln L1 y si queda bajo UMBRAL."""
    filas = []
    for i in range(len(p)):
        tr = p.drop(index=p.index[i])
        th = ajustar(tr)["theta"]
        th0 = ajustar(tr, "constante")["theta"]
        thn = ajustar(tr, "sin_tiempo")["theta"]
        thd = ajustar(tr, "desfase")["theta"]
        r = p.iloc[i]
        predd = np.log(r.l0) + thd[2] - k_de(thd, r.t_exp) * r.dias
        pred = np.log(r.l0) - k_de(th, r.t_exp) * r.dias
        pred0 = np.log(r.l0) - np.exp(th0[0]) * r.dias
        predn = np.log(r.l0) + thn[0]
        filas.append({"lote_id": r.lote_id, "ln_l1": np.log(r.l1), "censurado": r.ld1, "pred_temp": pred,
                      "pred_sin_temp": pred0, "pred_sin_tiempo": predn, "pred_desfase": predd})
    v = pd.DataFrame(filas)
    obs_bajo = v["ln_l1"] <= np.log(UMBRAL)
    res = {}
    for c in ["pred_temp", "pred_sin_temp", "pred_sin_tiempo", "pred_desfase"]:
        err = (v[c] - v["ln_l1"])[~v["censurado"]]
        res[c] = {"mae_ln": err.abs().mean(), "rmse_ln": np.sqrt((err ** 2).mean()),
                  "acierto_bajo_umbral": ((v[c] <= np.log(UMBRAL)) == obs_bajo).mean()}
    return v, pd.DataFrame(res).T


def tabla_espera(theta, boot=None, l0s=(0.2, 0.3, 0.5, 0.7, 1.0, 1.5), temps=(10, 15, 20, 25, 30)):
    filas = []
    for l0 in l0s:
        for T in temps:
            f = {"lactosa_inicial": l0, "t_media_semana": T, "dias": dias_hasta(l0, T, theta)}
            if boot is not None:
                d = dias_hasta(l0, T, boot.T)
                f["dias_ic_inf"], f["dias_ic_sup"] = np.percentile(d, [2.5, 97.5])
            filas.append(f)
    return pd.DataFrame(filas).round(2)


def transversal(u):
    """Todas las muestras UAT (sin remuestreo): proporción bajo UMBRAL y mediana de lactosa según T5."""
    x = u.dropna(subset=["t5"]).assign(t5_rango=lambda d: pd.cut(d["t5"], [-np.inf, 12, 17, 22, np.inf],
                                                                labels=["≤ 12 °C", "12 a 17 °C", "17 a 22 °C", "> 22 °C"]))
    g = x.groupby("t5_rango", observed=True)
    out = pd.DataFrame({"muestras": g.size(), "lactosa_mediana": g["lac"].median(),
                        "% bajo umbral": (100 * g["lac"].apply(lambda s: (s <= UMBRAL).mean())).round(1)}).reset_index()
    rho = stats.spearmanr(x["lac"], x["t5"])
    return out, rho


def armar(escribir=True, n_boot=1000):
    """Modelo principal: «desfase» (ln L1 = ln L0 + c − k(T)·días). c recoge la baja que aparece ya al primer
    día del remuestreo (el lote se remuestrea porque el primer valor dio alto) y no se usa para predecir la
    espera; la espera sale solo de k(T)."""
    u = muestras_uat()
    p = pares(u)
    modelos = ["desfase", "temperatura", "constante", "sin_tiempo"]
    fits = {m: ajustar(p, m) for m in modelos}
    th = fits["desfase"]["theta"]
    boot = bootstrap(p, n_boot, modelo="desfase")
    ic = lambda x: np.percentile(x, [2.5, 97.5])
    ic_b, ic_k20, ic_c = ic(boot[:, 1]), ic(np.exp(boot[:, 0])), ic(boot[:, 2])
    v, met = validar_loo(p)
    sin_b = optimize.minimize(lambda z: _nll([z[0], 0.0, z[1], z[2]], np.log(p["l0"].values), np.log(p["l1"].values),
                                              p["ld1"].values, p["dias"].values.astype(float), p["t_exp"].values, "desfase"),
                              [th[0], th[2], th[3]], method="Nelder-Mead", options={"maxiter": 20000}).fun
    lr = 2 * (sin_b - fits["desfase"]["nll"])
    rho_d = stats.spearmanr(np.log(p["l1"] / p["l0"]), p["dias"])
    trans, rho_t = transversal(u)
    resumen = pd.DataFrame([
        {"concepto": "muestras UAT con lactosa", "valor": len(u)},
        {"concepto": "lotes con remuestreo y L0 medido", "valor": len(p)},
        {"concepto": "días entre análisis (mediana, máx)", "valor": f"{p['dias'].median():.0f}, {p['dias'].max()}"},
        {"concepto": "remuestreos bajo LD (censurados)", "valor": int(p["ld1"].sum())},
        {"concepto": "Spearman ln(L1/L0) contra días", "valor": f"{rho_d.statistic:.2f} (p {rho_d.pvalue:.2g})"},
        {"concepto": "k a 20 °C (1/día)", "valor": f"{np.exp(th[0]):.3f} (IC {ic_k20[0]:.3f} a {ic_k20[1]:.3f})"},
        {"concepto": "desfase c (ln, baja inmediata del remuestreo)", "valor": f"{th[2]:.2f} (IC {ic_c[0]:.2f} a {ic_c[1]:.2f})"},
        {"concepto": "b: efecto de 1 °C sobre ln k", "valor": f"{th[1]:.4f} (IC {ic_b[0]:.4f} a {ic_b[1]:.4f})"},
        {"concepto": "k se multiplica por cada +10 °C", "valor": f"{np.exp(10 * th[1]):.2f} (IC {np.exp(10 * ic_b[0]):.2f} a {np.exp(10 * ic_b[1]):.2f})"},
        {"concepto": "razón de verosimilitud b = 0 (p)", "valor": f"{stats.chi2.sf(lr, 1):.3g}"},
        {"concepto": "AIC " + " / ".join(modelos), "valor": " / ".join(f"{fits[m]['aic']:.1f}" for m in modelos)},
        {"concepto": "Spearman lactosa contra T5, todas las UAT", "valor": f"{rho_t.statistic:.2f} (p {rho_t.pvalue:.2g})"},
    ])
    tabla = tabla_espera(th, boot)
    if escribir:
        import joblib
        R.preparar()
        R.MODELOS.mkdir(parents=True, exist_ok=True)
        joblib.dump({"theta": th, "boot": boot, "umbral": UMBRAL, "t_ref": T_REF, "n_lotes": len(p),
                     "rango_dias": (int(p["dias"].min()), int(p["dias"].max())),
                     "rango_t": (float(p["t_exp"].min()), float(p["t_exp"].max())),
                     "pares": p[["dias", "l0", "l1", "ld1", "t_exp"]].reset_index(drop=True)}, SALIDA_MODELO)
        cols = ["lote_id", "cliente", "f0", "f1", "dias", "l0", "l1", "ld1", "t5_0", "t5_1", "t_exp", "n_fechas"]
        with pd.ExcelWriter(SALIDA_XLSX) as w:
            resumen.to_excel(w, sheet_name="resumen", index=False)
            tabla.to_excel(w, sheet_name="dias_de_espera", index=False)
            met.round(3).to_excel(w, sheet_name="validacion_loo")
            trans.to_excel(w, sheet_name="transversal_t5", index=False)
            p[cols].to_excel(w, sheet_name="lotes_remuestreados", index=False)
            u.rename(columns={"Cliente": "cliente"}).to_excel(w, sheet_name="muestras_uat", index=False)
    return {"muestras": u, "pares": p, "ajustes": fits, "boot": boot, "loo": v, "metricas": met,
            "resumen": resumen, "tabla": tabla, "transversal": trans}


if __name__ == "__main__":
    r = armar()
    print(r["resumen"].to_string(index=False))
    print(r["metricas"].round(3).to_string())
    print(r["transversal"].to_string(index=False))
    print(r["tabla"].to_string(index=False))
