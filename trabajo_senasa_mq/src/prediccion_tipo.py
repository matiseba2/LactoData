"""Predecir el tipo de una muestra nueva a partir de lo que se le midió.

Pedido del usuario (2026-11-09): con los catálogos de Senasa (clasificación manual) y MQ (automática), y
sabiendo que hay muestras MQ cuya consistencia declarada no coincide con lo medido y que hay muestras de
calibración, ver si se puede predecir el tipo de una muestra nueva por sus parámetros medidos. Dos niveles:
primero la familia, después el subtipo dentro de la familia.

Muestras de calibración (no entrenan; se usan para evaluar): desc_tipo_muestra «control de laboratorio» o
descripción con calibración, patrón, referencia, repetibilidad / reproducibilidad o repetición. La
descripción se lee en memoria y nunca se imprime.

Entrenamiento: Senasa + MQ nuevas, sin calibración, sin «ensayo» ni «pool», sin «sin clasificar», sin las
excluidas por humedad + sólidos fuera de 100 ± 1 y sin las muestras cuyo declarado no coincide con lo medido
(declarado_vs_medido; «borde» sí entra).

Variables: los resultados numéricos de los análisis (NaN si no se midió) y, para cada uno, si se pidió:
qué análisis se piden también dice qué es la muestra. Modelo: HistGradientBoostingClassifier (admite NaN).
Validación cruzada agrupada por envío (las muestras de un envío nunca quedan a ambos lados).
"""

import re

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

import declarado_vs_medido as DVM
import rutas as R
from datos_clase41 import a_numero, leer_senasa

SALIDA_XLSX = R.EXCEL / "prediccion_tipo.xlsx"
SALIDA_MODELO = R.MODELOS / "prediccion_tipo.joblib"
UMBRAL_PROB = 0.80
MIN_CLASE = 20
SEMILLA = 42   # la cátedra pide semilla 42
SIN = "sin clasificar"

PATRON_CALIBRACION = re.compile(r"CALIBR|PATR[OÓ]N|REFERENCIA|REPETIB|REPRODUC|REPETICI", re.I)

# variable -> columnas «Resultado …» en orden de preferencia (la primera con dato gana)
VARIABLES = {
    "humedad": ["Resultado Humedad g/ 100g (*)"],
    "solidos": ["Resultado Sólidos totales g/ 100g_x"],
    "grasa": ["Resultado Materia Grasa g/ 100g (*)_x"],
    "grasa_es": ["Resultado Materia grasa en extracto seco g/ 100g (*)_x"],
    "proteinas": ["Resultado Proteínas g/ 100g (*)"],
    "proteinas_bs": ["Resultado Proteínas en base seca % (*)"],
    "cenizas": ["Resultado Cenizas g/ 100g"],
    "cloruros": ["Resultado Cloruros (NaCl g / 100g) (*)"],
    "acidez_naoh": ["Resultado Acidez (ml NaOH 0,1 N/ 10 g sólidos no grasos) (*)"],
    "acidez_lactico": ["Resultado Acidez (% Ácido Láctico) (*)", "Resultado Acidez (g ácido láctico/ 100g) (*)"],
    "lactosa": ["Resultado Lactosa g /100g (*)", "Resultado Contenido de Lactosa % (g/ 100ml) (*)",
                "Resultado Determinación del contenido de lactosa (g/ 100ml) (*)"],
    "galactosa": ["Resultado D - Galactosa g/ 100g (*)", "Resultado Galactosa g/ 100g (*)"],
    "azucares": ["Resultado Azucares totales g/100gr (*) (fructosa/glucosa/sacarosa/maltosa/lactosa)_x",
                 "Resultado Azucares totales g/100gr (*)", "Resultado Azúcares Totales % (*)",
                 "Resultado Azucares totales g/100gr (*) (fructosa/glucosa/sacarosa/maltosa/lactosa/galactosa)"],
    "ph": ["Resultado pH (*)", "Resultado pH (25°C) (*)"],
    "aw": ["Resultado Actividad de agua (% Aw) (*)"],
    "almidon": ["Resultado Almidón Total g/ 100g (*)"],
    "sodio": ["Resultado Sodio mg /100g (*)"],
    "calcio": ["Resultado Calcio mg/ 100g (*)"],
    "brix": ["Resultado Grado Brix (°Bx) (*)"],
    "solubilidad": ["Resultado Solubilidad % (*)"],
}

# subtipo dentro de cada familia: (columna, {valor original -> clase}); valores fuera del mapa quedan sin subtipo
SUBTIPO = {
    "queso": ("desc_consistencia", {"baja humedad": "baja humedad", "mediana humedad": "mediana humedad",
                                    "alta humedad": "alta humedad", "muy alta humedad": "alta humedad",
                                    "procesado": "procesado"}),
    "leche en polvo": ("desc_grasa", {"ENTERA": "entera", "DESCREMADA": "descremada",
                                      "PARCIALMENTE DESCREMADA": "descremada"}),
    "dulce de leche": ("desc_subtipo", {"FAMILIAR": "familiar", "REPOSTERO": "repostero", "HELADERO": "heladero"}),
    "suero / permeado": ("desc_subtipo", {"SUERO": "suero", "DESMINERALIZADO": "desmineralizado",
                                          "PERMEADO DE SUERO": "permeado", "PERMEADO": "permeado",
                                          "PERMEADO DE LECHE": "permeado", "CONCENTRADO": "concentrado", "WPC": "wpc"}),
    "leche fluida": ("desc_subtipo", {"UAT": "UAT", "CONCENTRADA": "concentrada", "CRUDA": "cruda",
                                      "CHOCOLATADA": "chocolatada", "PASTEURIZADA": "pasteurizada"}),
    "concentrado proteico": ("desc_subtipo", {"WPC": "WPC", "CASEINA": "caseína / caseinato",
                                              "CASEINATO DE SODIO": "caseína / caseinato",
                                              "CASEINATO DE CALCIO": "caseína / caseinato", "MPC": "MPC"}),
    "manteca / crema": ("desc_subtipo", {"MANTECA": "manteca", "CREMA": "crema", "CREMA DE LECHE": "crema",
                                         "DOBLE CREMA": "crema"}),
    "yogur / bebida lactea": ("desc_subtipo", {"YOGUR": "yogur", "BEBIDA LACTEA": "bebida láctea",
                                               "POLVO PARA PREPARAR": "polvo para preparar"}),
}


# ---------------- datos ----------------

def es_calibracion(tipo_muestra, descripcion):
    return tipo_muestra.eq("control de laboratorio") | descripcion.fillna("").str.contains(PATRON_CALIBRACION)


def _variables(d):
    x = pd.DataFrame(index=d.index)
    for v, cols in VARIABLES.items():
        s = pd.Series(np.nan, index=d.index)
        for c in cols:
            if c in d:
                s = s.fillna(a_numero(d[c]))
        x[v] = s
    x["humedad"] = x["humedad"].fillna(100 - x["solidos"])
    x["solidos"] = x["solidos"].fillna(100 - x["humedad"])
    x.loc[~x["humedad"].between(0, 100), ["humedad", "solidos"]] = np.nan
    # perfil de análisis pedidos (cuántos y cuáles)
    res = [c for c in d.columns if c.startswith("Resultado ")]
    x["n_analisis"] = d[res].notna().sum(axis=1)
    for v in VARIABLES:
        x[f"pide_{v}"] = x[v].notna().astype(int)
    return x


def datos():
    """Una fila por muestra (Senasa + MQ nuevas) con variables, etiquetas y marcas de conjunto."""
    S = leer_senasa()
    s = S.crudo.assign(origen="Senasa (manual)", desc_familia_fuente="senasa (manual)")
    q = pd.read_excel(DVM.MQ_NUEVAS).assign(origen="MQ nueva (automática)")
    meta = ["Identificador", "origen", "Cliente", "Envío", "desc_familia", "desc_familia_fuente", "desc_tipo_muestra"]
    filas = []
    for d in (s, q):
        y = d[meta].copy()
        y["calibracion"] = es_calibracion(d["desc_tipo_muestra"], d["Descripción"]).to_numpy()
        y["subtipo"] = np.nan
        for fam, (col, mapa) in SUBTIPO.items():
            m = d["desc_familia"].eq(fam)
            y.loc[m, "subtipo"] = d.loc[m, col].map(mapa)
        filas.append(pd.concat([y, _variables(d)], axis=1))
    t = pd.concat(filas, ignore_index=True)
    t["excluida_hum_sol"] = t["Identificador"].isin(S.excluidas)
    dvm = DVM.armar(escribir=False)[0]
    no = set(dvm.loc[dvm["grado"].eq("no coincide"), "Identificador"])
    t["declarado_no_coincide"] = t["Identificador"].isin(no)
    t["entrena"] = (~t["calibracion"] & t["desc_tipo_muestra"].isin(["produccion"]) & t["desc_familia"].ne(SIN)
                    & t["desc_familia"].notna() & ~t["excluida_hum_sol"] & ~t["declarado_no_coincide"]
                    & (t["n_analisis"] > 0))
    return t


def columnas_x(t):
    return [c for c in t.columns if c in VARIABLES or c.startswith("pide_") or c == "n_analisis"]


# ---------------- modelo ----------------

def _modelo():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0,
                                          class_weight="balanced", random_state=SEMILLA)


def _clases_validas(y):
    vc = y.value_counts()
    return vc.index[vc >= MIN_CLASE]


def validar(t, objetivo, filtro, n_folds=5):
    """Predicción fuera de pliegue para las filas de entrenamiento que cumplen `filtro`. Devuelve DataFrame."""
    d = t[t["entrena"] & filtro & t[objetivo].notna()]
    d = d[d[objetivo].isin(_clases_validas(d[objetivo]))]
    X, y, g = d[columnas_x(t)], d[objetivo].astype(str), d["Envío"].astype(str)
    oof = pd.DataFrame(index=d.index, columns=["pred", "prob"])
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=SEMILLA)
    for tr, te in cv.split(X, y, g):
        assert not set(g.iloc[tr]) & set(g.iloc[te]), "fuga de envíos entre pliegues"
        m = _modelo().fit(X.iloc[tr], y.iloc[tr])
        p = m.predict_proba(X.iloc[te])
        oof.iloc[te, 0] = m.classes_[p.argmax(1)]
        oof.iloc[te, 1] = p.max(1)
    oof["real"] = y
    oof["prob"] = oof["prob"].astype(float)
    oof["origen"] = d["origen"]
    return oof


def metricas(oof, nombre):
    filas = []
    for org, o in [("todas", oof)] + list(oof.groupby("origen")):
        seg = o["prob"] >= UMBRAL_PROB
        filas.append({"nivel": nombre, "origen": org, "muestras": len(o),
                      "exactitud": accuracy_score(o["real"], o["pred"]),
                      "F1 macro": f1_score(o["real"], o["pred"], average="macro"),
                      f"% con prob ≥ {UMBRAL_PROB}": 100 * seg.mean(),
                      f"exactitud con prob ≥ {UMBRAL_PROB}": accuracy_score(o.loc[seg, "real"], o.loc[seg, "pred"]) if seg.any() else np.nan})
    return pd.DataFrame(filas)


def confusion(oof):
    clases = sorted(set(oof["real"]) | set(oof["pred"]))
    return pd.DataFrame(confusion_matrix(oof["real"], oof["pred"], labels=clases), index=clases, columns=clases)


def fiabilidad(oof):
    b = pd.cut(oof["prob"], [0, .5, .6, .7, .8, .9, .95, 1.0001], right=False)
    return (oof.assign(ok=oof["real"] == oof["pred"]).groupby(b, observed=True)
            .agg(muestras=("ok", "size"), prob_media=("prob", "mean"), exactitud=("ok", "mean")).reset_index()
            .rename(columns={"prob": "rango_prob"}).astype({"rango_prob": str}))


def entrenar(t):
    """Modelo final: familia + uno por familia para el subtipo. Devuelve dict."""
    cx = columnas_x(t)
    d = t[t["entrena"]]
    d = d[d["desc_familia"].isin(_clases_validas(d["desc_familia"]))]
    modelos = {"familia": _modelo().fit(d[cx], d["desc_familia"].astype(str))}
    for fam in SUBTIPO:
        e = d[d["desc_familia"].eq(fam) & d["subtipo"].notna()]
        e = e[e["subtipo"].isin(_clases_validas(e["subtipo"]))]
        if e["subtipo"].nunique() >= 2:
            modelos[fam] = _modelo().fit(e[cx], e["subtipo"].astype(str))
    return {"modelos": modelos, "columnas": cx, "umbral": UMBRAL_PROB}


def predecir(paquete, X):
    """Familia y subtipo propuestos, con su probabilidad y la decisión con el umbral."""
    cx, ms, u = paquete["columnas"], paquete["modelos"], paquete["umbral"]
    pf = ms["familia"].predict_proba(X[cx])
    out = pd.DataFrame({"familia_pred": ms["familia"].classes_[pf.argmax(1)], "prob_familia": pf.max(1)}, index=X.index)
    out["subtipo_pred"], out["prob_subtipo"] = pd.Series(pd.NA, index=X.index, dtype=object), np.nan
    for fam, m in ms.items():
        if fam == "familia":
            continue
        sel = out["familia_pred"].eq(fam)
        if sel.any():
            p = m.predict_proba(X.loc[sel, cx])
            out.loc[sel, "subtipo_pred"] = m.classes_[p.argmax(1)]
            out.loc[sel, "prob_subtipo"] = p.max(1)
    out["decision"] = np.where(out["prob_familia"] >= u, "propuesta", "sin decidir")
    return out


def armar(escribir=True):
    t = datos()
    cx = columnas_x(t)
    oof_f = validar(t, "desc_familia", pd.Series(True, index=t.index))
    mets, confs, oofs_sub = [metricas(oof_f, "familia")], {"familia": confusion(oof_f)}, {}
    for fam in SUBTIPO:
        f = t["desc_familia"].eq(fam)
        if t.loc[t["entrena"] & f, "subtipo"].nunique() < 2:
            continue
        o = validar(t, "subtipo", f)
        if o["real"].nunique() < 2:
            continue
        oofs_sub[fam] = o
        mets.append(metricas(o, f"subtipo · {fam}"))
        confs[fam] = confusion(o)
    met = pd.concat(mets, ignore_index=True).round(3)
    paquete = entrenar(t)
    meta = ["Identificador", "origen", "Cliente", "desc_familia", "subtipo"]
    # calibración: predicción del modelo final (no entrenaron) contra su familia declarada
    cal = t[t["calibracion"] & (t["n_analisis"] > 0)]
    cal = pd.concat([cal[meta], predecir(paquete, cal)], axis=1)
    cal["acierta_familia"] = cal["familia_pred"].eq(cal["desc_familia"]).where(cal["desc_familia"].ne(SIN))
    # declarado no coincide: ¿el modelo apoya lo medido o lo declarado?
    nc = t[t["declarado_no_coincide"] & (t["n_analisis"] > 0)]
    nc = pd.concat([nc[meta + ["humedad"]], predecir(paquete, nc)], axis=1)
    nc["clase_por_humedad"] = np.where(nc["desc_familia"].eq("queso"), DVM.clase_queso(nc["humedad"]), DVM.clase_estado(nc["humedad"]))
    # sin clasificar: el modelo propone
    sc = t[t["desc_familia"].eq(SIN) & (t["n_analisis"] > 0)]
    sc = pd.concat([sc[["Identificador", "origen", "Cliente"]], predecir(paquete, sc)], axis=1)
    resumen = pd.DataFrame([
        {"concepto": "muestras totales", "valor": len(t)},
        {"concepto": "entrenamiento", "valor": int(t["entrena"].sum())},
        {"concepto": "calibración (fuera del entrenamiento)", "valor": int(t["calibracion"].sum())},
        {"concepto": "calibración por origen", "valor": t[t["calibracion"]].groupby("origen").size().to_dict()},
        {"concepto": "declarado no coincide (fuera del entrenamiento)", "valor": int(t["declarado_no_coincide"].sum())},
        {"concepto": "sin clasificar con análisis", "valor": len(sc)},
        {"concepto": "exactitud de familia en calibración (con familia conocida)", "valor": round(cal["acierta_familia"].mean(), 3)},
        {"concepto": f"ídem, solo con prob ≥ {UMBRAL_PROB}",
         "valor": round(cal.loc[cal["prob_familia"] >= UMBRAL_PROB, "acierta_familia"].mean(), 3)},
        {"concepto": "no coinciden: el subtipo predicho sigue a la humedad medida (quesos)",
         "valor": round((nc.loc[nc["desc_familia"].eq("queso"), "subtipo_pred"]
                         == nc.loc[nc["desc_familia"].eq("queso"), "clase_por_humedad"].replace("muy alta humedad", "alta humedad")).mean(), 3)},
    ])
    if escribir:
        R.preparar()
        R.MODELOS.mkdir(parents=True, exist_ok=True)
        joblib.dump(paquete, SALIDA_MODELO)
        with pd.ExcelWriter(SALIDA_XLSX) as w:
            resumen.astype(str).to_excel(w, sheet_name="resumen", index=False)
            met.to_excel(w, sheet_name="metricas", index=False)
            fiabilidad(oof_f).to_excel(w, sheet_name="fiabilidad_familia", index=False)
            for k, c in confs.items():
                c.to_excel(w, sheet_name=("conf " + k.replace("/", "-"))[:31])
            cal.to_excel(w, sheet_name="calibracion", index=False)
            nc.to_excel(w, sheet_name="no_coinciden", index=False)
            sc.to_excel(w, sheet_name="sin_clasificar", index=False)
    return {"datos": t, "oof_familia": oof_f, "oof_subtipo": oofs_sub, "metricas": met, "confusion": confs,
            "calibracion": cal, "no_coinciden": nc, "sin_clasificar": sc, "paquete": paquete, "resumen": resumen}


if __name__ == "__main__":
    r = armar()
    print(r["resumen"].to_string(index=False))
    print(r["metricas"].to_string(index=False))
