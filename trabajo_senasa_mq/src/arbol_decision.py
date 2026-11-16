"""Árbol de decisión en cascada (familia → subtipo) a partir de lo medido (Parte VII del cuaderno general).

Es el trabajo de otro integrante del grupo (cuadernos Arbol_V1 / V2 / V3, Algoritmos y Arbol_comparacion), llevado
al proyecto con el mismo código y las mismas reglas. Un primer árbol predice la familia con los 7 análisis de
composición (los faltantes quedan NaN: el árbol de scikit-learn ≥ 1.3 los maneja) y, dentro de la familia, otro
árbol predice el subtipo. Tres versiones:

- V1: solo arc (`dataset_senasa_revision.xlsx`); en cada árbol de subtipo se dejan afuera las clases con menos de
  5 muestras de entrenamiento (la regla original).
- V2: solo arc; en las familias con subtipos, las muestras sin subtipo forman la clase SIN CLASIFICAR.
- V3: 80 % de arc + todo MQ nuevas (filtrado); SIN CLASIFICAR solo se responde con probabilidad ≥ 0,85.

Cambios respecto de los cuadernos originales: semilla 42 (`SEMILLA` de datos_clase41) en vez de 2026; rutas de
rutas.py; el filtro «declarado vs medido» sale de `declarado_vs_medido.armar()` (el mismo cálculo que la planilla
que leía el original). La descripción se usa solo en memoria, para los filtros: nunca se muestra ni se guarda.
"""
from functools import lru_cache
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve
from sklearn.model_selection import (GridSearchCV, StratifiedGroupKFold, StratifiedKFold, cross_validate,
                                     train_test_split)
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier

import declarado_vs_medido as DVM
import rutas as R
from datos_clase41 import SEMILLA

ARC = R.EXCEL / "dataset_senasa_revision.xlsx"
MQ_NUEVAS = DVM.MQ_NUEVAS
SALIDA_MODELO = R.MODELOS / "arbol_decision.joblib"

VERSIONES = ["V1", "V2", "V3"]
TITULOS = {"V1": "V1 · solo arc", "V2": "V2 · arc + SIN CLASIFICAR", "V3": "V3 · arc + MQ, umbral 0,85"}
PREDICTORAS = ["humedad", "grasa", "proteinas", "cenizas", "solidos_totales", "lactosa", "galactosa"]
FAMILIA, SUBTIPO, AGRUPADO = "desc_familia", "desc_subtipo", "subtipo_agrupado"
SIN_CLASIFICAR = "sin clasificar"            # una familia: esas muestras no entrenan, se predicen al final
SUBTIPO_SIN_CLASIFICAR = "SIN CLASIFICAR"    # una clase de subtipo: muestras sin subtipo de una familia con subtipos
UMBRAL_SIN_CLASIFICAR = 0.85                 # V3: SIN CLASIFICAR solo si su probabilidad llega a esto
MINIMO_SUBTIPO = 20                          # dentro de cada familia, los subtipos con menos muestras van a «OTRO»
MINIMO_POR_CLASE = 5                         # V1: clases de subtipo con menos muestras de entrenamiento, afuera
CUADRICULA = {"max_depth": [2, 3, 4, 5, 6, 8, 10, None], "min_samples_leaf": [1, 2, 5, 10, 20, 50]}
PRUEBA = 0.2
N_PLIEGUES = 5
# Números de los cuadernos originales (semilla 2026), para comparar. Familia: diapositiva 6 de la presentación.
ORIGINAL = {"V1": {"familia: exactitud": 0.885, "familia: F1 macro": 0.697, "punta a punta (b)": 0.676,
                   "con familia real (b)": 0.765},
            "V2": {"familia: exactitud": 0.889, "familia: F1 macro": 0.693, "punta a punta (a)": 0.639,
                   "punta a punta (b)": 0.589, "con familia real (a)": 0.716, "con familia real (b)": 0.669,
                   "reales → SIN CLASIFICAR": 0.189},
            "V3": {"familia: exactitud": 0.896, "punta a punta (a)": 0.561, "punta a punta (b)": 0.621,
                   "reales → SIN CLASIFICAR": 0.097}}

# MQ: texto con coma decimal → números, con los nombres de arc; humedad = 100 − sólidos; sin lactosa ni galactosa
MQ_A_ARC = {"humedad": "humedad_derivada_100_menos_solidos", "grasa": "Resultado Materia Grasa g/ 100g (*)_x",
            "proteinas": "Resultado Proteínas g/ 100g (*)", "cenizas": "Resultado Cenizas g/ 100g",
            "solidos_totales": "Resultado Sólidos totales g/ 100g_x"}
PRINCIPALES_F = ["queso", "leche en polvo", "dulce de leche", "leche fluida", "suero / permeado", "manteca / crema",
                 "base nutricional", "concentrado proteico", "yogur / bebida lactea", SIN_CLASIFICAR]
LIQUIDOS = ["suero / permeado", "concentrado proteico", "base nutricional"]


def coma(x, d=3):
    """Número con coma decimal, para los textos."""
    return f"{x:.{d}f}".replace(".", ",")


# ------------------------------------------------------------------ datos
def _a_numero(serie):
    """Como el original: coma → punto y lo que no es número queda NaN («<LD» incluido)."""
    return pd.to_numeric(serie.str.replace(",", ".", regex=False), errors="coerce")


@lru_cache(maxsize=1)
def leer():
    """arc y MQ (llevado al formato de arc) en una sola tabla, con la columna `origen`. Se lee una vez por sesión
    (no modificar lo que devuelve)."""
    arc = pd.read_excel(ARC)
    q = pd.read_excel(MQ_NUEVAS, dtype=str)
    mq = q[["Identificador", "Envío", "Cliente", "Descripción", FAMILIA, SUBTIPO, "desc_tipo_muestra"]].copy()
    for col, origen in MQ_A_ARC.items():
        mq[col] = _a_numero(q[origen])
    mq["lactosa"] = mq["galactosa"] = np.nan
    mq.index = mq.index + 100_000          # índices propios, para no chocar con los de arc
    columnas = PREDICTORAS + [FAMILIA, SUBTIPO, "origen", "Identificador", "Envío", "Cliente", "Descripción",
                              "desc_tipo_muestra", "es_outlier", "excluida_hum_solidos"]
    union = pd.concat([arc.assign(origen="arc"), mq.assign(origen="MQ")]).reindex(columns=columnas)
    return arc, union


def filtros(union, arc, version):
    """Lista ordenada de (nombre, a qué origen se aplica, máscara de las filas que saca)."""
    base = [("sin ningún análisis", "arc y MQ", union[PREDICTORAS].isna().all(axis=1)),
            ("atípica o excluida por humedad y sólidos", "arc",
             (union["es_outlier"] == 1) | union["excluida_hum_solidos"].eq(True))]
    if version != "V3":
        return base
    desc = union["Descripción"].fillna("").str.upper().str.strip()
    fam, sol = union[FAMILIA], union["solidos_totales"]
    dvm = DVM.armar(escribir=False)[0]
    no_coinciden = set(dvm.loc[dvm["grado"] == "no coincide", "Identificador"].astype(str))
    errores = (((fam == "manteca / crema") & desc.str.contains(r"MANTECA POR \d+ ?G") & (union["grasa"] < 70))
               | desc.str.contains(r"CREMA PRUEBA-? ?MANTECA")
               | ((fam == "suero / permeado") & desc.str.contains("POLVO") & (sol < 50))
               | (desc.str.contains("WPC LIQUIDO") & (sol > 50))
               | ((fam == "yogur / bebida lactea") & (union["grasa"] > sol)))
    return base + [
        ("R1 control de laboratorio", "MQ", (union["desc_tipo_muestra"] == "control de laboratorio")
         & ~desc.str.contains("COMPOSICIÓN POR REFERENCIA|MEZCLA SUERO LIQUIDO")),
        ("R2 cliente IND-41", "arc y MQ", union["Cliente"] == "IND-41"),
        ("R3 IND-59 en serie (MUESTRA n, N°n, CREMA n)", "MQ",
         (union["Cliente"] == "IND-59") & desc.str.contains(r"(?:MUESTRA|N°|CREMA)\s*\d+$")),
        ("R4 serie por tiempo (n MIN, n')", "MQ", desc.str.contains(r"\d+\s*(?:MIN\b|')") & ~desc.str.contains(r"\bM\d+'")),
        ("R5 humedad + grasa + proteínas + cenizas > 102", "MQ",
         union[["humedad", "grasa", "proteinas", "cenizas"]].sum(axis=1) > 102),
        ("ensayo o pool", "MQ", union["desc_tipo_muestra"].isin(["ensayo", "pool"])),
        ("declarado vs medido: no coincide", "arc y MQ", union["Identificador"].astype(str).isin(no_coinciden)),
        ("líquido de planta (sólidos < 50)", "MQ", (sol < 50) & fam.isin(LIQUIDOS)),
        ("bebida láctea LIQUIDO R26 / R50 / D40L", "MQ",
         (fam == "yogur / bebida lactea") & desc.str.contains(r"LIQUIDO.*(?:R26|R50|D40L)")),
        ("suero de manteca", "MQ", desc.str.contains("SUERO DE MANTECA")),
        ("error de carga (las filas conocidas)", "MQ", errores),
        ("familia que no está en arc", "MQ", ~fam.isin(arc[FAMILIA].dropna().unique())),
    ]


def _aplicar(union, lista):
    """Saca las filas en orden; devuelve qué queda y la tabla de cuántas saca cada filtro (por origen y familia)."""
    es_mq = union["origen"] == "MQ"
    grupo = union[FAMILIA].where(union[FAMILIA].isin(PRINCIPALES_F), "otras")
    quedan = pd.Series(True, index=union.index)
    filas = [pd.concat([union["origen"].value_counts(), grupo.value_counts()]).rename("planillas completas")]
    for nombre, aplica, mascara in lista:
        sale = mascara & quedan & (es_mq if aplica == "MQ" else ~es_mq if aplica == "arc" else True)
        filas.append(pd.concat([union.loc[sale, "origen"].value_counts(), grupo[sale].value_counts()])
                     .rename(f"− {nombre} ({aplica})"))
        quedan &= ~sale
    filas.append(pd.concat([union.loc[quedan, "origen"].value_counts(), grupo[quedan].value_counts()]).rename("quedan"))
    columnas = [c for c in ["arc", "MQ"] if c in set(union["origen"])] + PRINCIPALES_F + ["otras"]
    conteos = pd.DataFrame(filas).reindex(columns=columnas).fillna(0).astype(int)
    return quedan, conteos


def agrupar(subtipos):
    """Dentro de una familia: los subtipos con menos de MINIMO_SUBTIPO muestras pasan a «OTRO»; los NaN quedan NaN."""
    cuentas = subtipos.value_counts()
    return subtipos.where(~subtipos.isin(cuentas[cuentas < MINIMO_SUBTIPO].index), "OTRO")


def preparar(version):
    """Filtra, agrupa subtipos, arma SIN CLASIFICAR (V2, V3) y parte 80 / 20 (solo arc; en V3 MQ entero entrena)."""
    arc, union = leer()
    if version != "V3":
        union = union[union["origen"] == "arc"]
    quedan, conteos = _aplicar(union, filtros(union, arc, version))
    d = union.loc[quedan].drop(columns=["Descripción", "es_outlier", "excluida_hum_solidos"])
    sin_par = {}
    if version == "V3":            # MQ solo usa subtipos que existen en arc para esa familia; los otros quedan sin subtipo
        pares_arc = set(zip(arc[FAMILIA], arc[SUBTIPO]))
        par = pd.Series(list(zip(d[FAMILIA], d[SUBTIPO])), index=d.index)
        m = (d["origen"] == "MQ") & d[SUBTIPO].notna() & ~par.isin(pares_arc)
        sin_par = d.loc[m].groupby([FAMILIA, SUBTIPO]).size().to_dict()
        d.loc[m, SUBTIPO] = np.nan
    sin_clasificar = d[d[FAMILIA] == SIN_CLASIFICAR].copy()
    d = d[d[FAMILIA] != SIN_CLASIFICAR].copy()
    d[AGRUPADO] = d.groupby(FAMILIA)[SUBTIPO].transform(agrupar)
    pasan_sc = pd.Series(False, index=d.index)
    if version != "V1":            # después de agrupar: las sin subtipo nunca van a «OTRO»
        con_subtipo = d.loc[d[SUBTIPO].notna(), FAMILIA].unique()
        pasan_sc = d[FAMILIA].isin(con_subtipo) & d[AGRUPADO].isna()
        d.loc[pasan_sc, AGRUPADO] = SUBTIPO_SIN_CLASIFICAR
    muestras, muestras_mq = d[d["origen"] == "arc"], d[d["origen"] == "MQ"]
    # estrato = familia + subtipo agrupado; los de una sola muestra se suman al más grande de su familia
    estrato = muestras[FAMILIA] + " | " + muestras[AGRUPADO].fillna("—")
    chicos = estrato.map(estrato.value_counts()) < 2
    mayor = estrato.groupby(muestras[FAMILIA]).agg(lambda s: s.value_counts().index[0])
    estrato[chicos] = muestras.loc[chicos, FAMILIA].map(mayor)
    ent_arc, pru = train_test_split(muestras, test_size=PRUEBA, random_state=SEMILLA, stratify=estrato)
    ent = pd.concat([ent_arc, muestras_mq])
    return SimpleNamespace(version=version, arc=len(arc), disponibles=len(union), conteos=conteos, modelado=d,
                           muestras=muestras, muestras_mq=muestras_mq, sin_clasificar=sin_clasificar,
                           pasan_sc=int(pasan_sc.sum()), sin_par=sin_par, ent=ent, ent_arc=ent_arc, pru=pru,
                           familias=sorted(muestras[FAMILIA].unique()))


# ------------------------------------------------------------------ entrenamiento
def pliegues_para(y):
    """Cinco pliegues estratificados, o menos si la clase más chica no alcanza."""
    return StratifiedKFold(n_splits=int(min(N_PLIEGUES, y.value_counts().min())), shuffle=True, random_state=SEMILLA)


def buscar(X, y):
    """Árbol elegido por validación cruzada (F1 macro) sobre la cuadrícula; best_estimator_ ya reentrenado con todo."""
    return GridSearchCV(DecisionTreeClassifier(random_state=SEMILLA), CUADRICULA, cv=pliegues_para(y),
                        scoring="f1_macro", n_jobs=-1).fit(X, y)


def y_subtipo(ent, familia, regla):
    """Las etiquetas de subtipo que entrenan el árbol de esa familia (V1: sin las clases con < 5 muestras)."""
    y = ent.loc[(ent[FAMILIA] == familia) & ent[AGRUPADO].notna(), AGRUPADO]
    return y[y.map(y.value_counts()) >= MINIMO_POR_CLASE] if regla == "V1" else y


def plan_subtipo(ent, familias, regla):
    """Qué modelo lleva cada familia: árbol, constante o nada (sin subtipo)."""
    filas = []
    for familia in familias:
        todas = ent.loc[(ent[FAMILIA] == familia) & ent[AGRUPADO].notna(), AGRUPADO].value_counts()
        cuentas = y_subtipo(ent, familia, regla).value_counts()
        fila = {"familia": familia, "entrenamiento": int(cuentas.sum()), "clases tras agrupar": len(todas)}
        if len(todas) == 0:
            decision, motivo = "sin subtipo", "la familia no tiene subtipos"
        elif regla == "V1":
            afuera = todas[todas < MINIMO_POR_CLASE]
            fila[f"afuera (< {MINIMO_POR_CLASE})"] = ", ".join(f"{k} ({v})" for k, v in afuera.items()) or "—"
            decision = "árbol" if len(cuentas) >= 2 else "constante"
            motivo = f"{len(cuentas)} clases con ≥ {MINIMO_POR_CLASE} muestras"
        else:
            fila[f"{SUBTIPO_SIN_CLASIFICAR} (entrenamiento)"] = int(todas.get(SUBTIPO_SIN_CLASIFICAR, 0))
            pliegues = int(min(N_PLIEGUES, cuentas.min()))
            if len(cuentas) == 1:
                decision, motivo = "constante", "una sola clase: no hay nada que decidir"
            elif pliegues < 2:
                decision, motivo = "constante", f"{cuentas.idxmin()} tiene 1 muestra de entrenamiento: no alcanza para 2 pliegues"
            else:
                decision = "árbol"
                motivo = f"{pliegues} pliegues" + ("" if pliegues == N_PLIEGUES else f" (la clase más chica, {cuentas.idxmin()}, tiene {cuentas.min()})")
        fila.update({"decisión": decision, "motivo": motivo,
                     "clases (entrenamiento)": ", ".join(f"{k} ({v})" for k, v in cuentas.items())})
        filas.append(fila)
    return pd.DataFrame(filas).set_index("familia")


def entrenar_cascada(ent, familias, regla, umbral=None, medianas=None):
    """La receta de las tres versiones: árbol de familia y, por familia, árbol / constante / nada.
    Devuelve (modelos, búsquedas, plan); `modelos` es lo que se guarda y lo que usa la app."""
    plan = plan_subtipo(ent, familias, regla)
    busquedas = {"familia": buscar(ent[PREDICTORAS], ent[FAMILIA])}
    sub = {}
    for familia, fila in plan.iterrows():
        y = y_subtipo(ent, familia, regla)
        if fila["decisión"] == "sin subtipo":
            sub[familia] = {"tipo": "sin subtipo", "motivo": fila["motivo"]}
        elif fila["decisión"] == "árbol":
            busquedas[familia] = buscar(ent.loc[y.index, PREDICTORAS], y)
            sub[familia] = {"tipo": "árbol", "modelo": busquedas[familia].best_estimator_}
        else:
            if len(y) == 0:
                y = ent.loc[(ent[FAMILIA] == familia) & ent[AGRUPADO].notna(), AGRUPADO]
            prop = y.value_counts(normalize=True)
            sub[familia] = {"tipo": "constante", "clase": prop.index[0], "probabilidades": prop.to_dict(),
                            "motivo": fila["motivo"]}
    modelos = {"predictoras": PREDICTORAS, "familia": busquedas["familia"].best_estimator_, "subtipo": sub,
               "minimo_subtipo": MINIMO_SUBTIPO,
               "subtipo_sin_clasificar": None if regla == "V1" else SUBTIPO_SIN_CLASIFICAR,
               "medianas": medianas, "umbral_sin_clasificar": umbral}
    return modelos, busquedas, plan


def entrenar(D):
    """La cascada de una versión, con su regla y su umbral."""
    return entrenar_cascada(D.ent, D.familias, D.version, umbral=UMBRAL_SIN_CLASIFICAR if D.version == "V3" else None,
                            medianas=D.muestras[PREDICTORAS].median().to_dict())


# ------------------------------------------------------------------ predicción
def predecir_subtipo(entrada, X, umbral=None):
    """Subtipo y su probabilidad para filas de una familia ya dada (o predicha).
    umbral: SIN CLASIFICAR solo si su probabilidad llega al umbral; si no, el subtipo real más probable.
    None: la predicción cruda del árbol."""
    if entrada["tipo"] == "sin subtipo":
        return np.full(len(X), None, dtype=object), np.full(len(X), np.nan)
    if entrada["tipo"] == "árbol":
        clases, proba = entrada["modelo"].classes_.astype(object), entrada["modelo"].predict_proba(X)
    else:                                   # constante: las proporciones de entrenamiento, iguales para todas
        clases = np.array(list(entrada["probabilidades"]), dtype=object)
        proba = np.tile([entrada["probabilidades"][k] for k in clases], (len(X), 1))
    pred, p = clases[proba.argmax(axis=1)], proba.max(axis=1)
    es_sc = clases == SUBTIPO_SIN_CLASIFICAR
    if umbral is not None and es_sc.any() and (~es_sc).any():
        reales = np.where(es_sc[None, :], -1.0, proba)
        cambia = (pred == SUBTIPO_SIN_CLASIFICAR) & (proba[:, es_sc][:, 0] < umbral)
        pred, p = np.where(cambia, clases[reales.argmax(axis=1)], pred), np.where(cambia, reales.max(axis=1), p)
    return pred, p


def predecir_cascada(modelos, X, familia_dada=None, crudo=False, umbral=None):
    """Nivel 1 → familia; nivel 2 → subtipo con el modelo de esa familia. familia_dada: la cascada ideal.
    crudo=True: sin umbral. umbral: otro umbral para probar; None usa el guardado."""
    X = X[modelos["predictoras"]]
    umbral = None if crudo else (umbral if umbral is not None else modelos.get("umbral_sin_clasificar"))
    familia = np.asarray(familia_dada) if familia_dada is not None else modelos["familia"].predict(X)
    p_familia = modelos["familia"].predict_proba(X).max(axis=1)
    subtipo, p_subtipo = np.full(len(X), None, dtype=object), np.full(len(X), np.nan)
    for f in np.unique(familia):
        m = familia == f
        subtipo[m], p_subtipo[m] = predecir_subtipo(modelos["subtipo"][f], X[m], umbral)
    return pd.DataFrame({"familia predicha": familia, "p̂ familia": p_familia, "subtipo predicho": subtipo,
                         "p̂ subtipo": p_subtipo}, index=X.index)


def camino(arbol, fila):
    """Las preguntas que recorre una fila (DataFrame de una fila), con sus valores, y la hoja a la que llega."""
    t = arbol.tree_
    nodos = arbol.decision_path(fila[PREDICTORAS]).indices
    pasos = []
    for nodo, siguiente in zip(nodos[:-1], nodos[1:]):
        col, corte = PREDICTORAS[t.feature[nodo]], t.threshold[nodo]
        valor = fila[col].iloc[0]
        respuesta = "sí" if siguiente == t.children_left[nodo] else "no"
        if np.isinf(corte):                 # umbral infinito: la pregunta es solo si se midió
            pasos.append(f"¿{col} medido? {'no' if pd.isna(valor) else 'sí'}")
        else:
            pasos.append(f"¿{col} ≤ {coma(corte, 2)}? {'no medido' if pd.isna(valor) else coma(valor, 2)}: {respuesta}")
    hoja = nodos[-1]
    cuentas = pd.Series((t.value[hoja][0] * t.weighted_n_node_samples[hoja]).round().astype(int), index=arbol.classes_)
    return pasos, hoja, cuentas[cuentas > 0].sort_values(ascending=False)


# ------------------------------------------------------------------ métricas
def metricas(y_real, y_pred, etiquetas=None):
    """etiquetas: las clases que entran en los promedios macro (None: las que aparecen)."""
    return {"exactitud": accuracy_score(y_real, y_pred),
            "precisión (macro)": precision_score(y_real, y_pred, labels=etiquetas, average="macro", zero_division=0),
            "exhaustividad (macro)": recall_score(y_real, y_pred, labels=etiquetas, average="macro", zero_division=0),
            "F1 (macro)": f1_score(y_real, y_pred, labels=etiquetas, average="macro", zero_division=0)}


def sin_sc(*ys):
    """Las clases que aparecen en ys, sin SIN CLASIFICAR: las que promedia la versión (b)."""
    return sorted(set().union(*map(set, ys)) - {SUBTIPO_SIN_CLASIFICAR})


def familia_vs_base(modelos, ent, pru):
    """Primer nivel sobre la prueba, contra la línea de base que siempre dice la familia más común."""
    base = DummyClassifier(strategy="most_frequent").fit(ent[PREDICTORAS], ent[FAMILIA])
    return pd.DataFrame({"línea de base (mayoritaria)": metricas(pru[FAMILIA], base.predict(pru[PREDICTORAS])),
                         "árbol de familia": metricas(pru[FAMILIA], modelos["familia"].predict(pru[PREDICTORAS]))}).T


def tabla_subtipo(modelos, busquedas, plan, ent, pru, regla):
    """Segundo nivel con la familia real y la predicción cruda: (a) todas las clases; (b) solo subtipo real."""
    filas = []
    for familia, fila in plan.iterrows():
        entrada = modelos["subtipo"][familia]
        if entrada["tipo"] == "sin subtipo":
            continue
        en_pru = (pru[FAMILIA] == familia) & pru[AGRUPADO].notna()
        ys_pru = pru.loc[en_pru, AGRUPADO]
        pred, _ = predecir_subtipo(entrada, pru.loc[en_pru, PREDICTORAS])
        y_ent = y_subtipo(ent, familia, regla)
        comun = y_ent.value_counts().index[0] if len(y_ent) else entrada.get("clase")
        base = np.full(len(ys_pru), comun, dtype=object)
        real = pru.loc[en_pru, SUBTIPO].notna().to_numpy()
        a, b = metricas(ys_pru, pred), metricas(ys_pru[real], pred[real], sin_sc(ys_pru[real], pred[real]))
        extra = {}
        if entrada["tipo"] == "árbol":
            g = busquedas[familia]
            extra = {"max_depth": g.best_params_["max_depth"], "min_samples_leaf": g.best_params_["min_samples_leaf"],
                     "pliegues": g.cv.get_n_splits(), "F1 macro validación": g.best_score_,
                     "hojas": g.best_estimator_.get_n_leaves()}
        filas.append({"familia": familia, "tipo": entrada["tipo"], "clases": fila["clases tras agrupar"],
                      "entrenamiento": len(y_ent), "prueba (a)": len(ys_pru), "prueba (b)": int(real.sum()), **extra,
                      "exactitud (a)": a["exactitud"], "F1 macro (a)": a["F1 (macro)"],
                      "F1 macro base (a)": metricas(ys_pru, base)["F1 (macro)"],
                      "exactitud (b)": b["exactitud"], "F1 macro (b)": b["F1 (macro)"],
                      "F1 macro base (b)": metricas(ys_pru[real], base[real], sin_sc(ys_pru[real], base[real]))["F1 (macro)"],
                      "% reales → SIN CLASIFICAR": (pred[real] == SUBTIPO_SIN_CLASIFICAR).mean() if real.any() else np.nan})
    return pd.DataFrame(filas).set_index("familia")


def evaluar(modelos, pru, crudo=False, umbral=None, por_familia=False):
    """Las métricas de la cascada sobre la prueba (punta a punta y con la familia real), en las dos versiones:
    (a) todas las clases, SIN CLASIFICAR incluida; (b) solo las muestras con subtipo real."""
    cas = predecir_cascada(modelos, pru, crudo=crudo, umbral=umbral)
    ide = predecir_cascada(modelos, pru, familia_dada=pru[FAMILIA], crudo=crudo, umbral=umbral)
    s_cas, s_ide = cas["subtipo predicho"].fillna("—"), ide["subtipo predicho"].fillna("—")
    ok_f = cas["familia predicha"] == pru[FAMILIA]
    con_sub, con_real = pru[AGRUPADO].notna(), pru[SUBTIPO].notna()
    tiene_sc = modelos.get("subtipo_sin_clasificar") is not None
    fila = {"familia: exactitud": accuracy_score(pru[FAMILIA], cas["familia predicha"]),
            "familia: F1 macro": f1_score(pru[FAMILIA], cas["familia predicha"], average="macro", zero_division=0)}
    for v, m in [("(a)", con_sub), ("(b)", con_real)]:
        nulo = v == "(a)" and not tiene_sc            # V1 no tiene la clase SIN CLASIFICAR: (a) no existe
        fila[f"punta a punta {v}"] = np.nan if nulo else (ok_f & (s_cas == pru[AGRUPADO]))[m].mean()
        fila[f"con familia real {v}"] = np.nan if nulo else (s_ide == pru[AGRUPADO])[m].mean()
    fila["reales → SIN CLASIFICAR"] = (s_cas == SUBTIPO_SIN_CLASIFICAR)[con_real].mean() if tiene_sc else np.nan
    fila["reales → SIN CLASIFICAR (familia real)"] = (s_ide == SUBTIPO_SIN_CLASIFICAR)[con_real].mean() if tiene_sc else np.nan
    if por_familia:
        for familia in sorted(pru.loc[con_sub, FAMILIA].unique()):
            en = (pru[FAMILIA] == familia) & con_sub
            fila[f"F1 subtipo (a) · {familia}"] = f1_score(pru.loc[en, AGRUPADO], s_ide[en], average="macro", zero_division=0)
            re_ = en & con_real
            if re_.any():
                fila[f"F1 subtipo (b) · {familia}"] = metricas(pru.loc[re_, AGRUPADO], s_ide[re_],
                                                              sin_sc(pru.loc[re_, AGRUPADO], s_ide[re_]))["F1 (macro)"]
    return fila


def muestra_a_muestra(modelos, pru):
    """Una fila por muestra de prueba (por Identificador): reales, predichos y aciertos (sin descripción)."""
    cas = predecir_cascada(modelos, pru)
    ok_f = cas["familia predicha"] == pru[FAMILIA]
    return pd.DataFrame({"familia real": pru[FAMILIA], "subtipo real": pru[AGRUPADO], "con subtipo real": pru[SUBTIPO].notna(),
                         "familia predicha": cas["familia predicha"], "subtipo predicho": cas["subtipo predicho"],
                         "acierto familia": ok_f, "acierto subtipo": ok_f & (cas["subtipo predicho"] == pru[AGRUPADO])}
                        ).set_axis(pru["Identificador"].astype(str).to_numpy())


def errores_frecuentes(t, n=15):
    """Los n pares real → predicho más repetidos: familia (toda la prueba) y subtipo (familia bien predicha)."""
    f = t[~t["acierto familia"]].groupby(["familia real", "familia predicha"]).size()
    s = t[t["acierto familia"] & t["subtipo real"].notna() & ~t["acierto subtipo"]]
    s = s.assign(**{"subtipo predicho": s["subtipo predicho"].fillna("—")}).groupby(["familia real", "subtipo real", "subtipo predicho"]).size()
    return (f.sort_values(ascending=False).head(n).rename("muestras").reset_index(),
            s.sort_values(ascending=False).head(n).rename("muestras").reset_index())


def roc_ovr(y_real, proba, clases):
    """ROC uno contra el resto. Omite las clases sin positivos (o sin negativos) en y_real."""
    y_real = np.asarray(y_real)
    Y = label_binarize(y_real, classes=clases)
    if len(clases) == 2:                    # label_binarize da una sola columna con dos clases
        Y = np.column_stack([1 - Y[:, 0], Y[:, 0]])
    curvas, omitidas = [], []
    for k, clase in enumerate(clases):
        pos = int(Y[:, k].sum())
        if pos == 0 or pos == len(Y):
            omitidas.append(clase)
            continue
        fpr, tpr, _ = roc_curve(Y[:, k], proba[:, k])
        curvas.append({"clase": str(clase), "auc": float(auc(fpr, tpr)), "soporte": pos, "fpr": fpr, "tpr": tpr})
    cols = [k for k, c in enumerate(clases) if c not in omitidas]
    if not cols:
        return None, omitidas
    fpr_m, tpr_m, _ = roc_curve(Y[:, cols].ravel(), proba[:, cols].ravel())
    return {"auc_macro": float(np.mean([c["auc"] for c in curvas])), "auc_micro": float(auc(fpr_m, tpr_m)),
            "fpr": fpr_m, "tpr": tpr_m, "clases": curvas}, omitidas


def compartidas(tablas):
    """Muestras de prueba que están en las tres versiones (por Identificador): transiciones acierto / error."""
    ids = sorted(set.intersection(*(set(t.index) for t in tablas.values())))
    S = {v: t.loc[ids] for v, t in tablas.items()}
    real_b = np.logical_and.reduce([S[v]["con subtipo real"].to_numpy() for v in S])

    def trans(a, b, col, mascara):
        A, B = S[a].loc[mascara, col].astype(bool), S[b].loc[mascara, col].astype(bool)
        return pd.Series({"acierto → acierto": int((A & B).sum()), "acierto → error": int((A & ~B).sum()),
                          "error → acierto": int((~A & B).sum()), "error → error": int((~A & ~B).sum())})

    todas = np.ones(len(ids), dtype=bool)
    pares = [("V1", "V2"), ("V2", "V3"), ("V1", "V3")]
    fam = pd.DataFrame({f"{a} → {b}": trans(a, b, "acierto familia", todas) for a, b in pares})
    sub = pd.DataFrame({f"{a} → {b}": trans(a, b, "acierto subtipo", real_b) for a, b in pares})
    for a, b in pares:
        A, B = S[a].loc[real_b, "acierto subtipo"], S[b].loc[real_b, "acierto subtipo"]
        sub.loc["de esas, acierto → SIN CLASIFICAR", f"{a} → {b}"] = int((A & ~B & (S[b].loc[real_b, "subtipo predicho"] == SUBTIPO_SIN_CLASIFICAR)).sum())
    neto = {}
    for nombre, col, m in [("familia", "acierto familia", todas), ("subtipo (b)", "acierto subtipo", real_b)]:
        g = S["V1"].loc[m, "familia real"]
        neto[nombre] = pd.DataFrame({"muestras": g.value_counts(),
                                     **{f"{a} → {b}": (S[b].loc[m, col].astype(int) - S[a].loc[m, col].astype(int)).groupby(g).sum()
                                        for a, b in pares[:2]}}).sort_values("muestras", ascending=False)
    exactitud = pd.DataFrame({v: {"familia": S[v]["acierto familia"].mean(),
                                  "familia y subtipo (b)": S[v].loc[real_b, "acierto subtipo"].mean()} for v in S})
    return SimpleNamespace(n=len(ids), n_b=int(real_b.sum()), familia=fam, subtipo=sub.astype(int), neto=neto,
                           exactitud=exactitud)


def por_envio(D, params):
    """El árbol de familia de V3 (mismos parámetros) con 5 pliegues al azar y con 5 pliegues agrupados por envío,
    sobre todo lo que modela V3 (arc + MQ). Mide cuánto ayuda que muestras del mismo envío caigan a los dos lados."""
    d = D.modelado
    X, y = d[PREDICTORAS], d[FAMILIA]
    grupos = d["Envío"].fillna(d["Identificador"]).astype(str)
    arbol = DecisionTreeClassifier(random_state=SEMILLA, **params)
    filas, tam = {}, {}
    for nombre, cv in [("al azar (estratificada)", StratifiedKFold(N_PLIEGUES, shuffle=True, random_state=SEMILLA)),
                       ("agrupada por envío", StratifiedGroupKFold(N_PLIEGUES, shuffle=True, random_state=SEMILLA))]:
        g = grupos if isinstance(cv, StratifiedGroupKFold) else None
        r = cross_validate(arbol, X, y, groups=g, cv=cv, scoring=["accuracy", "f1_macro"], n_jobs=-1)
        filas[nombre] = {"exactitud": r["test_accuracy"].mean(), "F1 macro": r["test_f1_macro"].mean(),
                         "exactitud (desvío entre pliegues)": r["test_accuracy"].std()}
        tam[nombre] = [(len(a), len(b)) for a, b in cv.split(X, y, g)]
    return pd.DataFrame(filas).T, tam, grupos.nunique()


def guardar(modelos):
    R.preparar()
    joblib.dump(modelos, SALIDA_MODELO)
    return SALIDA_MODELO
