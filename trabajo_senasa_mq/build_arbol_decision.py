"""Arma arbol_decision.ipynb: árbol de decisión en cascada (familia → subtipo), las tres versiones del compañero.

    cd trabajo_senasa_mq
    PYTHONIOENCODING=utf-8 ../.venv/Scripts/python.exe build_arbol_decision.py
    ../.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace arbol_decision.ipynb

El código está en src/arbol_decision.py; el notebook solo lo llama y muestra. Al final escribe y abre app_arbol.py
(celdas de src/celdas_app.py). Tarda unos 3 minutos. Es la Parte VII del cuaderno general.
"""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import celdas_app  # noqa: E402

SALIDA = Path(__file__).resolve().parent / "arbol_decision.ipynb"
APP = Path(__file__).resolve().parent / "app_arbol.py"
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: c.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Árbol de decisión en cascada: familia y subtipo

**Qué es.** El trabajo de otro integrante del grupo, llevado al proyecto con el mismo código y las mismas reglas.
Con solo los análisis de composición de una muestra (humedad, grasa, proteínas, cenizas, sólidos totales, lactosa y
galactosa), un primer árbol predice la **familia** del producto y, dentro de esa familia, un segundo árbol predice el
**subtipo**. Los análisis que no se midieron quedan vacíos: el árbol de scikit-learn aprende hacia qué lado mandarlos.

**Las tres versiones.**
- **V1, solo arc.** `dataset_senasa_revision.xlsx` (Senasa revisada). En cada árbol de subtipo se dejan afuera las
  clases con menos de 5 muestras de entrenamiento.
- **V2, arc con la clase SIN CLASIFICAR.** En las familias con subtipos, las muestras sin subtipo forman una clase más.
- **V3, arc + MQ.** El 80 % de arc más todas las MQ nuevas, ya filtradas, entrenan. SIN CLASIFICAR solo se responde si
  su probabilidad llega a 0,85; si no, se da el subtipo real más probable. Es la versión final.

En las tres:
- la prueba es un 20 % de arc, al azar, estratificado por familia y subtipo;
- cada árbol se elige por validación cruzada de 5 pliegues (F1 macro) sobre una cuadrícula de `max_depth` y
  `min_samples_leaf`;
- dentro de cada familia, los subtipos con menos de 20 muestras pasan a «OTRO».

**Qué cambia respecto de los cuadernos originales.**
- Semilla 42, como el resto del trabajo (el original usaba 2026).
- Rutas del proyecto.
- El filtro «declarado vs medido» sale del mismo cálculo del proyecto.
- Las descripciones de las muestras se usan solo para los filtros: no se muestran.

Quedan afuera:
- la exportación a JSON para la página web;
- el modelo guardado que venía con el trabajo (acá se vuelve a entrenar);
- las tablas con ejemplos de descripciones;
- la parte de `Algoritmos.ipynb` que repite a V3.
""")

code(r"""
import sys
import time
import warnings
from pathlib import Path
SRC = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "rutas.py").exists()) / "src"
sys.path.insert(0, str(SRC))
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.tree import plot_tree
import arbol_decision as AD
import datos_entrenamiento as DE
from arbol_decision import AGRUPADO, FAMILIA, PREDICTORAS, SUBTIPO, coma
from arbol_decision import SUBTIPO_SIN_CLASIFICAR as SC

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)   # clases que un pliegue no predice nunca
AZUL, AMBAR, GRIS, ROJO, VERDE = "#0055a5", "#f0a800", "#6b7280", "#c0392b", "#2e8b57"
COLORES = {"V1": GRIS, "V2": AMBAR, "V3": AZUL}
inicio = time.time()
print(f"Semilla {AD.SEMILLA} · prueba {coma(AD.PRUEBA * 100, 0)} % de arc · umbral de {SC} en V3: {coma(AD.UMBRAL_SIN_CLASIFICAR, 2)}")
""")

md("""
## 1. Datos

**Qué se saca.**
- V1 y V2: las filas de arc sin ningún análisis y las atípicas (o excluidas porque humedad + sólidos no da 100).
- V3 aplica además los filtros de MQ, en orden. La tabla muestra cuántas filas saca cada uno, por origen y por familia;
  cada fila ya descontó las anteriores.

**Después de filtrar.**
- MQ solo conserva los subtipos que existen en arc para esa familia.
- La familia «sin clasificar» queda aparte, para predecirla al final.
""")

code(r"""
D = {v: AD.preparar(v) for v in AD.VERSIONES}
D3 = D["V3"]
resumen_datos = pd.DataFrame({AD.TITULOS[v]: {
    "filas de las planillas": d.disponibles, "para modelar": len(d.modelado), "de MQ": len(d.muestras_mq),
    "«sin clasificar» (aparte)": len(d.sin_clasificar), "entrenamiento": len(d.ent), "prueba (arc)": len(d.pru),
    "con subtipo real": int(d.modelado[SUBTIPO].notna().sum()), f"pasan a {SC}": d.pasan_sc} for v, d in D.items()})
display(resumen_datos.style.format("{:.0f}").set_caption("Las tres versiones: cuántas muestras usa cada una"))
c1 = D["V1"].conteos
print("V1 y V2: " + " · ".join(f"{i.lstrip('− ')}: {c1.loc[i, 'arc']}" for i in c1.index))
display(D3.conteos.style.set_caption("V3: filas que saca cada filtro, en orden (por origen y por familia, los dos orígenes juntos)"))
print("V3, subtipos de MQ que no están en arc (quedan sin subtipo):",
      ", ".join(f"{f} / {s} ({n})" for (f, s), n in D3.sin_par.items()) or "ninguno")
m3 = D3.modelado
display(pd.concat([m3[FAMILIA].value_counts().rename("muestras"), D3.muestras_mq[FAMILIA].value_counts().rename("de MQ"),
                   m3.groupby(FAMILIA)[SUBTIPO].apply(lambda s: s.notna().sum()).rename("con subtipo"),
                   m3.groupby(FAMILIA)[AGRUPADO].apply(lambda s: (s == SC).sum()).rename(SC),
                   m3.groupby(FAMILIA)[PREDICTORAS].apply(lambda d: d.notna().mean() * 100).round(0).add_prefix("% medido: ")], axis=1)
        .fillna(0).style.format(precision=0).background_gradient(subset=[f"% medido: {p}" for p in PREDICTORAS], cmap="Blues", vmin=0, vmax=100)
        .set_caption("V3 (arc + MQ): muestras por familia, con subtipo, que pasan a SIN CLASIFICAR y % medido de cada análisis"))
crema = m3[AGRUPADO] == "CREMA"
print(f"CREMA (manteca / crema) solo está en MQ: entrena {crema.loc[D3.ent.index].sum()} · prueba {crema.loc[D3.pru.index].sum()}: no tiene evaluación")
""")

md("""
## 2. Entrenamiento de las tres versiones

Cada versión entrena su cascada con su propia regla para el segundo nivel.
- **V1.** Las clases con menos de 5 muestras de entrenamiento quedan afuera: no entrenan, pero en la prueba siguen
  estando y cuentan como error.
- **V2 y V3.** Árbol si hay al menos dos clases con dos muestras cada una. Constante si hay una sola clase, o si alguna
  tiene una sola muestra.

En las tres, las familias sin subtipos terminan en el primer nivel. La tabla es el plan de V3.
""")

code(r"""
t0 = time.time()
E = {v: AD.entrenar(D[v]) for v in AD.VERSIONES}          # (modelos, búsquedas, plan) de cada versión
modelos, busquedas, plan = E["V3"]
pru, ent = D3.pru, D3.ent
print(f"Las tres cascadas, entrenadas en {coma(time.time() - t0, 0)} s · árboles de subtipo: "
      + " · ".join(f"{v} {sum(e['tipo'] == 'árbol' for e in E[v][0]['subtipo'].values())}" for v in AD.VERSIONES))
with pd.option_context("display.max_colwidth", 160):
    display(plan.style.set_caption("V3: qué modelo lleva cada familia (clases con sus muestras de entrenamiento)"))
afuera = E["V1"][2][f"afuera (< {AD.MINIMO_POR_CLASE})"]
print("V1, clases que quedan afuera por tener menos de 5 muestras: " + "; ".join(f"{f}: {a}" for f, a in afuera.items() if isinstance(a, str) and a != "—"))
""")

md("""
## 3. Primer nivel: la familia (V3)

Se compara con la línea de base que siempre responde la familia más común. El mapa de calor muestra el F1 macro de
validación de cada combinación de la cuadrícula.
""")

code(r"""
cf = busquedas["familia"]
arbol_familia = modelos["familia"]
pred_familia = arbol_familia.predict(pru[PREDICTORAS])
resultados_familia = AD.familia_vs_base(modelos, ent, pru)
print(f"Mejor combinación: {cf.best_params_} · F1 macro de validación {coma(cf.best_score_)}")
print(f"Árbol de familia: profundidad {arbol_familia.get_depth()} · {arbol_familia.get_n_leaves()} hojas")
display(resultados_familia.style.format(precision=3, decimal=",").set_caption(f"Primer nivel sobre las {len(pru)} muestras de prueba"))
mapa = pd.DataFrame(cf.cv_results_).pivot(index="param_min_samples_leaf", columns="param_max_depth", values="mean_test_score")
mapa.columns = [str(int(v)) if pd.notna(v) else "∞" for v in mapa.columns]
mapa.index.name, mapa.columns.name = "min_samples_leaf", "max_depth"
display(mapa.style.format(precision=3, decimal=",").background_gradient(cmap="RdYlGn", axis=None)
        .set_caption("F1 macro de validación: filas min_samples_leaf, columnas max_depth"))
rep = classification_report(pru[FAMILIA], pred_familia, zero_division=0, output_dict=True)
reporte_familia = pd.DataFrame([{"clase": k, "precisión": v["precision"], "sensibilidad": v["recall"], "F1": v["f1-score"],
                                 "soporte": int(v["support"])} for k, v in rep.items()
                                if isinstance(v, dict) and k not in ("macro avg", "weighted avg")]).set_index("clase")
display(reporte_familia.style.format(precision=3, decimal=",").format("{:.0f}", subset=["soporte"])
        .background_gradient(subset=["F1"], cmap="RdYlGn", vmin=0, vmax=1).set_caption("Reporte de clasificación de la familia (prueba)"))
fig, ax = plt.subplots(figsize=(11, 9))
ConfusionMatrixDisplay.from_predictions(pru[FAMILIA], pred_familia, cmap="Blues", xticks_rotation=90, colorbar=False, ax=ax)
ax.set_xlabel("familia predicha"); ax.set_ylabel("familia real")
ax.set_title(f"Primer nivel sobre prueba: exactitud {coma(resultados_familia.loc['árbol de familia', 'exactitud'])} · "
             f"F1 macro {coma(resultados_familia.loc['árbol de familia', 'F1 (macro)'])}")
plt.tight_layout(); plt.show()
""")

md("### Cómo es el árbol de familia")
code(r"""
fig, ax = plt.subplots(figsize=(22, 9))
plot_tree(arbol_familia, max_depth=3, feature_names=PREDICTORAS, class_names=list(arbol_familia.classes_), filled=True,
          rounded=True, fontsize=7, impurity=False, proportion=False, ax=ax)
ax.set_title(f"Árbol de familia: los tres primeros niveles de {arbol_familia.get_depth()} (color: familia mayoritaria del nodo)")
plt.show()
importancias = pd.Series(arbol_familia.feature_importances_, index=PREDICTORAS).sort_values()
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.barh(importancias.index, importancias.values, color=AZUL)
for k, v in enumerate(importancias.values):
    ax.text(v + 0.005, k, coma(v), va="center", fontsize=8)
ax.set_xlabel("fracción de la reducción total de impureza"); ax.set_title("Importancia de cada análisis en el árbol de familia")
plt.tight_layout(); plt.show()
t_ = arbol_familia.tree_
print(f"Raíz: ¿{PREDICTORAS[t_.feature[0]]} ≤ {coma(t_.threshold[0], 2)}? · las faltantes van a la "
      f"{'izquierda' if t_.missing_go_to_left[0] else 'derecha'}")
""")

md("""
### Curvas ROC de la familia

Una curva por familia (uno contra el resto) con las probabilidades del árbol, más la curva micro (todas las clases
juntas). Un árbol da pocas probabilidades distintas, por eso las curvas tienen pocos escalones.
""")
code(r"""
roc_familia, omitidas_familia = AD.roc_ovr(pru[FAMILIA], arbol_familia.predict_proba(pru[PREDICTORAS]), arbol_familia.classes_)
fig, ax = plt.subplots(figsize=(9, 7.5))
colores = plt.cm.tab20(np.linspace(0, 1, len(roc_familia["clases"])))
for cu, col in zip(sorted(roc_familia["clases"], key=lambda x: -x["soporte"]), colores):
    ax.plot(cu["fpr"], cu["tpr"], color=col, lw=1.3, label=f"{cu['clase']} (AUC {coma(cu['auc'])}, n={cu['soporte']})")
ax.plot(roc_familia["fpr"], roc_familia["tpr"], color="black", lw=2.5, ls="--", label=f"micro (AUC {coma(roc_familia['auc_micro'])})")
ax.plot([0, 1], [0, 1], color=GRIS, lw=0.8, ls=":")
ax.set_xlabel("tasa de falsos positivos"); ax.set_ylabel("tasa de verdaderos positivos (sensibilidad)")
ax.set_title(f"ROC de la familia (uno contra el resto) · AUC macro {coma(roc_familia['auc_macro'])} · micro {coma(roc_familia['auc_micro'])}")
ax.legend(fontsize=7.5, loc="lower right"); plt.tight_layout(); plt.show()
print(f"AUC familia: macro {coma(roc_familia['auc_macro'])} · micro {coma(roc_familia['auc_micro'])}"
      + (f" · sin muestras en la prueba: {', '.join(map(str, omitidas_familia))}" if omitidas_familia else ""))
""")

md("""
## 4. Segundo nivel: el subtipo dentro de cada familia (V3)

Cada árbol se mide sobre las muestras de prueba de su familia **real**: es la calidad del segundo nivel aislado. Esta
tabla usa la predicción cruda del árbol, sin el umbral de SIN CLASIFICAR.
- **(a)** todas las clases, SIN CLASIFICAR incluida.
- **(b)** solo las muestras con subtipo real, que es lo comparable con V1.
""")
code(r"""
resultados_subtipo = AD.tabla_subtipo(modelos, busquedas, plan, ent, pru, "V3")
CON_ARBOL = list(resultados_subtipo.index[resultados_subtipo["tipo"] == "árbol"])
tabla_s = resultados_subtipo.copy()
tabla_s["max_depth"] = [("∞" if pd.isna(v) else f"{int(v)}") if t == "árbol" else "—" for v, t in zip(tabla_s["max_depth"], tabla_s["tipo"])]
ENTEROS = ["min_samples_leaf", "pliegues", "hojas"]
display(tabla_s.style.format({k: (lambda v: "—" if pd.isna(v) else f"{int(v)}") for k in ENTEROS})
        .format(precision=3, decimal=",", na_rep="—", subset=[k for k in tabla_s.columns if k not in ["tipo", "max_depth", *ENTEROS]])
        .background_gradient(subset=["F1 macro (a)", "F1 macro (b)"], cmap="RdYlGn", vmin=0, vmax=1)
        .set_caption("Segundo nivel por familia, con la familia real y sin umbral: (a) todas las clases; (b) solo subtipo real"))
tabla_imp = pd.DataFrame({"familia (nivel 1)": arbol_familia.feature_importances_,
                          **{f: modelos["subtipo"][f]["modelo"].feature_importances_ for f in CON_ARBOL}}, index=PREDICTORAS).T
display(tabla_imp.style.format(precision=2, decimal=",").background_gradient(cmap="Blues", axis=1, vmin=0, vmax=1)
        .set_caption("Importancia de cada análisis en cada árbol (cada fila suma 1)"))
""")

md("""
### ROC, matrices y reportes del subtipo

Son las muestras de prueba de cada familia real que tienen subtipo agrupado, sea un subtipo real o SIN CLASIFICAR.
- La ROC usa las probabilidades del árbol tal como salen.
- Las matrices y los reportes usan la predicción **con el umbral de 0,85**.
""")
code(r"""
roc_subtipo, conf_subtipo, reportes_subtipo, sin_prueba = {}, {}, {}, []
n_ = len(CON_ARBOL); columnas = 3; filas_g = int(np.ceil(n_ / columnas))
fig, ejes = plt.subplots(filas_g, columnas, figsize=(5 * columnas, 4.4 * filas_g), squeeze=False)
for ax, familia in zip(ejes.ravel(), CON_ARBOL):
    arbol = modelos["subtipo"][familia]["modelo"]
    en_pru = (pru[FAMILIA] == familia) & pru[AGRUPADO].notna()
    Xf, yf = pru.loc[en_pru, PREDICTORAS], pru.loc[en_pru, AGRUPADO]
    r, omit = AD.roc_ovr(yf, arbol.predict_proba(Xf), arbol.classes_) if len(yf) else (None, [])
    if r is None:
        sin_prueba.append(familia); ax.axis("off"); continue
    roc_subtipo[familia] = r
    for cu in r["clases"]:
        ax.plot(cu["fpr"], cu["tpr"], lw=1.1, label=f"{cu['clase'][:18]} ({coma(cu['auc'], 2)})")
    ax.plot(r["fpr"], r["tpr"], color="black", lw=2, ls="--", label=f"micro ({coma(r['auc_micro'], 2)})")
    ax.plot([0, 1], [0, 1], color=GRIS, lw=0.8, ls=":")
    ax.set_title(f"{familia} · n={len(yf)} · AUC macro {coma(r['auc_macro'], 2)}", fontsize=9)
    ax.legend(fontsize=6, loc="lower right"); ax.tick_params(labelsize=7)
    pred_u, _ = AD.predecir_subtipo(modelos["subtipo"][familia], Xf, AD.UMBRAL_SIN_CLASIFICAR)
    etiquetas = list(arbol.classes_)
    conf_subtipo[familia] = (etiquetas, confusion_matrix(yf, pred_u, labels=etiquetas))
    rep = classification_report(yf, pred_u, labels=etiquetas, zero_division=0, output_dict=True)
    reportes_subtipo[familia] = pd.DataFrame([{"clase": k, "precisión": rep[k]["precision"], "sensibilidad": rep[k]["recall"],
                                               "F1": rep[k]["f1-score"], "soporte": int(rep[k]["support"])} for k in etiquetas]).set_index("clase")
for ax in ejes.ravel()[n_:]:
    ax.axis("off")
fig.supxlabel("tasa de falsos positivos"); fig.supylabel("tasa de verdaderos positivos")
fig.suptitle("ROC del subtipo por familia (árbol aislado, uno contra el resto)", fontsize=12)
plt.tight_layout(); plt.show()
display(pd.DataFrame({f: {"muestras": sum(cu["soporte"] for cu in r["clases"]), "AUC macro": r["auc_macro"], "AUC micro": r["auc_micro"]}
                      for f, r in roc_subtipo.items()}).T
        .style.format(precision=3, decimal=",", subset=["AUC macro", "AUC micro"]).format("{:.0f}", subset=["muestras"])
        .set_caption("AUC del subtipo por familia"))
CON_PRUEBA = list(conf_subtipo)
fig, ejes = plt.subplots(int(np.ceil(len(CON_PRUEBA) / columnas)), columnas, figsize=(6 * columnas, 5.5 * int(np.ceil(len(CON_PRUEBA) / columnas))), squeeze=False)
for ax, familia in zip(ejes.ravel(), CON_PRUEBA):
    etiquetas, matriz = conf_subtipo[familia]
    ConfusionMatrixDisplay(matriz, display_labels=[e[:16] for e in etiquetas]).plot(cmap="Blues", xticks_rotation=90, colorbar=False, ax=ax)
    ax.set_title(familia, fontsize=10); ax.set_xlabel("subtipo predicho", fontsize=8); ax.set_ylabel("subtipo real", fontsize=8)
    ax.tick_params(labelsize=6)
    for t in ax.texts:
        t.set_fontsize(6)
for ax in ejes.ravel()[len(CON_PRUEBA):]:
    ax.axis("off")
fig.suptitle(f"Matrices de confusión del subtipo con la familia real (umbral {coma(AD.UMBRAL_SIN_CLASIFICAR, 2)})", fontsize=12)
plt.tight_layout(); plt.show()
for familia in resultados_subtipo.loc[CON_PRUEBA, "prueba (a)"].sort_values(ascending=False).index[:4]:
    display(reportes_subtipo[familia].style.format(precision=3, decimal=",").format("{:.0f}", subset=["soporte"])
            .background_gradient(subset=["F1"], cmap="RdYlGn", vmin=0, vmax=1).set_caption(f"Reporte del subtipo · {familia}"))
if sin_prueba:
    print("Familias con árbol pero sin ROC (sin muestras de prueba o con una sola clase):", ", ".join(sin_prueba))
""")

md("""
## 5. La cascada de punta a punta (V3, umbral 0,85)

Primero se predice la familia y después el subtipo, con el modelo de esa familia. Si el primer árbol se equivoca, el
subtipo va a un árbol que no corresponde. Cuenta como acierto solo si familia y subtipo son correctos.

«Con familia real» es la cascada ideal, en la que el segundo nivel recibe la familia correcta. Las familias sin subtipos
quedan afuera.
""")
code(r"""
T = {v: AD.muestra_a_muestra(E[v][0], D[v].pru) for v in AD.VERSIONES}     # una fila por muestra de prueba (Identificador)
t3 = T["V3"]
ev3 = pd.Series(AD.evaluar(modelos, pru))
ev3_crudo = pd.Series(AD.evaluar(modelos, pru, crudo=True))
familia_comun = ent[FAMILIA].mode()[0]
subtipo_comun = ent.loc[ent[FAMILIA] == familia_comun, AGRUPADO].mode()[0]
base = ((pru[FAMILIA] == familia_comun) & (pru[AGRUPADO] == subtipo_comun))
con_sub, con_real = pru[AGRUPADO].notna(), pru[SUBTIPO].notna()
resumen_cascada = pd.DataFrame({
    "(a) todas las clases": {"muestras de prueba": con_sub.sum(), "familia correcta": t3["acierto familia"].to_numpy()[con_sub.to_numpy()].mean(),
                             "familia y subtipo (punta a punta)": ev3["punta a punta (a)"], "subtipo con familia real": ev3["con familia real (a)"],
                             f"línea de base ({familia_comun} → {subtipo_comun})": base[con_sub].mean()},
    "(b) solo subtipo real": {"muestras de prueba": con_real.sum(), "familia correcta": t3["acierto familia"].to_numpy()[con_real.to_numpy()].mean(),
                              "familia y subtipo (punta a punta)": ev3["punta a punta (b)"], "subtipo con familia real": ev3["con familia real (b)"],
                              f"línea de base ({familia_comun} → {subtipo_comun})": base[con_real].mean()}})
display(resumen_cascada.style.format(precision=3, decimal=",").format("{:.0f}", subset=pd.IndexSlice["muestras de prueba", :])
        .set_caption("De punta a punta sobre la prueba, en las dos versiones"))
for v in ["(a)", "(b)"]:
    print(f"{v}: familia y subtipo correctos {coma(ev3[f'punta a punta {v}'] * 100, 1)} % · con la familia real "
          f"{coma(ev3[f'con familia real {v}'] * 100, 1)} % · se pierden "
          f"{coma((ev3[f'con familia real {v}'] - ev3[f'punta a punta {v}']) * 100, 1)} puntos por errores del primer nivel")
print(f"De las {con_real.sum()} muestras de prueba con subtipo real, la cascada da {SC} a {coma(ev3['reales → SIN CLASIFICAR'] * 100, 1)} % "
      f"con el umbral de {coma(AD.UMBRAL_SIN_CLASIFICAR, 2)}; sin umbral serían {coma(ev3_crudo['reales → SIN CLASIFICAR'] * 100, 1)} %")
g = t3["familia real"].to_numpy()
m_a, m_b = con_sub.to_numpy(), con_real.to_numpy()
por_familia = pd.DataFrame({"muestras (a)": pd.Series(m_a).groupby(g).sum(), "muestras (b)": pd.Series(m_b).groupby(g).sum(),
                            "familia correcta (b)": t3["acierto familia"][m_b].groupby(g[m_b]).mean(),
                            "familia y subtipo (a)": t3["acierto subtipo"][m_a].groupby(g[m_a]).mean(),
                            "familia y subtipo (b)": t3["acierto subtipo"][m_b].groupby(g[m_b]).mean(),
                            f"reales → {SC}": (t3["subtipo predicho"] == SC)[m_b].groupby(g[m_b]).mean()})
por_familia = por_familia[por_familia["muestras (a)"] > 0]
display(por_familia.style.format(precision=3, decimal=",", na_rep="—").format("{:.0f}", subset=["muestras (a)", "muestras (b)"])
        .background_gradient(subset=["familia y subtipo (a)", "familia y subtipo (b)"], cmap="RdYlGn", vmin=0, vmax=1)
        .set_caption("De punta a punta, por familia real: (a) todas las clases; (b) solo subtipo real"))
""")

md("""
## 6. Errores más frecuentes (V3)

Los 15 pares real → predicho más repetidos en cada nivel.
- En la familia, sobre toda la prueba.
- En el subtipo, entre las muestras cuya familia la cascada acertó, con el umbral de 0,85.

El original mostraba descripciones de ejemplo; acá no se muestran.
""")
code(r"""
errores_familia, errores_subtipo = AD.errores_frecuentes(t3)
display(errores_familia.style.hide(axis="index").set_caption("Los errores más frecuentes de la familia"))
display(errores_subtipo.style.hide(axis="index").set_caption("Los errores más frecuentes del subtipo (familia bien predicha)"))
""")

md("""
## 7. Qué aporta MQ y qué aporta el umbral

Viene de la sección C.2 de `Algoritmos.ipynb`. Son dos cascadas medidas sobre la misma prueba, el 20 % de arc de V3:
- la de **solo arc**, entrenada con el 80 % de arc y la receta de V3;
- la de **arc + MQ**, que es la de V3.

Cada una se mide sin umbral, con 0,7 (solo como prueba) y con 0,85, el umbral que se guarda. Las dos usan las mismas
clases. En la tabla, el F1 de subtipo de cada familia se calcula con la familia real.
""")
code(r"""
t0 = time.time()
modelos_solo_arc, _, _ = AD.entrenar_cascada(D3.ent_arc, D3.familias, "V3", umbral=AD.UMBRAL_SIN_CLASIFICAR)
UMBRALES = [None, 0.7, AD.UMBRAL_SIN_CLASIFICAR]
comparacion = pd.DataFrame({f"{nombre} · {'sin umbral' if u is None else 'umbral ' + coma(u, 2).rstrip('0')}":
                            AD.evaluar(m, pru, crudo=u is None, umbral=u, por_familia=True)
                            for nombre, m in [("solo arc", modelos_solo_arc), ("arc + MQ", modelos)] for u in UMBRALES})
print(f"Entrenamiento: solo arc {len(D3.ent_arc)} · arc + MQ {len(ent)} · prueba, la misma para todas: {len(pru)} de arc "
      f"({coma(time.time() - t0, 0)} s)")
display(comparacion.style.format(precision=3, decimal=",", na_rep="—").background_gradient(cmap="RdYlGn", axis=1)
        .set_caption(f"Las dos cascadas con tres umbrales, sobre las {len(pru)} muestras de prueba de arc"))
""")

md("""
## 8. Las tres versiones, una al lado de la otra

Viene de `Arbol_comparacion.ipynb`. Las columnas «original» son los números de los cuadernos del compañero, que usaban
la semilla 2026: las de familia son las de la diapositiva 6 de la presentación. Las tres pruebas no son idénticas,
porque V3 filtra más; diferencias de 1 o 2 puntos no son concluyentes.
""")
code(r"""
EV = {v: AD.evaluar(E[v][0], D[v].pru) for v in AD.VERSIONES}
ROC = {v: AD.roc_ovr(D[v].pru[FAMILIA], E[v][0]["familia"].predict_proba(D[v].pru[PREDICTORAS]), E[v][0]["familia"].classes_)[0]
       for v in AD.VERSIONES}
filas_v = {}
for v in AD.VERSIONES:
    filas_v[v] = {**EV[v], "familia: AUC macro": ROC[v]["auc_macro"], "familia: AUC micro": ROC[v]["auc_micro"],
                  "muestras de prueba": len(D[v].pru)}
    filas_v[f"{v} original"] = {k: AD.ORIGINAL[v].get(k, np.nan) for k in filas_v[v]}
versiones = pd.DataFrame(filas_v).drop(index="reales → SIN CLASIFICAR (familia real)")
display(versiones.style.format(precision=3, decimal=",", na_rep="—").format("{:.0f}", subset=pd.IndexSlice["muestras de prueba", :], na_rep="—")
        .set_caption("Métricas principales de las tres versiones (sobre su propia prueba) y las del original"))
for v in AD.VERSIONES:
    print(f"{v}: exactitud de familia {coma(EV[v]['familia: exactitud'] * 100, 1)} % (original {coma(AD.ORIGINAL[v]['familia: exactitud'] * 100, 1)} %)")

fig, ejes = plt.subplots(1, 2, figsize=(14, 5.5))
for v in AD.VERSIONES:
    ejes[0].plot(ROC[v]["fpr"], ROC[v]["tpr"], color=COLORES[v], lw=2, label=f"{AD.TITULOS[v]} (micro {coma(ROC[v]['auc_micro'])}, macro {coma(ROC[v]['auc_macro'])})")
ejes[0].plot([0, 1], [0, 1], color=GRIS, lw=0.8, ls=":")
ejes[0].set(xlabel="tasa de falsos positivos", ylabel="tasa de verdaderos positivos", title="ROC micro de la familia")
ejes[0].legend(fontsize=8, loc="lower right")
f1_fam = pd.DataFrame({v: pd.Series({r_: d_["f1-score"] for r_, d_ in classification_report(
    D[v].pru[FAMILIA], E[v][0]["familia"].predict(D[v].pru[PREDICTORAS]), zero_division=0, output_dict=True).items()
    if isinstance(d_, dict) and r_ not in ("macro avg", "weighted avg")}) for v in AD.VERSIONES})
f1_fam = f1_fam.sort_values("V3")
y_ = np.arange(len(f1_fam))
for k, v in enumerate(AD.VERSIONES):
    ejes[1].barh(y_ + (k - 1) * 0.27, f1_fam[v], 0.27, color=COLORES[v], label=v)
ejes[1].set_yticks(y_, f1_fam.index, fontsize=8); ejes[1].set_xlim(0, 1)
ejes[1].set(xlabel="F1 en la prueba", title="F1 de cada familia, por versión"); ejes[1].legend(fontsize=8, loc="lower right")
plt.tight_layout(); plt.show()
""")

md("""
### Las mismas muestras en las tres versiones

Para comparar muestra por muestra se toman los Identificadores que están en **las tres** pruebas. Están cargadas
hacia las familias grandes, así que sirven para ver **qué cambió**, no para reemplazar las métricas de arriba.
- En el subtipo solo entran las que tienen subtipo real en las tres, es decir, el universo de (b).
- «acierto → SIN CLASIFICAR» cuenta las que la versión nueva dejó sin subtipo.
""")
code(r"""
comp = AD.compartidas(T)
print(f"Muestras de prueba compartidas por las tres versiones: {comp.n} · con subtipo real en las tres: {comp.n_b}")
display(comp.exactitud.style.format(precision=3, decimal=",").set_caption("Exactitud sobre las mismas muestras"))
display(comp.familia.style.set_caption("Familia: transiciones entre versiones (muestras compartidas)"))
display(comp.subtipo.style.set_caption("Familia y subtipo (b): transiciones entre versiones"))
fig, ejes = plt.subplots(1, 2, figsize=(14, 4.8))
for ax, (nombre, t_) in zip(ejes, comp.neto.items()):
    y_ = np.arange(len(t_))
    ax.barh(y_ - 0.2, t_["V1 → V2"], 0.4, color=COLORES["V2"], label="V1 → V2")
    ax.barh(y_ + 0.2, t_["V2 → V3"], 0.4, color=COLORES["V3"], label="V2 → V3")
    ax.set_yticks(y_, [f"{i} ({n})" for i, n in t_["muestras"].items()], fontsize=8); ax.invert_yaxis()
    ax.axvline(0, color="k", lw=0.8); ax.grid(axis="x", alpha=0.3); ax.legend(fontsize=8)
    ax.set(xlabel="aciertos ganados − perdidos", title=f"Cambio neto en aciertos de {nombre}")
plt.tight_layout(); plt.show()
""")

md("""
## 9. ¿Cuánto ayuda partir al azar?

Las tres versiones parten al azar. Pero las muestras de un mismo envío se parecen, porque son el mismo producto en días
seguidos, y al azar caen a los dos lados. En la Parte V (`prediccion_tipo.ipynb`) la validación se agrupa por envío.

Se toma el árbol de familia de V3, con los mismos parámetros, y se valida en 5 pliegues de dos maneras, sobre todo lo
que modela V3:
- al azar;
- agrupando por envío.
""")
code(r"""
envio, tam_envio, n_envios = AD.por_envio(D3, cf.best_params_)
display(envio.style.format(precision=3, decimal=",").set_caption(f"Árbol de familia de V3, 5 pliegues, {n_envios} envíos"))
caida = envio.loc["al azar (estratificada)", "exactitud"] - envio.loc["agrupada por envío", "exactitud"]
print(f"Agrupando por envío la exactitud baja {coma(caida * 100, 1)} puntos y el F1 macro "
      f"{coma((envio.iloc[0]['F1 macro'] - envio.iloc[1]['F1 macro']) * 100, 1)}.")
""")

md("""
## 10. Las muestras «sin clasificar»

Las muestras de la familia «sin clasificar», de arc y de MQ, no entrenan: acá la cascada de V3 les asigna familia y
subtipo. Para las de arc se muestra la cadena de preguntas del árbol de familia, identificadas por su código de
muestra.
""")
code(r"""
sc_todas = D3.sin_clasificar
pred_sc = AD.predecir_cascada(modelos, sc_todas)
pred_sc_crudo = AD.predecir_cascada(modelos, sc_todas, crudo=True)
print(f"«sin clasificar» predichas: {len(sc_todas)} (arc {(sc_todas['origen'] == 'arc').sum()} · MQ {(sc_todas['origen'] == 'MQ').sum()}) · "
      f"subtipo {SC}: sin umbral {(pred_sc_crudo['subtipo predicho'] == SC).sum()} → con umbral {(pred_sc['subtipo predicho'] == SC).sum()}")
display(pd.crosstab(pred_sc["familia predicha"], sc_todas["origen"], margins=True, margins_name="total")
        .sort_values("total", ascending=False).style.set_caption("Familia que la cascada da a las muestras «sin clasificar», por origen"))
display(pd.concat([sc_todas["origen"], pred_sc[["familia predicha", "subtipo predicho"]].fillna("—")], axis=1)
        .value_counts().head(15).rename("muestras").to_frame().style.set_caption("Las 15 combinaciones familia / subtipo más frecuentes"))
sc_arc = sc_todas[sc_todas["origen"] == "arc"]
display(pd.concat([sc_arc[["Identificador"] + PREDICTORAS], pred_sc.loc[sc_arc.index]], axis=1).set_index("Identificador")
        .style.format(precision=2, decimal=",", na_rep="—").set_caption(f"Las {len(sc_arc)} «sin clasificar» de arc, con lo que les da la cascada"))
for idx in sc_arc.index:
    pasos, hoja, cuentas = AD.camino(arbol_familia, sc_arc.loc[[idx]])
    print(f"{sc_arc.loc[idx, 'Identificador']} → {pred_sc.loc[idx, 'familia predicha']} / {pred_sc.loc[idx, 'subtipo predicho'] or '—'}")
    print("   " + " → ".join(pasos))
    print(f"   hoja {hoja}: " + ", ".join(f"{k} {v}" for k, v in cuentas.head(4).items()))
""")

md("""
## 11. Guardar el modelo

Se guarda la cascada de V3 en `resultados/modelos/arbol_decision.joblib`, que es lo que carga la aplicación. Después se
comprueba que, al cargarla, da las mismas predicciones.
""")
code(r"""
modelos["version"] = "V3"
ruta_modelo = AD.guardar(modelos)
cargado = joblib.load(ruta_modelo)
iguales = AD.predecir_cascada(cargado, pru).fillna("—").equals(AD.predecir_cascada(modelos, pru).fillna("—"))
print(f"Modelo guardado: {ruta_modelo.relative_to(ruta_modelo.parents[2]).as_posix()} · mismas predicciones tras guardar y cargar: {iguales}")
""")

md("""
## Datos usados en cada entrenamiento

Una fila por cada árbol de familia, con las mismas columnas que en las otras partes.
- **V3:** además, una fila por cada árbol de subtipo.
- **V1, V2 y la cascada de solo arc de la sección 7:** los árboles de subtipo van en una sola fila por versión.

Entrenamiento y prueba son los de la partición 80 / 20. La validación cruzada de 5 pliegues se hace dentro del
entrenamiento y solo sirve para elegir los parámetros.
""")
code(r"""
def miles(n):
    return f"{int(n):,}".replace(",", ".")

PARTICION = "80 / 20 al azar, estratificada por familia y subtipo (solo arc); 5 pliegues dentro del entrenamiento para elegir parámetros"
filas = []
for v in AD.VERSIONES:
    d, (m, b, p) = D[v], E[v]
    origenes = [o for o in ["arc", "MQ"] if o in d.conteos.columns]
    motivos = [f"{i.lstrip('− ').rsplit(' (', 1)[0]}: {miles(d.conteos.loc[i, origenes].sum())}"
               for i in d.conteos.index[1:-1] if d.conteos.loc[i, origenes].sum() > 0]
    motivos.append(f"familia «sin clasificar» (se predice al final): {miles(len(d.sin_clasificar))}")
    filas.append(DE.fila(
        parte="VII", modelo=f"{v}: árbol de familia", datos="arc" + (" + MQ nuevas" if v == "V3" else ""),
        disponibles=d.disponibles, usadas=len(d.modelado), motivo="; ".join(motivos), entrenamiento=len(d.ent), prueba=len(d.pru),
        particion=PARTICION, nota=("MQ entero va al entrenamiento; la prueba es solo de arc" if v == "V3" else "")))
    arboles = [f for f, e in m["subtipo"].items() if e["tipo"] == "árbol"]
    if v == "V3":
        for f in arboles:
            y = AD.y_subtipo(d.ent, f, v)
            en_pru = (d.pru[FAMILIA] == f) & d.pru[AGRUPADO].notna()
            filas.append(DE.fila(
                parte="VII", modelo=f"V3: árbol de subtipo · {f}", datos=f"muestras de {f} (arc + MQ)",
                disponibles=(d.modelado[FAMILIA] == f).sum(), usadas=len(y) + en_pru.sum(), entrenamiento=len(y), prueba=en_pru.sum(),
                particion=PARTICION, nota=f"incluye la clase {SC} ({int((y == SC).sum())} de entrenamiento)"))
    else:
        y_tot = sum(len(AD.y_subtipo(d.ent, f, v)) for f in arboles)
        p_tot = int(((d.pru[FAMILIA].isin(arboles)) & d.pru[AGRUPADO].notna()).sum())
        disp = int(d.modelado[FAMILIA].isin(arboles).sum())
        filas.append(DE.fila(
            parte="VII", modelo=f"{v}: {len(arboles)} árboles de subtipo (uno por familia)", datos="muestras de esas familias (arc)",
            disponibles=disp, usadas=y_tot + p_tot, entrenamiento=y_tot, prueba=p_tot, particion=PARTICION,
            motivo=("sin subtipo; en el entrenamiento, clases con menos de 5 muestras" if v == "V1" else ""),
            nota=("" if v == "V1" else f"incluye la clase {SC}")))
arboles_sa = [f for f, e in modelos_solo_arc["subtipo"].items() if e["tipo"] == "árbol"]
filas.append(DE.fila(parte="VII", modelo="V3 solo arc (sección 7): árbol de familia", datos="arc, con los filtros de V3",
                     disponibles=len(D3.muestras), usadas=len(D3.muestras), entrenamiento=len(D3.ent_arc), prueba=len(pru),
                     particion=PARTICION, nota="solo para comparar: no se guarda"))
y_sa = sum(len(AD.y_subtipo(D3.ent_arc, f, "V3")) for f in arboles_sa)
p_sa = int((pru[FAMILIA].isin(arboles_sa) & pru[AGRUPADO].notna()).sum())
filas.append(DE.fila(parte="VII", modelo=f"V3 solo arc (sección 7): {len(arboles_sa)} árboles de subtipo", datos="muestras de esas familias (arc)",
                     disponibles=int(D3.muestras[FAMILIA].isin(arboles_sa).sum()), usadas=y_sa + p_sa, entrenamiento=y_sa, prueba=p_sa,
                     particion=PARTICION, nota="solo para comparar: no se guarda"))
for nombre in envio.index:
    filas.append(DE.fila(parte="VII", modelo=f"comprobación: árbol de familia de V3, validación {nombre}", datos="arc + MQ nuevas (lo que modela V3)",
                         disponibles=len(D3.modelado), usadas=len(D3.modelado),
                         entrenamiento=round(np.mean([a for a, _ in tam_envio[nombre]])),
                         prueba=len(D3.modelado) - round(np.mean([a for a, _ in tam_envio[nombre]])),
                         particion=f"validación cruzada de 5 pliegues {nombre}", nota="entrenamiento y prueba son el promedio por pliegue"))
with pd.option_context("display.max_colwidth", 120):
    display(DE.guardar(filas, "arbol_decision"))
""")

md("""
## Lectura

**La familia sale bien con un solo árbol.**
- V3 acierta la familia en 8,9 de cada 10 muestras de prueba (exactitud 0,889, contra 0,257 de la línea de base);
  F1 macro 0,67; AUC macro 0,90 y micro 0,98.
- El árbol elegido es profundo: sin límite de profundidad y con 5 muestras por hoja como mínimo, llega a 37 niveles y
  unas 1 100 hojas. La primera pregunta es la humedad (≤ 30,6 %).
- Se acierta muy bien dulce de leche (F1 0,98), queso (0,94) y manteca / crema (0,93).
- Fallan las familias chicas o con pocos análisis: agua / hisopado y panificados no se aciertan nunca, y yogur tiene
  un F1 de 0,29.

**El subtipo es lo difícil.**
- De punta a punta, con el umbral de 0,85, familia y subtipo salen bien en el 65 % de las muestras con subtipo real
  (b), y en el 57 % si se cuenta SIN CLASIFICAR (a).
- Con la familia real serían 73 % y 64 %: los errores del primer nivel cuestan unos 7 u 8 puntos.
- Va bien en base nutricional (94 %), manteca / crema (93 %), leche fluida (81 %) y dulce de leche (80 %).
- Va mal en queso (50 %), suero / permeado (41 %), leche en polvo (29 %) y yogur (12 %). Son subtipos que no se
  distinguen por la composición.

**El umbral es un intercambio.** Sin umbral, el 19 % de las muestras con subtipo real recibe SIN CLASIFICAR; con 0,85,
el 5,6 %. Eso sube (b) del 58 % al 65 % y baja (a) del 63 % al 57 %.

**Sumar MQ no mejora la familia en arc.** Sobre la misma prueba, la cascada de solo arc acierta 0,898 y la de arc + MQ
0,889 (F1 macro 0,70 contra 0,67).
- MQ ayuda en el subtipo de concentrado proteico (F1 0,82 → 0,89) y de suero / permeado (0,41 → 0,47).
- Empeora el de manteca / crema (0,49 → 0,33), donde MQ suma CREMA, que la prueba de arc no tiene.

**Contra los números del compañero (semilla 2026).** La exactitud de familia da 89,3 / 89,2 / 88,9 % para V1 / V2 / V3,
contra 88,5 / 88,9 / 89,6 % en la diapositiva 6.
- Todas las diferencias son menores que un punto. Una prueba de 3 000 muestras tiene un error estándar de unos
  0,6 puntos, y con otra semilla la prueba es otra.
- En V3, además, el archivo de MQ no es el mismo: el filtro de errores de carga encuentra 3 filas y no 10.
- Las conclusiones no cambian:
  - V2 pierde en (b) porque usa SIN CLASIFICAR como refugio (18 % de las reales);
  - V3 recupera buena parte con el umbral, aunque queda por debajo de V1 en (b) (0,651 contra 0,666);
  - sobre las 1 456 muestras que están en las tres pruebas, V3 es la mejor en el subtipo (0,73 contra 0,72 y 0,63).

**La partición al azar casi no infla la familia.** Agrupando por envío, la exactitud del árbol de familia baja apenas
1,1 puntos (0,880 → 0,869). La diferencia con el ~79 % de la Parte V (`prediccion_tipo.ipynb`) no viene de la
partición. Las dos partes usan muestras, filtros y familias distintas, y otro modelo.

**Las «sin clasificar».** Las 1 026 muestras (9 de arc) reciben una familia; 405 se quedan sin subtipo, porque
SIN CLASIFICAR supera el umbral.

Modelo: `resultados/modelos/arbol_decision.joblib`, con `AD.predecir_cascada(joblib.load(...), X)` sobre las 7
columnas de `AD.PREDICTORAS`. Lo usa `app_arbol.py`.
""")

code(r"""
assert set(D3.modelado["Cliente"].dropna().astype(str).str.match(r"^IND-\d+$")) <= {True}
assert "Descripción" not in D3.modelado.columns and "Descripción" not in D3.sin_clasificar.columns
assert AD.SALIDA_MODELO.exists() and (DE.CARPETA / "arbol_decision.csv").exists() and iguales
assert not set(D3.pru.index) & set(D3.ent.index) and (D3.pru["origen"] == "arc").all()
print(f"Verificación: todo en orden · la parte tardó {coma((time.time() - inicio) / 60, 1)} min")
""")

c.extend(celdas_app.celdas(APP, "(¿qué producto es esta muestra?, con la cascada de V3 y la cadena de preguntas de cada\n"
                                "árbol)"))

nb = nbf.v4.new_notebook(); nb["cells"] = c
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
if __name__ == "__main__":
    nbf.write(nb, SALIDA); print(f"Escrito: {SALIDA} ({len(c)} celdas)")
