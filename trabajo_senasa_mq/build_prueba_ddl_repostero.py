"""Arma prueba_ddl_repostero.ipynb: ¿el dulce de leche repostero puede tener su propio subtag (DDL_REPOSTERO)?

No editar el .ipynb a mano: se regenera con este script y se ejecuta desde esta carpeta.

    cd trabajo_senasa_mq
    ../.venv/bin/python build_prueba_ddl_repostero.py
    ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace prueba_ddl_repostero.ipynb

Cuaderno de diagnóstico, en las dos versiones (solo Senasa y Senasa + MQ). Toma las muestras del tag DDL de
resultados/tablas/clusters_*.csv y el subtipo de referencia de mejoras_clustering.preparar (que no entra en el
agrupamiento). No modifica nada: escribe solo los PNG en resultados/prueba_ddl_repostero/.
"""
from pathlib import Path

import nbformat as nbf

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI / "prueba_ddl_repostero.ipynb"

c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: c.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# ¿Se puede separar el dulce de leche repostero? (prueba)
**Aprendizaje Automático y Grandes Datos · UNRaf · 2026**

El segundo nivel del tag **DDL** que había cuando se hizo esta prueba (mezcla de gaussianas sobre humedad, grasa, proteínas y cenizas reescaladas dentro
del tag, k = 6 por BIC; `src/mejoras_clustering.py`, `subtags`) deja dos subtags: **DDL_FAMILIAR** y **DDL_HELADERO**.
El resto queda como «DDL». La pregunta es si el **repostero** puede tener su propio subtag.

**Nota (después de las pruebas por familia):** el notebook único reemplazó esta mezcla por el segundo nivel de `src/segundo_nivel.py` (en el DDL, la regresión logística de `prueba_subtipo_ddl.ipynb`, que sí da DDL_REPOSTERO). Esta prueba recalcula la mezcla (es determinista) y compara contra ella, como cuando se hizo.


Este cuaderno es **de prueba** y no cambia nada del notebook completo. Pasos:

| | qué se mira |
|---|---|
| 1 | conteos y distribución de cada análisis por subtipo dentro del DDL: ¿dónde queda el repostero? |
| 2 | **diagnóstico supervisado** (LDA y regresión logística con validación cruzada estratificada). No es el modelo: si ni un supervisado lo separa, ningún agrupamiento lo va a separar |
| 3 | en qué componentes de la mezcla actual cae el repostero; la mezcla con k de 1 a 12 (y más) y covarianza full / diag |
| 4 | una regla simple que lo aísle, con su pureza y cobertura |
| 5 | causa probable: análisis fuera de la composición centesimal, texto de las descripciones y separabilidad por cliente (IND-nn) |

El subtipo de referencia es `desc_subtipo` de Senasa; si falta, el de `desc_producto` y, si no, el de
`mq_tipo_muestra.csv` (`mejoras_clustering.subtipo_referencia`). **Pureza** = % de repostero entre las muestras elegidas
que tienen subtipo conocido. **Cobertura** = % de los reposteros conocidos que quedan elegidos. Semilla 42. Al final,
el **veredicto**.
""")

md("""
## 0 · Rutas, parámetros y funciones
""")

code(r"""
import sys
import warnings
from pathlib import Path
SRC = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "rutas.py").exists()) / "src"
sys.path.insert(0, str(SRC))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

import rutas as R
import mejoras_clustering as M
from datos_clase41 import COLUMNAS, SEMILLA, a_numero, leer_mq, leer_senasa

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)

SALIDA = R.RESULTADOS / "prueba_ddl_repostero"; SALIDA.mkdir(parents=True, exist_ok=True)
SUBTIPOS = ["FAMILIAR", "REPOSTERO", "HELADERO", "sin subtipo"]
COLOR = {"FAMILIAR": "#2a78d6", "REPOSTERO": "#eb6834", "HELADERO": "#1baf7a", "sin subtipo": "#b0aea5"}
TINTA, TINTA2 = "#0b0b0b", "#52514e"
plt.rcParams.update({"figure.dpi": 100, "savefig.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "font.size": 9})
EJEMPLOS = {"MQ-34408-01": "repostero", "MQ-35246-01": "familiar", "MQ-34696-01": "heladero"}
REGLA = dict(humedad_max=27.6, cenizas_max=2.0)       # la regla simple de la sección 4 (umbrales redondeados)
KS = range(1, 13)
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
CV_REP = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEMILLA)
MODELOS = {"LDA": lambda: make_pipeline(StandardScaler(), LinearDiscriminantAnalysis()),
           "regresión logística": lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))}
OTROS = {"aw": "Resultado Actividad de agua (% Aw) (*)", "carbohidratos": "Resultado Carbohidratos Totales g/ 100g (*)",
         "lactosa": "Resultado Lactosa g /100g (*)", "sodio": "Resultado Sodio mg /100g (*)"}
S = leer_senasa()
MQ_CRUDO, _ = leer_mq()
MQ_CRUDO = MQ_CRUDO.set_index("codigo")
RES = {}


def cargar_ddl(v, con_mq):
    # Muestras del tag DDL tal como quedaron en clusters_{v}.csv (con los outliers, que también llevan tag) + subtipo,
    # cliente (IND-nn), otros análisis de Senasa y de MQ. Recalcula el subtag de la mezcla (el segundo nivel anterior).
    muestras, _ = M.preparar(con_mq, S)
    cl = pd.read_csv(R.TABLAS / f"clusters_{v}.csv", encoding="utf-8-sig")
    d = cl[cl["tag"] == "DDL"].merge(muestras[["Identificador", "subtipo_ref", "fuente_subtipo"]], on="Identificador",
                                    how="left", validate="one_to_one").reset_index(drop=True)
    cr = S.crudo.set_index("Identificador")
    d["cliente"] = d["Identificador"].map(cr["Cliente"])
    for k, col in OTROS.items():
        d[k] = a_numero(d["Identificador"].map(cr[col]))
    for k in ["cloruro_nacl_g100g", "cloruro_mg100g", "solidos", "caseina", "ceniza_acida", "grasa_libre"]:
        d[f"mq_{k}"] = d["Identificador"].map(MQ_CRUDO[k])
    d["subtipo"] = d["subtipo_ref"].fillna("sin subtipo")
    X = d[COLUMNAS].to_numpy()
    sub, modelos = M.subtags(X, np.array(["DDL"] * len(d), dtype=object), d["subtipo_ref"])
    # El notebook único ya no usa esta mezcla (su segundo nivel sale de segundo_nivel.py, con la logística que salió de
    # prueba_subtipo_ddl). Acá se recalcula la mezcla, que es determinista, y su subtag reemplaza al de clusters_*.csv:
    # la prueba compara contra el segundo nivel que había cuando se hizo.
    d["subtag"] = sub
    m = modelos["DDL"]
    d["componente"] = m["gmm"].predict(m["escalador"].transform(X))
    return d, m


def pureza_cobertura(elegidas, subtipo, objetivo="REPOSTERO"):
    elegidas = np.asarray(elegidas, dtype=bool); s = pd.Series(np.asarray(subtipo, dtype=object))
    conocido = s != "sin subtipo"
    n_c = int((elegidas & conocido).sum()); n_o = int((elegidas & (s == objetivo)).sum())
    return {"elegidas": int(elegidas.sum()), "con subtipo": n_c, "sin subtipo": int((elegidas & ~conocido).sum()),
            f"{objetivo.lower()}s": n_o, "% pureza": round(100 * n_o / n_c, 1) if n_c else np.nan,
            "% cobertura": round(100 * n_o / (s == objetivo).sum(), 1)}


def supervisado(d, positivo, negativos):
    s = d[d["subtipo"].isin([positivo] + negativos)]
    y = (s["subtipo"] == positivo).astype(int).to_numpy(); Xs = s[COLUMNAS].to_numpy()
    filas, coefs, matrices = [], {}, {}
    for nombre, f in MODELOS.items():
        pred = cross_val_predict(f(), Xs, y, cv=CV)
        rep = cross_val_score(f(), Xs, y, cv=CV_REP, scoring="balanced_accuracy")
        filas.append({"modelo": nombre, "n": len(y), f"n {positivo.lower()}": int(y.sum()),
                      "exactitud balanceada (5 pliegues)": round(balanced_accuracy_score(y, pred), 3),
                      "media 5×10": round(rep.mean(), 3), "desvío 5×10": round(rep.std(), 3)})
        matrices[nombre] = pd.DataFrame(confusion_matrix(y, pred), index=[f"real: otro", f"real: {positivo}"],
                                        columns=[f"predicho: otro", f"predicho: {positivo}"])
        ajuste = f().fit(Xs, y)[-1]
        coefs[nombre] = (ajuste.scalings_[:, 0] if nombre == "LDA" else ajuste.coef_[0])
    return pd.DataFrame(filas).set_index("modelo"), matrices, pd.DataFrame(coefs, index=COLUMNAS).round(2)


def guardar(fig, nombre):
    fig.savefig(SALIDA / nombre, dpi=150, bbox_inches="tight", facecolor="white")
    print("PNG:", (SALIDA / nombre).relative_to(R.RAIZ))
""")


def celdas_version(v, titulo, numero, con_mq):
    h = numero
    md(f"""
## {h} · {titulo}
### {h}.1 · Conteos por subtipo dentro del DDL
Las muestras con tag DDL de `resultados/tablas/clusters_{v}.csv` (incluidos los outliers, que también llevan tag). El
subtag es el de la mezcla, recalculada con `M.subtags` (el segundo nivel anterior; el de `clusters_{v}.csv` ya es otro).
""")
    code(fr"""
v = "{v}"
d, modelo = cargar_ddl(v, {con_mq})
print(f"{{len(d)}} muestras con tag DDL · mezcla con k = {{modelo['k_bic']}} · subtag de la mezcla recalculado")
display(pd.crosstab(d["subtipo"], d["subtag"], margins=True, margins_name="total").reindex(SUBTIPOS + ["total"]))
print("Fuente del subtipo:"); display(pd.crosstab(d["subtipo"], d["fuente_subtipo"].fillna("-")))
""")

    md(f"""
### {h}.2 · Distribución por subtipo
Medianas y percentiles 10 y 90 de los cuatro análisis. Abajo, los violines (con la mediana marcada) y la actividad de
agua (Aw), que es el único análisis extra con suficientes muestras de DDL. Los cloruros, la caseína, la ceniza ácida y la
grasa libre de MQ **no tienen ningún valor** en las muestras de DDL; los sólidos de MQ son 100 − humedad y no agregan nada.
""")
    code(r"""
resumen = d.groupby("subtipo")[COLUMNAS].describe(percentiles=[.1, .5, .9]).loc[SUBTIPOS]
display(pd.concat({c_: resumen[c_][["count", "10%", "50%", "90%"]] for c_ in COLUMNAS}, axis=1).round(2))
otros = d.groupby("subtipo")[list(OTROS)].agg(["count", "median"]).reindex(SUBTIPOS).round(3)
print("Otros análisis de Senasa (n con valor y mediana):"); display(otros)
print("Análisis de MQ con valor en el DDL:", {k_: int(d[k_].notna().sum()) for k_ in d.columns if k_.startswith("mq_")})
display(d[d["Identificador"].isin(EJEMPLOS)][["Identificador", "subtipo", "subtag", "componente"] + COLUMNAS])

fig, axs = plt.subplots(1, 5, figsize=(15, 3.6))
for ax, var in zip(axs, COLUMNAS + ["aw"]):
    grupos = [d.loc[d["subtipo"] == s_, var].dropna().to_numpy() for s_ in SUBTIPOS]
    pos = [i for i, g_ in enumerate(grupos) if len(g_) > 1]
    partes = ax.violinplot([grupos[i] for i in pos], positions=pos, showmedians=True, showextrema=False, widths=0.8)
    for cuerpo, i in zip(partes["bodies"], pos):
        cuerpo.set_facecolor(COLOR[SUBTIPOS[i]]); cuerpo.set_edgecolor("white"); cuerpo.set_alpha(0.75)
    partes["cmedians"].set_color(TINTA); partes["cmedians"].set_linewidth(1.5)
    ax.set_xticks(range(len(SUBTIPOS))); ax.set_xticklabels([f"{s_.lower()}\n(n={len(g_)})" for s_, g_ in zip(SUBTIPOS, grupos)], fontsize=7)
    ax.set_title({"aw": "actividad de agua (Aw)"}.get(var, f"{var} (g/100 g)"), fontsize=9, color=TINTA); ax.grid(axis="x", visible=False)
fig.suptitle(f"DDL por subtipo · {v}", x=0.01, ha="left", color=TINTA)
fig.tight_layout(); guardar(fig, f"violines_{v}.png"); plt.show()

fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
for ax, (x_, y_) in zip(axs, [("humedad", "proteinas"), ("humedad", "cenizas"), ("proteinas", "grasa")]):
    for s_ in ["sin subtipo", "FAMILIAR", "HELADERO", "REPOSTERO"]:
        q = d[d["subtipo"] == s_]
        ax.scatter(q[x_], q[y_], s=9, color=COLOR[s_], alpha=0.55 if s_ != "sin subtipo" else 0.35, lw=0, label=f"{s_.lower()} ({len(q)})")
    for cod, rot in EJEMPLOS.items():
        q = d[d["Identificador"] == cod]
        if len(q) and x_ == "humedad":
            ax.scatter(q[x_], q[y_], s=70, facecolors="none", edgecolors=TINTA, lw=1.2)
            ax.annotate(rot, (q[x_].iloc[0], q[y_].iloc[0]), xytext={"repostero": (8, 8), "familiar": (8, -12), "heladero": (-8, -14)}[rot],
                        textcoords="offset points", fontsize=8, color=TINTA, ha="left" if rot != "heladero" else "right",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    if x_ == "humedad":
        ax.set_xlim(18, 34)
    ax.set_xlabel(f"{x_} (g/100 g)"); ax.set_ylabel(f"{y_} (g/100 g)")
    if (x_, y_) == ("humedad", "cenizas"):
        ax.axvline(REGLA["humedad_max"], color=TINTA2, lw=0.8, ls="--"); ax.axhline(REGLA["cenizas_max"], color=TINTA2, lw=0.8, ls="--")
        ax.text(REGLA["humedad_max"] - 0.2, REGLA["cenizas_max"] - 0.04, "regla: humedad ≤ 27,6 y cenizas ≤ 2,0", ha="right", va="top", fontsize=7, color=TINTA2)
axs[0].legend(fontsize=7, markerscale=1.8, frameon=False, loc="upper left")
fig.suptitle(f"DDL: dispersión por subtipo (círculos: los tres ejemplos, en los dos primeros paneles) · {v}", x=0.01, ha="left", color=TINTA)
fig.tight_layout(); guardar(fig, f"dispersion_{v}.png"); plt.show()
""")

    md(f"""
### {h}.3 · Diagnóstico supervisado
LDA y regresión logística (con peso de clase balanceado) sobre las cuatro columnas estandarizadas. Validación cruzada
estratificada de 5 pliegues para la matriz de confusión, y 5 pliegues × 10 repeticiones para la media y el desvío de la
**exactitud balanceada** (0,5 = azar). Tres preguntas: repostero contra familiar, repostero contra el resto (familiar +
heladero) y, como referencia de algo que sí se separa, heladero contra el resto. Los coeficientes están en unidades de
desvío: positivo = empuja hacia el primer subtipo.
""")
    code(r"""
sup = {}
for pos_, negs_ in [("REPOSTERO", ["FAMILIAR"]), ("REPOSTERO", ["FAMILIAR", "HELADERO"]), ("HELADERO", ["FAMILIAR", "REPOSTERO"])]:
    nombre = f"{pos_.lower()} contra {' + '.join(n_.lower() for n_ in negs_)}"
    tabla, mats, coefs = supervisado(d, pos_, negs_); sup[nombre] = tabla
    print(nombre); display(tabla)
    for k_, m_ in mats.items():
        print(f"  matriz de confusión ({k_}):"); display(m_)
    print("  coeficientes:"); display(coefs.T)
""")

    md(f"""
### {h}.4 · ¿En qué componentes de la mezcla actual cae el repostero?
Componente × subtipo de la mezcla (la que guardaba el joblib antes del segundo nivel por familia). El subtag sale de `proponer_subtag`: un componente lo
lleva si su subtipo mayoritario tiene pureza de al menos 60 % y cobertura de al menos 50 %, con suficientes muestras de
subtipo conocido.
""")
    code(r"""
display(pd.crosstab(d["componente"], d["subtipo"], margins=True, margins_name="total")[SUBTIPOS + ["total"]])
display(modelo["descripcion"])
may_rep = [g_ for g_, f_ in modelo["descripcion"].iterrows() if f_["subtipo mayoritario"] == "REPOSTERO"]
print("Componentes con mayoría de repostero:", may_rep)
union_actual = pureza_cobertura(d["componente"].astype(str).isin(may_rep), d["subtipo"])
print("Si se fusionan esos componentes en un DDL_REPOSTERO:", union_actual)

fig, ax = plt.subplots(figsize=(6.5, 3.6))
t_ = pd.crosstab(d["componente"], d["subtipo"])[SUBTIPOS]
izq = np.zeros(len(t_))
for s_ in SUBTIPOS:
    ax.barh(t_.index.astype(str), t_[s_], left=izq, color=COLOR[s_], edgecolor="white", lw=1, label=s_.lower(), height=0.7)
    izq += t_[s_].to_numpy()
sub_ = modelo["descripcion"]["subtag"]
ax.set_yticks(range(len(t_))); ax.set_yticklabels([f"comp. {g_} · {sub_.get(str(g_), '')}" for g_ in t_.index], fontsize=8)
ax.invert_yaxis(); ax.set_xlabel("muestras"); ax.grid(axis="y", visible=False)
ax.legend(fontsize=7, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.45, -0.16))
ax.set_title(f"Mezcla actual (k = {modelo['k_bic']}): componente × subtipo · {v}", loc="left", fontsize=9, color=TINTA)
fig.tight_layout(); guardar(fig, f"componentes_{v}.png"); plt.show()
""")

    md(f"""
### {h}.5 · La mezcla con un rango de k más amplio (1 a 12) y covarianza full / diag
Misma escala que la actual (dentro del tag) y `n_init = 3`. Para cada k: BIC, el componente con más pureza de repostero
(de al menos 20 muestras) y la **fusión** de todos los componentes con mayoría de repostero entre las de subtipo
conocido. Con covarianza full se siguen probando k hasta 20, para ver si el BIC llega a un mínimo o sigue en meseta.
""")
    code(r"""
Z = modelo["escalador"].transform(d[COLUMNAS].to_numpy())
filas, bic = [], {}
for cov in ["full", "diag"]:
    for k_ in (range(1, 21) if cov == "full" else KS):
        g_ = GaussianMixture(n_components=k_, covariance_type=cov, n_init=3, random_state=SEMILLA).fit(Z)
        bic[(cov, k_)] = g_.bic(Z)
        if k_ not in KS:
            continue
        e_ = g_.predict(Z); t_ = pd.crosstab(e_, d["subtipo"]).reindex(columns=SUBTIPOS, fill_value=0)
        conoc = t_[["FAMILIAR", "REPOSTERO", "HELADERO"]].sum(axis=1)
        pur = (t_["REPOSTERO"] / conoc.replace(0, np.nan)).where(t_.sum(axis=1) >= 20)
        mejor = pur.idxmax() if pur.notna().any() else None
        mayoria = [i for i in t_.index if conoc[i] > 0 and t_.loc[i, "REPOSTERO"] == t_.loc[i, ["FAMILIAR", "REPOSTERO", "HELADERO"]].max()]
        u_ = pureza_cobertura(np.isin(e_, mayoria), d["subtipo"])
        filas.append({"covarianza": cov, "k": k_, "BIC": round(bic[(cov, k_)]),
                      "mejor comp.: muestras": int(t_.loc[mejor].sum()) if mejor is not None else 0,
                      "mejor comp.: % pureza": round(100 * pur[mejor], 1) if mejor is not None else np.nan,
                      "mejor comp.: % cobertura": round(100 * t_.loc[mejor, "REPOSTERO"] / t_["REPOSTERO"].sum(), 1) if mejor is not None else np.nan,
                      "fusión: componentes": len(mayoria), "fusión: % pureza": u_["% pureza"], "fusión: % cobertura": u_["% cobertura"]})
barrido = pd.DataFrame(filas).set_index(["covarianza", "k"]); display(barrido)
for cov in ["full", "diag"]:
    b_ = {k_: bic[(cov, k_)] for (c_, k_) in bic if c_ == cov}
    kmin = min(b_, key=b_.get)
    print(f"{cov}: BIC mínimo en k = {kmin} (de {min(b_)}–{max(b_)} probados); "
          f"k con BIC a menos de 100 del mínimo: {[k_ for k_ in b_ if b_[k_] - b_[kmin] < 100]}")

fig, ax = plt.subplots(figsize=(6.5, 3.4))
for cov, col_ in [("full", "#2a78d6"), ("diag", "#eb6834")]:
    ks_ = sorted(k_ for (c_, k_) in bic if c_ == cov)
    ax.plot(ks_, [bic[(cov, k_)] for k_ in ks_], marker="o", ms=4, lw=2, color=col_, label=f"covarianza {cov}")
ax.axvline(modelo["k_bic"], color=TINTA2, lw=0.8, ls="--"); ax.text(modelo["k_bic"] + 0.2, ax.get_ylim()[1], "k actual", va="top", fontsize=7, color=TINTA2)
ax.set_xlabel("k (componentes)"); ax.set_ylabel("BIC (menor es mejor)"); ax.legend(frameon=False, fontsize=8)
ax.set_xticks(range(1, 21))
ax.set_title(f"BIC de la mezcla dentro del DDL · {v}", loc="left", fontsize=9, color=TINTA)
fig.tight_layout(); guardar(fig, f"bic_{v}.png"); plt.show()
""")

    md(f"""
### {h}.6 · Una regla simple
Un árbol de decisión de profundidad 1 a 3 (peso de clase balanceado, hojas de al menos 30 muestras), entrenado solo con
las muestras de subtipo conocido, propone los cortes; su exactitud balanceada se mide con validación cruzada. Después se
evalúa la regla fija redondeada **humedad ≤ {{REGLA['humedad_max']}} y cenizas ≤ {{REGLA['cenizas_max']}}** (la misma en
las dos versiones) sobre todo el DDL, y la regla aplicada después del heladero (solo a lo que la mezcla no manda a
DDL_HELADERO).
""".replace("{{REGLA['humedad_max']}}", "27,6").replace("{{REGLA['cenizas_max']}}", "2,0"))
    code(r"""
con = d[d["subtipo"] != "sin subtipo"]; y_ = (con["subtipo"] == "REPOSTERO").astype(int)
for prof in [1, 2, 3]:
    arbol = lambda: DecisionTreeClassifier(max_depth=prof, min_samples_leaf=30, class_weight="balanced", random_state=SEMILLA)
    ba = cross_val_score(arbol(), con[COLUMNAS], y_, cv=CV_REP, scoring="balanced_accuracy")
    print(f"árbol de profundidad {prof}: exactitud balanceada {ba.mean():.3f} ± {ba.std():.3f}")
    if prof == 2:
        print(export_text(arbol().fit(con[COLUMNAS], y_), feature_names=COLUMNAS, decimals=2))
regla = (d["humedad"] <= REGLA["humedad_max"]) & (d["cenizas"] <= REGLA["cenizas_max"])
reglas = {"regla: humedad ≤ 27,6 y cenizas ≤ 2,0": regla,
          "regla, fuera de DDL_HELADERO": regla & (d["subtag"] != "DDL_HELADERO"),
          "regla + (humedad ≤ 29 y proteínas > 6,8)": regla | ((d["humedad"] <= 29) & (d["proteinas"] > 6.8) & (d["cenizas"] <= REGLA["cenizas_max"])),
          "fusión de componentes de la mezcla actual": d["componente"].astype(str).isin(may_rep)}
t_reglas = pd.DataFrame({n_: pureza_cobertura(r_, d["subtipo"]) for n_, r_ in reglas.items()}).T
display(t_reglas)
print("Qué subtipo queda adentro de la regla:"); display(pd.crosstab(regla.map({True: "dentro", False: "fuera"}), d["subtipo"])[SUBTIPOS])

# esquema completo propuesto: DDL_HELADERO de la mezcla; después la regla → DDL_REPOSTERO; después DDL_FAMILIAR de la mezcla
propuesto = np.where(d["subtag"] == "DDL_HELADERO", "DDL_HELADERO", np.where(regla, "DDL_REPOSTERO", d["subtag"]))
print("Esquema propuesto (heladero → regla de repostero → familiar) contra el subtipo:")
display(pd.crosstab(pd.Series(propuesto, name="subtag propuesto"), d["subtipo"], margins=True, margins_name="total")[SUBTIPOS + ["total"]])
esquema = pd.DataFrame({st_: pureza_cobertura(propuesto == f"DDL_{st_}", d["subtipo"], st_) for st_ in ["FAMILIAR", "REPOSTERO", "HELADERO"]}).T
esquema = esquema[["elegidas", "con subtipo", "sin subtipo", "% pureza", "% cobertura"]]
print("Pureza y cobertura de cada subtag con el esquema propuesto (antes: familiar 90 / 78, heladero 100 / 80):"); display(esquema)
anio = pd.to_datetime(d["Identificador"].map(S.crudo.set_index("Identificador")["Fecha de Recepción"]), errors="coerce", dayfirst=True).dt.year
print("La regla por año de recepción (pureza / cobertura del repostero):")
display(pd.DataFrame({int(a_): pureza_cobertura(regla[anio == a_], d.loc[anio == a_, "subtipo"].reset_index(drop=True))
                      for a_ in sorted(anio.dropna().unique())}).T[["con subtipo", "% pureza", "% cobertura"]])
RES[v] = dict(n=len(d), esquema=esquema, conteos=d["subtipo"].value_counts().to_dict(), medianas=d.groupby("subtipo")[COLUMNAS].median().round(2),
              sup=sup, regla=t_reglas, barrido=barrido, union_actual=union_actual, d=d)
""")

    md(f"""
### {h}.7 · Causa probable: ¿otra cosa que no sea la composición, o la etiqueta?
Tres miradas. (a) La **actividad de agua** y los carbohidratos, que no entran en el agrupamiento. (b) El **texto** de
`Descripción`, `desc_producto` y `desc_atributos`: ¿se menciona algún espesante o ingrediente (almidón, agar, gelificante,
carragenina, pectina)? ¿`desc_consistencia` dice algo? (c) La separabilidad **por cliente** (solo el código IND-nn):
regresión logística repostero contra familiar dentro de cada cliente con al menos 15 de cada uno, y las medianas.
""")
    code(r"""
cr = S.crudo.set_index("Identificador")
texto = d["Identificador"].map(cr[["Descripción", "desc_producto", "desc_atributos"]].astype("string").fillna("").agg(" ".join, axis=1)).fillna("").str.upper()
palabras = {"almidón": "ALMID", "agar": r"\bAGAR", "espesante": "ESPES", "gelificante": "GELIF", "carragenina": "CARRAG",
            "pectina": "PECTIN", "horneable": "HORNE", "alfajorero": "ALFAJOR", "balde / cuñete": r"BALDE|CU[NÑ]ETE"}
display(pd.DataFrame({n_: d.loc[texto.str.contains(p_, regex=True), "subtipo"].value_counts().reindex(SUBTIPOS, fill_value=0)
                      for n_, p_ in palabras.items()}).T)
print("desc_consistencia con valor en el DDL:", int(d["Identificador"].map(cr["desc_consistencia"]).notna().sum()))

print("Repostero dentro de la regla, por cliente, y separabilidad repostero / familiar:")
filas = []
for cli, q in d.groupby("cliente"):
    n_r, n_f = (q["subtipo"] == "REPOSTERO").sum(), (q["subtipo"] == "FAMILIAR").sum()
    if n_r < 5:
        continue
    f_ = {"cliente": cli, "reposteros": int(n_r), "familiares": int(n_f), "heladeros": int((q["subtipo"] == "HELADERO").sum()),
          "sin subtipo": int((q["subtipo"] == "sin subtipo").sum())}
    for s_ in ["REPOSTERO", "FAMILIAR"]:
        for c_ in ["humedad", "proteinas"]:
            f_[f"{c_} mediana {s_.lower()}"] = round(q.loc[q["subtipo"] == s_, c_].median(), 2) if (q["subtipo"] == s_).any() else np.nan
    f_["% repostero en la regla"] = round(100 * (regla[q.index] & (q["subtipo"] == "REPOSTERO")).sum() / n_r, 0)
    if n_r >= 15 and n_f >= 15:
        qq = q[q["subtipo"].isin(["REPOSTERO", "FAMILIAR"])]
        f_["exactitud balanceada R/F"] = round(cross_val_score(MODELOS["regresión logística"](), qq[COLUMNAS], (qq["subtipo"] == "REPOSTERO").astype(int),
                                                               cv=CV_REP, scoring="balanced_accuracy").mean(), 3)
    filas.append(f_)
por_cliente = pd.DataFrame(filas).set_index("cliente").sort_values("reposteros", ascending=False); display(por_cliente)
RES[v]["por_cliente"] = por_cliente

# el repostero tiene dos modos (se ve en la dispersión): ¿en qué componentes cae cada cliente?
rep = d[d["subtipo"] == "REPOSTERO"]
print("Reposteros: componente de la mezcla actual × cliente (clientes con al menos 5 reposteros), con mediana por componente:")
tc = pd.crosstab(rep["componente"], rep["cliente"])
display(tc[[c_ for c_ in por_cliente.index if c_ in tc]].join(rep.groupby("componente")[COLUMNAS].median().round(2)))
es_alf = texto.str.contains("ALFAJOR")
print("Alfajorero, por componente:", d.loc[es_alf, "componente"].value_counts().to_dict())

# sin el cliente más grande, ¿se sigue separando?
grande = por_cliente.index[0]
resto = d[(d["cliente"] != grande)]
t_, _, _ = supervisado(resto, "REPOSTERO", ["FAMILIAR"])
print(f"Repostero contra familiar SIN {grande}:"); display(t_)
RES[v]["sin_grande"] = (grande, t_)

fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True, sharey=True)
top = por_cliente[por_cliente["exactitud balanceada R/F"].notna()].index[:2] if "exactitud balanceada R/F" in por_cliente else []
for ax, cli in zip(axs, top):
    q = d[d["cliente"] == cli]
    for s_ in ["FAMILIAR", "HELADERO", "REPOSTERO"]:
        qq = q[q["subtipo"] == s_]
        ax.scatter(qq["humedad"], qq["proteinas"], s=10, color=COLOR[s_], alpha=0.6, lw=0, label=f"{s_.lower()} ({len(qq)})")
    ax.set_xlim(18, 34); ax.set_xlabel("humedad (g/100 g)"); ax.set_ylabel("proteínas (g/100 g)")
    ax.set_title(f"{cli} · exactitud balanceada R/F {por_cliente.loc[cli, 'exactitud balanceada R/F']:.2f}", loc="left", fontsize=9, color=TINTA)
    ax.legend(fontsize=7, frameon=False, markerscale=1.8, loc="upper left")
fig.suptitle(f"Repostero y familiar dentro de los dos clientes con más muestras · {v}", x=0.01, ha="left", color=TINTA)
fig.tight_layout(); guardar(fig, f"por_cliente_{v}.png"); plt.show()
""")


celdas_version("senasa", "Solo Senasa", 1, False)
celdas_version("senasa_mq", "Senasa + MQ", 2, True)

md("""
## 3 · Las dos versiones, lado a lado
""")
code(r"""
filas = []
for v_, r_ in RES.items():
    f_ = {"versión": v_, "muestras DDL": r_["n"]} | {f"n {k_.lower()}": r_["conteos"].get(k_, 0) for k_ in SUBTIPOS}
    for n_, t_ in r_["sup"].items():
        if n_.startswith("repostero"):
            f_[f"BA {n_} (LR, 5×10)"] = t_.loc["regresión logística", "media 5×10"]
            f_[f"BA {n_} (LDA, 5×10)"] = t_.loc["LDA", "media 5×10"]
    f_["BA repostero/familiar sin el cliente mayor (LR)"] = r_["sin_grande"][1].loc["regresión logística", "media 5×10"]
    for n_ in r_["regla"].index:
        f_[f"{n_}: pureza / cobertura"] = f"{r_['regla'].loc[n_, '% pureza']} / {r_['regla'].loc[n_, '% cobertura']}"
    b_ = r_["barrido"].loc["full"]
    f_["mezcla full: k con BIC mínimo (1–12)"] = int(b_["BIC"].idxmin())
    filas.append(f_)
display(pd.DataFrame(filas).set_index("versión").T)
for v_, r_ in RES.items():
    print(v_, "medianas por subtipo:"); display(r_["medianas"])
""")

md("""
## 4 · Veredicto
### ¿Se puede crear DDL_REPOSTERO?

**Sí, pero no con el agrupamiento actual.** Lo que funciona es una **regla** sobre dos análisis, y la señal viene casi
toda de un solo cliente.

- **Regla propuesta**: dentro del tag DDL y fuera de DDL_HELADERO, **humedad ≤ 27,6 y cenizas ≤ 2,0 g/100 g** →
  DDL_REPOSTERO. En solo Senasa da **82,6 % de pureza y 78,9 % de cobertura** (670 muestras: 489 reposteros, 93
  familiares, 10 heladeros y 78 sin subtipo). En Senasa + MQ (MQ encolumnado, sin las MO) da 82,4 % y 77,7 %. Primero va el heladero (mezcla), después
  la regla y al final el familiar (mezcla). Con ese orden, DDL_FAMILIAR sube de 90 % a 93 % de pureza, pero su
  cobertura baja de 78 % a 69 %. DDL_HELADERO no cambia.
- Un árbol de profundidad 2 llega a cortes parecidos (exactitud balanceada 0,86 en validación cruzada): primero humedad
  ≤ 27,6, igual que la regla; del lado seco ya no corta por cenizas sino por humedad (> 22,9 es repostero, lo que deja
  afuera al heladero) y del lado húmedo por proteínas > 6,8. Los umbrales se
  eligieron **mirando la etiqueta**, así que es una regla supervisada y no un hallazgo del agrupamiento. Hay que
  presentarla así.

### Evidencia

1. **Dónde está.** El repostero queda entre el familiar y el heladero. Medianas (humedad / grasa / proteínas / cenizas):
   familiar 28,7 / 6,8 / 5,8 / 1,36; repostero 26,6 / 6,8 / 6,6 / 1,50; heladero 21,8 / 7,0 / 6,7 / 2,56. Tiene menos agua
   que el familiar y más proteína, pero sin las cenizas del heladero. Además es **bimodal**: un modo seco y magro
   (humedad ≈ 25, proteínas ≈ 5,2, cenizas ≈ 1,2) y otro con humedad ≈ 27, proteínas ≈ 7,2 y cenizas ≈ 1,6.
2. **Supervisado.** Repostero contra familiar: exactitud balanceada **0,87** (LDA y logística). Repostero contra el resto:
   **0,82 y 0,83**. Pesan la humedad (−) y las proteínas (+), y contra el resto también las cenizas (−). Como referencia, el
   heladero da 0,96 a 0,99. Se separa bastante, pero no tan limpio como el heladero.
3. **La mezcla.** El repostero se reparte en cuatro componentes con mayoría de repostero (y 68 de los 620 caen en el del
   familiar). En solo Senasa, uno es 98 % puro pero cubre solo el 26 %; otro tiene 74 % de pureza y 43 % de cobertura (con
   MQ, 97 % y 26 %, y 74 % y 46 %). Ningún componente puro llega al 50 % de cobertura, así que `proponer_subtag` lo rechaza con
   razón. Con k de 6 a 12, full o diag, el componente más puro nunca pasa de alrededor del 26 % de cobertura; con k menores
   es menos puro (83 % con k = 5). Fusionar los
   componentes con mayoría de repostero da 74 % de pureza y 89 % de cobertura, peor que la regla. **El BIC no tiene un
   mínimo real**: con full el mínimo cae en k = 8 en las dos versiones, pero queda en meseta desde k ≈ 6 (en solo Senasa, de 6 a 12
   están a menos de 100 del mínimo; con MQ, 8 y de 10 a 15) y con diag sigue bajando hasta el borde del rango (k = 12 en solo
   Senasa, k = 11 con MQ). El k = 6 no es un número que el BIC haya encontrado.

### Causa probable

- **La etiqueta depende del cliente.** El cliente mayor (IND-51) aporta 459 de los 620 reposteros, y su repostero es una
  receta propia, separable del familiar del mismo cliente (exactitud balanceada 0,93). Los dos modos y el componente
  puro son suyos. En IND-11, el repostero **tiene la misma composición** que el familiar (medianas 27,8 contra 27,9 de
  humedad; exactitud balanceada 0,59; la regla captura el 48 %). Sin IND-51, el supervisado baja a 0,72 con la logística (0,74 con LDA). La pureza de la
  regla también varía por año: 57,5 % y 73,7 % en 2020–2021, entre 74,7 % y 95,5 % después.
- **Sí hay un rasgo físico.** La actividad de agua del repostero es más baja (mediana 0,804 contra 0,836, con 220 y 89
  muestras): es más concentrado y más firme. Eso concuerda con un producto pensado por textura o consistencia. No hay
  forma de ver espesantes: ninguna descripción menciona almidón, agar, gelificante, carragenina ni pectina, y
  `desc_consistencia` está vacío en todo el DDL. Los cloruros, la caseína y la ceniza ácida de MQ no tienen ningún valor
  en el DDL.

### Dudas

- Si DDL_REPOSTERO entra, entra **como regla documentada**, no como subtag del agrupamiento. En la práctica describe el
  repostero de IND-51. Una muestra repostera de otro cliente, con fórmula de familiar, va a caer en DDL_FAMILIAR o en DDL.
- 595 muestras de DDL (625 en Senasa + MQ) no tienen subtipo, 326 de ellas de IND-38, que casi no rotula. La pureza se
  mide solo sobre las que tienen subtipo conocido.
- Alfajorero (27 muestras) cae casi todo en un componente mixto. Habría que decidir si es un repostero o un subtipo aparte.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = c
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
if __name__ == "__main__":
    nbf.write(nb, SALIDA)
    print(f"Escrito: {SALIDA} ({len(c)} celdas)")
