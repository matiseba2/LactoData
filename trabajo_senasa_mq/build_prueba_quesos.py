"""Arma prueba_clustering_quesos.ipynb: agrupamiento propio de los QUESOS, aparte del notebook completo.

No editar el .ipynb a mano: se regenera con este script y se ejecuta desde esta carpeta.

    cd trabajo_senasa_mq
    ../.venv/bin/python build_prueba_quesos.py
    ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace prueba_clustering_quesos.ipynb

El código está en src/clustering_quesos.py (reusa datos_clase41.py y mejoras_clustering.py). No toca build_notebook.py,
trabajo_senasa_mq.ipynb ni las salidas actuales: escribe solo en resultados/prueba_quesos/.
"""
from pathlib import Path

import nbformat as nbf

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI / "prueba_clustering_quesos.ipynb"

c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: c.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Prueba: agrupamiento propio de los quesos
**Aprendizaje Automático y Grandes Datos · UNRaf · 2026**

Cuaderno **de prueba**, separado de `trabajo_senasa_mq.ipynb`. En el k-medias general el queso casi no entra (hacen
falta humedad, grasa, proteína y cenizas), y los quesos que confirma el catálogo van ahora a una **rama propia** (humedad y
materia grasa, subtag por la clase de humedad del CAA). Acá se agrupan **los mismos quesos de esa rama** (los que tienen
materia grasa y sólidos totales; en Senasa + MQ: Senasa con `desc_familia = queso` y solo MQ únicamente si el rótulo
nombra la variedad; las MO no entran por MQ, ya están en Senasa), para ver si aparecen tipos de queso reales y, si
corresponde, proponer subtags `QUESO_<algo>`.

**Criterio de variables (lo fijó el usuario).**

* En queso, Senasa pide **materia grasa (MG)** y **sólidos totales (ST)**. La humedad (= 100 − ST) y la grasa en
  extracto seco (MGES = MG / ST × 100) son *funciones* de esos dos: no se usan como variables independientes junto con
  MG y ST (sería contar dos veces la misma información). **El espacio base tiene 2 dimensiones.**
* **NaCl, proteína y cenizas** son análisis de MQ (el NaCl es casi exclusivo de quesos). Van en un **nivel extendido**,
  solo para las muestras MQ que los tienen.

**Referencias para validar.** (1) El **subtipo** del rótulo (`desc_subtipo`: mozzarella, cremoso, reggianito…) y (2) la
**consistencia** del rótulo (`desc_consistencia`: baja / mediana / alta humedad, procesado, untable, mezcla; la arma
`src/compartido/catalogo.py` a partir de la variedad nombrada). Las dos son **independientes** de lo medido. (3) Las
**clases del CAA** (art. 605: humedad y MGES) son una función determinista de las mismas dos variables, así que el ARI
contra el CAA mide si los grupos naturales coinciden con los cortes legales: **no es una validación independiente**.

El código está en `src/clustering_quesos.py`; se regenera con `build_prueba_quesos.py`. Escribe solo en
`resultados/prueba_quesos/`. Semilla 42. Al final, el **veredicto**.
""")

md("""
## 0 · Rutas, parámetros y lectura
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score
import rutas as R
import clustering_quesos as Q
from datos_clase41 import SEMILLA, leer_senasa

warnings.filterwarnings("ignore", message=".*covariance matrix.*full rank.*")
warnings.filterwarnings("ignore", message=".*Determinant has increased.*")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 80)
SALIDA = R.RESULTADOS / "prueba_quesos"; SALIDA.mkdir(parents=True, exist_ok=True)
PCT = 3                                   # corte de outliers por consenso (clase 46)
TAMANOS = (20, 30, 40, 50, 60)            # min_cluster_size de HDBSCAN
KS_KM, KS_GMM = range(2, 11), range(1, 9)
DPI = 150

S = leer_senasa()
quesos, _, _sin_confirmar, conteo = Q.rama_queso(S, True)   # misma poblacion que la rama del notebook unico: solo lo que confirma el catalogo, sin las MO
pd.Series(conteo, name="muestras").to_frame()
""")

md("""
## 1 · Conteos
Quesos con MG y ST por origen. «Senasa completada con MQ»: muestra de Senasa a la que le faltaba MG o ST y lo completó
MQ (Senasa manda; MQ solo llena huecos). «solo MQ»: familia queso en `MQ encolumnado.xlsx` (que sale de
`mq_tipo_muestra.csv`, sin revisar a mano), que no está en Senasa y cuyo **rótulo nombra una variedad de queso**; las
solo MQ de familia queso que el rótulo no confirma (1128) quedan afuera, igual que en el notebook principal. Las MO de la
tabla ancha no entran por MQ (las que tienen código en Senasa entran por Senasa; una fila por Identificador). Exclusiones: humedad + ST fuera de 100 ± 1 donde
Senasa midió las dos (regla del proyecto; ningún queso con MG y ST la viola) y MG > ST (MGES > 100, imposible).
""")

code(r"""
ext = quesos[Q.EXTRA].notna()
t = pd.DataFrame({"con MG y ST": quesos.groupby("origen").size(),
                  "+ NaCl": ext["nacl"].groupby(quesos["origen"]).sum(),
                  "+ proteína": ext["proteina"].groupby(quesos["origen"]).sum(),
                  "+ cenizas": ext["cenizas"].groupby(quesos["origen"]).sum(),
                  "+ NaCl y proteína": (ext["nacl"] & ext["proteina"]).groupby(quesos["origen"]).sum(),
                  "+ las tres": ext.all(axis=1).groupby(quesos["origen"]).sum(),
                  "con subtipo": quesos["subtipo"].notna().groupby(quesos["origen"]).sum(),
                  "con consistencia": quesos["consistencia"].notna().groupby(quesos["origen"]).sum()})
t.loc["total"] = t.sum()
t.astype(int)
""")

code(r"""
print("fuente del subtipo:", quesos["fuente_subtipo"].value_counts(dropna=False).to_dict())
sub = quesos["subtipo"].value_counts()
print(f"{sub.size} subtipos distintos; los 25 más frecuentes:")
display(pd.DataFrame({"muestras": sub.head(25),
                      "consistencia más común": [quesos.loc[quesos["subtipo"] == s, "consistencia"].mode().iat[0]
                                                 if quesos.loc[quesos["subtipo"] == s, "consistencia"].notna().any() else "-"
                                                 for s in sub.head(25).index]}).T)
pd.crosstab(quesos["consistencia"].fillna("sin dato"), quesos["caa_humedad"].fillna("-"), margins=True)
""")

md("""
## 2 · Parametrización del espacio base
Las dos parametrizaciones posibles son equivalentes en información (una es función biyectiva de la otra), pero no en
geometría: el escalado y las distancias cambian.

* **(MG, ST)** es lo medido. Pero en queso MG y ST están muy correlacionadas (un queso más seco tiene más grasa por
  100 g): el eje principal mezcla «qué tan seco» con «qué tan graso».
* **(humedad, MGES)** es la del CAA: la humedad separa duro / semiduro / blando y la MGES dice qué tan graso es el
  queso *independientemente* de cuánta agua tenga. Casi no están correlacionadas: cada eje dice una cosa distinta.

Se elige **(humedad, MGES)**: ejes casi ortogonales, interpretables y en la escala de la norma. En la sección 3.4 se repite
todo en (MG, ST) para ver que el resultado no depende de la elección.
""")

code(r"""
pd.DataFrame({"r de Pearson": [quesos[["mg", "st"]].corr().iat[0, 1], quesos[["humedad", "mges"]].corr().iat[0, 1]]},
             index=["MG vs ST", "humedad vs MGES"]).round(2)
""")

md("""
## 3 · Nivel base: humedad y MGES, todos los quesos
### 3.1 · Outliers por consenso
`mejoras_clustering.outliers_consenso`: factor local (LOF, 50 vecinos), bosque de aislamiento y elipse, cada uno marca el
3 % de menor puntaje; outlier si lo marcan al menos 2 de 3. Todos los quesos son un solo grupo (el tag QUESO).
""")

code(r"""
X_todas = Q.matriz(quesos, Q.BASE)
es_out, votos = Q.quitar_outliers(X_todas, pct=PCT)
base = quesos[~es_out].reset_index(drop=True)
print(f"outliers: {es_out.sum()} de {len(quesos)} ({100 * es_out.mean():.1f} %); quedan {len(base)}")
quesos.loc[es_out, ["Identificador", "origen", "mg", "st", "humedad", "mges", "subtipo"]].round(1).sort_values("humedad").head(12)
""")

md("""
### 3.2 · Los tres algoritmos
* **k-medias**: k por silueta máxima (2 a 10).
* **Mezcla de gaussianas** (covarianza completa, n_init 3): k por BIC mínimo (1 a 8).
* **HDBSCAN**: min_cluster_size 20 a 60. Se elige *sin mirar las referencias*: el número de grupos más frecuente en la
  grilla y, entre los que lo dan, el menor tamaño.
""")

code(r"""
def correr(d, variables):
    # Los tres algoritmos sobre d[variables] escalado; etiquetas renumeradas por humedad media
    Z, _ = Q.escalar(Q.matriz(d, variables))
    km, k_km, sil, _ = Q.kmedias(Z, KS_KM)
    gm, k_gm, bic, mod = Q.mezcla(Z, KS_GMM)
    hd = Q.hdbscan_sensibilidad(Z, TAMANOS)
    m_hd, e_hd = Q.elegir_hdbscan(hd)
    h = d["humedad"].to_numpy()
    et = {f"k-medias (k = {k_km})": Q.ordenar(km, h), f"GMM (k = {k_gm}, BIC)": Q.ordenar(gm, h),
          f"HDBSCAN (min {m_hd})": Q.ordenar(e_hd, h)}
    return dict(Z=Z, et=et, sil=sil, bic=bic, hd=hd, m_hd=m_hd, gmm=mod, k_km=k_km, k_gm=k_gm)

def referencias(d):
    return {"subtipo": d["subtipo"], "consistencia": d["consistencia"], "CAA": d["caa"]}

def tabla(r, d, extra=True):
    filas = {n: Q.evaluar(r["Z"], e, referencias(d)) for n, e in r["et"].items()}
    if extra:   # referencia: la regla del CAA aplicada directamente (sin agrupar)
        for n, col in [("regla CAA humedad (4 clases)", "caa_humedad"), ("regla CAA humedad × MGES", "caa")]:
            e = pd.factorize(d[col])[0]
            filas[n] = Q.evaluar(r["Z"], e, referencias(d))
    return pd.DataFrame(filas).T

rb = correr(base, Q.BASE)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(list(rb["sil"]), list(rb["sil"].values()), "o-", color=Q.PALETA[0]); ax[0].set_xlabel("k"); ax[0].set_ylabel("silueta")
ax[0].set_title("k-medias: silueta por k", fontsize=10)
ax[1].plot(list(rb["bic"]), list(rb["bic"].values()), "o-", color=Q.PALETA[1]); ax[1].set_xlabel("k"); ax[1].set_ylabel("BIC")
ax[1].set_title("GMM: BIC por k (menor es mejor)", fontsize=10)
for a in ax: a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(SALIDA / "base_seleccion_k.png", dpi=DPI); plt.show()
""")

md("""
El BIC de la mezcla baja hasta el máximo probado (k = 8) y, si se sigue, continúa bajando un poco (sección 3.5): la mezcla
aproxima una densidad continua, no prueba que haya ocho tipos. Por eso los subtags se proponen solo donde un componente
coincide con un tipo real (sección 5).

**Sensibilidad de HDBSCAN a min_cluster_size:**
""")

code(r"""
filas = {}
for m, e in rb["hd"].items():
    f = Q.evaluar(rb["Z"], e, referencias(base))
    filas[m] = {k: f[k] for k in ["grupos", "ruido", "silueta", "silueta macro", "ARI subtipo", "ARI consistencia", "ARI CAA"]}
pd.DataFrame(filas).T.rename_axis("min_cluster_size").round(3)
""")

md("""
**Métricas y acuerdo con las referencias.** Silueta, Davies-Bouldin y Calinski-Harabasz sin el ruido; ARI y NMI contando
el ruido de HDBSCAN como un grupo más. Las dos últimas filas no son agrupamientos: son las clases del CAA aplicadas
directo, como vara para saber si agrupar agrega algo sobre el corte legal.
""")

code(r"""
tb = tabla(rb, base)
tb.round(3)
""")

md("""
### 3.3 · Qué hay en cada cluster
Centroides en g/100 g (MGES en g/100 g de extracto seco); subtipo, consistencia y clase CAA mayoritarios con su pureza
(% sobre las muestras del cluster que tienen ese dato). Clusters numerados por humedad creciente.
""")

code(r"""
for n, e in rb["et"].items():
    print(n); display(Q.describir(base, e))
""")

code(r"""
top = base["subtipo"].value_counts().head(15).index
for n, e in rb["et"].items():
    print(n, "· tabla cruzada con los 15 subtipos más frecuentes (columnas = cluster)")
    t = pd.crosstab(base["subtipo"].where(base["subtipo"].isin(top), "otros / sin dato"), np.where(e == -1, "ruido", e.astype(str)))
    display(t)
""")

code(r"""
fig, axs = plt.subplots(1, 3, figsize=(18, 5.6), sharex=True, sharey=True)
for a, (n, e) in zip(axs, rb["et"].items()):
    d = base.assign(cl=np.where(e == -1, "ruido", e.astype(str)))
    Q.dispersion_caa(a, d, "cl", n, max_cat=12, orden=sorted(set(d["cl"]) - {"ruido"}, key=int))
fig.tight_layout(); fig.savefig(SALIDA / "base_clusters.png", dpi=DPI); plt.show()

fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True)
Q.dispersion_caa(axs[0], base, "subtipo", "subtipo del rótulo (los 10 más frecuentes)", max_cat=10)
Q.dispersion_caa(axs[1], base, "consistencia", "consistencia del rótulo", max_cat=6)
fig.tight_layout(); fig.savefig(SALIDA / "base_referencias.png", dpi=DPI); plt.show()
""")

md("""
### 3.4 · ¿Depende de la parametrización? Lo mismo en (MG, ST)
Mismas muestras (los mismos outliers), escaladas en (MG, ST). Se compara cada algoritmo con su par de la sección 3.2
(ARI entre las dos particiones: 1 = idénticas).
""")

code(r"""
ra = correr(base, Q.BASE_ALT)
ta = tabla(ra, base, extra=False)
comp = pd.DataFrame({
    "k (H, MGES)": [rb["k_km"], rb["k_gm"], len(set(rb["et"][list(rb["et"])[2]]) - {-1})],
    "k (MG, ST)": [ra["k_km"], ra["k_gm"], len(set(ra["et"][list(ra["et"])[2]]) - {-1})],
    "ARI subtipo (H, MGES)": tb["ARI subtipo"].iloc[:3].values, "ARI subtipo (MG, ST)": ta["ARI subtipo"].values,
    "ARI consistencia (H, MGES)": tb["ARI consistencia"].iloc[:3].values, "ARI consistencia (MG, ST)": ta["ARI consistencia"].values,
    "ARI entre parametrizaciones": [adjusted_rand_score(a, b) for a, b in zip(rb["et"].values(), ra["et"].values())]},
    index=["k-medias", "GMM", "HDBSCAN"])
display(comp.round(3))
ta.round(3)
""")

md("""
### 3.5 · Estabilidad de la mezcla de gaussianas
La misma mezcla con otras semillas (k fijo en el elegido por BIC) y el BIC hasta k = 14 (solo para ver la forma).
""")

code(r"""
from sklearn.mixture import GaussianMixture
k_gm = rb["k_gm"]; e0 = rb["et"][f"GMM (k = {k_gm}, BIC)"]
otras = {s: Q.ordenar(GaussianMixture(n_components=k_gm, n_init=3, random_state=s).fit(rb["Z"]).predict(rb["Z"]),
                      base["humedad"]) for s in range(SEMILLA + 1, SEMILLA + 6)}
print("ARI contra la semilla 42:", {s: round(adjusted_rand_score(e0, e), 2) for s, e in otras.items()})
bic_largo = {k: GaussianMixture(n_components=k, n_init=3, random_state=SEMILLA).fit(rb["Z"]).bic(rb["Z"]) for k in range(6, 15)}
print("BIC k = 6..14:", {k: round(v) for k, v in bic_largo.items()})
""")

md("""
## 4 · Nivel extendido: + NaCl, proteína y cenizas (solo MQ)
Cada variante usa solo las muestras con todas sus variables; para esas mismas muestras se corre también el nivel base
(humedad, MGES), así la comparación es justa. Outliers por consenso dentro de cada variante, con sus variables.
""")

code(r"""
VARIANTES = {"+ NaCl": ["nacl"], "+ NaCl + proteína": ["nacl", "proteina"], "+ proteína": ["proteina"],
             "+ proteína + cenizas": ["proteina", "cenizas"]}
cols = ["silueta", "silueta macro", "ARI subtipo", "ARI consistencia", "ARI CAA"]
filas, res_ext = [], {}
for nombre, extra in VARIANTES.items():
    d = quesos.dropna(subset=extra).reset_index(drop=True)
    o, _ = Q.quitar_outliers(Q.matriz(d, Q.BASE + extra), pct=PCT)
    d = d[~o].reset_index(drop=True)
    for nivel, vs in [("base", Q.BASE), ("extendido", Q.BASE + extra)]:
        r = correr(d, vs)
        res_ext[(nombre, nivel)] = (d, r)
        for alg, e in r["et"].items():
            f = Q.evaluar(r["Z"], e, referencias(d))
            filas.append({"variante": nombre, "n": len(d), "nivel": nivel, "algoritmo": alg.split(" (")[0],
                          "k": f["grupos"], "ruido": f["ruido"], **{c: f[c] for c in cols}})
te = pd.DataFrame(filas).set_index(["variante", "n", "algoritmo", "nivel"]).sort_index(level=[0, 2, 3], sort_remaining=False)
te.round(3)
""")

md("""
**¿Por qué ayudan poco?** En queso, ST ≈ MG + proteína + cenizas (+ lactosa residual): la proteína queda casi determinada
por MG y ST. Y el NaCl acompaña a la humedad (los quesos duros están más salados). Lo nuevo que aportan es poco y, al
escalarlas, pesan lo mismo que humedad y MGES y diluyen la estructura.
""")

code(r"""
d = quesos.dropna(subset=["proteina"])
r2 = LinearRegression().fit(d[["mg", "st"]], d["proteina"]).score(d[["mg", "st"]], d["proteina"])
d2 = quesos.dropna(subset=["nacl"])
print(f"R² de proteína ~ MG + ST: {r2:.2f} (n = {len(d)})")
print(f"r de Pearson NaCl vs humedad: {d2[['nacl', 'humedad']].corr().iat[0, 1]:.2f}; NaCl vs MGES: {d2[['nacl', 'mges']].corr().iat[0, 1]:.2f} (n = {len(d2)})")
d2.groupby(d2["consistencia"].fillna("sin dato"))["nacl"].describe().round(2)
""")

code(r"""
d_ext, r_ext = res_ext[("+ NaCl + proteína", "extendido")]
n_gm = [n for n in r_ext["et"] if n.startswith("GMM")][0]
e_ext = r_ext["et"][n_gm]
desc_ext = Q.describir(d_ext, e_ext, cols=("mg", "st", "humedad", "mges", "nacl", "proteina"))
display(desc_ext.join(Q.proponer_subtags(desc_ext)[["subtag", "criterio"]]).fillna({"subtag": "QUESO"}))
fig, axs = plt.subplots(1, 3, figsize=(18, 5.6))
de = d_ext.assign(cl=e_ext.astype(str))
Q.dispersion_caa(axs[0], de, "cl", f"extendido (+ NaCl + proteína) · {n_gm}", max_cat=12, orden=sorted(set(de["cl"]), key=int), s=12)
Q.dispersion_caa(axs[1], d_ext, "consistencia", "mismas muestras · consistencia del rótulo", max_cat=6, s=12)
for i, cons in enumerate(["baja humedad", "mediana humedad", "alta humedad", "procesado"]):
    sel = d_ext["consistencia"] == cons
    axs[2].scatter(d_ext.loc[sel, "humedad"], d_ext.loc[sel, "nacl"], s=12, alpha=0.7, c=Q.PALETA[i], lw=0, label=f"{cons} ({sel.sum()})")
axs[2].set_xlabel("humedad (g/100 g)"); axs[2].set_ylabel("NaCl (g/100 g)"); axs[2].legend(fontsize=7)
axs[2].set_title("NaCl vs humedad, por consistencia", fontsize=10, pad=14)
fig.tight_layout(); fig.savefig(SALIDA / "extendido.png", dpi=DPI); plt.show()
""")

md("""
## 5 · Subtags
Regla (`clustering_quesos.proponer_subtags`): un cluster lleva subtag solo si corresponde a un tipo real, en este orden:
1. **subtipo** del rótulo con pureza ≥ 70 % (con ≥ 20 muestras con subtipo, que sean al menos la mitad del cluster) →
   `QUESO_<SUBTIPO>`;
2. si no, **consistencia** del rótulo con pureza ≥ 70 % → `QUESO_DURO`, `QUESO_SEMIDURO`, `QUESO_PASTA_BLANDA`,
   `QUESO_PROCESADO`, `QUESO_UNTABLE`;
3. si no, **clase de humedad del CAA** con pureza ≥ 70 % *y* la consistencia del rótulo compatible en ≥ 50 % (la clase
   legal sola no alcanza: sale de las mismas dos variables).

Se aplica a los tres algoritmos del nivel base; para el CSV se usa la **mezcla de gaussianas** (ver veredicto).
""")

code(r"""
props = {}
for n, e in rb["et"].items():
    desc = Q.describir(base, e)
    p = Q.proponer_subtags(desc)
    props[n] = p
    print(n); display(desc[["muestras", "humedad", "mges", "subtipo mayoritario", "% subtipo", "consistencia mayoritario",
                            "% consistencia", "CAA humedad mayoritario", "% CAA humedad"]].join(p[["subtag", "criterio"]]).fillna({"subtag": "QUESO"}))
""")

code(r"""
st0 = Q.subtag_por_muestra(e0, props[f"GMM (k = {k_gm}, BIC)"])
res = pd.DataFrame({"subtag": st0, "consistencia": base["consistencia"].fillna("sin dato")})
display(pd.crosstab(res["subtag"], res["consistencia"], margins=True))
por = res.groupby("subtag").size().to_frame("muestras")
por["subtipos más frecuentes"] = [", ".join(f"{k} {v}" for k, v in base.loc[st0 == s, "subtipo"].value_counts().head(5).items()) for s in por.index]
por
""")

md("""
### 5.1 · ¿Son estables los subtags?
La misma regla con la mezcla corrida con otras semillas (k = 8 fijo) y con HDBSCAN en cada min_cluster_size. Cada fila
es una corrida: cuántas muestras recibe cada subtag y el ARI de sus subtags contra los de la semilla 42.
""")

code(r"""
filas = {}
for s_, e in {SEMILLA: e0, **otras}.items():
    t = Q.subtag_por_muestra(e, Q.proponer_subtags(Q.describir(base, e)))
    filas[f"GMM semilla {s_}"] = {**pd.Series(t).value_counts().to_dict(), "ARI vs 42": adjusted_rand_score(st0, t)}
for m, e in rb["hd"].items():
    e = Q.ordenar(e, base["humedad"])
    t = Q.subtag_por_muestra(e, Q.proponer_subtags(Q.describir(base, e)))
    filas[f"HDBSCAN min {m}"] = {**pd.Series(t).value_counts().to_dict(), "ARI vs 42": adjusted_rand_score(st0, t)}
estab = pd.DataFrame(filas).T
estab = estab[["ARI vs 42", "QUESO"] + sorted(c for c in estab if c.startswith("QUESO_"))]
estab.round(2).fillna("")
""")

md("""
La partición fina cambia con la semilla y con eso cambian los subtags: con la semilla 42 salen duro, azul, pasta blanda y
port salut, pero cuatro de cada diez muestras quedan sin subtag (QUESO) y, con otras semillas, entre un tercio y casi
seis de cada diez; aparecen además otros (QUESO_SEMIDURO, QUESO_MUY_ALTA_HUMEDAD). Lo único que sale en las seis corridas
es **QUESO_DURO**; **QUESO_MUY_ALTA_HUMEDAD** sale en cinco y **QUESO_AZUL** (humedad ≈ 43 %, MGES ≈ 52 %) en cuatro.

### 5.2 · La alternativa sin agrupar: el corte del CAA
El subtag sale directo de la clase de humedad del CAA (humedad = 100 − ST). Se compara con la consistencia del rótulo,
la referencia independiente.
""")

code(r"""
st_caa = Q.subtag_regla_caa(base["caa_humedad"])
ok = base["consistencia"].notna().to_numpy()
print(f"ARI contra la consistencia del rótulo · regla CAA: {adjusted_rand_score(base['consistencia'][ok], st_caa[ok]):.2f}"
      f" · subtags de la mezcla (semilla 42): {adjusted_rand_score(base['consistencia'][ok], st0[ok]):.2f}")
pd.crosstab(pd.Series(st_caa, name="subtag (regla CAA)"), base["consistencia"].fillna("sin dato"), margins=True)
""")

md("""
## 6 · Muestras que no son queso y tienen NaCl (informativo)
¿Hay sueros en polvo con NaCl? ¿Se confundirían con quesos en el nivel extendido? Se toman las muestras MQ con NaCl cuya
familia (Senasa si está, si no `mq_tipo_muestra.csv`) no es queso, y las que tienen MG y ST se proyectan sobre la mezcla
del nivel «+ NaCl» (la variante con más muestras): se miran su componente y su log-verosimilitud contra la de los quesos.
""")

code(r"""
nq = Q.no_quesos_con_nacl(S)
display(nq.groupby("familia").agg(muestras=("Identificador", "size"), con_MG_y_ST=("st", lambda s: s.notna().sum()),
                                   ST_mediana=("st", "median"), ST_max=("st", "max"), NaCl_mediana=("nacl", "median")).round(1))
print("¿suero / permeado en polvo con NaCl?:", int(((nq["familia"] == "suero / permeado") | (nq["st"] > 90)).sum()))
d_n, r_n = res_ext[("+ NaCl", "extendido")]
esc = Q.escalar(Q.matriz(d_n, Q.BASE + ["nacl"]))[1]
pr = nq.dropna(subset=["mg", "st"]).query("mg <= st").reset_index(drop=True)
if len(pr):
    zq = esc.transform(Q.matriz(d_n, Q.BASE + ["nacl"])); zn = esc.transform(Q.matriz(pr, Q.BASE + ["nacl"]))
    umbral = np.percentile(r_n["gmm"].score_samples(zq), 1)
    pr["log-verosimilitud"] = r_n["gmm"].score_samples(zn)
    pr["como queso"] = pr["log-verosimilitud"] >= umbral
    print(f"umbral = percentil 1 de los quesos: {umbral:.1f}")
    display(pr.groupby("familia").agg(muestras=("Identificador", "size"), se_confundirian=("como queso", "sum"),
                                       humedad=("humedad", "median"), MGES=("mges", "median"), NaCl=("nacl", "median")).round(1))
""")

md("""
## 7 · CSV de resultados
Una fila por queso: Identificador, origen, subtipo, consistencia, clases del CAA, cluster base de cada algoritmo
(los outliers por consenso quedan marcados y su cluster de mezcla es el que les asigna el modelo con `predict`), cluster
extendido (mezcla, variante + NaCl + proteína; vacío si la muestra no está en ese nivel), el subtag que da la mezcla
(`subtag_gmm`, semilla 42; los outliers quedan QUESO) y el **subtag propuesto** (`subtag_propuesto`: el corte de
humedad del CAA, lo que recomienda el veredicto).
""")

code(r"""
salida = quesos[["Identificador", "origen", "mg", "st", "humedad", "mges", "nacl", "proteina", "cenizas", "subtipo",
                 "fuente_subtipo", "consistencia", "caa_humedad", "caa_mges"]].copy().round(2)
salida["outlier_base"] = es_out
idx = np.flatnonzero(~es_out)
for n, e in rb["et"].items():
    col = "cluster_base_" + n.split(" (")[0].lower().replace("-", "")
    salida[col] = pd.array([pd.NA] * len(salida), dtype="Int64"); salida.loc[idx, col] = e
# outliers: componente de la mezcla por predict (con la misma escala), renumerado igual
esc_b = Q.escalar(Q.matriz(base, Q.BASE))[1]
crudo = rb["gmm"].predict(rb["Z"]); mapa = dict(zip(crudo, e0))
pred = rb["gmm"].predict(esc_b.transform(X_todas[es_out]))
salida.loc[es_out, "cluster_base_gmm"] = [mapa.get(p, -1) for p in pred]
salida["cluster_ext_gmm"] = salida["Identificador"].map(dict(zip(d_ext["Identificador"], e_ext))).astype("Int64")
salida["subtag_gmm"] = "QUESO"; salida.loc[idx, "subtag_gmm"] = st0     # el del agrupamiento (inestable, ver 5.1)
salida["subtag_propuesto"] = Q.subtag_regla_caa(salida["caa_humedad"])  # lo que recomienda el veredicto
salida.to_csv(SALIDA / "quesos_clusters.csv", index=False, encoding="utf-8-sig")
tb.round(4).to_csv(SALIDA / "metricas_base.csv", encoding="utf-8-sig")
te.round(4).to_csv(SALIDA / "metricas_extendido.csv", encoding="utf-8-sig")
print(salida.shape, "→", SALIDA / "quesos_clusters.csv")
print(salida["subtag_propuesto"].value_counts().to_dict())
sorted(p.name for p in SALIDA.iterdir())
""")

md("""
## Veredicto
""")

code(r"""
# números que cita el veredicto (se imprimen para que el texto no quede desfasado de la ejecución)
g = f"GMM (k = {k_gm}, BIC)"
print(f"n base = {len(base)} (de {len(quesos)}; {es_out.sum()} outliers) · n extendido (+ NaCl + proteína) = {len(d_ext)}")
print(tb.loc[[*rb['et'], 'regla CAA humedad (4 clases)', 'regla CAA humedad × MGES'], ['grupos', 'silueta', 'silueta macro', 'ARI subtipo', 'ARI consistencia', 'ARI CAA']].round(2))
print("ARI GMM (H, MGES) vs GMM (MG, ST):", round(comp.loc['GMM', 'ARI entre parametrizaciones'], 2))
print("subtags de la mezcla (42):", pd.Series(st0).value_counts().to_dict())
print("ARI de los subtags con otras semillas:", estab.loc[[i for i in estab.index if i.startswith("GMM") and str(SEMILLA) not in i], "ARI vs 42"].round(2).tolist())
print(f"ARI contra la consistencia · regla CAA {adjusted_rand_score(base['consistencia'][ok], st_caa[ok]):.2f} · subtags de la mezcla {adjusted_rand_score(base['consistencia'][ok], st0[ok]):.2f}")
print("ARI entre parametrizaciones (H, MGES) vs (MG, ST):", comp["ARI entre parametrizaciones"].round(2).to_dict())
gm_runs = estab.loc[[i for i in estab.index if i.startswith("GMM")]]
print(f"GMM, {len(gm_runs)} semillas · corridas en las que sale cada subtag:", {c: int(gm_runs[c].notna().sum()) for c in gm_runs if c.startswith("QUESO_")})
print("GMM, % sin subtag por semilla:", (100 * gm_runs["QUESO"] / len(base)).round(0).astype(int).tolist(), "· HDBSCAN, % de ruido por min_cluster_size:", {m: round(float(100 * (e == -1).mean()), 1) for m, e in rb["hd"].items()})
print("GMM extendido, ARI base → extendido (por variante):")
print(te.xs("GMM", level="algoritmo")[["ARI subtipo", "ARI consistencia"]].round(2).unstack("nivel").to_string())
proc = base.loc[base["consistencia"] == "procesado"]
print(f"procesado: {(proc['caa_humedad'] == 'muy alta humedad').sum()} de {len(proc)} están en muy alta humedad")
print(f"cenizas: {int(quesos['cenizas'].notna().sum())} quesos · sin especificar con NaCl evaluados: {len(pr)}, dentro de la densidad de los quesos: {int(pr['como queso'].sum())}")
""")

md("""
**No conviene sumar al notebook completo subtags de queso que salgan de un agrupamiento.** Los subtags de queso salen
del **corte de humedad del CAA** (art. 605) sobre humedad = 100 − ST, en una rama propia para los quesos que confirma el
catálogo con MG y ST, no dentro del k-medias general (que exige los cuatro análisis): `QUESO_DURO` (< 36 %),
`QUESO_SEMIDURO` (36–46 %), `QUESO_PASTA_BLANDA` (46–55 %) y `QUESO_MUY_ALTA_HUMEDAD` (≥ 55 %). Es lo que hace hoy el notebook
principal y la columna `subtag_propuesto` del CSV (2891 quesos: 1195 semiduros, 693 pasta blanda, 539 duros y 464 de muy
alta humedad).

* **Hay estructura, pero es la de la humedad.** Con los 2 810 quesos sin outliers, k-medias y la mezcla llegan a un ARI de
  0,28 y 0,36 contra el subtipo (HDBSCAN, 0,02). Pero lo que separan es sobre todo la humedad: los grupos se ordenan duro →
  semiduro → blando → untable, y la **regla del CAA aplicada directo** coincide con la consistencia del rótulo (la
  referencia independiente) **mejor** que cualquier agrupamiento (ARI 0,63 contra 0,37 de la mezcla, 0,31 de sus subtags y
  0,31 de k-medias).
* **Los subtags del agrupamiento son inestables.** Con la semilla 42 la mezcla de gaussianas da duro, azul, pasta blanda y
  port salut, pero deja el 40 % sin subtag; con otras cinco semillas los subtags cambian (ARI 0,26 a 0,86 contra la
  semilla 42; ninguna repite la partición) y entre el 33 y el 58 % queda sin subtag. Solo el duro sale en las seis corridas. HDBSCAN cambia con
  min_cluster_size (de 4 a 6 grupos) y manda entre el 5 y el 9 % al ruido, con ARI de 0,02 contra el subtipo. k-medias
  depende de la parametrización (k = 7 en humedad y MGES, k = 2 en MG y ST). El BIC no tiene mínimo claro: sigue
  bajando, con algún repunte (k = 12), hasta k = 14.
* **Si igual se quisiera un agrupamiento:** mezcla de gaussianas sobre **humedad y MGES** (las dos variables de Senasa,
  sin agregar nada de MQ). Depende poco de la parametrización (ARI 0,57 entre (humedad, MGES) y (MG, ST); k-medias,
  0,17; HDBSCAN, 0,94, pero casi no se parece al subtipo: 0,02) y es la que más se parece a la consistencia entre los
  agrupamientos (0,37). Los grupos que más se repiten en las corridas son el duro (las seis), el de muy alta humedad
  (cinco) y un grupo **azul** (humedad ≈ 43 %, MGES ≈ 52 %; cuatro). El azul es el único subtipo con composición propia.
* **El nivel extendido no mejora de forma pareja.** Para las mismas muestras, con la mezcla el ARI contra el subtipo baja
  con NaCl (0,33 a 0,26) y con NaCl + proteína (0,42 a 0,38, sobre 359 muestras), y sube con proteína sola (0,29 a 0,43) y
  con proteína + cenizas (0,03 a 0,13, solo 74 muestras). Contra la consistencia el signo es el mismo en las cuatro
  variantes. Como el nivel base de cada variante también cambia mucho (la mezcla elige entre k = 5 y 8 según las muestras),
  parte de la diferencia es ruido de la partición y no efecto de la variable. La proteína está casi determinada por MG y ST
  (R² ≈ 0,85) y el NaCl acompaña a la humedad (r ≈ −0,6). Hay pocas cenizas (107 quesos). Con NaCl y proteína aparecen
  grupos chicos más puros (reggianito con 14 muestras al 100 %, port salut magro con 29 al 100 %, cremoso con 98 al 68 %, y
  30 de muy alta humedad con poca proteína, ≈ 7 g/100 g, en su mayoría procesados). Pero son 359 muestras, todas con
  análisis de MQ, y la mejora no es consistente: no alcanza para un subtag del notebook completo.
* **Lo que no sale:** subtipos finos (en k-medias el azul y el cremoso caen en el mismo grupo; el cheddar se reparte en
  cuatro) y **procesado**, que se mezcla con los demás de muy alta humedad (162 de 211 están ahí) y el resto va a otras
  clases: se define por la tecnología y no por la composición. No se propone `QUESO_MOZZARELLA` ni
  `QUESO_PROCESADO`.
* **Sueros en polvo con NaCl: no hay.** Las muestras MQ con NaCl que no son queso son manteca o crema, «sin
  especificar» y unas pocas de base nutricional y cárnicos. Ninguna tiene ST > 90. Pero de las 25 «sin especificar» que se
  evaluaron, 8 caen dentro de la densidad de los quesos con el umbral usado (percentil 1 de los quesos): podrían ser quesos
  sin familia asignada.
""")

md("""
### Dudas
- La población coincide con la rama del notebook principal (2891 quesos). La familia de las solo MQ sale de un modelo
  (`mq_tipo_muestra.csv`) sin revisar a mano y solo se aceptan las que el rótulo confirma con una variedad de queso; las
  1128 que no (697 con MG y ST) quedan afuera. El subtipo de las solo MQ sale del catálogo normalizado
  (`MQ encolumnado - solo nuevas.xlsx`).
- La consistencia del rótulo sale de la variedad con la tabla de `catalogo.py`, que se contrastó con humedades medidas.
  Por eso no es del todo independiente de la humedad, y eso favorece a la regla del CAA en la comparación.
- Las «sin especificar» con NaCl podrían contener quesos sin familia asignada (44 tienen MG y ST); con el criterio usado
  8 de las 25 evaluadas caen dentro de la densidad de los quesos. No se revisaron a mano.
- Hay NaCl negativos en MQ: se tomaron como 0, igual que la proteína.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = c
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
if __name__ == "__main__":
    nbf.write(nb, SALIDA)
    print(f"Escrito: {SALIDA} ({len(c)} celdas)")
