"""Arma prueba_clustering_polvos.ipynb: agrupamiento propio de los PRODUCTOS EN POLVO, aparte del notebook completo.

No editar el .ipynb a mano: se regenera con este script y se ejecuta desde esta carpeta.

    cd trabajo_senasa_mq
    ../.venv/bin/python build_prueba_polvos.py
    ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace prueba_clustering_polvos.ipynb

El código está en src/clustering_polvos.py (reusa clustering_quesos.py, datos_clase41.py y mejoras_clustering.py). Lee
resultados/tablas/clusters_*.csv (sección 6), así que se ejecuta después del notebook único. No toca build_notebook.py,
trabajo_senasa_mq.ipynb ni las salidas actuales: escribe solo en resultados/prueba_polvos/.
"""
from pathlib import Path

import nbformat as nbf

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI / "prueba_clustering_polvos.ipynb"

c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: c.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Prueba: agrupamiento propio de los productos en polvo
**Aprendizaje Automático y Grandes Datos · UNRaf · 2026**

Cuaderno **de prueba**, separado de `trabajo_senasa_mq.ipynb`. En el agrupamiento general los polvos quedan en tres tags:
**POLVO_GRASO** (leche en polvo entera), **POLVO_MAGRO** (todo lo demás: sueros, concentrados de suero, leche
descremada) y **PROT_ALTA** (caseínas y concentrados de más de 70 % de proteína). Acá se agrupan solo los polvos, para ver
si aparecen tipos reales dentro de esos tags y, si corresponde, proponer subtags `POLVO_<tipo>`. Es la misma prueba que
`prueba_clustering_quesos.ipynb`, con el mismo método y los mismos algoritmos.

**Qué es un polvo.** Las familias **leche en polvo**, **suero / permeado** y **concentrado proteico** con los cuatro
análisis (humedad, grasa, proteínas y cenizas; Senasa manda y MQ completa; MQ sale de `MQ encolumnado.xlsx`, sin las MO) y **humedad ≤ 15 g/100 g**. Hay un hueco
natural entre 12 y 25: lo que está arriba (humedad de 25 a 99) son sueros, permeados y concentrados **líquidos**, que no
son polvos.

**Criterio de variables (el mismo de los quesos).** En un polvo la humedad es de proceso (2 a 6 g/100 g) y lo que define
el tipo es **de qué están hechos los sólidos**. El espacio base es **grasa, proteína y cenizas en base seca** (g/100 g de
extracto seco): 3 dimensiones. La lactosa no entra: en base seca es casi exactamente 100 menos las otras tres (el
«resto por diferencia»), así que sería contar dos veces la misma información. La parametrización medida (humedad, grasa,
proteínas, cenizas) se prueba como control.

**Referencias para validar.** (1) El **tipo** del rótulo, independiente de lo medido: el subtipo (WPC, caseína,
caseinato, suero, suero desmineralizado, permeado, leche modificada...) y, en la leche en polvo, que casi nunca trae
subtipo, `desc_grasa` (entera, descremada). (2) La **familia**. (3) En la leche en polvo, la **clase de grasa del CAA**
(art. 567): es función de la grasa medida, así que no es una validación independiente.

El código está en `src/clustering_polvos.py`; se regenera con `build_prueba_polvos.py`. Escribe solo en
`resultados/prueba_polvos/`. Semilla 42. Al final, el **veredicto**.
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
from IPython.display import display
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
import rutas as R
import clustering_polvos as P
from datos_clase41 import COLUMNAS, N_INIT_KMEDIAS, SEMILLA, leer_senasa

warnings.filterwarnings("ignore", message=".*covariance matrix.*full rank.*")
warnings.filterwarnings("ignore", message=".*Determinant has increased.*")
warnings.filterwarnings("ignore", message=".*y_pred contains classes not in y_true.*")
warnings.filterwarnings("ignore", message=".*least populated class.*")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 80)
SALIDA = R.RESULTADOS / "prueba_polvos"; SALIDA.mkdir(parents=True, exist_ok=True)
PCT = 3                                   # corte de outliers por consenso (clase 46)
TAMANOS = (20, 30, 40, 50, 60)            # min_cluster_size de HDBSCAN
KS_KM, KS_GMM = range(2, 11), range(1, 13)
TIPO_MIN = 10                             # tipos con menos muestras no entran en la validación supervisada
DPI = 150

S = leer_senasa()
polvos, conteo = P.preparar_polvos(S)
pd.Series(conteo, name="muestras").to_frame()
""")

md("""
## 1 · Conteos
Polvos por familia y origen. «Senasa completada con MQ»: muestra de Senasa a la que le faltaba algún análisis y lo
completó MQ. «solo MQ»: no está en Senasa y su familia sale de `mq_tipo_muestra.csv` (un desglose que **no se revisó a
mano**). La mayoría de las leches en polvo son solo MQ.
""")

code(r"""
display(pd.crosstab(polvos["familia"], polvos["origen"], margins=True))
t = pd.crosstab(polvos["tipo"].fillna("sin tipo"), polvos["familia"], margins=True)
t.sort_values("All", ascending=False)
""")

md("""
**Los líquidos que quedan afuera** (humedad > 15): por familia y subtipo, con su humedad mediana.
""")

code(r"""
from datos_clase41 import leer_mq, unir
_, mq_ = leer_mq()
sen_ = S.crudo[["Identificador", "desc_familia", "desc_subtipo"] + COLUMNAS]
tipo_mq_ = pd.read_csv(R.MQ_TIPO).drop_duplicates("codigo").set_index("codigo")
u_, _ = unir(sen_, mq_)
u_["desc_familia"] = u_["desc_familia"].where(u_["origen"] != "solo MQ", u_["Identificador"].map(tipo_mq_["desc_familia"]))
liq = u_[u_["desc_familia"].isin(P.FAMILIAS)].dropna(subset=COLUMNAS).query("humedad > @P.HUMEDAD_MAX")
print(f"humedad de los polvos: hasta {polvos['humedad'].max():.1f} · humedad de los líquidos: de {liq['humedad'].min():.1f} a {liq['humedad'].max():.1f} (n = {len(liq)})")
liq.groupby(["desc_familia", liq["desc_subtipo"].fillna("sin subtipo")]).agg(
    muestras=("Identificador", "size"), humedad=("humedad", "median"), proteinas=("proteinas", "median")).round(1)
""")

md("""
## 2 · Parametrización del espacio base
* **Base seca (grasa, proteína, cenizas en g/100 g de extracto seco).** Saca la humedad, que en un polvo es de proceso y
  no del tipo, y deja la composición de los sólidos. El «resto» (100 − las tres) es casi todo lactosa.
* **Lo medido (humedad, grasa, proteínas, cenizas).** Lo que usa el agrupamiento general. Se repite todo así en la
  sección 3.4.

¿Hace falta la lactosa como variable? Se compara la lactosa medida con el resto por diferencia, en las muestras que la
tienen.
""")

code(r"""
display(polvos[P.BASE + ["resto_bs"]].corr().round(2))
q = polvos.dropna(subset=["lactosa"])
igual = (q["resto_bs"] - q["lactosa_bs"]).abs() < 0.05
print(f"con lactosa medida: {len(q)} (origen: {q['origen'].value_counts().to_dict()})")
print(f"r de Pearson lactosa (base seca) vs resto por diferencia: {np.corrcoef(q['lactosa_bs'], q['resto_bs'])[0, 1]:.3f}")
print(f"lactosa = resto por diferencia (±0,05): {igual.sum()} de {len(q)}")
q.loc[~igual, ["Identificador", "tipo", "lactosa_bs", "resto_bs"]].round(1).sort_values("tipo")
""")

md("""
La lactosa informada en estos polvos es, en la mayoría, **el resto por diferencia** (coincide dentro de ±0,05). No es un análisis independiente y son pocas muestras: **no hay nivel extendido** en los polvos (en los
quesos lo daban NaCl, proteína y cenizas de MQ; acá ya están todos en la base).

## 3 · Nivel base: grasa, proteína y cenizas en base seca
### 3.1 · Outliers por consenso
`mejoras_clustering.outliers_consenso`: factor local (LOF, 50 vecinos), bosque de aislamiento y elipse, cada uno marca el
3 % de menor puntaje; outlier si lo marcan al menos 2 de 3. Todos los polvos son un solo grupo.
""")

code(r"""
X_todas = P.matriz(polvos, P.BASE)
es_out, votos = P.quitar_outliers(X_todas, pct=PCT)
base = polvos[~es_out].reset_index(drop=True)
print(f"outliers: {es_out.sum()} de {len(polvos)} ({100 * es_out.mean():.1f} %); quedan {len(base)}")
print("por tipo:", polvos.loc[es_out, "tipo"].fillna("sin tipo").value_counts().to_dict())
polvos.loc[es_out, ["Identificador", "Cliente", "familia", "tipo", "humedad"] + P.BASE].round(1).sort_values(["tipo", "proteina_bs"])
""")

md("""
Casi todos los outliers son **tipos reales con pocas muestras**: suero y suero reengrasado con grasa de 25 a 56 % en
base seca (8 muestras), caseína ácida (4; humedad ≈ 11), suero desmineralizado con cenizas < 1 (6) y dos concentrados con
83 % de proteína (uno sin tipo). Hay además dos permeados sin tipo con cenizas de 19 a 23 en base seca, que podrían ser
un error de carga. Quedan fuera del agrupamiento por ser pocos; el CSV los conserva marcados.

### 3.2 · Los tres algoritmos
* **k-medias**: k por silueta máxima (2 a 10).
* **Mezcla de gaussianas** (covarianza completa, n_init 3): k por BIC mínimo (1 a 12).
* **HDBSCAN**: min_cluster_size 20 a 60, elegido *sin mirar las referencias* (el número de grupos más frecuente en la
  grilla y, entre los que lo dan, el menor tamaño).
""")

code(r"""
def correr(d, variables):
    # Los tres algoritmos sobre d[variables] escalado; etiquetas renumeradas por proteína en base seca
    Z, _ = P.escalar(P.matriz(d, variables))
    km, k_km, sil, _ = P.kmedias(Z, KS_KM)
    gm, k_gm, bic, mod = P.mezcla(Z, KS_GMM)
    hd = P.hdbscan_sensibilidad(Z, TAMANOS)
    m_hd, e_hd = P.elegir_hdbscan(hd)
    pr = d["proteina_bs"].to_numpy()
    et = {f"k-medias (k = {k_km})": P.ordenar(km, pr), f"GMM (k = {k_gm}, BIC)": P.ordenar(gm, pr),
          f"HDBSCAN (min {m_hd})": P.ordenar(e_hd, pr)}
    return dict(Z=Z, et=et, sil=sil, bic=bic, hd=hd, m_hd=m_hd, gmm=mod, k_km=k_km, k_gm=k_gm)

def referencias(d):
    return {"tipo": d["tipo"], "familia": d["familia"]}

tg = P.tags_generales()
base = base.join(tg, on="Identificador")

def tabla(r, d, extra=True):
    filas = {n: P.evaluar(r["Z"], e, referencias(d)) for n, e in r["et"].items()}
    if extra:   # vara: los tags del agrupamiento general (Senasa + MQ) como partición de estas mismas muestras
        filas["tags del agrupamiento general"] = P.evaluar(r["Z"], pd.factorize(d["tag_senasa_mq"].fillna("no está"))[0], referencias(d))
    return pd.DataFrame(filas).T

rb = correr(base, P.BASE)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
ax[0].plot(list(rb["sil"]), list(rb["sil"].values()), "o-", color=P.PALETA[0]); ax[0].set_xlabel("k"); ax[0].set_ylabel("silueta")
ax[0].set_title("k-medias: silueta por k", fontsize=10)
ax[1].plot(list(rb["bic"]), list(rb["bic"].values()), "o-", color=P.PALETA[1]); ax[1].set_xlabel("k"); ax[1].set_ylabel("BIC")
ax[1].set_title("GMM: BIC por k (menor es mejor)", fontsize=10)
for a in ax: a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(SALIDA / "base_seleccion_k.png", dpi=DPI); plt.show()
""")

md("""
**Sensibilidad de HDBSCAN a min_cluster_size:**
""")

code(r"""
filas = {}
for m, e in rb["hd"].items():
    f = P.evaluar(rb["Z"], e, referencias(base))
    filas[m] = {k: f[k] for k in ["grupos", "ruido", "silueta", "silueta macro", "ARI tipo", "ARI familia"]}
pd.DataFrame(filas).T.rename_axis("min_cluster_size").round(3)
""")

md("""
**Métricas y acuerdo con las referencias.** Silueta, Davies-Bouldin y Calinski-Harabasz sin el ruido; ARI y NMI contando
el ruido de HDBSCAN como un grupo más. La última fila no es un agrupamiento de esta prueba: son los tags del
agrupamiento general (POLVO_GRASO, POLVO_MAGRO, PROT_ALTA) sobre las mismas muestras, como vara para saber si agrupar
solo los polvos agrega algo.
""")

code(r"""
tb = tabla(rb, base)
tb.round(3)
""")

md("""
### 3.3 · Qué hay en cada cluster
Centroides en g/100 g de extracto seco (humedad en g/100 g); tipo, familia y clase CAA de la leche mayoritarios con su
pureza (% sobre las muestras del cluster que tienen ese dato) y, para el tipo, su cobertura (% de ese tipo que cae en el
cluster). Clusters numerados por proteína en base seca creciente.
""")

code(r"""
for n, e in rb["et"].items():
    print(n); display(P.describir(base, e))
""")

code(r"""
for n, e in rb["et"].items():
    print(n, "· tipo del rótulo (filas) × cluster (columnas)")
    display(pd.crosstab(base["tipo"].fillna("sin tipo"), np.where(e == -1, "ruido", e.astype(str))))
""")

code(r"""
fig, axs = plt.subplots(2, 3, figsize=(18, 10.5))
for j, (n, e) in enumerate(rb["et"].items()):
    d = base.assign(cl=np.where(e == -1, "ruido", e.astype(str)))
    orden = sorted(set(d["cl"]) - {"ruido"}, key=int)
    P.dispersion(axs[0, j], d, "cl", f"{n} · proteína × grasa", max_cat=12, orden=orden)
    P.dispersion(axs[1, j], d, "cl", f"{n} · proteína × cenizas", y="cenizas_bs", max_cat=12, orden=orden)
fig.tight_layout(); fig.savefig(SALIDA / "base_clusters.png", dpi=DPI); plt.show()

fig, axs = plt.subplots(1, 3, figsize=(18, 5.4))
P.dispersion(axs[0], base, "tipo", "tipo del rótulo · proteína × grasa", max_cat=11)
P.dispersion(axs[1], base, "tipo", "tipo del rótulo · proteína × cenizas", y="cenizas_bs", max_cat=11)
P.dispersion(axs[2], base, "tag_senasa_mq", "tag del agrupamiento general · proteína × grasa", max_cat=4)
fig.tight_layout(); fig.savefig(SALIDA / "base_referencias.png", dpi=DPI); plt.show()
""")

md("""
### 3.4 · ¿Depende de la parametrización? Lo mismo con lo medido
Mismas muestras (los mismos outliers), escaladas en humedad, grasa, proteínas y cenizas (g/100 g). Se compara cada
algoritmo con su par de la sección 3.2 (ARI entre las dos particiones: 1 = idénticas).
""")

code(r"""
ra = correr(base, P.BASE_ALT)
ta = tabla(ra, base, extra=False)
n_grupos = lambda r: [len(set(e) - {-1}) for e in r["et"].values()]
comp = pd.DataFrame({
    "k (base seca)": n_grupos(rb), "k (medido)": n_grupos(ra),
    "ARI tipo (base seca)": tb["ARI tipo"].iloc[:3].values, "ARI tipo (medido)": ta["ARI tipo"].values,
    "silueta (base seca)": tb["silueta"].iloc[:3].values, "silueta (medido)": ta["silueta"].values,
    "ARI entre parametrizaciones": [adjusted_rand_score(a, b) for a, b in zip(rb["et"].values(), ra["et"].values())]},
    index=["k-medias", "GMM", "HDBSCAN"])
display(comp.round(3))
n_km_a = list(ra["et"])[0]
print(n_km_a); display(P.describir(base, ra["et"][n_km_a], cols=("humedad", "grasa", "proteinas", "cenizas")))
""")

md("""
### 3.5 · Estabilidad
k-medias y la mezcla con otras semillas (k fijo en el elegido), y el BIC de la mezcla hasta k = 16 (solo para ver la forma).
""")

code(r"""
n_km = list(rb["et"])[0]; n_gm = list(rb["et"])[1]; n_hd = list(rb["et"])[2]
k_km, k_gm = rb["k_km"], rb["k_gm"]
pr = base["proteina_bs"]
otras_km = {s: P.ordenar(KMeans(n_clusters=k_km, n_init=N_INIT_KMEDIAS, random_state=s).fit_predict(rb["Z"]), pr) for s in range(SEMILLA + 1, SEMILLA + 6)}
otras_gm = {s: P.ordenar(GaussianMixture(n_components=k_gm, n_init=3, random_state=s).fit(rb["Z"]).predict(rb["Z"]), pr)
            for s in range(SEMILLA + 1, SEMILLA + 6)}
print("k-medias · ARI contra la semilla 42:", {s: round(adjusted_rand_score(rb["et"][n_km], e), 2) for s, e in otras_km.items()})
print("GMM · ARI contra la semilla 42:", {s: round(adjusted_rand_score(rb["et"][n_gm], e), 2) for s, e in otras_gm.items()})
bic_largo = {k: GaussianMixture(n_components=k, n_init=3, random_state=SEMILLA).fit(rb["Z"]).bic(rb["Z"]) for k in range(8, 17)}
print("BIC k = 8..16:", {k: round(v) for k, v in bic_largo.items()})
""")

md("""
### 3.6 · ¿Los clusters son tipos o clientes?
Con pocos clientes por tipo, un cluster puede ser la receta de un cliente y no un tipo de producto (pasó con el DDL
repostero de IND-51). Tabla cruzada cluster × cliente (clientes con al menos 15 muestras; «sin cliente» = sin dato).
""")

code(r"""
e = rb["et"][n_km]
cli = base["Cliente"].fillna("sin cliente")
t = pd.crosstab(pd.Series(e, name="cluster k-medias"), cli)
display(t.loc[:, t.sum() >= 15])
wpc = base["tipo"].eq("WPC")
print("WPC por cliente y cluster:"); display(pd.crosstab(cli[wpc].to_numpy(), e[wpc], rownames=["Cliente"], colnames=["cluster"]))
display(base[wpc].groupby(cli[wpc])[["proteina_bs", "grasa_bs", "cenizas_bs", "humedad"]].median().round(2))
""")

md("""
## 4 · Subtags
Regla (`clustering_polvos.proponer_subtags`): un cluster lleva subtag solo si corresponde a un tipo real:
1. **tipo** del rótulo con pureza ≥ 70 % (con ≥ 20 muestras con tipo, que sean al menos la mitad del cluster) y
   **cobertura** ≥ 50 % del tipo, sumando los clusters puros de ese tipo (un tipo puede ocupar dos regiones, como el WPC
   de dos clientes) → `POLVO_<TIPO>`;
2. si no, **familia** con pureza ≥ 70 % → `POLVO_LECHE`, `POLVO_SUERO_PERMEADO`, `POLVO_CONCENTRADO`.

Se aplica a los tres algoritmos del nivel base.
""")

code(r"""
props = {}
for n, e in rb["et"].items():
    desc = P.describir(base, e)
    p = P.proponer_subtags(desc)
    props[n] = p
    print(n); display(desc[["muestras", "clientes", "grasa_bs", "proteina_bs", "cenizas_bs", "tipo mayoritario", "% tipo",
                            "% cobertura tipo", "familia mayoritario", "% familia"]].join(p[["subtag", "criterio", "% cobertura"]])
                      .fillna({"subtag": "POLVO"}))
""")

code(r"""
st_km = P.subtag_por_muestra(rb["et"][n_km], props[n_km])
res = pd.DataFrame({"subtag": st_km, "tipo": base["tipo"].fillna("sin tipo")})
display(pd.crosstab(res["subtag"], res["tipo"], margins=True))
""")

md("""
### 4.1 · ¿Son estables los subtags?
La misma regla con k-medias y la mezcla corridas con otras semillas, y con HDBSCAN en cada min_cluster_size. Cada fila
es una corrida: cuántas muestras recibe cada subtag y el ARI de sus subtags contra los de k-medias con la semilla 42.
""")

code(r"""
filas = {}
corridas = {**{f"k-medias semilla {s}": e for s, e in {SEMILLA: rb["et"][n_km], **otras_km}.items()},
            **{f"GMM semilla {s}": e for s, e in {SEMILLA: rb["et"][n_gm], **otras_gm}.items()},
            **{f"HDBSCAN min {m}": P.ordenar(e, pr) for m, e in rb["hd"].items()}}
for nombre, e in corridas.items():
    t = P.subtag_por_muestra(e, P.proponer_subtags(P.describir(base, e)))
    filas[nombre] = {**pd.Series(t).value_counts().to_dict(), "ARI vs k-medias 42": adjusted_rand_score(st_km, t)}
estab = pd.DataFrame(filas).T
estab = estab[["ARI vs k-medias 42", "POLVO"] + sorted(c for c in estab if c.startswith("POLVO_"))]
estab.round(2).fillna("")
""")

md("""
## 5 · Las alternativas sin agrupar
### 5.1 · La leche en polvo: el corte de grasa del CAA
En la leche en polvo el tipo lo define la norma (CAA art. 567, sobre la grasa en g/100 g): descremada ≤ 1,5;
parcialmente descremada > 1,5 y < 26; entera ≥ 26. Se compara con `desc_grasa` del rótulo.
""")

code(r"""
le = polvos[polvos["familia"].eq("leche en polvo")]
display(pd.crosstab(le["tipo"].fillna("sin tipo"), le["clase_caa"].fillna("-"), margins=True))
ok = le["tipo"].isin(["LECHE ENTERA", "LECHE DESCREMADA", "LECHE PARCIALMENTE DESCREMADA"])
coincide = (le.loc[ok, "tipo"].str.replace("LECHE ", "").str.lower() == le.loc[ok, "clase_caa"]).mean()
print(f"rótulo (entera / descremada / parcialmente descremada) = clase del CAA: {100 * coincide:.1f} % de {ok.sum()}")
le.loc[ok & (le["tipo"].str.replace("LECHE ", "").str.lower() != le["clase_caa"]), "grasa"].describe().round(2)
""")

md("""
### 5.2 · Todos los polvos: ¿un modelo supervisado acierta el tipo en clientes no vistos?
No es el modelo propuesto: sirve para saber cuánto tipo hay en la composición. Tipos con al menos 10 muestras,
validación **por cliente** (se deja afuera un cliente por vez; las sin cliente van juntas) y estratificada (5 pliegues).
Métrica: exactitud balanceada. Se comparan (a) k-medias con cada cluster mapeado al tipo mayoritario *en el pliegue de
entrenamiento*, (b) un árbol de profundidad 3 (una regla legible) y (c) la regresión logística multinomial.
""")

code(r"""
cuenta = base["tipo"].value_counts()
sup = base[base["tipo"].isin(cuenta[cuenta >= TIPO_MIN].index)].reset_index(drop=True)
Xs, ys, gs = P.matriz(sup, P.BASE), sup["tipo"].to_numpy(), sup["Cliente"].fillna("sin cliente").to_numpy()
print(f"{len(sup)} muestras, {sup['tipo'].nunique()} tipos, {len(set(gs))} grupos de cliente:", cuenta[cuenta >= TIPO_MIN].to_dict())

class KmMapeado:
    # k-medias (k de la sección 3) + cada cluster → tipo mayoritario del entrenamiento
    def fit(self, X, y):
        self.esc = StandardScaler().fit(X); self.km = KMeans(n_clusters=k_km, n_init=N_INIT_KMEDIAS, random_state=SEMILLA).fit(self.esc.transform(X))
        self.mapa = pd.Series(y).groupby(self.km.labels_).agg(lambda s: s.mode()[0]).to_dict(); return self
    def predict(self, X):
        return np.array([self.mapa.get(g, "?") for g in self.km.predict(self.esc.transform(X))], dtype=object)

modelos = {"(a) k-medias mapeado": KmMapeado,
           "(b) árbol, profundidad 3": lambda: DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=SEMILLA),
           "(c) regresión logística": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, class_weight="balanced"))}

def validar(fabrica, cortes):
    pred = np.empty(len(ys), dtype=object)
    for tr, te in cortes:
        pred[te] = fabrica().fit(Xs[tr], ys[tr]).predict(Xs[te])
    return pred

cortes_cli = list(LeaveOneGroupOut().split(Xs, ys, gs))
cortes_est = list(StratifiedKFold(5, shuffle=True, random_state=SEMILLA).split(Xs, ys))
preds, filas = {}, {}
for n, f in modelos.items():
    preds[n] = validar(f, cortes_cli)
    filas[n] = {"estratificada": balanced_accuracy_score(ys, validar(f, cortes_est)), "por cliente": balanced_accuracy_score(ys, preds[n])}
val = pd.DataFrame(filas).T
display(val.round(3))
mejor = val["por cliente"].idxmax()
print(f"recall por tipo, por cliente · {mejor}:")
pd.crosstab(pd.Series(ys, name="real"), pd.Series(preds[mejor], name="predicho"))
""")

code(r"""
arbol = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=SEMILLA).fit(Xs, ys)
print(export_text(arbol, feature_names=[P.NOMBRES[v] for v in P.BASE], decimals=1))
""")

md("""
## 6 · Comparación con el agrupamiento general
¿Qué cambia contra los tags POLVO_GRASO, POLVO_MAGRO y PROT_ALTA del agrupamiento general (Senasa + MQ)? Filas: tag
general; columnas: subtag de esta prueba (k-medias).
""")

code(r"""
display(pd.crosstab(base["tag_senasa_mq"].fillna("no está"), pd.Series(st_km, name="subtag (polvos)"), margins=True))
print("Dentro de POLVO_MAGRO, tipo del rótulo × subtag de esta prueba:")
mg = base["tag_senasa_mq"].eq("POLVO_MAGRO").to_numpy()
pd.crosstab(base.loc[mg, "tipo"].fillna("sin tipo").to_numpy(), st_km[mg], rownames=["tipo"], colnames=["subtag (polvos)"], margins=True)
""")

md("""
## 7 · CSV de resultados
Una fila por polvo: Identificador, cliente, origen, familia, tipo del rótulo, clase CAA de la leche, composición (medida y
en base seca), tag del agrupamiento general, outlier por consenso, cluster de cada algoritmo (los outliers quedan con
el cluster que les asigna k-medias con `predict` en la misma escala) y el **subtag propuesto** (`subtag_propuesto`:
k-medias + `proponer_subtags`; los outliers quedan POLVO).
""")

code(r"""
salida = polvos[["Identificador", "Cliente", "origen", "fuente_familia", "familia", "tipo", "subtipo", "desc_grasa",
                 "clase_caa", "anio"] + COLUMNAS + P.BASE + ["resto_bs"]].copy().round(2)
salida = salida.join(tg, on="Identificador")
salida["outlier"] = es_out
idx = np.flatnonzero(~es_out)
for n, e in rb["et"].items():
    col = "cluster_" + n.split(" (")[0].lower().replace("-", "")
    salida[col] = pd.array([pd.NA] * len(salida), dtype="Int64"); salida.loc[idx, col] = e
esc_b = P.escalar(P.matriz(base, P.BASE))[1]
km_b = KMeans(n_clusters=k_km, n_init=N_INIT_KMEDIAS, random_state=SEMILLA).fit(rb["Z"])
mapa = dict(zip(km_b.labels_, rb["et"][n_km]))
salida.loc[es_out, "cluster_kmedias"] = [mapa[p] for p in km_b.predict(esc_b.transform(X_todas[es_out]))]
salida["subtag_propuesto"] = "POLVO"; salida.loc[idx, "subtag_propuesto"] = st_km
salida.to_csv(SALIDA / "polvos_clusters.csv", index=False, encoding="utf-8-sig")
tb.round(4).to_csv(SALIDA / "metricas_base.csv", encoding="utf-8-sig")
val.round(4).to_csv(SALIDA / "validacion_supervisada.csv", encoding="utf-8-sig")
print(salida.shape, "→", SALIDA / "polvos_clusters.csv")
print(salida["subtag_propuesto"].value_counts().to_dict())
sorted(p.name for p in SALIDA.iterdir())
""")

md("""
## Veredicto
""")

code(r"""
# números que cita el veredicto (se imprimen para que el texto no quede desfasado de la ejecución)
print(f"n = {len(polvos)} polvos; base = {len(base)} ({es_out.sum()} outliers)")
print(tb[["grupos", "ruido", "silueta", "silueta macro", "ARI tipo", "ARI familia"]].round(2))
print(comp[["k (base seca)", "k (medido)", "ARI tipo (base seca)", "ARI tipo (medido)", "ARI entre parametrizaciones"]].round(2))
print("subtags (k-medias 42):", pd.Series(st_km).value_counts().to_dict())
print("ARI de los subtags en las otras corridas:", estab["ARI vs k-medias 42"].round(2).to_dict())
print(f"leche en polvo · rótulo = CAA: {100 * coincide:.1f} %")
print(val.round(2))
r_ = estab["ARI vs k-medias 42"]
print("ARI de los subtags por algoritmo (min a max):", {a: f"{r_[[i for i in r_.index if i.startswith(a)]].min():.2f} a {r_[[i for i in r_.index if i.startswith(a)]].max():.2f}" for a in ["k-medias", "GMM", "HDBSCAN"]})
rango = lambda a: f"{min(a):.2f} a {max(a):.2f}"
print(f"ARI de la partición entre semillas: k-medias {rango([adjusted_rand_score(rb['et'][n_km], e) for e in otras_km.values()])} · GMM {rango([adjusted_rand_score(rb['et'][n_gm], e) for e in otras_gm.values()])}")
print(f"HDBSCAN: ruido {int((rb['et'][n_hd] == -1).sum())} ({100 * (rb['et'][n_hd] == -1).mean():.0f} %)")
print(P.describir(base, rb["et"][n_km])[["muestras", "grasa_bs", "proteina_bs", "cenizas_bs", "tipo mayoritario", "% tipo", "con tipo"]].to_string())
d_ = res["tipo"].eq("LECHE DESCREMADA")
print(f"leches descremadas: {int(d_.sum())}; sin subtag (POLVO): {int((res.loc[d_, 'subtag'] == 'POLVO').sum())}")
print("subtags de k-medias 42 + outliers:", {**pd.Series(st_km).value_counts().to_dict(), "outliers (POLVO)": int(es_out.sum())})
print("tags generales × subtag (k-medias):"); print(pd.crosstab(base["tag_senasa_mq"].fillna("no está"), pd.Series(st_km, name="subtag")).to_string())
""")

md("""
**Sí conviene un segundo nivel en los polvos, y lo da k-medias en base seca.** Con 1152 polvos (22 outliers por
consenso, base de 1130), k-medias elige k = 8 por silueta (0,78; macro 0,73) y su acuerdo con el tipo del rótulo (ARI 0,76) supera
al de los tags del agrupamiento general sobre las mismas muestras (0,70; silueta 0,64). HDBSCAN llega a un ARI parecido
(0,75) dejando 151 muestras (13 %) como ruido; la mezcla de gaussianas no sirve acá: el BIC sigue bajando hasta k = 12 (el
máximo probado) y más allá, la silueta cae a 0,45 (macro 0,49) y la partición cambia con la semilla (ARI 0,65 a 0,88 entre
semillas). k-medias da exactamente la misma partición con las seis semillas (ARI 1,0).

**Subtags propuestos (k-medias + `proponer_subtags`)**, en la columna `subtag_propuesto` de `polvos_clusters.csv`:

| subtag | muestras | qué hay |
|---|---|---|
| `POLVO_LECHE_ENTERA` | 605 | 521 entera + 49 modificada + 28 sin tipo + 7 de otros tipos; es el POLVO_GRASO del general (605 de las 605) |
| `POLVO_WPC` | 267 | 256 WPC + 7 suero + 2 proteínas lácteas + 2 de otros tipos; une dos clusters que son dos clientes (IND-06 e IND-68, cenizas 6,4 contra 5,3) |
| `POLVO_SUERO_PERMEADO` | 85 | suero desmineralizado, suero y permeado: por familia, no por tipo |
| `POLVO_CASEINATO` | 41 | 26 caseinato + 2 proteínas lácteas + 4 WPC + 9 sin tipo (81 % sobre los que tienen tipo) |
| `POLVO` | 132 + 22 outliers | sin subtag: la leche descremada mezclada con WPC y suero, y las caseínas y concentrados de más de 70 % de proteína que no son caseinato (5 caseína, 3 MPC) |

Ya no sale `POLVO_CONCENTRADO` (con k = 10 era un cluster de 6 muestras con 5 caseína y 1 MPC): con k = 8 esas caseínas
quedan en un cluster de 18 muestras, sin tipo mayoritario puro (36 % caseína), y no llevan subtag.

Los subtags son estables: ARI 1,0 entre semillas de k-medias, 0,91 a 0,93 con HDBSCAN en toda la grilla y 0,82 a
0,96 con la mezcla.

**Lo que no sale.** (1) **Leche descremada contra WPC**: en base seca la leche descremada y el WPC 35 tienen casi la
misma proteína (cluster de descremada 37,7; clusters de WPC 37,0 y 37,4), y los rangos se pisan; lo que las separa son las
cenizas (8,0 contra 5,3 a 6,4) y la grasa (2,6 contra 3,5 a 3,8). El cluster de k-medias queda con 64 % de descremada porque
se le suman muestras de un concentrado de suero de IND-70 (28 muestras, 26 rotuladas WPC) con 46 % de proteína y 7,8 de
cenizas en base seca, no el WPC 35, y 10 sueros; esas muestras no llevan subtag, y de las 74 leches descremadas 72 caen en
`POLVO`. HDBSCAN las aísla (93 %, 69 muestras en el cluster) pero deja 151 muestras como ruido; la mezcla de gaussianas las
junta peor (68 %: 89 muestras, 59 de las 74 leches descremadas) y con silueta 0,45. **En el notebook principal (`segundo_nivel.py`) esto se resuelve con una regla de composición antes del k-medias**:
grasa ≤ 2,5, cenizas ≥ 7 y proteína 30 a 40 en base seca → `POLVO_LECHE_DESCREMADA`. (2) **Leche modificada contra entera**:
la composición es la misma; la diferencia está en la receta, no en los cuatro análisis. (3) **Suero contra suero
desmineralizado**: van juntos por familia.

**La parametrización medida** (humedad, grasa, proteínas, cenizas) da k = 7 y un ARI con el tipo algo mayor (0,81),
porque no parte el WPC por cliente; la partición es casi la misma (ARI 0,92 con la de base seca) y los subtags, que ya
juntan los dos WPC, no cambian de fondo. Se queda la base seca por el criterio de variables (no mezclar derivadas con las
de origen) y por su silueta (0,78 contra 0,67).

**Sin agrupar.** En la leche en polvo, el corte del CAA sobre la grasa coincide con el rótulo en 95,8 % de 598: alcanza
para asignar las 32 leches sin tipo (la mayoría son enteras). Para todos los polvos, una regresión logística da exactitud
balanceada 0,88 en validación estratificada pero 0,58 con clientes no vistos; k-medias mapeado al tipo cae a 0,45 y un
árbol de profundidad 3, a 0,17. Hay tipo en la composición, pero buena parte del acierto viene de reconocer al cliente.

**No hay nivel extendido.** La lactosa informada (95 muestras, todas completadas con MQ) es el resto por diferencia en 74.

## Dudas
- La familia de las 570 muestras solo MQ (513 de las 679 leches en polvo) sale de `mq_tipo_muestra.csv` (MQ encolumnado),
  que no se revisó a mano. Si hay sueros rotulados como leche, ensucian `POLVO_LECHE_ENTERA` y el cluster de descremada.
- El WPC del rótulo junta concentraciones distintas (35, 60, 80): el cluster de proteína ≈ 37 es WPC35. Los WPC de
  más proteína no tienen un cluster propio: en k-medias 6 de los 291 caen en los clusters de más de 75 % de proteína (el de
  18 muestras, sin subtag, y el del caseinato) y con HDBSCAN 55 quedan como ruido.
  ¿Vale la pena un subtag por concentración?
- Caseinato, caseína y MPC tienen pocos clientes (1 a 4): `POLVO_CASEINATO` puede ser la receta de IND-70 más que un
  tipo. Con validación por cliente, el caseinato no se reconoce nunca (los 26 caen en WPC).
- El suero reengrasado (suero con oleomargarina; grasa 25 a 56 % y proteína 5 a 10 en base seca) acá queda como outlier
  o con la leche entera (4 con ese tipo; 3 son outliers y 1 cae en `POLVO_LECHE_ENTERA`; otros 5 sueros sin ese rótulo, con
  grasa de 25 a 51 %, también son outliers). En el notebook principal lo toma una regla antes del k-medias (grasa ≥ 15 y proteína ≤ 12 en
  base seca → `POLVO_SUERO_REENGRASADO`).
- ¿Qué es el concentrado de suero de IND-70 con 46 % de proteína y 7,8 de cenizas? No es WPC 35 ni leche descremada.
- La población cambió respecto de la versión anterior: se quitaron las 1839 MO de la tabla ancha y MQ sale de
  `MQ encolumnado.xlsx`. Por eso las cifras de acá (n = 1152) no son comparables una a una con las de antes (n = 1197).
""")

nb = nbf.v4.new_notebook()
nb["cells"] = c
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
if __name__ == "__main__":
    nbf.write(nb, SALIDA)
    print(f"Escrito: {SALIDA} ({len(c)} celdas)")
