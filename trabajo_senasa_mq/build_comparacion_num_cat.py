"""Arma comparacion_numerico_categorico.ipynb: clustering numérico (k-medias) contra clustering categórico (cat_*).

No editar el .ipynb a mano: se regenera con este script y se ejecuta desde esta carpeta.

    cd trabajo_senasa_mq
    ../.venv/bin/python build_comparacion_num_cat.py
    ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace comparacion_numerico_categorico.ipynb

Lee resultados/tablas/clusters_*.csv, categorias_*.csv y cortes_categorias.csv (no los modifica) y escribe en
resultados/comparacion_num_cat/. El código está en src/clustering_categorico.py.
"""
import sys
from pathlib import Path

import nbformat as nbf

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI / "src"))
import clustering_categorico as _CC          # solo para leer K_KMODAS al armar el texto fijo (el cuaderno lo vuelve a leer al ejecutarse)

K = _CC.K_KMODAS                             # k fijo del k-modas (5); el texto de los markdown sale de acá
K_SENS = _CC.K_SENSIBILIDAD                  # k de sensibilidad: el que se usó hasta el 2026-11-11 (6)
SALIDA = AQUI / "comparacion_numerico_categorico.ipynb"

c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: c.append(nbf.v4.new_code_cell(s.strip("\n")))

md(f"""
# Clustering con números contra clustering con categorías
**Trabajo Senasa + MQ · UNRaf · 2026**

Dos algoritmos sobre **las mismas muestras**: las que entran al k-medias actual (secciones 3 y 4 de
`trabajo_senasa_mq.ipynb`: humedad, grasa, proteínas y cenizas completas, suma ≤ 102, sin los outliers por consenso), en las dos
versiones (solo Senasa y Senasa + MQ). Las familias no lácteas (cárnicos, panificados / otros, agua / hisopado, base
nutricional) ya vienen excluidas de esos CSV, y los quesos que confirma el catálogo tampoco están: van a su rama propia
(humedad y materia grasa), sin cluster, y esta comparación solo toma las filas con cluster.

| | numérico | categórico |
|---|---|---|
| entrada | los cuatro análisis en g/100 g, escalados | los mismos cuatro análisis pasados a categorías (`cat_*`) |
| algoritmo | k-medias (euclídea) | k-modas (Hamming); como control, aglomerativo con Hamming y k-medias sobre one-hot |
| k | el de la clase 41 (por silueta: 5 en las dos versiones) | **fijo en {K}** en las dos versiones: el mismo k que el numérico y que el codo de solo Senasa (se explica en la sección 3); k = {K_SENS} se muestra solo como sensibilidad |

Las categorías van en dos variantes, porque los cortes de `cortes_categorias.csv` dependen de la familia:

- **(a) con familia**: las `cat_*` de `categorias_*.csv`, tal cual. «humedad = alto» es el tercio de arriba **dentro de
  su familia** (en un polvo, más de ~4 %; en un dulce de leche, más de ~29 %), y algunas clases del CAA existen solo en
  una familia (grasa «dulce de leche», «entera», humedad del queso). La categoría ya trae información de la familia:
  es una **fuga** si después se compara contra la familia.
- **(b) globales**: los mismos cortes para todas las muestras: los cortes «*» de `cortes_categorias.csv` (terciles
  sobre todo Senasa), sin familia.

**Familia, tag y subtipo no entran en ningún algoritmo**: quedan aparte, solo para evaluar.

Secciones: 0 preparación · 1 las muestras y las dos variantes de categorías · 2 el numérico reproducido · 3 el k del
categórico (fijo en {K}) y su estabilidad · 4 tabla comparativa · 5 tags numéricos contra clusters categóricos · 6 perfil y tag de cada cluster
categórico · 7 subtipos · 8 control con otros algoritmos categóricos · 9 muestras incompletas («no medido») ·
10 guardar · 11 datos usados en cada entrenamiento · Conclusión (sus números se calculan en la propia celda).
""")

md("""
## 0 · Preparación
Rutas de `src/rutas.py`, semilla 42. Se verifica si el paquete `kmodes` está en el entorno: **no está, y no se
instala**. Por eso el k-modas es una implementación corta en numpy dentro de `src/clustering_categorico.py` (el
algoritmo de Huang, 1998, el mismo que usa `kmodes` con `init="Huang"`), y los controles salen de sklearn
(`AgglomerativeClustering(metric="hamming")` y `KMeans` sobre one-hot).
""")

code(r"""
import sys, importlib.util
from pathlib import Path
SRC = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "rutas.py").exists()) / "src"
sys.path.insert(0, str(SRC))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import rutas as R
import clustering_categorico as CC
from datos_clase41 import COLUMNAS, SEMILLA

pd.set_option("display.width", 160); pd.set_option("display.max_columns", 30)
AZUL, ROJO, VERDE, AMBAR = "#0055a5", "#c0392b", "#2e8b57", "#f0a800"
K_SENS = CC.K_SENSIBILIDAD       # k de sensibilidad: el que se usó hasta el 2026-11-11 (el fijo es CC.K_KMODAS)


def ks_de(a, d):
    # los k que se muestran en las tablas: el fijo, el del numérico (si es otro) y el de sensibilidad
    return list(dict.fromkeys([d["k_opt"], a["K"], K_SENS]))


def rotulo(k, a, d):
    if k == d["k_opt"]:
        return "(fijo; = k del numérico)" if k == a["K"] else "(fijo)"
    return "(= k del numérico)" if k == a["K"] else "(sensibilidad)"


print("semilla:", SEMILLA, "· kmodes instalado:", importlib.util.find_spec("kmodes") is not None)
""")

md("""
## 1 · Las muestras y las dos variantes de categorías
`CC.analizar(version)` lee las muestras con tag de `clusters_<version>.csv`, les pega las `cat_*` de
`categorias_<version>.csv` (variante a) y calcula las globales (variante b, con `categorizar.categorizar` y solo los
cortes «*»). Después reproduce el k-medias y corre el barrido de k del categórico (secciones 2 y 3). Tarda menos de un
minuto por versión.
""")

code(r"""
A = {v: CC.analizar(v) for v in CC.VERSIONES}
for v, a in A.items():
    t = a["t"]
    print(f"{CC.VERSIONES[v]}: {len(t)} muestras · k numérico = {a['K']} · familias conocidas {int(t['familia_conocida'].sum())} · "
          f"con subtipo de evaluación {int(t['subtipo_eval'].notna().sum())}")
    for clave, d in a["var"].items():
        print(f"   variante ({clave}) {d['nombre']:12}: niveles por análisis {[len(n) for n in d['niveles'].values()]} · "
              f"perfiles distintos {d['perfiles']}")
""")

md("""
### La fuga, a la vista
En la variante (a), la misma categoría cubre rangos numéricos muy distintos según la familia, y la grasa tiene clases
que solo existen en una familia. En la (b) cada categoría es un solo rango para todas. La tabla muestra el rango de
humedad de «alto», «medio» y «bajo» por familia (Senasa + MQ, familias con 20 muestras o más) y las clases de grasa.
""")

code(r"""
t = A["senasa_mq"]["t"]
grandes = t["familia"].value_counts()[lambda s: s >= 20].index
rango = lambda s: f"{s.min():.1f} – {s.max():.1f}"
print("Humedad (g/100 g) de cada categoría, por familia · variante (a) con familia:")
display(t[t["familia"].isin(grandes)].pivot_table(index="familia", columns="cat_humedad", values="humedad", aggfunc=rango, fill_value=""))
print("Variante (b) globales:")
display(t[t["familia"].isin(grandes)].pivot_table(index="familia", columns="catg_humedad", values="humedad", aggfunc=rango, fill_value=""))
print("Grasa, variante (a): clases contra familia")
display(pd.crosstab(t["cat_materia_grasa"], t["familia"]).loc[:, lambda d: d.sum() >= 20])
""")

md("""
## 2 · El algoritmo numérico, reproducido
`CC.reproducir_numerico` repite la receta de `celdas_clase41.py` sobre las mismas muestras (escalado, `KMeans` con
`n_init=50` y semilla 42, k del CSV) y nombra los grupos con `etiquetar()` de `datos_clase41.py`. Falla si algún
tag no coincide con el de `clusters_*.csv`: si esta celda corre, el numérico es exactamente el del trabajo.
""")

code(r"""
from sklearn.metrics import silhouette_score
for v, a in A.items():
    t = a["t"]
    print(f"{CC.VERSIONES[v]}: k = {a['K']} · silueta euclídea {silhouette_score(a['Z'], a['et_num']):.3f} · "
          f"tags {t['tag_num'].value_counts().to_dict()}")
""")

md(f"""
## 3 · El k del categórico: fijo en {K}
Para cada versión y variante se corre el barrido de k de 2 a 14 con cuatro métodos: **k-modas** (el principal), aglomerativo con
Hamming (`average` y `complete`) y k-medias sobre one-hot. **El k del k-modas no se elige con la silueta: queda fijo en
`CC.K_KMODAS = {K}`**, en las dos versiones y en las dos variantes. Es una decisión (Matías, 2026-11-11), y se apoya en esto:

- **Es el mismo k que el del k-medias numérico**, que da 5 en las dos versiones: las dos particiones tienen el mismo número de
  grupos y se comparan de igual a igual.
- **Coincide con el codo del costo** (suma de desacuerdos del k-modas) de solo Senasa, en las dos variantes. En Senasa + MQ el
  codo da 5 en la variante (a) y 6 en la (b); la celda de abajo imprime los codos y la baja del costo.
- **La silueta de Hamming no sirve para elegirlo.** Con cuatro variables categóricas hay pocos perfiles distintos (la celda de
  abajo imprime cuántos): cada muestra es una de esas combinaciones de «bajo / medio / alto». Si se piden tantos grupos como
  combinaciones hay, cada grupo es una copia de un perfil y la silueta de Hamming da casi 1. Por eso **la silueta sube sola al
  aumentar k**, y el k de mayor silueta cae en el borde del rango probado (se ve en la tabla y en la figura de abajo).
- Mismo k en las dos versiones: las dos salidas son comparables entre sí.
- Contra: no sale de un criterio único, porque el codo de Senasa + MQ (b) está en 6 y no en 5, y porque hasta el 2026-11-11 el
  k fijo era 6 («el mejor punto»). Con la semilla 42 y `n_init = 50` el numérico pasó a 5 en las dos versiones y el codo ya no
  alcanza para sostener el 6, así que se cambió a {K}. **k = {K_SENS} queda como sensibilidad**: se calcula y se muestra al lado
  en las tablas de las secciones 4 a 6 y en la conclusión, para ver qué cambia.

**Reinicios.** El k-modas arranca de modas al azar y el costo no siempre baja al subir k con pocos reinicios, así
que se usan `CC.N_INIT = 1000` reinicios (≈ 1 s por corrida): con 1000, las cinco semillas probadas llegan al mismo costo
mínimo. Pero llegar al mismo costo no es llegar a la misma partición: hay particiones distintas con el mismo costo
(muestras que están a la misma distancia de dos modas). La estabilidad de la sección siguiente lo mide.
""")

code(r"""
for v, a in A.items():
    for clave, d in a["var"].items():
        b = d["barrido"]
        print(f"{CC.VERSIONES[v]} · variante ({clave}) {d['nombre']} · k del k-modas (fijo) = {d['k_opt']} (k del numérico = {a['K']}) · "
              f"k de mayor silueta = {d['k_sil']} (borde del rango: {CC.KS[-1]}) · codo del costo = {d['k_codo']} · perfiles distintos {d['perfiles']}")
        tab = b.pivot(index="k", columns="metodo", values="silueta_hamming").round(3)
        tab["grupo mínimo (k-modas)"] = b[b["metodo"] == "kmodas"].set_index("k")["grupo_min"]
        tab["costo k-modas"] = b[b["metodo"] == "kmodas"].set_index("k")["costo_kmodas"].astype(int)
        tab["baja del costo"] = tab["costo k-modas"].diff()
        display(tab.style.apply(lambda f: ["font-weight: bold" if f.name == d["k_opt"] else "" for _ in f], axis=1))
        sil, kf = tab["kmodas"], d["k_opt"]
        print(f"   k-modas: silueta {sil[kf - 1]:.3f} (k = {kf - 1}) → {sil[kf]:.3f} (k = {kf}) → {sil[kf + 1]:.3f} (k = {kf + 1}) · "
              f"baja del costo de {kf - 1} a {kf}: {-tab.loc[kf, 'baja del costo']:.0f}, de {kf} a {kf + 1}: {-tab.loc[kf + 1, 'baja del costo']:.0f}, "
              f"de {kf + 1} a {kf + 2}: {-tab.loc[kf + 2, 'baja del costo']:.0f} · grupo mínimo con k = {kf}: {int(tab.loc[kf, 'grupo mínimo (k-modas)'])} muestras "
              f"({100 * tab.loc[kf, 'grupo mínimo (k-modas)'] / len(a['t']):.1f} %)")

fig, ejes = plt.subplots(2, 2, figsize=(14, 8))
for fila, (v, a) in enumerate(A.items()):
    for col, (clave, d) in enumerate(a["var"].items()):
        ax = ejes[fila, col]; b = d["barrido"]
        for m, color in zip(CC.METODOS, [AMBAR, VERDE, AZUL, ROJO]):
            s = b[b["metodo"] == m]
            ax.plot(s["k"], s["silueta_hamming"], marker="o", color=color, label=m, lw=2.5 if m == "kmodas" else 1)
        if a["K"] == d["k_opt"]:
            ax.axvline(d["k_opt"], color=AZUL, ls="--", label=f"k fijo del k-modas = k del numérico ({d['k_opt']})")
        else:
            ax.axvline(a["K"], color="gray", ls=":", label=f"k del numérico ({a['K']})"); ax.axvline(d["k_opt"], color=AZUL, ls="--", label=f"k fijo del k-modas ({d['k_opt']})")
        ax.axvline(d["k_codo"], color=ROJO, ls=":", lw=1, label=f"codo del costo ({d['k_codo']})")
        ax.set_xticks(CC.KS)
        ax2 = ax.twinx(); s = b[b["metodo"] == "kmodas"]
        ax2.plot(s["k"], s["costo_kmodas"], color="black", alpha=0.35, ls="-."); ax2.set_ylabel("costo k-modas (gris)")
        ax.set_title(f"{CC.VERSIONES[v]} · ({clave}) {d['nombre']}"); ax.set_xlabel("k"); ax.set_ylabel("silueta (Hamming)")
ejes[0, 0].legend(fontsize=8)
plt.tight_layout(); plt.show()
""")

md(f"""
### Estabilidad con k = {K}
El k-modas con la variante (b) y k = {K}, corrido con cinco semillas distintas (`CC.estabilidad`): costo, grupo más chico y
**ARI entre todos los pares de corridas** (1 = misma partición). Si el costo es el mismo pero el ARI no llega a 1, hay
varias particiones empatadas en costo (muestras que están a la misma distancia de dos modas y van a una u otra).
Debajo, qué cambia de una semilla a otra con k = {K}: los clusters que salen MIXTO y a qué cluster va la mayoría de los DDL
heladeros. **Los conteos por cluster de este cuaderno valen para la semilla 42** (la de `SEMILLA`): con otra semilla los
totales por tag del numérico no cambian, pero la composición de los clusters que mezclan productos sí.
""")

code(r"""
ESTAB, HEL_SEM = {}, {}
for v, a in A.items():
    tab, aris = CC.estabilidad(a["var"]["b"]["X"])
    ESTAB[v] = (tab, aris)
    print(f"{CC.VERSIONES[v]} · variante (b), k = {CC.K_KMODAS}, {CC.N_INIT} reinicios · ARI entre corridas: mínimo {aris.min():.2f} · "
          f"media {aris.mean():.2f} · máximo {aris.max():.2f}")
    display(tab.set_index("semilla").T)
    # qué cambia entre semillas: los clusters MIXTO y dónde caen los DDL heladeros (dependen de la partición exacta)
    t, X = a["t"], a["var"]["b"]["X"]
    hel = (t["subtipo_eval"] == "DDL heladero").values
    filas = []
    for sem in tab["semilla"]:
        et, _, _ = CC.kmodas(X, CC.K_KMODAS, semilla=int(sem))
        _, tags_ = CC.perfil(t, et, a["var"]["b"]["cols"])
        cuenta = pd.Series(et[hel]).value_counts()
        filas.append({"semilla": sem, "clusters MIXTO": ", ".join(sorted(x for x in tags_.values() if x.startswith("MIXTO"))) or "-",
                      "heladeros en el cluster con más": f"{int(cuenta.iloc[0])} de {int(hel.sum())} ({tags_[cuenta.index[0]]})"})
    HEL_SEM[v] = pd.DataFrame(filas).set_index("semilla")
    display(HEL_SEM[v])
""")

md(f"""
## 4 · Tabla comparativa
Para cada versión: el numérico, y el k-modas en las dos variantes con k = {K} (fijo; es también el k del numérico) y con k = {K_SENS} como sensibilidad.

- **silueta propia**: en el espacio con el que se armó (euclídea escalada para el numérico; Hamming sobre sus
  categorías para el categórico). **Macro**: el promedio de la silueta media de cada grupo (un grupo chico pesa igual
  que uno grande).
- **silueta en el otro espacio**: el numérico medido con Hamming sobre las categorías de la variante (a o b, según la
  fila de comparación) y el categórico medido con euclídea sobre los análisis escalados.
- **ARI / NMI familia**: solo muestras con familia conocida (acá todas la traen; la de las solo MQ sale de MQ encolumnado, que la asigna un modelo).
- **ARI / NMI subtipo**: DDL familiar / repostero / heladero y la clase de humedad del CAA en el queso (los quesos de la rama propia no
  entran; quedan unos pocos con familia queso en Senasa + MQ, los que el catálogo no confirma: sección 7); **subtipo DDL**: lo mismo, solo dentro del dulce de leche.
- **ARI / NMI con el numérico**: acuerdo entre la partición categórica y la numérica (1 = idénticas, 0 = al azar).
- La fila **«+ subtags del DDL»** es el numérico con el segundo nivel de las secciones 3 y 4 (`subtag` de
  `clusters_*.csv`: DDL_HELADERO, DDL_FAMILIAR, DDL_REPOSTERO y el resto del DDL como DDL). Su silueta es baja a propósito: los subtags
  son un segundo nivel dentro del DDL, no grupos del primero; lo que importa de esa fila es el ARI contra el subtipo.

Una silueta propia **no es comparable** entre los dos espacios (Hamming y euclídea miden cosas distintas); la silueta
en el espacio del otro sí lo es dentro de cada espacio.
""")

code(r"""
filas = []
for v, a in A.items():
    t, Z = a["t"], a["Z"]
    for clave, d in a["var"].items():
        m = CC.metricas(t, a["et_num"], Z, d["X"], "euclidea")
        filas.append({"versión": CC.VERSIONES[v], "algoritmo": f"k-medias numérico (otro espacio = Hamming {clave})",
                      **m, "ARI con el numérico": 1.0, "NMI con el numérico": 1.0})
    et_sub = pd.factorize(t["subtag"])[0]          # el numérico con el segundo nivel del DDL (DDL_HELADERO, DDL_FAMILIAR)
    m = CC.metricas(t, et_sub, Z, a["var"]["b"]["X"], "euclidea")
    filas.append({"versión": CC.VERSIONES[v], "algoritmo": "k-medias numérico + subtags del DDL (otro espacio = Hamming b)",
                  **m, "ARI con el numérico": adjusted_rand_score(a["et_num"], et_sub),
                  "NMI con el numérico": normalized_mutual_info_score(a["et_num"], et_sub)})
    for clave, d in a["var"].items():
        for k in ks_de(a, d):
            et = d["part"][("kmodas", k)]
            m = CC.metricas(t, et, Z, d["X"], "hamming")
            filas.append({"versión": CC.VERSIONES[v],
                          "algoritmo": f"k-modas ({clave}) {d['nombre']}, k = {k} " + rotulo(k, a, d),
                          **m, "ARI con el numérico": adjusted_rand_score(a["et_num"], et),
                          "NMI con el numérico": normalized_mutual_info_score(a["et_num"], et)})
comparacion = pd.DataFrame(filas).set_index(["versión", "algoritmo"])
# las dos filas del numérico solo difieren en la silueta en el otro espacio
comparacion.round(3)
""")

md(f"""
## 5 · Tags numéricos contra clusters categóricos
Tabla cruzada con k = {K} (fijo; el del numérico) y con k = {K_SENS} (sensibilidad): filas = tag del k-medias, columnas = cluster del k-modas
(numerado por tamaño, 0 = el más grande). Si las dos particiones coincidieran, cada fila tendría una sola celda llena.
""")

code(r"""
for v, a in A.items():
    for clave, d in a["var"].items():
        for k in ks_de(a, d):
            et = d["part"][("kmodas", k)]
            print(f"{CC.VERSIONES[v]} · variante ({clave}) {d['nombre']} · k = {k} {rotulo(k, a, d)}")
            display(pd.crosstab(pd.Series(a["t"]["tag_num"].values, name="tag numérico"), pd.Series(et, name="cluster categórico"), margins=True, margins_name="total"))
""")

md(f"""
## 6 · Perfil y tag de cada cluster categórico
Por cluster (k = {K}, el fijo y el del numérico; y k = {K_SENS} como sensibilidad): categoría modal de cada análisis y qué fracción del grupo la tiene, familia
mayoritaria (entre las de familia conocida), tag numérico mayoritario y el centroide numérico en g/100 g (la media de
los análisis originales de las muestras del grupo, que el algoritmo **no** vio). El **tag** sale de ese centroide con
las mismas reglas que el numérico (`etiquetar()`: DDL, POLVO_GRASO, POLVO_MAGRO, PROT_ALTA, LIQUIDO, QUESO,
ALTA_HUMEDAD; si dos grupos caen en la misma regla, el de menor humedad y grasa conserva el nombre y el otro lleva `_2`,
`_3`: por ejemplo, los clusters de DDL del k-modas se llaman DDL, DDL_2, DDL_3…). Cuando el grupo junta productos
distintos (el tag numérico mayoritario cubre menos del 60 %), el centroide es un promedio sin sentido físico y el grupo
se rotula **`MIXTO_<tag>`** (por ejemplo líquidos con dulce de leche, o polvos magros con dulce de leche), para no hacer
pasar por un producto un grupo que no lo es. Había también un tag `POLVO` para los grupos que mezclaban solo polvos;
no apareció en ninguna de las 52 particiones probadas (2 versiones × 2 cortes × k de 2 a 14) y se quitó.
""")

code(r"""
PERFILES, TAGS_CAT = {}, {}
for v, a in A.items():
    for clave, d in a["var"].items():
        for k in ks_de(a, d):
            et = d["part"][("kmodas", k)]
            p, _ = CC.perfil(a["t"], et, d["cols"])
            PERFILES[(v, clave, k)] = p; TAGS_CAT[(v, clave, k)] = p["tag"].to_dict()
            print(f"{CC.VERSIONES[v]} · variante ({clave}) {d['nombre']} · k = {k} {rotulo(k, a, d)}")
            display(p)
""")

md(f"""
## 7 · Subtipos: ¿quién separa el dulce de leche familiar, repostero y heladero?
Dentro del dulce de leche con subtipo conocido: la composición media de cada subtipo, y cómo se reparten en los grupos
del numérico (tag y subtag) y del k-modas (k = {K}). Los quesos que confirma el catálogo no están en el k-medias; los que quedan con familia queso (solo MQ, sin confirmar por el rótulo) se muestran
aparte.
""")

code(r"""
for v, a in A.items():
    t = a["t"]; ddl = t["subtipo_eval"].str.startswith("DDL", na=False).values
    print(f"{CC.VERSIONES[v]} · DDL con subtipo: {int(ddl.sum())}")
    display(t[ddl].groupby("subtipo_eval")[COLUMNAS].agg(["mean", "std"]).round(2))
    partes = {"numérico (tag)": t["tag_num"].values, "numérico (subtag)": t["subtag"].values}
    for clave, d in a["var"].items():
        partes[f"k-modas ({clave})"] = d["part"][("kmodas", d["k_opt"])]
    tablas = [pd.crosstab(pd.Series(t.loc[ddl, "subtipo_eval"].values, name="subtipo"), pd.Series(np.asarray(e)[ddl], name=n)) for n, e in partes.items()]
    display(pd.concat(tablas, axis=1, keys=list(partes)))
q = A["senasa_mq"]["t"]; q = q[q["familia"] == "queso"]
print("Quesos (Senasa + MQ):")
display(q[["subtipo", "subtipo_eval", "tag_num"] + COLUMNAS].assign(
    **{f"k-modas ({c})": A["senasa_mq"]["var"][c]["part"][("kmodas", CC.K_KMODAS)][q.index] for c in "ab"}))
""")

md(f"""
## 8 · Control: otros algoritmos categóricos
¿El resultado es del k-modas o de las categorías? Con k = {K}: aglomerativo con Hamming (`average` y
`complete`) y k-medias sobre one-hot, contra el k-modas y contra el numérico. Si los cuatro dan ARI parecidos contra la
familia y contra el numérico, la conclusión depende de la codificación, no del algoritmo.
""")

code(r"""
filas = []
for v, a in A.items():
    for clave, d in a["var"].items():
        ref = d["part"][("kmodas", d["k_opt"])]
        for m in CC.METODOS:
            et = d["part"][(m, d["k_opt"])]
            ari_f, _ = CC.contra(et, a["t"]["familia"].where(a["t"]["familia_conocida"]))
            ari_s, _ = CC.contra(et, a["t"]["subtipo_eval"])
            filas.append({"versión": CC.VERSIONES[v], "variante": f"({clave}) {d['nombre']}", "método": m,
                          "silueta Hamming": d["barrido"].set_index(["metodo", "k"]).loc[(m, d["k_opt"]), "silueta_hamming"],
                          "grupo mínimo": int(np.bincount(et).min()),
                          "ARI con k-modas": adjusted_rand_score(ref, et), "ARI con el numérico": adjusted_rand_score(a["et_num"], et),
                          "ARI familia": ari_f, "ARI subtipo": ari_s})
CONTROL = pd.DataFrame(filas).set_index(["versión", "variante", "método"])
CONTROL.round(3)
""")

md(f"""
## 9 · Extra: el categórico admite muestras incompletas («no medido»)
El k-medias necesita los cuatro análisis. Con categorías, un análisis que falta puede ser una categoría más,
**«no medido»**. `CC.cargar_ampliado` toma, además de las muestras del k-medias, las que tienen entre 1 y 3 de los
cuatro análisis (sin las excluidas por humedad + sólidos). Para el clustering se piden **al menos 2 de 4** (con uno
solo, la muestra es casi toda «no medido»). Se corre el k-modas con k = {K} y se mira (i) cuántas muestras más
entran, (ii) si las muestras completas quedan agrupadas como sin las incompletas (ARI) y (iii) cuánto se agrupa por
«qué se midió» en vez de por composición (un grupo con «no medido» como moda es un grupo armado por el patrón de
faltantes, no por el producto).
""")

code(r"""
AMPLIADO, CONTEOS, AMP_ARI = {}, {}, {}
for v, a in A.items():
    amp, conteo = CC.cargar_ampliado(v)
    CONTEOS[v] = conteo
    print(f"{CC.VERSIONES[v]} · muestras por cantidad de análisis medidos (de humedad, grasa, proteínas, cenizas):")
    display(conteo)
    print(f"   entran con ≥ 2 de 4: {len(amp)} ({int(amp['completa'].sum())} completas + {int((~amp['completa']).sum())} incompletas, "
          f"×{len(amp) / amp['completa'].sum():.1f})")
    for clave, (nombre, cols) in CC.VARIANTES.items():
        X, _ = CC.codificar(amp, cols)
        et, _, _ = CC.kmodas(X, CC.K_KMODAS)
        AMPLIADO[(v, clave)] = (amp, et)
        sel = amp["completa"].values
        base = pd.Series(a["var"][clave]["part"][("kmodas", CC.K_KMODAS)], index=a["t"]["Identificador"])
        ari_c = adjusted_rand_score(base[amp.loc[sel, "Identificador"]].values, et[sel])
        ari_f, _ = CC.contra(et, amp["familia"].where(amp["familia"] != CC.SIN_FAMILIA))
        no_med = amp[cols].eq(CC.NO_MEDIDO).groupby(et).mean().round(2)
        no_med.columns = [c.split("_", 1)[1] for c in cols]
        AMP_ARI[(v, clave)] = ari_c
        print(f"   variante ({clave}) {nombre}: ARI de las completas contra el k-modas sin incompletas {ari_c:.2f} · ARI familia {ari_f:.2f}")
        display(no_med.assign(muestras=np.bincount(et), familia_mayoritaria=amp.groupby(et)["familia"].agg(lambda s: s.value_counts().index[0]))
                .rename_axis("cluster").add_prefix("").rename(columns=lambda c: c if c in ("muestras", "familia_mayoritaria") else f"% no medido {c}"))
""")

md(f"""
## 10 · Guardar
`resultados/comparacion_num_cat/comparacion_<version>.csv`: una fila por muestra del k-medias, con Identificador,
familia, subtipo, origen, tag numérico y, para cada variante y k (el fijo, {K}; el del numérico, si es otro; y {K_SENS} como sensibilidad), el cluster categórico y su tag.
Además la tabla comparativa (`tabla_comparativa.csv`) y los perfiles (`perfiles_<version>.csv`).
""")

code(r"""
CC.SALIDA.mkdir(parents=True, exist_ok=True)
for v, a in A.items():
    t = a["t"].copy(); cols = []
    for clave, d in a["var"].items():
        for k in ks_de(a, d):
            et = d["part"][("kmodas", k)]; nom = f"{clave}_k{k}"
            t[f"cluster_cat_{nom}"] = et; t[f"tag_cat_{nom}"] = [TAGS_CAT[(v, clave, k)][g] for g in et]
            cols += [f"cluster_cat_{nom}", f"tag_cat_{nom}"]
    ruta = CC.guardar(t, cols, v)
    pd.concat({f"({c}) k={k}": p for (vv, c, k), p in PERFILES.items() if vv == v}, names=["variante", "cluster"]).to_csv(
        CC.SALIDA / f"perfiles_{v}.csv", encoding="utf-8-sig")
    print("Guardado:", ruta.relative_to(R.RAIZ), pd.read_csv(ruta).shape)
comparacion.round(4).to_csv(CC.SALIDA / "tabla_comparativa.csv", encoding="utf-8-sig")
print("Guardado:", (CC.SALIDA / "tabla_comparativa.csv").relative_to(R.RAIZ))
""")

md(f"""
## 11 · Datos usados en cada entrenamiento
Cuántas muestras entran a cada modelo de este cuaderno, con el mismo formato que en los demás (parte II del cuaderno
general). No hay partición en entrenamiento y prueba: son agrupamientos, se entrenan y se evalúan con las mismas
muestras. Los números salen de las variables de arriba, no están escritos a mano. Los modelos que se entrenan son el
k-medias numérico de referencia, el k-modas (variantes a y b, con k = {K} y k = {K_SENS} de sensibilidad), los dos controles categóricos
(aglomerativo con Hamming y k-medias sobre one-hot) y el k-modas con «no medido» de la sección 9.
""")

code(r"""
import datos_entrenamiento as DE
from datos_clase41 import N_INIT_KMEDIAS

PART = "sin partición: no supervisado, se entrena y se evalúa con las mismas muestras"
filas, DISP = [], {}
for v, a in A.items():
    cl = pd.read_csv(R.TABLAS / f"clusters_{v}.csv", encoding="utf-8-sig")
    rama = cl["rama"] == "k-medias"
    n_disp, n_usa, n_queso = int(rama.sum()), len(a["t"]), int((~rama).sum())
    assert n_usa == int(cl["cluster"].notna().sum()), "las muestras del cuaderno no son las del k-medias"
    DISP[v] = n_disp
    nom = CC.VERSIONES[v]
    datos = f"{nom}: muestras de la rama k-medias (cuatro análisis completos, suma ≤ 102, lácteas)"
    comun = dict(disponibles=n_disp, usadas=n_usa, motivo=f"{n_disp - n_usa} outliers por consenso del k-medias", particion=PART)
    fuera = f"los {n_queso} quesos de la rama propia no están"
    barrido = f"k de {CC.KS[0]} a {CC.KS[-1]} en las dos variantes"
    filas += [
        DE.fila("II", f"k-medias numérico de referencia, {nom}", datos + ", escaladas",
                nota=f"k = {a['K']}, n_init = {N_INIT_KMEDIAS}, semilla {SEMILLA}; reproduce el del cuaderno principal; {fuera}", **comun),
        DE.fila("II", f"k-modas (a) con familia, {nom}", datos + ", con los cuatro análisis pasados a categorías (cortes por familia)",
                nota=f"k = {CC.K_KMODAS} (fijo; el del numérico es {a['K']}) y k = {K_SENS} (sensibilidad); barrido de {barrido}; {CC.N_INIT} reinicios por corrida", **comun),
        DE.fila("II", f"k-modas (b) globales, {nom}", datos + ", con los cuatro análisis pasados a categorías (cortes «*» globales)",
                nota=f"k = {CC.K_KMODAS} (fijo; el del numérico es {a['K']}) y k = {K_SENS} (sensibilidad); barrido de {barrido}; {CC.N_INIT} reinicios por corrida; "
                     f"estabilidad con {len(ESTAB[v][0])} semillas", **comun),
        DE.fila("II", f"aglomerativo con Hamming (average y complete), {nom}", datos + ", categorías de las variantes (a) y (b)",
                nota=f"control; k = {CC.K_KMODAS} en las tablas y barrido de {barrido}", **comun),
        DE.fila("II", f"k-medias sobre one-hot, {nom}", datos + ", categorías de las variantes (a) y (b) en one-hot",
                nota=f"control; n_init = {N_INIT_KMEDIAS}; k = {CC.K_KMODAS} en las tablas y barrido de {barrido}", **comun),
    ]
    cont, amp = CONTEOS[v], AMPLIADO[(v, "b")][0]
    n_tot, n_amp, n_comp = int(cont.loc["total", "total"]), len(amp), int(amp["completa"].sum())
    filas.append(DE.fila(
        "II", f"k-modas con «no medido» (a) y (b), k = {CC.K_KMODAS}, {nom}",
        f"{nom}: muestras con 1 a 4 de los cuatro análisis (las completas del k-medias y las incompletas, sin las excluidas)",
        n_tot, n_amp, motivo=f"{n_tot - n_amp} con un solo análisis medido (se piden al menos 2 de 4)", particion=PART,
        nota=f"{n_comp} completas + {n_amp - n_comp} incompletas; {n_amp} de {n_tot}; el análisis que falta es una categoría más"))
display(DE.guardar(filas, "comparacion_numerico_categorico"))
""")

md("""
## Conclusión
Los números de esta conclusión **se calculan en la celda de abajo** con las salidas de este mismo cuaderno (semilla 42):
si cambian los datos o la semilla, cambian solos. Lo que no cambia solo son las afirmaciones, que hay que revisar.
""")

code(r"""
from IPython.display import Markdown

cm = lambda x, d=2: f"{round(float(x), d) + 0:.{d}f}".replace(".", ",")      # coma decimal, sin «-0,00»
pc = lambda x: f"{100 * x:.0f} %"
S1, S2 = "senasa", "senasa_mq"
NOM, KF = CC.VERSIONES, CC.K_KMODAS
n_muestras = {v: len(a["t"]) for v, a in A.items()}
n_out = {v: DISP[v] - n_muestras[v] for v in A}
K_NUM = A[S1]["K"]


def met(v, clave, k):
    # métricas del k-modas (variante `clave`, k) y el ARI contra el numérico
    a = A[v]; d = a["var"][clave]; et = d["part"][("kmodas", k)]
    m = CC.metricas(a["t"], et, a["Z"], d["X"], "hamming")
    m["ARI numérico"] = adjusted_rand_score(a["et_num"], et)
    return m


MET = {(v, c, k): met(v, c, k) for v in A for c in "ab" for k in dict.fromkeys([KF, A[v]["K"], K_SENS])}
NUM = {v: CC.metricas(a["t"], a["et_num"], a["Z"], a["var"]["b"]["X"], "euclidea") for v, a in A.items()}
SUBTAG = {v: CC.metricas(a["t"], pd.factorize(a["t"]["subtag"])[0], a["Z"], a["var"]["b"]["X"], "euclidea") for v, a in A.items()}
SIL_NUM = {v: silhouette_score(a["Z"], a["et_num"]) for v, a in A.items()}


def donde_caen(v, k=KF):
    # dónde caen los tags numéricos que no son DDL en el k-modas (b), y cuánto DDL arrastra ese cluster
    t = A[v]["t"]; et = A[v]["var"]["b"]["part"][("kmodas", k)]
    ct = pd.crosstab(t["tag_num"], et); tags = TAGS_CAT[(v, "b", k)]
    hel = (t["subtipo_eval"] == "DDL heladero").values
    por_g = {}
    for tag in ["POLVO_GRASO", "POLVO_MAGRO", "PROT_ALTA", "LIQUIDO"]:
        if tag not in ct.index:
            continue
        g = ct.loc[tag].idxmax()
        if ct.loc[tag, g] / ct.loc[tag].sum() >= 0.5:
            por_g.setdefault(g, []).append(f"{ct.loc[tag, g]} de {ct.loc[tag].sum()} {tag}")
    return "; ".join(f"{', '.join(x)} van al cluster {g} (`{tags[g]}`, {ct[g].sum()} muestras, con {ct.loc['DDL', g]} DDL y {int((hel & (et == g)).sum())} heladeros)"
                     for g, x in por_g.items())


def ddl_partido(v, k=KF):
    # cuántos clusters son DDL casi puro y cuáles mezclan
    t = A[v]["t"]; et = A[v]["var"]["b"]["part"][("kmodas", k)]
    ct = pd.crosstab(t["tag_num"], et); share = ct.loc["DDL"] / ct.sum(); tags = TAGS_CAT[(v, "b", k)]
    puros = int((share >= 0.95).sum())
    mezcla = ", ".join(f"`{tags[g]}` ({pc(share[g])} de DDL)" for g in ct.columns if share[g] < 0.95)
    return f"{puros} de {k} clusters son DDL casi puro (95 % o más) y los otros mezclan: {mezcla}"


def n_puros(v, k=KF):
    t = A[v]["t"]; et = A[v]["var"]["b"]["part"][("kmodas", k)]; ct = pd.crosstab(t["tag_num"], et)
    return int(((ct.loc["DDL"] / ct.sum()) >= 0.95).sum())


def mixtos(v, k):
    m = [x for x in TAGS_CAT[(v, "b", k)].values() if x.startswith("MIXTO")]
    return f"{len(m)}" + (f" (`{'`, `'.join(m)}`)" if m else "")


def ddl_en_tag(v):
    t = A[v]["t"]; ddl = t["familia"] == "dulce de leche"; en = ddl & (t["tag_num"] == "DDL")
    otros = t.loc[ddl & ~en, "tag_num"].value_counts()
    resto = ", ".join(f"{n} en {tg}" for tg, n in otros.items()) or "ninguno en otro tag"
    return f"{int(en.sum())} de los {int(ddl.sum())} DDL en el tag DDL ({resto})"


def subtipos(v):
    t = A[v]["t"]; ddl = t["subtipo_eval"].str.startswith("DDL", na=False)
    tab = pd.crosstab(t.loc[ddl, "subtipo_eval"], t.loc[ddl, "subtag"]); tot = tab.sum(axis=1)
    return (f"DDL_HELADERO junta {tab.loc['DDL heladero', 'DDL_HELADERO']} de los {tot['DDL heladero']} heladeros y "
            f"{tab['DDL_HELADERO'].sum() - tab.loc['DDL heladero', 'DDL_HELADERO']} muestras de otros subtipos; DDL_FAMILIAR junta "
            f"{tab.loc['DDL familiar', 'DDL_FAMILIAR']} de los {tot['DDL familiar']} familiares, con {tab.loc['DDL repostero', 'DDL_FAMILIAR']} reposteros; "
            f"DDL_REPOSTERO junta {tab.loc['DDL repostero', 'DDL_REPOSTERO']} de los {tot['DDL repostero']} reposteros")


def heladeros_k(v, k=KF):
    t = A[v]["t"]; et = A[v]["var"]["b"]["part"][("kmodas", k)]; tags = TAGS_CAT[(v, "b", k)]
    hel = (t["subtipo_eval"] == "DDL heladero").values; cuenta = pd.Series(et[hel]).value_counts().head(2)
    return " y ".join(f"{n} de {int(hel.sum())} en el cluster {g} (`{tags[g]}`)" for g, n in cuenta.items())


def mayoria(n, k=KF):
    return "la mayoría de los clusters" if 2 * n > k else "menos de la mitad de los clusters"


def liquidos_en_ddl(v, k=KF):
    # el cluster donde cae la mayoría de los LIQUIDO (b): cuánto de ese cluster es DDL
    t = A[v]["t"]; et = A[v]["var"]["b"]["part"][("kmodas", k)]; ct = pd.crosstab(t["tag_num"], et)
    g = ct.loc["LIQUIDO"].idxmax()
    return f"{int(ct.loc['LIQUIDO', g])} de {int(ct.loc['LIQUIDO'].sum())} LIQUIDO en un cluster de {int(ct[g].sum())} muestras con {pc(ct.loc['DDL', g] / ct[g].sum())} de DDL"


def semillas_hel(v):
    # con las 5 semillas: cuántos clusters distintos (por tag) se llevan la mayoría de los heladeros y cuántos conjuntos MIXTO distintos salen
    h = HEL_SEM[v]["heladeros en el cluster con más"].str.extract(r"\((.*)\)")[0]
    return h.nunique(), ", ".join(f"`{x}`" for x in h.unique()), HEL_SEM[v]["clusters MIXTO"].nunique()


def familias_por_tag(v):
    t = A[v]["t"]; ct = pd.crosstab(t["tag_num"], t["familia"])
    return (ct.max(axis=1) / ct.sum(axis=1)), ct


def faltantes(v):
    # clusters armados por «qué se midió»: un análisis (o dos) en «no medido» en el 75 % o más del grupo
    amp, et = AMPLIADO[(v, "b")]; nm = amp[CC.CAT_B].eq(CC.NO_MEDIDO).groupby(et).mean(); out, n = [], 0
    for g in nm.index:
        mal = [c for c in CC.CAT_B if nm.loc[g, c] >= 0.75]
        if mal:
            m = nm.loc[g, mal].min(); n += int((et == g).sum())
            out.append(f"un cluster de {int((et == g).sum())} muestras con {' y '.join(c.split('_', 1)[1] for c in mal)} «no medido» "
                       f"{'en el' if m >= 0.995 else 'en al menos el'} {pc(m)}")
    return "; ".join(out), n


def perfil_a(v, k=KF):
    # familia mayoritaria de cada cluster de la variante (a): cuántos son de dulce de leche y en qué rango de %
    fm = PERFILES[(v, "a", k)]["familia mayoritaria"]
    nombre = fm.str.replace(r" \(\d+ %\)", "", regex=True); pct = fm.str.extract(r"\((\d+) %\)")[0].astype(int)
    d = nombre == "dulce de leche"
    return f"{int(d.sum())} de {len(fm)} clusters tienen al dulce de leche como familia mayoritaria ({pct[d].min()} a {pct[d].max()} %)"


cortes = pd.read_csv(R.TABLAS / "cortes_categorias.csv", encoding="utf-8-sig")
cortes = cortes[(cortes["familia"] == "*") & (cortes["analisis"] == "humedad")].set_index("categoria")["hasta"]
polvos = lambda v: A[v]["t"]["tag_num"].isin(["POLVO_GRASO", "POLVO_MAGRO", "PROT_ALTA"]).values
polvo_bajo = {v: (A[v]["t"].loc[polvos(v), "catg_humedad"] == "bajo").mean() for v in A}
import re
ddl_sh = {v: (A[v]["t"]["familia"] == "dulce de leche").mean() for v in A}
hum_ddl = A[S2]["t"].loc[A[S2]["t"]["familia"] == "dulce de leche", "humedad"].quantile([0.25, 0.5, 0.75])
alto_a = A[S2]["t"].groupby("familia")["humedad"].apply(lambda x: x[A[S2]["t"].loc[x.index, "cat_humedad"] == "alto"].min())
ce_hel = {v: A[v]["t"].loc[A[v]["t"]["subtipo_eval"] == "DDL heladero", "cenizas"].mean() for v in A}
ce_ddl = {v: A[v]["t"].loc[A[v]["t"]["subtipo_eval"].isin(["DDL familiar", "DDL repostero"]), "cenizas"].mean() for v in A}
modas = [int(re.search(r"\((\d+) %", x).group(1)) for v in A for c in ["humedad", "materia_grasa", "proteinas", "cenizas"] for x in PERFILES[(v, "b", KF)][c]]
cg_ddl = A[S2]["t"].loc[A[S2]["t"]["cat_materia_grasa"] == "dulce de leche", "familia"].value_counts()
nq_amp = {v: int((AMPLIADO[(v, "b")][0]["familia"] == "queso").sum()) for v in A}

fam_tag = {v: familias_por_tag(v) for v in A}
sin_liq = pd.concat([f.drop("LIQUIDO", errors="ignore") for f, _ in fam_tag.values()])
liq = {v: (fam_tag[v][0]["LIQUIDO"], fam_tag[v][1].loc["LIQUIDO"].sort_values(ascending=False)) for v in A}
nq = int((A[S2]["t"]["familia"] == "queso").sum())

b = {v: A[v]["var"]["b"] for v in A}
cuerpo = {v: b[v]["barrido"][b[v]["barrido"]["metodo"] == "kmodas"].set_index("k") for v in A}
baja = {v: (cuerpo[v].loc[KF - 1, "costo_kmodas"] - cuerpo[v].loc[KF, "costo_kmodas"], cuerpo[v].loc[KF, "costo_kmodas"] - cuerpo[v].loc[KF + 1, "costo_kmodas"]) for v in A}
n_codo = sum(d["k_codo"] == KF for v in A for d in A[v]["var"].values())
codo_v = {v: " y ".join(f"{A[v]['var'][c]['k_codo']} en ({c})" for c in "ab") for v in A}
chico = {v: cuerpo[v].loc[KF, "grupo_min"] for v in A}
codos = ", ".join(f"{d['k_codo']} en {NOM[v]} ({c})" for v, a in A.items() for c, d in a["var"].items())
aris = {v: ESTAB[v][1] for v in A}
mismo_costo = all(ESTAB[v][0]["costo"].nunique() == 1 for v in A)
n_sem = len(ESTAB[S1][0])
agl = CONTROL.xs("aglomerativo-average", level="método")["grupo mínimo"]
agl_txt = ", ".join(f"{n} en {var} de {ver}" for (ver, var), n in agl.items())
rng_ctrl = CONTROL[["ARI familia", "ARI con el numérico"]]
amp_n = {v: len(AMPLIADO[(v, "b")][0]) for v in A}
amp_ari = [x for x in AMP_ARI.values()]

texto = f'''
Los números citados son los de la tabla de la sección 4 (k-modas con k = {KF}, fijo, salvo que se diga otra cosa), con las
muestras del k-medias actual: {n_muestras[S1]} en solo Senasa y {n_muestras[S2]} en Senasa + MQ, ya sin los outliers por consenso
({n_out[S1]} y {n_out[S2]}), sin las familias no lácteas ni los quesos de la rama propia. La silueta propia **no se compara**
entre algoritmos: una es euclídea sobre números y la otra es de Hamming sobre 4 símbolos. Todo con la semilla {SEMILLA}.

**En qué coinciden, a medias.** Con categorías globales (b) y k = {KF}, los polvos se juntan en pocos clusters, pero ninguno queda limpio: todos
arrastran dulce de leche, y los líquidos no tienen cluster propio (en solo Senasa, {liquidos_en_ddl(S1)}; en Senasa + MQ, {liquidos_en_ddl(S2)}). En solo Senasa: {donde_caen(S1)}. En Senasa + MQ: {donde_caen(S2)}. Los quesos ya no forman parte de esta comparación (van a su rama
propia): en Senasa + MQ quedan {nq} muestras con familia queso, las que el catálogo no confirma, así que no se puede ver si el k-modas separa
las clases de humedad del CAA.

**En qué difieren.** El k-medias usa sus {K_NUM} grupos para los **tipos de producto**: ARI contra la familia de {cm(NUM[S1]["ARI familia"])} en solo
Senasa y {cm(NUM[S2]["ARI familia"])} en Senasa + MQ, y casi todo el dulce de leche en un solo tag (en solo Senasa, {ddl_en_tag(S1)}; en Senasa + MQ,
{ddl_en_tag(S2)}). El k-modas dedica {mayoria(n_puros(S1))} (solo Senasa) y {mayoria(n_puros(S2))} (Senasa + MQ) a **partir el dulce de leche**: en solo Senasa, {ddl_partido(S1)};
en Senasa + MQ, {ddl_partido(S2)}. Eso pasa porque el dulce de leche es el {pc(ddl_sh[S1])} de las muestras de solo Senasa y el {pc(ddl_sh[S2])} de Senasa + MQ, y los cortes
globales de humedad ({cm(cortes["bajo"])} y {cm(cortes["medio"])}) caen en medio de su rango (mediana {cm(hum_ddl[0.5], 1)}, 50 % central entre {cm(hum_ddl[0.25], 1)} y {cm(hum_ddl[0.75], 1)} en
Senasa + MQ): el {pc(polvo_bajo[S1])} (solo Senasa) y el {pc(polvo_bajo[S2])} (Senasa + MQ) de los polvos quedan en «bajo» y el DDL se reparte en las tres categorías. El acuerdo entre las dos particiones es bajo: ARI de
{cm(MET[(S1, "b", KF)]["ARI numérico"])} en solo Senasa y {cm(MET[(S2, "b", KF)]["ARI numérico"])} en Senasa + MQ con k = {KF} ({cm(MET[(S1, "b", K_SENS)]["ARI numérico"])} y {cm(MET[(S2, "b", K_SENS)]["ARI numérico"])} con k = {K_SENS}, la sensibilidad).
Además salen clusters MIXTO (su centroide es un promedio de productos distintos): con k = {KF}, {mixtos(S1, KF)} en solo Senasa y
{mixtos(S2, KF)} en Senasa + MQ; con k = {K_SENS}, {mixtos(S1, K_SENS)} y {mixtos(S2, K_SENS)}. Cuántos MIXTO salen y cuáles dependen del k y, en Senasa + MQ, de la semilla (ver la
tabla de estabilidad de la sección 3).

**La fuga de la variante (a) no hace lo que se esperaba.** Esas categorías son relativas a la familia: «alto» en una
leche en polvo es más de {cm(alto_a["leche en polvo"], 1)} % de humedad y en un DDL más de {cm(alto_a["dulce de leche"], 1)} % (Senasa + MQ), pero para el algoritmo es el mismo símbolo. El k-modas agrupa
entonces por el **perfil dentro de la familia** («alto en proteína y cenizas para lo que es») y mezcla familias dentro de cada cluster: en solo Senasa, {perfil_a(S1)}; en
Senasa + MQ, {perfil_a(S2)}. El ARI
contra la familia da {cm(MET[(S1, "a", KF)]["ARI familia"])} en solo Senasa y {cm(MET[(S2, "a", KF)]["ARI familia"])} en Senasa + MQ, y contra el numérico {cm(MET[(S1, "a", KF)]["ARI numérico"])} y {cm(MET[(S2, "a", KF)]["ARI numérico"])}. La fuga más directa
es la clase de grasa («dulce de leche», «entera», «descremada»), que nombra a la familia ({cg_ddl.iloc[0]} muestras con la clase «dulce de leche» y todas de familia {cg_ddl.index[0]}), pero es una variable de
cuatro. Conclusión práctica: para agrupar sin familia hay que usar la variante (b), y con (a) cualquier comparación
contra la familia está contaminada en los dos sentidos.

**Subtipos.** Con el tag solo, el numérico no separa nada dentro del DDL (ARI subtipo DDL = {cm(NUM[S1]["ARI subtipo DDL"])} en solo Senasa y {cm(NUM[S2]["ARI subtipo DDL"])} en Senasa + MQ: los
tres subtipos tienen el mismo tag). Con el **subtag** (el segundo nivel de las secciones 3 y 4 del trabajo) el ARI sube a
{cm(SUBTAG[S1]["ARI subtipo DDL"])} en solo Senasa y {cm(SUBTAG[S2]["ARI subtipo DDL"])} en Senasa + MQ. En solo Senasa, {subtipos(S1)}. Con MQ, {subtipos(S2)}. El categórico (b) con k = {KF} llega a
{cm(MET[(S1, "b", KF)]["ARI subtipo DDL"])} en solo Senasa y {cm(MET[(S2, "b", KF)]["ARI subtipo DDL"])} en Senasa + MQ ({cm(MET[(S1, "b", K_SENS)]["ARI subtipo DDL"])} y {cm(MET[(S2, "b", K_SENS)]["ARI subtipo DDL"])} con k = {K_SENS}). Los heladeros tienen
cenizas de {cm(ce_hel[S1], 1)} en solo Senasa ({cm(ce_ddl[S1], 1)} los otros DDL) y {cm(ce_hel[S2], 1)} en Senasa + MQ ({cm(ce_ddl[S2], 1)}), pero no tienen un cluster propio: en solo Senasa, {heladeros_k(S1)}; en Senasa + MQ, {heladeros_k(S2)}. En qué cluster caen
**depende de la semilla** (sección 3, con k = {KF} y la variante b): con las {n_sem} semillas, la mayoría de los heladeros de solo Senasa va a {semillas_hel(S1)[0]} clusters distintos ({semillas_hel(S1)[1]})
y la de Senasa + MQ a {semillas_hel(S2)[0]} ({semillas_hel(S2)[1]}). Es un costo de medir con
Hamming sobre 4 símbolos. Para los subtipos del DDL, el subtag numérico separa mejor que el categórico.

**Interpretabilidad.** Cada cluster categórico se lee como una regla («humedad medio, grasa medio, proteína alta,
cenizas medio»), con modas que cubren del {min(modas)} al {max(modas)} % del grupo. Pero las categorías globales son terciles de todas las muestras de Senasa, no de un
producto, así que la regla dice poco del producto. El numérico es más interpretable como **tipo de
producto**: tags con {pc(sin_liq.min())} a {pc(sin_liq.max())} de una familia (salvo LIQUIDO, con {pc(liq[S1][0])} en solo Senasa, {int(fam_tag[S1][1].loc["LIQUIDO"].sum())} muestras, y {pc(liq[S2][0])} en Senasa + MQ, donde
junta {liq[S2][1].index[0]}, {liq[S2][1].index[1]} y {liq[S2][1].index[2]}), silueta de {cm(SIL_NUM[S1], 3)} y {cm(SIL_NUM[S2], 3)}. La familia de las solo MQ sale de un modelo, no está revisada a mano. Medido
en el espacio del otro, el numérico sigue siendo positivo con Hamming (b), más débil en solo Senasa: {cm(NUM[S1]["silueta en el otro espacio"])} y {cm(NUM[S2]["silueta en el otro espacio"])}. El
categórico, medido con euclídea, da {cm(MET[(S1, "b", KF)]["silueta en el otro espacio"])} y {cm(MET[(S2, "b", KF)]["silueta en el otro espacio"])} con k = {KF}.

**Elegir k.** El k del k-modas queda **fijo en {KF}**, en las dos versiones y las dos variantes (decisión de Matías, 2026-11-11), y no sale de la
silueta. Se eligió {KF} porque coincide con el k del k-medias numérico ({K_NUM} en las dos versiones) y con el codo del costo de solo Senasa. El codo da {codo_v[S1]} en solo Senasa y {codo_v[S2]} en
Senasa + MQ: {KF} en {n_codo} de las 4 combinaciones de versión y variante. Hasta el 2026-11-11 el k fijo era {K_SENS}; con la semilla {SEMILLA} el numérico da {K_NUM} y el codo ya no
alcanza para sostener el {K_SENS}. La silueta con Hamming no sirve para elegir: con cuatro variables categóricas hay solo {b[S1]["perfiles"]} (solo Senasa) y {b[S2]["perfiles"]} (Senasa + MQ) perfiles
distintos en la variante (b), y la silueta sube sola con k porque cada cluster se vuelve una copia de un perfil: el k de mayor silueta (variante b) es {b[S1]["k_sil"]} en solo Senasa y
{b[S2]["k_sil"]} en Senasa + MQ, en el borde del rango probado (de {CC.KS[0]} a {CC.KS[-1]}; los k que dejan un grupo de menos del 1 % no cuentan). Con k = {KF} la baja del costo de {KF - 1} a {KF} es de
{baja[S1][0]:.0f} (solo Senasa) y {baja[S2][0]:.0f} (Senasa + MQ), y de {KF} a {KF + 1} de {baja[S1][1]:.0f} y {baja[S2][1]:.0f}. El grupo más chico tiene {chico[S1]} muestras ({cm(100 * chico[S1] / n_muestras[S1], 1)} %) en solo Senasa
y {chico[S2]} ({cm(100 * chico[S2] / n_muestras[S2], 1)} %) en Senasa + MQ. {"Con " + str(CC.N_INIT) + " reinicios, las " + str(n_sem) + " semillas llegan al mismo costo mínimo" if mismo_costo else "Con " + str(CC.N_INIT) + " reinicios, las " + str(n_sem) + " semillas no llegan al mismo costo"};
las particiones empatadas en costo no son idénticas entre semillas (ARI entre corridas de {cm(aris[S1].min())} a {cm(aris[S1].max())} en solo Senasa y de {cm(aris[S2].min())} a {cm(aris[S2].max())} en Senasa + MQ).

**Sensibilidad: k = {K_SENS}.** Con un grupo más, el ARI contra el numérico es de {cm(MET[(S1, "b", K_SENS)]["ARI numérico"])} en solo Senasa y {cm(MET[(S2, "b", K_SENS)]["ARI numérico"])} en Senasa + MQ (con k = {KF}: {cm(MET[(S1, "b", KF)]["ARI numérico"])} y {cm(MET[(S2, "b", KF)]["ARI numérico"])}).
Clusters MIXTO con k = {K_SENS}: {mixtos(S1, K_SENS)} en solo Senasa y {mixtos(S2, K_SENS)} en Senasa + MQ. Heladeros con k = {K_SENS}: en solo Senasa, {heladeros_k(S1, K_SENS)}; en Senasa + MQ, {heladeros_k(S2, K_SENS)}.
Clusters de DDL casi puro con k = {K_SENS}: {n_puros(S1, K_SENS)} de {K_SENS} en solo Senasa y {n_puros(S2, K_SENS)} de {K_SENS} en Senasa + MQ.

**Los controles** (aglomerativo con Hamming y k-medias sobre one-hot) dan el mismo cuadro: ARI contra la familia y
contra el numérico entre {cm(rng_ctrl.min().min())} y {cm(rng_ctrl.max().max())}. El aglomerativo `average` deja grupos mínimos chicos: {agl_txt}. La conclusión
depende de la codificación, no del algoritmo.

**Muestras incompletas.** Con «no medido», el categórico pasa de {n_muestras[S1]} a {amp_n[S1]} muestras en solo Senasa (×{cm(amp_n[S1] / n_muestras[S1], 1)}) y de {n_muestras[S2]}
a {amp_n[S2]} en Senasa + MQ (×{cm(amp_n[S2] / n_muestras[S2], 1)}); entran también muestras de familia queso ({nq_amp[S1]} en solo Senasa y {nq_amp[S2]} en Senasa + MQ). Pero parte de los grupos se arma por **qué se midió**, no por el producto: en
solo Senasa sale {faltantes(S1)[0]} ({pc(faltantes(S1)[1] / amp_n[S1])} de las muestras del clustering); en Senasa + MQ, {faltantes(S2)[0]} ({pc(faltantes(S2)[1] / amp_n[S2])}). Las muestras completas conservan su partición
solo en parte (ARI entre {cm(min(amp_ari))} y {cm(max(amp_ari))} contra el k-modas sin incompletas). Sirve para no descartar muestras, pero el
patrón de faltantes manda.

**Veredicto.** El k-modas no reemplaza al k-medias: el acuerdo con el numérico es bajo (ARI de {cm(MET[(S1, "b", KF)]["ARI numérico"])} y {cm(MET[(S2, "b", KF)]["ARI numérico"])} con los
cortes globales y k = {KF}), parte el dulce de leche ({n_puros(S1)} de {KF} clusters de DDL casi puro en solo Senasa y {n_puros(S2)} de {KF} en Senasa + MQ; el resto del DDL queda repartido en clusters que mezclan) y deja polvos y líquidos
mezclados con DDL. Los heladeros no tienen cluster propio y su cluster cambia con la semilla. Y la
silueta de Hamming no sirve para elegir k: sube con k hasta el borde del rango.

**Recomendación.** El k-medias numérico queda como la partición de tipos de producto, y su subtag como la forma de
separar el heladero y el familiar dentro del DDL. El categórico con cortes globales (b) sirve para describir el dulce de
leche, no para reemplazarlo. La variante (a) no conviene como entrada de un clustering que se va a validar contra la
familia.
'''
display(Markdown(texto))
""")

nb = nbf.v4.new_notebook()
nb["cells"] = c
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
if __name__ == "__main__":
    nbf.write(nb, SALIDA)
    print(f"Escrito: {SALIDA} ({len(c)} celdas)")
