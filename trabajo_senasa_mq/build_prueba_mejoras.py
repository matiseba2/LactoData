"""Arma prueba_mejoras_clustering.ipynb: prueba, aparte del notebook completo, tres mejoras del agrupamiento.

No editar el .ipynb a mano: se regenera con este script y se ejecuta desde esta carpeta.

    cd trabajo_senasa_mq
    ../.venv/bin/python build_prueba_mejoras.py
    ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace prueba_mejoras_clustering.ipynb

Las tres mejoras (clases 43 y 46), en las dos versiones (solo Senasa y Senasa + MQ):
  1. outliers por consenso (LOF, IsolationForest, EllipticEnvelope) dentro de cada grupo preliminar, contra la regla IQR;
  2. segundo nivel (GaussianMixture por BIC y HDBSCAN) dentro de DDL, QUESO, POLVO_GRASO y POLVO_MAGRO;
  3. métricas sin el sesgo del grupo más grande (silueta macro, Davies-Bouldin, Calinski-Harabasz).
El código está en src/mejoras_clustering.py. No toca build_notebook.py, trabajo_senasa_mq.ipynb ni las salidas actuales:
escribe solo en resultados/prueba_mejoras/.
"""
from pathlib import Path

import nbformat as nbf

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI / "prueba_mejoras_clustering.ipynb"

c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: c.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Prueba de mejoras al agrupamiento (clases 43 y 46)
**Aprendizaje Automático y Grandes Datos · UNRaf · 2026**

Cuaderno **de prueba**, separado de `trabajo_senasa_mq.ipynb`. Toma las mismas muestras que las secciones 3 y 4 del
notebook completo (misma lectura, mismas correcciones, misma unión con MQ (`MQ encolumnado.xlsx`, sin las MO de la tabla ancha) y mismo filtro de familias no lácteas; los quesos que
confirma el catálogo van a su rama propia y no están acá: `src/datos_clase41.py`, `src/mejoras_clustering.py`) y prueba tres cambios contra la regla de outliers anterior. El notebook
completo ya usa el consenso con corte en 3 % (la mejora 1); acá se conserva la comparación contra la regla IQR anterior, y
no se cambia nada de eso.

| | qué se prueba | de dónde sale |
|---|---|---|
| 1 | **outliers por consenso** dentro de cada grupo preliminar: factor local (LOF, 50 vecinos), bosque de aislamiento (200 árboles) y elipse (EllipticEnvelope), cada uno cortado en un percentil fijo (2, 3 y 5 %); outlier si lo marcan al menos 2 de 3. Contra la regla anterior (Q3 + 1,5·IQR de la distancia al centroide) | clase 46, celdas 12, 24, 26 y 31 |
| 2 | **segundo nivel** dentro de DDL, QUESO, POLVO_GRASO y POLVO_MAGRO: mezcla de gaussianas con k por BIC (1 a 6) y HDBSCAN (min_cluster_size 20 a 50), comparados con el subtipo conocido | clase 46 celda 15; clase 43 celda 19 |
| 3 | **métricas sin sesgo de desbalance**: silueta global, silueta macro (cada grupo pesa igual), peor grupo, Davies-Bouldin y Calinski-Harabasz | clase 43, celdas 33 y 35 |

El código está en `src/mejoras_clustering.py` (pensado para sumarse tal cual al notebook completo). Se regenera con
`build_prueba_mejoras.py`. Escribe solo en `resultados/prueba_mejoras/`. Semilla 42. Al final, el **veredicto**.
""")

md("""
## 0 · Rutas, parámetros y lectura
`PCTS` son los percentiles de corte que se prueban; `PCT_V2` el que se usa para la versión 2 (segundo nivel y csv).
`ACTUAL` son los números del agrupamiento actual del notebook completo (consenso 3 %): el cuaderno verifica que la variante
«consenso 3 %» los reproduce (y las mismas muestras quitadas y los mismos tags por muestra que `clusters_*.csv`) antes de comparar.
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
import rutas as R
import mejoras_clustering as M
from datos_clase41 import COLUMNAS, SEMILLA, leer_senasa

warnings.filterwarnings("ignore", message=".*covariance matrix.*full rank.*")   # elipse en grupos chicos y casi planos
warnings.filterwarnings("ignore", message=".*Determinant has increased.*")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
AZUL, AMBAR, GRIS, ROJO, VERDE, VIOLETA = "#0055a5", "#f0a800", "#6b7280", "#c0392b", "#2e8b57", "#7c3aed"

PRUEBA = R.RESULTADOS / "prueba_mejoras"; PRUEBA.mkdir(parents=True, exist_ok=True)
PCTS = [2, 3, 5]
PCT_V2 = 3
TAMANOS = (20, 30, 40, 50)
TAGS_SEGUNDO = ["DDL", "QUESO", "POLVO_GRASO", "POLVO_MAGRO"]
ACTUAL = {"senasa": dict(muestras=2597, k=5, silueta=0.854, tags={"DDL", "POLVO_MAGRO", "POLVO_GRASO", "PROT_ALTA", "LIQUIDO"}),
          "senasa_mq": dict(muestras=3836, k=5, silueta=0.835, tags={"DDL", "POLVO_GRASO", "POLVO_MAGRO", "LIQUIDO", "PROT_ALTA"})}
S = leer_senasa()
RES = {}
print("Semilla:", SEMILLA, "· salidas en", PRUEBA.relative_to(R.RAIZ))
""")


def celdas_version(v, titulo, numero, con_mq):
    h = f"{numero}"
    md(f"""
## {h} · {titulo}
### {h}.1 · Las muestras y la regla actual
`M.preparar` arma las mismas muestras que el notebook completo y les suma, **solo para comparar**, el subtipo conocido
(`desc_subtipo`; en dulce de leche sin subtipo, el de `desc_producto`; si no, el de `mq_tipo_muestra.csv`) y las marcas
de DDL especial (REPOSTERO, HELADERO, reducido en grasa / descremado / light, buscadas en `desc_producto`, `desc_subtipo`,
`desc_grasa`, `desc_atributos` y `Descripción`). Nada de eso entra en el agrupamiento. Se rehace la regla IQR, que era
la del notebook completo y ahora queda solo para comparar.
""")
    code(fr"""
v = "{v}"
muestras, S = M.preparar({con_mq}, S)
X = muestras[COLUMNAS].to_numpy()
Z_todas, _, _ = M.escalar(X)                       # escala común: todas las muestras antes de quitar outliers
esp = M.especiales_ddl(muestras)
actual = pd.read_csv(R.TABLAS / f"clusters_{{v}}.csv", encoding="utf-8-sig").set_index("Identificador").loc[muestras["Identificador"]]
out_iqr, pre, K_PRE = M.outliers_iqr(Z_todas)
print(f"{{len(muestras)}} muestras · k-medias preliminar con k = {{K_PRE}} · regla IQR (la anterior): {{int(out_iqr.sum())}} outliers "
      f"({{100 * out_iqr.mean():.1f}} %)")
print("Subtipo de referencia, por fuente:", muestras["fuente_subtipo"].fillna("sin subtipo").value_counts().to_dict())
print("DDL especiales:", esp.sum().to_dict())
""")

    md(f"""
### {h}.2 · Outliers por consenso dentro de cada grupo preliminar
Mismo punto de partida que la regla actual: el k-medias preliminar (k por silueta). Dentro de **cada grupo** se reescala
(media 0 y desvío 1 con las muestras del grupo, para que un grupo chico y compacto no quede aplastado por la escala
global) y se calculan los tres puntajes de la clase 46, donde más bajo es más raro: `negative_outlier_factor_` del factor
local (50 vecinos), `score_samples` del bosque de aislamiento (200 árboles) y `score_samples` de la elipse. Cada método
marca el p % de menor puntaje del grupo, y es outlier lo que marcan **al menos 2 de 3** (el consenso de la celda 31,
con voto en vez de rango promedio). Los grupos de menos de 30 muestras no se filtran.

La tabla cruzada dice cuánto coinciden con la regla IQR. Con el corte en 3 %, el consenso tiene que marcar exactamente los
mismos outliers que `resultados/tablas/clusters_{v}.csv`: la celda lo verifica.
""")
    code(r"""
cons = {p: M.outliers_consenso(X, pre, pct=p) for p in PCTS}
assert (cons[PCT_V2][0] == actual["outlier"].to_numpy()).all(), f"el consenso {PCT_V2} % no reproduce los outliers de clusters_{v}.csv"
quitar = {"IQR (anterior)": out_iqr} | {f"consenso {p} %": cons[p][0] for p in PCTS}
display(pd.DataFrame({n: {"quitadas": int(o.sum()), "% quitado": round(100 * o.mean(), 1),
                          "también las quita IQR": int((o & out_iqr).sum()), "solo este método": int((o & ~out_iqr).sum()) if n != "IQR (anterior)" else 0}
                      for n, o in quitar.items()}).T)
print("% quitado por grupo preliminar:")
display(pd.DataFrame({n: pd.Series(o).groupby(pre).mean().mul(100).round(1) for n, o in quitar.items()})
        .assign(muestras=np.bincount(pre)).rename_axis("grupo preliminar"))
for p in PCTS:
    print(f"IQR contra consenso {p} %:"); display(pd.crosstab(pd.Series(out_iqr, name="IQR"), pd.Series(cons[p][0], name=f"consenso {p} %"), margins=True))
votos = cons[PCT_V2][1]
print(f"Con el corte en {PCT_V2} %, cuántas marca cada método y cuántas comparten (el consenso es >= 2):")
metodos = ["factor local", "bosque", "elipse"]
display(pd.DataFrame([[int((votos[a] & votos[b]).sum()) for b in metodos] for a in metodos], index=metodos, columns=metodos))
""")

    md(f"""
### {h}.3 · ¿Qué pasa con los dulces de leche especiales?
El repostero, el heladero y el reducido en grasa son DDL legítimos con composición propia (el heladero tiene menos
humedad y más cenizas). Una regla que mide distancia al centroide del DDL los ve lejos y los saca; un detector local
(LOF) o de densidad no tiene por qué. Cuántos **se conservan** con cada método:
""")
    code(r"""
filas = {}
for n in list(esp.columns) + ["todo el DDL"]:
    sel = esp[n].to_numpy() if n in esp else (muestras["desc_familia"] == "dulce de leche").to_numpy()
    filas[n] = {"muestras": int(sel.sum())} | {f"conserva · {m}": int((sel & ~o).sum()) for m, o in quitar.items()}
especiales = pd.DataFrame(filas).T
display(especiales)

ddl = (muestras["desc_familia"] == "dulce de leche").to_numpy(); o2 = quitar[f"consenso {PCT_V2} %"]
fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (x, y) in zip(ejes, [("humedad", "cenizas"), ("grasa", "proteinas")]):
    for sel, color, rot, s in [(ddl & ~out_iqr & ~o2, GRIS, "quedan con los dos", 5), (ddl & out_iqr & ~o2, AMBAR, "solo los quita IQR", 14),
                               (ddl & ~out_iqr & o2, VIOLETA, f"solo los quita el consenso {PCT_V2} %", 22), (ddl & out_iqr & o2, ROJO, "los quitan los dos", 22)]:
        ax.scatter(muestras.loc[sel, x], muestras.loc[sel, y], s=s, color=color, alpha=0.6, label=f"{rot} ({sel.sum()})")
    he = ddl & esp["heladero"].to_numpy()
    ax.scatter(muestras.loc[he, x], muestras.loc[he, y], s=40, facecolors="none", edgecolors="black", lw=0.6, label=f"heladero ({he.sum()})")
    ax.set_xlabel(f"{x} (g/100 g)"); ax.set_ylabel(f"{y} (g/100 g)")
ejes[0].legend(fontsize=8)
fig.suptitle(f"Familia dulce de leche: qué quita la regla IQR y qué quita el consenso {PCT_V2} %")
plt.tight_layout(); plt.show()
""")

    md(f"""
### {h}.4 · Ejemplos: lo que quita solo uno de los dos
Con el corte en `PCT_V2` %. `votos` es cuántos de los tres métodos la marcan; `grupo_pre`, el grupo preliminar.
""")
    code(r"""
ver = ["Identificador", "desc_familia", "desc_producto", "subtipo_ref"] + COLUMNAS
tab = muestras[ver].assign(grupo_pre=pre, votos=cons[PCT_V2][1]["votos"].to_numpy())
solo_iqr = out_iqr & ~o2; solo_cons = o2 & ~out_iqr
print(f"Solo los quita IQR: {solo_iqr.sum()} · por familia:", muestras.loc[solo_iqr, "desc_familia"].value_counts().head(8).to_dict())
display(tab[solo_iqr].sample(min(10, solo_iqr.sum()), random_state=SEMILLA).sort_values("grupo_pre"))
print(f"Solo los quita el consenso {PCT_V2} %: {solo_cons.sum()} · por familia:", muestras.loc[solo_cons, "desc_familia"].value_counts().head(8).to_dict())
display(tab[solo_cons].sort_values("grupo_pre").head(12))
print("Los quitan los dos (los más claros):")
display(tab[out_iqr & o2].sort_values("votos", ascending=False).head(8))
""")

    md(f"""
### {h}.5 · El k-medias final con cada filtro, y sus métricas
Con las muestras que quedan se reimputa, se reescala y se elige k por silueta (2 a 10), como el pipeline; los tags salen
de `etiquetar`. Cada variante se mide tres veces:

- **sobre sus propias muestras** (lo que reporta el pipeline): favorece al filtro que quita más, porque saca justo lo
  que está entre grupos;
- **sobre todas** las muestras (cada modelo las asigna con `predict`, en la escala común);
- **sobre las comunes**, las que ninguna variante quita: el mismo conjunto para todos, la comparación más justa.

La **silueta macro** promedia las medias de cada grupo (cada grupo pesa igual): el DDL, que es más de la mitad de las
muestras y está muy bien separado, deja de tapar a los grupos débiles. **Peor grupo** es la menor de esas medias.
""")
    code(r"""
comunes = ~np.logical_or.reduce(list(quitar.values()))
resultados, filas, et_todas = {}, {}, {}
for n, o in quitar.items():
    resultados[n], filas[n], et_todas[n] = M.variante(X, o, Z_todas, comunes)
    for e in esp.columns:
        filas[n][f"conserva {e}"] = int((esp[e].to_numpy() & ~o).sum())
comparacion = pd.DataFrame(filas).T.infer_objects()
r0 = resultados[f"consenso {PCT_V2} %"]
assert r0["K"] == ACTUAL[v]["k"] and set(r0["tags"].values()) == ACTUAL[v]["tags"] and len(r0["Z"]) == ACTUAL[v]["muestras"]
assert round(comparacion.loc[f"consenso {PCT_V2} %", "silueta global"], 3) == ACTUAL[v]["silueta"]
assert (r0["tag_de"] == actual.loc[~cons[PCT_V2][0], "tag"].to_numpy()).all(), "los tags actuales no se reproducen"
print(f"Actual (consenso {PCT_V2} %) reproducido: {len(r0['Z'])} muestras, k = {r0['K']}, silueta {comparacion.loc[f'consenso {PCT_V2} %', 'silueta global']:.3f}, mismos tags por muestra.")
print(f"Muestras comunes (no las quita ninguna variante): {comunes.sum()}")
display(comparacion.style.format(precision=3).format({"% quitado": "{:.1f}", "Calinski-Harabasz": "{:.0f}"}))
print("Silueta media por tag, en cada variante (sobre sus propias muestras):")
display(pd.concat({n: M.silueta_por_grupo(r["Z"], r["tag_de"])["silueta media"] for n, r in resultados.items()}, axis=1).round(3))
print("Muestras por tag:")
display(pd.concat({n: pd.Series(r["tag_de"]).value_counts() for n, r in resultados.items()}, axis=1).fillna(0).astype(int))
""")

    md(f"""
### {h}.6 · Segundo nivel dentro de cada tag
Sobre el agrupamiento con el filtro por consenso en `PCT_V2` %, que conserva los DDL especiales (si se usara el de IQR,
casi no habría heladeros para encontrar). Dentro de cada tag con al menos 30 muestras se reescala y se prueban:

- **mezcla de gaussianas** con k por BIC entre 1 y 6 (`n_init = 3`, como la clase 46): si BIC elige 1, no hay estructura;
- **HDBSCAN** con `min_cluster_size` 20, 30, 40 y 50 (clase 43): cuántos grupos, cuánto ruido, y si es estable.

Contra el **subtipo conocido** (DDL: familiar/clásico, repostero, heladero; quesos: subtipo y, para los de familia queso
de Senasa, la clase de humedad del CAA, `cat_humedad` de `categorias_{v}.csv`; polvos: WPC, desmineralizado, suero...):
el **ARI** (Rand ajustado; 0 = azar, 1 = igual) sin el ruido y **con el ruido como un grupo más** (el honesto: no premia
mandar lo difícil al ruido), y la silueta y Davies-Bouldin dentro del tag, sin el ruido.

Se queda la partición de mayor ARI con el ruido. Cada subgrupo se describe con su centroide en g/100 g, su subtipo
mayoritario, la **pureza** (% del subgrupo que es de ese subtipo) y la **cobertura** (% de ese subtipo que cae en el
subgrupo). Regla para proponer un **subtag**: ARI >= 0,2; en el subgrupo, al menos 20 muestras con subtipo conocido y
que sean al menos la mitad del subgrupo; pureza >= 60 % y cobertura >= 50 %.
Si ningún subgrupo la cumple, no hay estructura que valga un subtag.
""")
    code(r"""
nombre_v2 = f"consenso {PCT_V2} %"
r2 = resultados[nombre_v2]
v2 = muestras[~quitar[nombre_v2]].reset_index(drop=True)
v2["cluster"] = r2["grupos"]; v2["tag"] = r2["tag_de"]
cat = pd.read_csv(R.TABLAS / f"categorias_{v}.csv", usecols=["Identificador", "familia", "cat_humedad"], encoding="utf-8-sig", low_memory=False)
v2["humedad_caa"] = v2["Identificador"].map(cat[cat["familia"] == "queso"].set_index("Identificador")["cat_humedad"])
v2["subtag"] = np.nan; v2["subtag"] = v2["subtag"].astype(object)
segundo, propuestas, sensibilidades = {}, {}, []
for tag in TAGS_SEGUNDO:
    sel = (v2["tag"] == tag).to_numpy()
    if sel.sum() < 30:
        print(f"{tag}: {sel.sum()} muestras, no se analiza."); continue
    sub = v2[sel].reset_index(drop=True); Xs = sub[COLUMNAS].to_numpy()
    refs = {"subtipo": sub["subtipo_ref"]}
    if tag == "QUESO":
        refs["humedad CAA"] = sub["humedad_caa"]
    print(f"\n━━━━━━━━ {tag}: {sel.sum()} muestras · con subtipo conocido {sub['subtipo_ref'].notna().sum()} · "
          f"subtipos: {sub['subtipo_ref'].value_counts().head(6).to_dict()}")
    for nref, ref in refs.items():
        sn = M.segundo_nivel(Xs, ref, tamanos=TAMANOS)
        if nref == "subtipo":
            print("BIC por cantidad de gaussianas (menor es mejor):", {k: round(b) for k, b in sn["bic"].items()}, "→ k =", sn["k_bic"])
        print(f"Contra {nref} ({ref.notna().sum()} conocidas):")
        display(sn["sensibilidad"].round(3))
        sensibilidades.append(sn["sensibilidad"].reset_index().assign(version=v, tag=tag, referencia=nref))
        particion, et = M.elegir_particion(sn)
        desc = M.describir_subgrupos(Xs, et, ref)
        ari = sn["sensibilidad"].loc[particion, "ARI con el ruido"]
        print(f"Partición elegida: {particion} (ARI con el ruido {ari:.3f}):"); display(desc)
        top = ref.value_counts().head(6).index
        display(pd.crosstab(pd.Series(np.where(et == -1, "ruido", et.astype(str)), name="subgrupo"), ref.where(ref.isin(top), "otro").fillna("sin dato").rename(nref)))
        prop = M.proponer_subtag(tag, desc, ari)
        print("Subtags propuestos:", prop if prop else "ninguno: no hay estructura que se corresponda con un subtipo real")
        segundo[(tag, nref)] = dict(sn=sn, particion=particion, desc=desc, ari=ari)
        if prop and not any(t == tag for t, _ in propuestas):      # la primera referencia que da subtags manda
            propuestas[(tag, nref)] = prop
            idx = np.flatnonzero(sel)
            ets = np.where(et == -1, "ruido", et.astype(str))
            for g, st in prop.items():
                v2.loc[idx[ets == g], "subtag"] = st
print("\nSubtags propuestos en", v, ":", {t: p for (t, _), p in propuestas.items()} or "ninguno")
print(v2["subtag"].value_counts().to_dict())
""")

    md(f"""
### {h}.7 · Tabla comparativa y csv de prueba
La tabla de 5.5 más una fila para la versión 2 con los subtags (cada muestra con subtag cuenta como su propio grupo; las
demás, como su tag). Se guardan en `resultados/prueba_mejoras/`: `clusters_{v}_v2.csv` (una fila por muestra que
entra, con los dos filtros, los votos del consenso, el tag actual, y el cluster, tag, subtag y silueta de la versión 2),
`comparacion_{v}.csv` y `segundo_nivel_{v}.csv`.
""".replace("5.5", f"{h}.5"))
    code(r"""
etiqueta2 = v2["subtag"].fillna(v2["tag"]).to_numpy()
m2 = M.metricas(r2["Z"], etiqueta2)
fila = comparacion.loc[nombre_v2].copy()
fila.update({"k": m2["grupos"], "tags": " ".join(sorted(set(etiqueta2))), "silueta global": m2["silueta global"], "silueta macro": m2["silueta macro"],
             "peor grupo": m2["peor grupo"], "Davies-Bouldin": m2["Davies-Bouldin"], "Calinski-Harabasz": m2["Calinski-Harabasz"]})
for col in ["silueta global (todas)", "silueta macro (todas)", "silueta global (comunes)", "silueta macro (comunes)", "Davies-Bouldin (comunes)"]:
    fila[col] = np.nan
tabla = comparacion.copy(); tabla.loc[f"{nombre_v2} + subtags"] = fila
display(tabla.style.format(precision=3, na_rep="-").format({"% quitado": "{:.1f}", "Calinski-Harabasz": "{:.0f}"}))
v2["silueta"] = np.round(M.silhouette_samples(r2["Z"], r2["grupos"]), 4)
salida = muestras[["Identificador", "desc_familia", "origen"] + COLUMNAS].rename(columns={"desc_familia": "familia"})
salida["outlier_iqr"] = out_iqr; salida[f"outlier_consenso_{PCT_V2}"] = quitar[nombre_v2]; salida["votos_consenso"] = cons[PCT_V2][1]["votos"].to_numpy()
salida["tag_actual"] = actual["tag"].to_numpy()
salida = salida.merge(v2[["Identificador", "cluster", "tag", "subtag", "silueta"]], on="Identificador", how="left")
salida["cluster"] = salida["cluster"].astype("Int64")
salida.to_csv(PRUEBA / f"clusters_{v}_v2.csv", index=False, encoding="utf-8-sig")
tabla.rename_axis("variante").to_csv(PRUEBA / f"comparacion_{v}.csv", encoding="utf-8-sig")
pd.concat(sensibilidades).to_csv(PRUEBA / f"segundo_nivel_{v}.csv", index=False, encoding="utf-8-sig")
print(f"Guardado en {PRUEBA.relative_to(R.RAIZ)}: clusters_{v}_v2.csv ({len(salida)} filas, {int(salida['tag'].notna().sum())} con tag, "
      f"{int(salida['subtag'].notna().sum())} con subtag), comparacion_{v}.csv, segundo_nivel_{v}.csv")
RES[v] = dict(tabla=tabla, especiales=especiales, propuestas=propuestas, segundo=segundo)
""")


celdas_version("senasa", "Solo Senasa", 1, False)
celdas_version("senasa_mq", "Senasa + MQ", 2, True)

md("""
## 3 · Las dos versiones juntas
""")
code(r"""
cols = ["% quitado", "muestras", "k", "conserva repostero", "conserva heladero", "conserva reducido en grasa", "silueta global", "silueta macro", "peor grupo",
        "Davies-Bouldin", "Calinski-Harabasz", "silueta global (comunes)", "silueta macro (comunes)", "Davies-Bouldin (comunes)"]
juntas = pd.concat({v: RES[v]["tabla"][cols] for v in RES}, names=["versión", "variante"])
display(juntas.style.format(precision=3, na_rep="-").format({"% quitado": "{:.1f}", "Calinski-Harabasz": "{:.0f}"}))
filas = []
for v in RES:
    for (tag, ref), s in RES[v]["segundo"].items():
        filas.append({"versión": v, "tag": tag, "referencia": ref, "BIC elige k": s["sn"]["k_bic"], "partición elegida": s["particion"],
                      "ARI con el ruido": round(s["ari"], 3), "subtags": ", ".join(RES[v]["propuestas"].get((tag, ref), {}).values())})
display(pd.DataFrame(filas).set_index(["versión", "tag", "referencia"]))
""")

md(r"""
## 4 · Veredicto
Números de la ejecución con semilla 42 (tablas de las secciones 1.7, 2.7 y 3). Si se cambian los datos o las
correcciones, hay que volver a leerlos.

### Conviene sumar al notebook completo

**1 · Outliers por consenso, con corte en 3 %, en lugar de la regla IQR.** Se calcula dentro de cada grupo del k-medias
preliminar, reescalando dentro del grupo. Los parámetros son `M.outliers_consenso(X, grupos_preliminares, pct=3,
minimo=2, n_vecinos=50, tam_min=30)`: LOF con 50 vecinos, IsolationForest con 200 árboles y EllipticEnvelope, y es
outlier lo que marcan al menos 2 de 3.

| | solo Senasa: IQR | solo Senasa: consenso 3 % | Senasa + MQ: IQR | Senasa + MQ: consenso 3 % |
|---|---|---|---|---|
| % quitado | 8,6 | 2,4 | 9,2 | 2,6 |
| DDL heladero que se conservan (de 176) | 36 | 152 | 35 | 172 |
| DDL reducido en grasa que se conservan (de 11) | 0 | 3 | 2 | 11 |
| DDL repostero que se conservan (de 623) | 616 | 620 | 613 | 622 |
| k y tags | 5, los mismos | 5, los mismos | 5, los mismos | 5, los mismos |
| silueta global / macro (sus muestras) | 0,869 / 0,791 | 0,854 / 0,760 | 0,862 / 0,779 | 0,835 / 0,754 |
| silueta global / macro y DB (muestras comunes) | 0,879 / 0,802 / 0,301 | igual | 0,867 / 0,786 / 0,346 | igual |

- **La regla IQR saca casi todos los heladeros.** El heladero tiene una composición propia (humedad ~22 g/100 g y
  cenizas ~2,6, contra ~28,5 y ~1,3 del familiar) y la regla lo ve lejos del centroide del DDL. El consenso los conserva.
- Sobre las muestras que ninguna variante quita, **los cuatro filtros dan lo mismo**: misma silueta global y macro,
  y mismo Davies-Bouldin. La silueta más baja «sobre sus muestras» no indica un modelo peor. Es el costo de conservar
  muestras legítimas que están en el borde del grupo. Por eso la comparación justa es la de las muestras comunes.
- El 90 % (Senasa, 57 de 63) y el 87 % (Senasa + MQ, 88 de 101) de lo que quita el consenso también lo quita la regla IQR.
  Lo que quita solo el consenso (6 y 13 muestras) no incluye ningún dulce de leche. En Senasa son polvos raros dentro de su
  grupo: 4 permeados de suero, un suero parcialmente desmineralizado y un polvo para bebida láctea. En MQ son 4 sueros, 4
  sin clasificar, 3 arroces con leche y 2 concentrados (uno de ellos con 98 % de humedad, casi agua).
- Con el corte en 3 %, k y tags son los mismos con el consenso y con IQR, en las dos versiones. Con **2 %** se conserva
  algo más: en solo Senasa, 166 heladeros y los 11 reducidos en grasa (contra 152 y 3 con 3 %); en Senasa + MQ, 176 y 11
  (contra 172 y 11). Pero en solo Senasa el 2 % cambia el agrupamiento (k = 6: parte POLVO_GRASO en 4 y 84 muestras, y aparece
  el tag POLVO_GRASO_2). Con **5 %** se vuelven a perder heladeros (quedan 133 y 151). Se deja el 3 %, el corte que ya usa el
  notebook completo; el 2 % es la alternativa que quita menos (1,4 y 1,9 %).

**2 · Métricas sin sesgo de desbalance.** Conviene reportar siempre, junto a la silueta global, la **silueta macro**, la del
**peor grupo**, Davies-Bouldin y Calinski-Harabasz (`M.metricas`). Para comparar filtros, conviene medir sobre las
**muestras comunes** (`M.variante`). La silueta del DDL (0,88–0,90) tapa a PROT_ALTA (0,59–0,62) y, en Senasa + MQ, a POLVO_MAGRO (0,63–0,65): con el consenso en 3 %, la silueta macro es 0,09
más baja que la global en solo Senasa y 0,08 en Senasa + MQ (QUESO ya no está en el k-medias: los quesos van a su rama).

**3 · Segundo nivel, solo en DDL: `DDL_HELADERO` y `DDL_FAMILIAR`.** Se arma con `M.segundo_nivel`: GaussianMixture con
k por BIC entre 1 y 6 y `n_init=3`, sobre el DDL reescalado y después del filtro por consenso. El ARI contra el subtipo
es 0,40 en solo Senasa y 0,41 con MQ.
- `DDL_HELADERO`: es 100 % heladero y reúne el 80 % de los heladeros. Su centroide es humedad 21,7, grasa 6,6–6,8,
  proteínas 6,8–6,9 y cenizas 2,6.
- `DDL_FAMILIAR`: es 89–90 % familiar/clásico y reúne el 77–78 % de los familiares. Su centroide es humedad 28,5, grasa 6,7,
  proteínas 5,8 y cenizas 1,3.
- El **repostero no tiene estructura propia**: se reparte en cuatro componentes grandes. La más pura (97–98 %) junta solo el
  26 % de los reposteros. No lleva subtag.

### Candidato, a confirmar antes de sumarlo

**`POLVO_MAGRO_WPC` y `POLVO_MAGRO_DESMINERALIZADO`**, solo en Senasa + MQ, con HDBSCAN y `min_cluster_size` = 30.
- WPC: pureza 100 % y cobertura 83 %, con humedad 4,0, grasa 3,3, proteínas 35,6 y cenizas 5,7.
- Desmineralizado: pureza 83 % y cobertura 92 %, con humedad 2,6, grasa 1,7, proteínas 13,1 y cenizas 4,9.
- El ARI con el ruido es 0,60, pero depende del parámetro: con `min_cluster_size` 20 cae a 0,19 (con 40 o 50 queda en 0,57).
- Queda un tercer grupo de 71 muestras de MQ sin subtipo, con proteínas 34 y cenizas 7,8, y 80 muestras como ruido.
- En solo Senasa no aparece: hay solo 15 desmineralizados y el ARI es 0,15.

### No conviene sumar

- **Consenso con 5 %**, por lo que se ve más arriba: pierde heladeros (133 y 151). El 2 % conserva algo más, pero se
  mantiene el 3 % del notebook completo.
- **HDBSCAN para el DDL**: deja un grupo gigante (casi el 90 % del DDL) y manda entre 110 y 155 muestras al ruido.
  El ARI con el ruido es ~0,24–0,25. La mezcla de gaussianas lo supera (0,40–0,41).
- **Subtags de QUESO**: ya no se analizan acá. Los quesos que confirma el catálogo salen del k-medias y van a una rama
  propia (humedad y materia grasa), con el subtag de la clase de humedad del CAA; el tag QUESO no aparece en el primer
  nivel de ninguna de las dos versiones. La prueba de ese agrupamiento es `prueba_clustering_quesos.ipynb`.
- **Subtags de POLVO_GRASO**: ningún subgrupo cumple la regla, en las dos versiones. En Senasa + MQ (644 muestras, 70 con
  subtipo conocido) el más grande (MODIFICADA, 239 muestras) llega a 96 % de pureza pero tiene subtipo conocido en solo el
  21 % de sus muestras (51), y los otros con subtipo son de 5 muestras (suero, pureza 60 %) y de 10 (polvo para preparar,
  100 %, con 7 con subtipo), demasiado chicos para un subtag. En solo Senasa el tag tiene 87 muestras (la leche en polvo y
  unos sueros reengrasados) y solo 17 con subtipo conocido: el único subgrupo puro (polvo para preparar, 7 muestras) no
  llega a las 20 que pide la regla, y el grupo grande (57 muestras) no tiene ningún subtipo conocido. HDBSCAN manda todo
  al ruido en las dos versiones.
- **Evaluar los subtags con la silueta global**: la fila «+ subtags» baja a 0,30 y 0,44, y el DDL residual (lo que
  queda sin subtag) da una silueta negativa. Es esperable, porque los subtags son un segundo nivel dentro del tag y no
  grupos del primer nivel. Se juzgan por ARI, pureza y cobertura.

### Dudas abiertas

- En DDL, en POLVO_MAGRO y en POLVO_GRASO de Senasa + MQ el BIC elige el máximo probado (k = 6; en POLVO_GRASO de solo
  Senasa elige 5). La mezcla
  de gaussianas aproxima una densidad continua: no prueba que haya seis tipos. Por eso los subtags se proponen solo para las componentes que coinciden con
  un subtipo real.
- El subtipo de referencia usa `mq_tipo_muestra.csv` en 319 muestras de Senasa + MQ (y 2 de solo Senasa). Una parte sale del clasificador de familias y no
  está revisada a mano.
- «Reducido en grasa» incluye DESCREMADO y LIGHT (11 DDL en total). Se buscó en `desc_producto`, `desc_subtipo`,
  `desc_grasa`, `desc_atributos` y `Descripción`.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = c
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
if __name__ == "__main__":
    nbf.write(nb, SALIDA)
    print(f"Escrito: {SALIDA} ({len(c)} celdas)")
