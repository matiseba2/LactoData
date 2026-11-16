"""Celdas de la sección 6 de trabajo_senasa_mq.ipynb: los análisis pasados a categorías e ítems, y una primera corrida
de reglas de asociación (estilo de la clase 44: TransactionEncoder, fpgrowth, association_rules).

Ex build_categorias_asociacion.py (original en _archivo/proyecto/), que armaba categorias_asociacion.ipynb (archivado).
La categorización está en src/categorizar.py (las decisiones, como constantes al principio). Al ejecutarse, la sección
vuelve a escribir resultados/tablas/categorias_senasa.csv, categorias_senasa_mq.csv y las dos matrices
transacciones_*.parquet.

    celdas()   # lista de celdas que build_notebook.py pone en el notebook único
"""
import re

import nbformat as nbf

c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(re.sub(r"(?m)^(#+) ", r"#\1 ", s.strip("\n"))))   # un nivel más abajo
code = lambda s: c.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# 6 · Categorías de los análisis e ítems para reglas de asociación

Cada resultado numérico de una muestra pasa a una categoría (por ejemplo `humedad=alta humedad` o
`materia_grasa=entera`) con la tabla `cortes_categorias.csv` (criterio «CAA por familia»; ver
`docs/cortes_categorias.md`; la tabla se regenera en la sección 5). Con las categorías, la familia y el tag del cluster se arma una canasta por muestra y se buscan
reglas de asociación como en la clase 44.

**Tag y subtag.** Cada muestra agrupada lleva su `tag=` (el primer nivel: DDL, POLVO_GRASO...), igual que antes, y las
que tienen subtag (secciones 3 y 4, segundo nivel por familia: DDL_FAMILIAR, DDL_REPOSTERO, DDL_HELADERO, POLVO_WPC,
POLVO_LECHE_ENTERA...) suman además el ítem `subtag=`. Se eligió sumar en vez de reemplazar porque así `tag=DDL` sigue
siendo todo el dulce de leche (las reglas del primer nivel no cambian de significado y se comparan con las de antes) y el
subtag agrega lo que el tag no ve: el heladero tiene menos humedad y más cenizas que el familiar, justo lo que miden las
categorías. Una regla que junta el tag con su propio subtag (`subtag=DDL_HELADERO ⇒ tag=DDL`) es trivial («mismo
análisis»). El subtag del queso (QUESO_SEMIDURO...) **no** entra como ítem: es la clase de humedad del CAA, que ya está
como `humedad=`; queda en `clusters_senasa_mq.csv`.

- Lectura y correcciones: `src/datos_clase41.py` (las mismas de la clase 41, sección 1).
- Categorización: `src/categorizar.py`. Las decisiones están como constantes al principio del módulo.
- Salidas, en `resultados/tablas/`: `categorias_senasa.csv`, `categorias_senasa_mq.csv` (una fila por muestra) y
  `transacciones_senasa.parquet`, `transacciones_senasa_mq.parquet` (matriz one-hot muestra × ítem).
""")

code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
import warnings; warnings.filterwarnings("ignore")

import categorizar as cz

AZUL, AMBAR, GRIS, ROJO, VERDE = "#0055a5", "#f0a800", "#6b7280", "#c0392b", "#2e8b57"
pd.set_option("display.width", 160); pd.set_option("display.max_colwidth", 110); pd.set_option("display.max_rows", 200)
nombre = lambda s: "{" + ", ".join(sorted(s)) + "}"

print("Decisiones (categorizar.py):")
for k in ["HUMEDAD_QUESO_4_CLASES", "NO_SON_ITEMS", "USAR_CORTES_GLOBALES", "GLOBALES_PARA_SIN_FAMILIA",
          "MARCAR_DDL_GRASA_BAJO_MINIMO", "LACTOSA_REFERENCIA", "SUBTIPO_COMO_ITEM", "SUBTAG_COMO_ITEM", "LIFT_MIN"]:
    print(f"   {k} = {getattr(cz, k)}")
print("   desc_consistencia: queda como columna, no es ítem · terciles por subtipo: no")
""")

md("""
## 1 · Las dos tablas

Solo Senasa, y Senasa + MQ (MQ sale de `MQ encolumnado.xlsx`: Senasa manda; MQ completa huecos de humedad, grasa,
proteínas y cenizas y agrega las muestras que solo están en MQ, con la familia que les asigna MQ encolumnado). Los quesos
que confirma el catálogo llevan `tag=QUESO` (rama propia, fuera del k-medias de cuatro análisis); su subtag no es un ítem,
porque sale de la humedad. Las muestras excluidas por humedad + sólidos fuera de 100 ± 1 quedan en la
tabla, sin categorías y con el motivo.
""")

code("""
tablas, matrices = {}, {}
for version in ["senasa", "senasa_mq"]:
    tablas[version], matrices[version] = cz.guardar(version)
A = cz.ANALISIS_LISTA
resumen = []
for version, t in tablas.items():
    n_val = int(t[A].notna().sum().sum()); n_cat = int(t[[f"cat_{a}" for a in A]].notna().sum().sum())
    resumen.append({"versión": version, "muestras": len(t), "con algún análisis": int(t[A].notna().any(axis=1).sum()),
                    "transacciones (con ítems)": len(matrices[version]), "ítems distintos": matrices[version].shape[1],
                    "valores": n_val, "categorizados": n_cat, "% categorizado": round(100 * n_cat / n_val, 2),
                    "excluidas": int(t["excluida"].sum())})
display(pd.DataFrame(resumen).set_index("versión"))
t = tablas["senasa"]; M = matrices["senasa"]
print("Orígenes en Senasa + MQ:", tablas["senasa_mq"]["origen"].value_counts().to_dict())
print("Dos muestras con sus ítems:")
for i in t.index[t["items"].map(len) >= 6][:2]:
    print(f"   {t.at[i, 'Identificador']} ({t.at[i, 'familia']}): {t.at[i, 'items']}")
""")

md("""
## 2 · Los cortes que se aplicaron

Para cada análisis y familia: qué juego de cortes se usó (familia + subtipo, familia o los globales `*`), cada
categoría con sus límites (desde incluido, hasta excluido), si está verificada en el CAA y cuántas muestras cayeron.
""")

code("""
cortes = cz.leer_cortes()
filas = []
for a in A:
    x = t[t[f"cat_{a}"].notna()]
    g = x.groupby(["familia", f"juego_{a}", f"cat_{a}"]).size().reset_index(name="n")
    g.columns = ["familia", "juego", "categoria", "n"]; g.insert(0, "analisis", a)
    filas.append(g)
aplicados = pd.concat(filas, ignore_index=True)
aplicados["familia_corte"] = np.where(aplicados["juego"] == "globales (*)", "*", aplicados["familia"])
aplicados = aplicados.merge(cortes[["analisis", "familia", "categoria", "desde", "hasta", "verificado", "fuente"]]
                            .rename(columns={"familia": "familia_corte"}), on=["analisis", "familia_corte", "categoria"], how="left")
assert aplicados["desde"].notna().all()
aplicados["fuente"] = aplicados["fuente"].str.replace(r" \\(http.*\\)", "", regex=True)
print(f"{len(aplicados)} combinaciones análisis × familia × categoría con muestras · juegos usados:",
      aplicados.groupby("juego")["n"].sum().to_dict())
display(aplicados[["analisis", "familia", "juego", "categoria", "desde", "hasta", "verificado", "n", "fuente"]])
""")

code("""
print("Familias que usan los cortes globales (*), por análisis (n muestras):")
g = aplicados[aplicados["juego"] == "globales (*)"].groupby(["analisis", "familia"])["n"].sum()
for a, s in g.groupby(level=0):
    print(f"   {a}: " + ", ".join(f"{f} ({n})" for (_, f), n in s.items()))
""")

md("""
## 3 · Conteos por categoría y familia
""")

code("""
fig, ax = plt.subplots(figsize=(11, 4.5))
p = pd.DataFrame({"valores": t[A].notna().sum(), "categorizados": t[[f"cat_{a}" for a in A]].notna().sum().values}, index=A)
p = p.sort_values("valores")
ax.barh(p.index, p["valores"], color=AZUL); ax.set_xscale("log"); ax.set_xlabel("muestras con valor (escala log)")
ax.set_title("Muestras con valor por análisis (Senasa). Las que no tienen categoría son las 8 excluidas")
for y, (v, k) in enumerate(zip(p["valores"], p["categorizados"])):
    ax.text(v * 1.05, y, f"{k}/{v}", va="center", fontsize=8, color=GRIS)
plt.tight_layout(); plt.show()
""")

code("""
for a in ["humedad", "materia_grasa", "proteinas", "cenizas", "lactosa", "galactosa", "mg_extracto_seco"]:
    x = t[t[f"cat_{a}"].notna()]
    tab = pd.crosstab(x[f"cat_{a}"], x["familia"], margins=True, margins_name="total")
    print(f"{a}: {len(x)} muestras"); display(tab)
""")

code("""
print("Análisis con pocos datos (casi todos de dulce de leche):")
chicos = [a for a in A if t[f"cat_{a}"].notna().sum() < 400]
display(pd.DataFrame({a: t[f"cat_{a}"].value_counts() for a in chicos}).fillna(0).astype(int).T)
""")

md("""
## 4 · Requisitos del CAA y marcas

Los requisitos que no son clases (`cortes_categorias.md`, decisión 3) entran como ítems `req_<requisito>=cumple / no
cumple`. Además se marcan los dulces de leche con grasa menor a 6 (conservan la categoría «bajo el mínimo (<6)») y la
lactosa «bajo / reducido / normal» de la leche fluida, que depende de la referencia 4,83 g/100 g (no verificada en el
CAA).
""")

code("""
req = pd.DataFrame({f"req_{r[0]}": t[f"req_{r[0]}"].value_counts() for r in cz.REQUISITOS}).fillna(0).astype(int).T
req["% no cumple"] = (100 * req.get("no cumple", 0) / req.sum(axis=1)).round(1)
req["fuente"] = [r[6] for r in cz.REQUISITOS]
display(req)
ddl = t[t["ddl_grasa_bajo_minimo"]]
print(f"Dulce de leche con grasa < 6: {len(ddl)} · por subtipo: {ddl['subtipo'].fillna('sin subtipo').value_counts().to_dict()}")
print(f"Lactosa con categoría que depende de la referencia {cz.LACTOSA_REFERENCIA} (no verificada): {int(t['lactosa_sin_verificar'].sum())} muestras",
      t.loc[t["lactosa_sin_verificar"], "cat_lactosa"].value_counts().to_dict())
""")

md("""
## 5 · Muestras sin categoría y por qué

Cada valor sin categoría tiene su motivo en la columna `sin_categoria` («análisis: motivo»). Motivos posibles: muestra
excluida (humedad + sólidos), valor fuera del rango físico, valor negativo, familia sin cortes ni globales, globales
desactivados.
""")

code("""
for version, tv in tablas.items():
    s = tv["sin_categoria"].dropna().str.split("; ").explode()
    mot = s.str.split(": ", n=1).str[1].str.replace(r"valor [-\\d.e+]+", "valor X", regex=True)
    print(f"{version}: {tv['sin_categoria'].notna().sum()} muestras con algún valor sin categoría · {len(s)} valores")
    display(pd.crosstab(s.str.split(": ", n=1).str[0], mot))
    # comprobación: todo valor tiene categoría o motivo
    for a in A:
        sin = tv[a].notna() & tv[f"cat_{a}"].isna()
        assert tv.loc[sin, "sin_categoria"].fillna("").str.contains(f"{a}: ", regex=False).all(), a
print("Comprobado: cada valor sin categoría tiene su motivo.")
display(t.loc[t["sin_categoria"].notna(), ["Identificador", "familia", "humedad", "solidos_totales", "sin_categoria"]])
""")

md("""
## 6 · Las categorías frente a los tags de cluster (clase 41)

El tag sale del agrupamiento de humedad, grasa, proteínas y cenizas (solo para las muestras con los cuatro análisis; los
outliers no tienen tag ni subtag). Si las categorías y los clusters cuentan la misma historia, cada tag debería
concentrarse en pocas categorías. Las tablas por categoría usan el subtag (que fuera del DDL y de los polvos es el mismo
tag), para ver también en qué se diferencian los subtipos del DDL y los tipos de polvo.
""")

code("""
for version, tv in tablas.items():
    print(f"{version} · estado del cluster:", tv["estado_cluster"].value_counts().to_dict())
x = t[t["tag"].notna()]
display(pd.crosstab(x["familia"], x["tag"], margins=True, margins_name="total"))
print("Subtags del DDL contra el subtipo (Senasa):")
display(pd.crosstab(x.loc[x["tag"] == "DDL", "subtag"], x.loc[x["tag"] == "DDL", "subtipo"].fillna("sin subtipo"), margins=True, margins_name="total"))
for a in ["humedad", "materia_grasa", "proteinas", "cenizas"]:
    y = x[x[f"cat_{a}"].notna()]
    print(f"{a} × subtag (Senasa, fila = familia | categoría; % de cada subtag, que es el tag fuera del DDL y de los polvos):")
    tab = pd.crosstab([y["familia"], y[f"cat_{a}"]], y["subtag"], normalize="columns").mul(100).round(0)
    display(tab[(tab > 0).any(axis=1)])
""")

md("""
## 7 · La matriz transaccional

Una fila por muestra con al menos un análisis categorizado; una columna por ítem. Como en la clase 44, la matriz sale de
`TransactionEncoder`. Los sólidos totales están categorizados pero no son ítems (son ≈ 100 − humedad).
""")

code("""
soportes = M.mean().sort_values(ascending=False)
print(f"Matriz Senasa: {M.shape[0]} transacciones × {M.shape[1]} ítems · ítems por muestra: {M.sum(axis=1).describe().round(1).to_dict()}")
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [2, 1]})
top = soportes.head(30)
a.barh(top.index, top.values, color=AZUL); a.invert_yaxis(); a.set_xlabel("soporte (fracción de muestras)"); a.set_title("Los 30 ítems más frecuentes")
a.tick_params(axis="y", labelsize=8)
tam = M.sum(axis=1).value_counts().sort_index(); b.bar(tam.index, tam.values, color=AZUL)
b.set_xlabel("ítems por muestra"); b.set_ylabel("muestras"); b.set_title("Tamaño de las canastas")
plt.tight_layout(); plt.show()
M.head().astype(int)
""")

md("""
## 8 · Reglas de asociación (Senasa)

`fpgrowth` (da los mismos conjuntos que `apriori`, más rápido) con soporte mínimo 1 % (la celda imprime cuántas
transacciones son) y hasta 3 ítems; `association_rules` con confianza mínima 0,6.

Muchas reglas son triviales. `cz.filtrar_reglas` las marca:

- **mismo análisis**: el consecuente repite un análisis del antecedente (humedad=alto ⇒ req_ddl_humedad_max_30=no
  cumple repite la definición del corte);
- **cuenta**: la materia grasa en extracto seco junto con la grasa y la humedad, de las que se calcula;
- **define la familia**: materia_grasa=helado de crema ⇒ familia=helado (esa categoría solo existe en el helado);
- **familia redundante**: familia=queso junto con una categoría que solo existe en el queso;
- **cortes globales en la familia**: familia=X junto con un análisis que en X usa los cortes globales (familias chicas
  y muestras solo MQ): `cenizas=alto` de una muestra sin familia no se compara con el tercil de cada familia;
- **explicada por la familia o el panel**: la elevación global es alta solo porque la categoría es propia de una
  familia o porque los análisis se miden juntos (cenizas y proteínas van en el mismo panel). Se recalcula la elevación
  sobre las muestras que pueden tener la regla (las familias donde aparecen sus ítems, con sus análisis medidos):
  `lift_ajustado`. Si es menor que 1,2, la regla es trivial.

Las reglas útiles se ordenan por elevación (`lift`); al lado, `lift_ajustado`.
""")

code("""
MIN_SOPORTE, MIN_CONFIANZA, MAX_LEN = 0.01, 0.6, 3
columnas = ["regla", "support", "confidence", "lift", "lift_ajustado", "familias"]

def reglas_de(M_):
    frec = fpgrowth(M_, min_support=MIN_SOPORTE, use_colnames=True, max_len=MAX_LEN)
    r = association_rules(frec, metric="confidence", min_threshold=MIN_CONFIANZA)
    r = cz.filtrar_reglas(r, M_)
    r["regla"] = [f"{nombre(a)} ⇒ {nombre(b)}" for a, b in zip(r["antecedents"], r["consequents"])]
    return frec, r      # los listados ordenan por elevación y desempatan por soporte y por texto: sin eso, el orden de los empates cambia entre corridas

frec, reglas = reglas_de(M)
print(f"Soporte mínimo {MIN_SOPORTE:.0%} de {len(M)} transacciones = {MIN_SOPORTE * len(M):.0f} muestras (Senasa) · y de {len(matrices['senasa_mq'])} = {MIN_SOPORTE * len(matrices['senasa_mq']):.0f} (Senasa + MQ)")
print(f"{len(frec)} conjuntos frecuentes (por tamaño: {frec['itemsets'].apply(len).value_counts().sort_index().to_dict()}) → {len(reglas)} reglas con confianza ≥ {MIN_CONFIANZA}")
print("Reglas por motivo:", reglas["motivo_trivial"].fillna("útil").value_counts().to_dict())
for m in ["mismo análisis", "cuenta", "define la familia", "familia redundante", "cortes globales en la familia", "explicada por la familia o el panel"]:
    ej = reglas[reglas["motivo_trivial"] == m].sort_values(["lift", "support", "regla"], ascending=[False, False, True], kind="mergesort")
    if len(ej):
        print(f"Triviales «{m}» (las 3 de mayor elevación):"); display(ej[columnas].head(3).round(3).reset_index(drop=True))
""")

code("""
utiles = reglas[reglas["motivo_trivial"].isna()].sort_values(["lift", "support", "regla"], ascending=[False, False, True], kind="mergesort").reset_index(drop=True)
print(f"{len(utiles)} reglas no triviales, ordenadas por elevación:")
display(utiles[columnas].round(3))
""")

code("""
fig, ax = plt.subplots(figsize=(9, 4.8))
tr = reglas["motivo_trivial"].notna()
ax.scatter(reglas.loc[tr, "confidence"], reglas.loc[tr, "lift"], s=12, alpha=0.5, color=GRIS, label=f"triviales ({tr.sum()})")
ax.scatter(reglas.loc[~tr, "confidence"], reglas.loc[~tr, "lift"], s=30, color=AZUL, label=f"no triviales ({(~tr).sum()})")
ax.axhline(1, color=GRIS, ls=":", label="elevación 1: azar"); ax.set_yscale("log")
ax.set_xlabel("confianza"); ax.set_ylabel("elevación (log)"); ax.legend(fontsize=8)
ax.set_title("La elevación global alta no alcanza: la mayoría de esas reglas son triviales")
plt.tight_layout(); plt.show()
""")

md("""
## 9 · Reglas con Senasa + MQ

Las muestras solo MQ llevan la familia de MQ encolumnado (la asigna un modelo o, si el modelo no decide y la descripción es clara, una regla por descripción; no está revisada a mano). En las familias con terciles por estado (sección 5) cada muestra se compara con las de su mismo estado. Las «sin clasificar» usan los cortes globales y solo tienen los cuatro análisis principales.
""")

code("""
M2 = matrices["senasa_mq"]
frec2, reglas2 = reglas_de(M2)
print(f"Senasa + MQ: {M2.shape[0]} transacciones × {M2.shape[1]} ítems · {len(frec2)} frecuentes → {len(reglas2)} reglas")
print("Reglas por motivo:", reglas2["motivo_trivial"].fillna("útil").value_counts().to_dict())
utiles2 = reglas2[reglas2["motivo_trivial"].isna()].sort_values(["lift", "support", "regla"], ascending=[False, False, True], kind="mergesort").reset_index(drop=True)
display(utiles2[columnas].head(20).round(3))
nuevas = sorted(set(utiles2["regla"]) - set(utiles["regla"]))
print(f"Reglas útiles que aparecen con MQ y no solo con Senasa: {len(nuevas)} (las primeras 15):")
for r in nuevas[:15]:
    print("   ", r)
""")

md("""
## 10 · Lo que queda abierto

- Las categorías por terciles son relativas a la familia: `humedad=alto` no es el mismo número en el queso que en el
  dulce de leche. Por eso las reglas que mezclan familias se leen con `lift_ajustado`.
- El soporte mínimo de 1 % deja afuera a las familias chicas (manteca, helado, yogur). Para ellas conviene correr las
  reglas dentro de cada familia.
- Por la misma razón, no todos los subtags llegan al 1 % de las transacciones: quedan afuera los polvos chicos y, según
  la versión, hasta algún subtag del DDL. La celda de abajo cuenta cuáles entran y cuáles
  no, en cada versión. Para ver las reglas de los que quedan afuera hay que correrlas dentro de la familia o bajar el
  soporte.
- Las decisiones de `cortes_categorias.md` siguen abiertas (bases de la leche fluida, del yogur y del helado; lactosa con
  referencia 4,83; cortes globales en las familias chicas y en las muestras sin familia; dulce de leche con grasa < 6).
  Se cambian en las constantes de `categorizar.py`.
""")

code("""
for version, Mx in [("senasa", M), ("senasa_mq", M2)]:
    corte = MIN_SOPORTE * len(Mx)
    n_sub = Mx[[c for c in Mx.columns if c.startswith("subtag=")]].sum().sort_values(ascending=False)
    n_sub.index = n_sub.index.str.replace("subtag=", "", regex=False)
    print(f"{version} (corte: {corte:.0f} de {len(Mx)} transacciones)")
    print("   entran:   ", ", ".join(f"{k} ({v})" for k, v in n_sub[n_sub >= corte].items()))
    print("   quedan afuera:", ", ".join(f"{k} ({v})" for k, v in n_sub[n_sub < corte].items()))
""")



def celdas():
    return list(c)
