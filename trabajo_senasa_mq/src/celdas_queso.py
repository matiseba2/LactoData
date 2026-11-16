"""Celdas de la rama de queso de las secciones 3 y 4 de trabajo_senasa_mq.ipynb (las arma celdas_clase41.celdas).

Los quesos que el catálogo confirma NO pasan por el k-medias de cuatro análisis (los quesos de Senasa casi nunca traen
proteína ni cenizas): van a una rama propia, con humedad (100 − sólidos totales) y materia grasa, y su subtag sale de la
clase de humedad del CAA (art. 605). La lógica está en clustering_quesos.py (rama_queso, no_supervisado) y en
segundo_nivel.py (subtag_queso); acá solo están las celdas.

    celdas(con_mq, texto)    texto = el markdown con la respuesta numérica a «¿se puede diferenciar sin un tag previo?»
"""
import re

import nbformat as nbf


def _bajar(s):
    return re.sub(r"(?m)^(#+) ", r"#\1 ", s)


RAMA_MD = """
### Rama de queso: fuera del k-medias de cuatro análisis
Los quesos no pasan por el k-medias: casi ninguno trae proteína ni cenizas (Senasa no las pide en queso), así que con la
regla de los cuatro análisis quedaban afuera o, si entraban, formaban un grupo débil entre el dulce de leche y los
polvos. Van a una **rama propia** con lo que sí tienen: **humedad** (100 − sólidos totales) y **materia grasa**. Solo
entra como queso **lo que confirma el catálogo**:

- **Senasa**: `desc_familia = queso`.
- **Solo MQ** (con `MQ encolumnado.xlsx`): la familia de las muestras solo MQ sale de un modelo (`mq_tipo_muestra.csv`,
  sin revisar a mano), así que no alcanza. Hace falta que el **rótulo** nombre una variedad de queso (el subtipo
  normalizado con el catálogo). Las que el modelo llama queso pero el rótulo no confirma **no entran** a la rama y siguen
  en el k-medias si tienen los cuatro análisis (desconfiar por defecto: no se les asume categoría).

Una muestra tiene **una sola asignación**: los quesos confirmados salen del k-medias (se quitan antes de elegir k y de
buscar outliers), de modo que el tag QUESO sale solo de esta rama. El **subtag** es la clase de humedad del CAA (art. 605):
< 36 duro, 36 a 46 semiduro, 46 a 55 pasta blanda, ≥ 55 muy alta humedad (`segundo_nivel.subtag_queso`; en
`prueba_clustering_quesos.ipynb` coincide con la consistencia del rótulo mejor que cualquier agrupamiento). Se piden MG y
ST y se descartan los imposibles (MG > ST, humedad + ST fuera de 100 ± 1). Código: `clustering_quesos.rama_queso`.
"""

RAMA_CODE = r"""
rama = rama.copy()
rama["subtag"] = N.subtag_queso(rama["humedad"]); rama["criterio_subtag"] = N.C_QUESO
assert (rama["subtag"] == rama["caa_humedad"].map(lambda c_: "QUESO_" + Q.NOMBRE_CAA[c_])).all()   # misma regla que el CAA
print("Cuántos quesos confirma el catálogo y cuántos entran a la rama:")
print({k_: v_ for k_, v_ in conteo_q.items()})
print(f"Rama de queso: {len(rama)} muestras · por origen: {rama['origen'].value_counts().to_dict()}")
print("Subtag:", rama["subtag"].value_counts().to_dict())
display(pd.crosstab(rama["subtag"], rama["origen"], margins=True, margins_name="total"))
print("Subtag (filas) contra la consistencia del rótulo (columnas); el rótulo no entra, solo se compara:")
display(pd.crosstab(rama["subtag"], rama["consistencia"].fillna("sin dato"), margins=True, margins_name="total"))
__SIN_CONFIRMAR__
"""

SIN_CONFIRMAR_MQ = r"""
ids_ = set(sin_confirmar["Identificador"])
en_pre = muestras_pre["Identificador"].isin(ids_)
en_km = muestras["Identificador"].isin(ids_).to_numpy()
print(f"Solo MQ de familia queso sin confirmar por el rótulo: {len(sin_confirmar)} ({int(sin_confirmar[['mg', 'st']].notna().all(axis=1).sum())} con MG y ST). "
      f"No entran a la rama. Con los cuatro análisis entran al k-medias {int(en_pre.sum())}; {int(muestras_pre.loc[en_pre, 'outlier'].sum())} salen como outliers; "
      f"las otras {int(en_km.sum())} reciben el tag:", pd.Series(tag_de[en_km]).value_counts().to_dict())
"""

SIN_CONFIRMAR_SENASA = r"""
print("Con solo Senasa no hay solo MQ: todos los quesos son de desc_familia = queso.")
"""

NOSUP_MD = """
### ¿Se pueden diferenciar los quesos en clusters sin aprender de un tag previo?
Se agrupan los quesos de la rama usando **solo humedad y MGES** (grasa en el extracto seco = 100·MG/ST), estandarizadas y
sin los outliers por consenso (3 %). No se usa ningún rótulo ni el CAA para ajustar nada: **k-medias** con k por silueta
y **mezcla de gaussianas** con k por BIC. Recién después se comparan con la **clase de humedad del CAA**, la **clase de
MGES del CAA** y la **consistencia** y el **tipo** del rótulo: Rand ajustado (ARI, 0 = azar, 1 = igual) e información
mutua normalizada (NMI), tablas cruzadas y estabilidad con otras semillas (`clustering_quesos.no_supervisado`). La fila
«regla CAA de humedad» es el corte del CAA aplicado directo, sin agrupar: es el techo contra el que mirar.
"""

NOSUP_CODE = r"""
ns = Q.no_supervisado(rama)
print(f"{len(rama)} quesos · {ns['n_outliers']} outliers por consenso · se agrupan {len(ns['base'])}")
print("k-medias, silueta por k:", {k_: round(float(v_), 3) for k_, v_ in ns["silueta"].items()}, "→ k =", ns["k_km"])
print("Mezcla de gaussianas, BIC por k:", {k_: round(float(v_)) for k_, v_ in ns["bic"].items()}, "→ k =", ns["k_gm"])
display(ns["tabla"].round(2).T)
for (alg_, ref_), t_ in ns["cruzadas"].items():
    print(f"{alg_} (filas) contra {ref_} (columnas):")
    display(t_)
print("Estabilidad con 10 semillas: ARI de cada partición contra la de la semilla base (mismo k) y k que elige cada criterio:")
display(ns["estabilidad"].round(2))
print("Promedio:", ns["estabilidad"].mean().round(3).to_dict())

fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
for ax_, (nombre_, et_) in zip(ejes, ns["et"].items()):
    ax_.scatter(ns["base"]["humedad"], ns["base"]["mges"], c=[COLORES[e_ % len(COLORES)] for e_ in et_], s=6, alpha=0.6)
    Q.lineas_caa(ax_); ax_.set_xlabel("humedad (g/100 g)"); ax_.set_ylabel("MGES (g/100 g de extracto seco)"); ax_.set_title(nombre_, fontsize=10)
fig.suptitle("Quesos agrupados sin tag previo (humedad y MGES); las líneas son los cortes del CAA")
plt.tight_layout(); plt.show()
"""


def celdas(con_mq, texto_nosup):
    c = []
    c.append(nbf.v4.new_markdown_cell(_bajar(RAMA_MD.strip("\n"))))
    c.append(nbf.v4.new_code_cell(RAMA_CODE.replace("__SIN_CONFIRMAR__", (SIN_CONFIRMAR_MQ if con_mq else SIN_CONFIRMAR_SENASA).strip("\n")).strip("\n")))
    c.append(nbf.v4.new_markdown_cell(_bajar(NOSUP_MD.strip("\n"))))
    c.append(nbf.v4.new_code_cell(NOSUP_CODE.strip("\n")))
    c.append(nbf.v4.new_markdown_cell(_bajar(texto_nosup.strip("\n"))))
    return c
