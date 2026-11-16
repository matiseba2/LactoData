"""Arma trabajo_senasa_mq.ipynb, el notebook único del trabajo Senasa + MQ.

No editar el .ipynb a mano: se regenera con este script y se ejecuta desde esta carpeta.

    cd trabajo_senasa_mq
    ../.venv/bin/python build_notebook.py
    ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace trabajo_senasa_mq.ipynb

Secciones: 0 rutas y parámetros · 1 lectura y correcciones · 2 Excel de inconsistencias · 3 agrupamiento Senasa ·
4 agrupamiento Senasa + MQ (con el contenido didáctico de la clase 41) · 5 cortes CAA · 6 categorías y reglas de
asociación · 7 MQ encolumnado · 8 datos usados en cada entrenamiento · 9 verificación. El código está en src/; el notebook solo llama a sus funciones
(y las celdas de las secciones 1, 3, 4 y 6 salen de src/celdas_clase41.py y src/celdas_asociacion.py).
"""
import sys
from pathlib import Path

import nbformat as nbf

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI / "src"))
import celdas_asociacion  # noqa: E402
import celdas_clase41  # noqa: E402

SALIDA = AQUI / "trabajo_senasa_mq.ipynb"

c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: c.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# Trabajo Senasa + MQ: revisión, agrupamiento, categorías y reglas de asociación
**Aprendizaje Automático y Grandes Datos · UNRaf · 2026**

Un solo cuaderno para todo el trabajo actual. Lee las entradas de `datos_entrada/`, usa el código de `src/` y escribe
todo en `resultados/` (`excel/`, `tablas/`, `modelos/`). Se regenera con `build_notebook.py` (no editar a mano).

| sección | qué hace | escribe |
|---|---|---|
| 0 | rutas y parámetros | |
| 1 | lectura de Senasa y correcciones | |
| 2 | Excel de inconsistencias para revisión (solo Senasa; Senasa + MQ) | `excel/dataset_senasa_revision.xlsx`, `excel/dataset_senasa_mq_revision.xlsx` |
| 3 | agrupamiento solo Senasa (outliers por consenso, k-medias, tags, métricas y segundo nivel por familia: DDL y polvos; el queso va a una rama propia con la humedad del CAA, fuera del k-medias) | `tablas/clusters_senasa.csv`, `modelos/agrupamiento_senasa_clase41.joblib` |
| 4 | agrupamiento Senasa + MQ (MQ sale de `MQ encolumnado.xlsx`, sin las MO), con la clase 41 completa (y lo mismo que la sección 3, con la rama de queso y su agrupamiento sin tag previo) | `tablas/clusters_senasa_mq.csv`, `modelos/agrupamiento_senasa_mq_clase41.joblib` |
| 5 | cortes para categorizar (CAA por familia, terciles) | `tablas/cortes_categorias.csv` |
| 6 | categorías, ítems y reglas de asociación | `tablas/categorias_*.csv`, `tablas/transacciones_*.parquet` |
| 7 | MQ con el formato de Senasa encolumnado (solo si `REGENERAR_MQ = True`) | `excel/MQ encolumnado*.xlsx` |
| 8 | datos usados en cada entrenamiento (cuántas muestras entran a cada modelo, qué se excluye y cómo se reparte) | `resultados/datos_entrenamiento/trabajo_senasa_mq.csv` |
| 9 | verificación de las salidas | |
""")

md("""
## 0 · Rutas y parámetros
Todas las rutas salen de `src/rutas.py` y son relativas a la carpeta `trabajo_senasa_mq/` (se puede cambiar con la
variable de entorno `TRABAJO_SENASA_MQ_RAIZ`). `REGENERAR_MQ = False` lee los dos Excel de MQ encolumnado ya generados;
con `True` los vuelve a armar desde los crudos de MQ (tarda más y los reemplaza). Las secciones 3 y 4 leen `MQ encolumnado.xlsx` (la sección 7 lo escribe al final), así que con `True` hay que ejecutar el cuaderno dos veces para que el agrupamiento use el Excel nuevo.
""")

code(r"""
import sys
from pathlib import Path
SRC = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "rutas.py").exists()) / "src"
sys.path.insert(0, str(SRC))
import rutas as R
R.preparar()

REGENERAR_MQ = False

print("RAIZ:", R.RAIZ)
for nombre in ["SENASA", "CORRECCIONES", "MQ_ANCHA", "MQ_TIPO", "MQ_CRUDOS", "EXCEL", "TABLAS", "MODELOS"]:
    ruta = getattr(R, nombre)
    print(f"   {nombre:13} {ruta.relative_to(R.RAIZ)}{'' if ruta.exists() else '   (FALTA)'}")
""")

c.extend(celdas_clase41.celdas_lectura())

md("""
## 2 · Excel de inconsistencias
`src/revision_excel.py` (ex `exportar_dataset_senasa_mq.py`) arma un agrupamiento propio con el mismo método de las secciones 3 y 4
(outliers por consenso y k-medias por silueta), pero **sobre otra población**: no saca las familias que no son lácteas ni
los quesos de la rama de queso. Por eso su k y sus cantidades (muestras, outliers) **no coinciden** con las de esas
secciones; sirve para revisar las muestras, no para leer el agrupamiento. Marca las inconsistencias de cada muestra con reglas (suma > 102, humedad + sólidos, Senasa contra MQ,
todo cerca de 0...). Escribe dos Excel con las hojas `dataset`, `inconsistencias`, `resumen` y `notas`: uno solo con
Senasa y otro con Senasa + MQ. Las correcciones que salen de revisarlos van a `datos_entrada/correcciones_clase41.csv`.
""")

code(r"""
import revision_excel
revision_excel.exportar()
""")

c.extend(celdas_clase41.celdas("senasa", didactico=False))
c.extend(celdas_clase41.celdas("senasa_mq", didactico=True))

md("""
## 5 · Cortes para categorizar (CAA por familia)
`src/cortes_caa.py` (ex `build_cortes_categorias.py`) arma `resultados/tablas/cortes_categorias.csv`: donde el Código
Alimentario Argentino define clases, esas; donde no, terciles dentro de la familia; con menos de 30 valores, cortes
globales (familia `*`). La tabla no se edita a mano: se regenera acá. Criterios y decisiones abiertas:
`docs/cortes_categorias.md`.

**Terciles por estado.** En suero / permeado, concentrado proteico, yogur / bebida láctea, base nutricional y
manteca / crema la composición cambia por completo entre el polvo y el líquido (o entre la manteca y la crema), así que
los terciles se calculan aparte para cada estado (columna `estado`). El estado de una muestra sale de la humedad medida
(≤ 15 polvo, ≥ 25 líquido; en manteca / crema, < 30 es manteca) y, si no hay humedad, del estado declarado. Donde
Senasa tiene menos de 30 valores de un estado (casi no tiene cremas ni sueros líquidos) se suman las muestras nuevas
de MQ. Una muestra de esas familias sin estado no se categoriza en los análisis que dependen del estado.
""")

code(r"""
import cortes_caa
cortes = cortes_caa.armar()
display(cortes.groupby(["analisis", "fuente"], sort=False).size().rename("filas").reset_index()
        .assign(fuente=lambda t: t["fuente"].str.replace(r" \(http.*\)", "", regex=True)))
""")

c.extend(celdas_asociacion.celdas())

md("""
## 7 · MQ con el formato de Senasa encolumnado
`src/mq_encolumnado.py` (ex `build_mq_encolumnado.py`) pone las muestras de tipo MQ con las 177 columnas de
`Senasa encolumnado.xlsx` (más los analitos que solo tiene MQ), para sumarlas al entrenamiento del modelo supervisado.
Usa los crudos de `datos_entrada/mq/crudos/`, `muestras_ancha.csv`, `mq_tipo_muestra.csv` y el extractor de
`src/compartido/Filtrado_de_Datos_2_ordenado.ipynb`; lo que escribe pasa por `src/compartido/anonimizar.py`.
Con `REGENERAR_MQ = False` (por defecto) solo lee los dos Excel ya generados.

La familia de las muestras nuevas la asigna un modelo. Las que el modelo deja «sin clasificar» pasan por las reglas de
`src/reglas_descripcion.py`: si la descripción nombra un producto de una sola familia (crema, manteca, ricota, WPC,
salame...) se clasifica; si es ambigua, queda «sin clasificar». `desc_familia_fuente` dice de dónde sale cada familia.
""")

code(r"""
import mq_encolumnado
mq_todas, mq_nuevas = mq_encolumnado.armar(REGENERAR_MQ)
print(f"MQ encolumnado: {mq_todas.shape} · solo nuevas: {mq_nuevas.shape}")
display(mq_todas["desc_familia"].value_counts().rename("muestras MQ").to_frame()
        .join(mq_nuevas["desc_familia"].value_counts().rename("nuevas (no están en Senasa)")).fillna(0).astype(int))
display(mq_nuevas["desc_familia_fuente"].value_counts().rename("muestras nuevas").to_frame())
""")

md("""
### 7.1 · Lo declarado contra la humedad medida
`src/declarado_vs_medido.py` compara lo que la clasificación afirma sobre la humedad con el resultado: la consistencia
del queso (baja / mediana / alta, donde «alta» incluye «muy alta») contra la clase del CAA art. 605, y el estado
(polvo ≤ 15, líquido ≥ 25). «borde» es quedar fuera de la clase por 1 g/100 g o menos. El detalle de las que no
coinciden queda en `resultados/excel/declarado_vs_medido.xlsx`.
""")

code(r"""
import declarado_vs_medido
dvm, dvm_resumen = declarado_vs_medido.armar()
display(dvm_resumen)
q = dvm[dvm["eje"] == "consistencia del queso"]
display(q.groupby(["origen", "origen_declarado"]).agg(muestras=("coincide", "size"), coincide=("coincide", "mean"))
        .assign(coincide=lambda t: (100 * t["coincide"]).round(1)).rename(columns={"coincide": "% coincide"}))
""")

md("""
## 8 · Datos usados en cada entrenamiento
Cuántas muestras entran a cada modelo que se entrena en este cuaderno, con la misma tabla que en los demás cuadernos:
cuántas hay antes de los filtros propios del modelo, cuántas se excluyen y por qué, cuántas se usan y cómo se reparten
entre entrenamiento, validación y prueba. Los números salen de las variables de las secciones 3, 4 y 6 (no están escritos
a mano). Una fila por modelo y por población (solo Senasa; Senasa + MQ). Los agrupamientos no tienen prueba: se ajustan
con todas las muestras que usan. La tabla también se guarda en `resultados/datos_entrenamiento/trabajo_senasa_mq.csv`.
""")

code(r"""
import datos_entrenamiento as DE

ETIQUETAS = {"senasa": "solo Senasa", "senasa_mq": "Senasa + MQ"}
SIN_PARTICION = "sin partición: se ajusta con todas las muestras que usa"
filas = []
for clave, etiqueta in ETIQUETAS.items():
    e = ENTRENOS[clave]
    origen = "muestras de Senasa" if clave == "senasa" else "muestras de Senasa y de MQ"
    umbral = str(N.UMBRAL_DDL).replace(".", ",")
    filas += [
        DE.fila("I", f"k-medias preliminar y consenso de outliers, {etiqueta}", f"{origen} con los cuatro análisis (humedad, grasa, proteínas y cenizas)",
                e["inicial"], e["pre"],
                motivo=(f"{e['n_hs']} con humedad + sólidos incoherentes; {e['n_nolact']} de familias no lácteas; {e['n_quesos']} quesos (van a su rama); "
                        f"{e['n_sin4']} sin los cuatro análisis; {e['n_suma']} suman más de {SUMA_MAX}"),
                particion=SIN_PARTICION,
                nota=(f"k-medias preliminar con k = {e['k_pre']} (silueta, {N_INIT_KMEDIAS} reinicios, semilla {SEMILLA}); el consenso (factor local, bosque de "
                      f"aislamiento y elipse, 2 de 3 votos, dentro de cada grupo preliminar) marca {e['n_out']} outliers")),
        DE.fila("I", f"k-medias final, {etiqueta}", "muestras del k-medias preliminar sin los outliers", e["pre"], e["final"],
                motivo=f"{e['n_out']} outliers por consenso", particion=SIN_PARTICION,
                nota=f"k = {e['k']} por silueta máxima ({e['silueta']:.3f}); {N_INIT_KMEDIAS} reinicios, semilla {SEMILLA}"),
        DE.fila("I", f"segundo nivel del DDL: regresión logística, {etiqueta}", "muestras con tag DDL del k-medias final", e["ddl_n"], e["ddl_lab"],
                motivo="sin subtipo conocido: no se usan para ajustar (el modelo les propone un subtag)",
                particion="sin partición en este cuaderno (se ajusta con todas las que tienen subtipo); la validación por cliente, con clientes no vistos, está en prueba_subtipo_ddl.ipynb",
                nota=(f"subtipo conocido por clase: {e['ddl_clases']}; decide en {e['ddl_decide']} de {e['ddl_n']} (probabilidad ≥ {umbral} y sin novedad; "
                      f"la novedad se detecta con Mahalanobis robusta y LOF)")),
        DE.fila("I", f"segundo nivel de polvos: k-medias en base seca, {etiqueta}", "muestras con tag POLVO_GRASO, POLVO_MAGRO o PROT_ALTA",
                e["pol_tag"], e["pol_rama"] - e["pol_out"],
                motivo=f"{e['pol_tag'] - e['pol_rama']} con humedad > 15 (no son polvo); {e['pol_out']} outliers de la rama (consenso)",
                particion=SIN_PARTICION,
                nota=(f"k = {e['pol_k']} por silueta ({e['pol_sil']:.2f}) en grasa, proteína y cenizas en base seca; además dos reglas de composición "
                      f"(leche descremada y suero reengrasado) que no se entrenan")),
        DE.fila("I", f"rama de queso: regla del CAA, no se entrena, {etiqueta}", "quesos que confirma el catálogo", e["q_leidos"], e["q_rama"],
                motivo=(f"{e['q_leidos'] - e['q_mgst']} sin materia grasa o sin sólidos totales; {e['q_mgst'] - e['q_rama']} imposibles "
                        f"(MG > ST o humedad + sólidos fuera de 100 ± 1)"),
                entrenamiento=0, particion="no se entrena: el subtag sale de la clase de humedad del CAA (art. 605)",
                nota=(f"entran las {e['q_rama']}, sin quitar outliers" +
                      (f"; {e['q_sin_confirmar']} solo MQ de familia queso que el rótulo no confirma no entran" if e["q_sin_confirmar"] else ""))),
        DE.fila("I", f"quesos sin tag previo: k-medias y mezcla de gaussianas, {etiqueta}", "quesos de la rama (humedad y MGES)", e["q_rama"], e["q_base"],
                motivo=f"{e['q_out']} outliers por consenso", particion=SIN_PARTICION + "; estabilidad con 10 semillas",
                nota=f"k-medias con k = {e['q_k_km']} (silueta) y mezcla con k = {e['q_k_gm']} (BIC); es un ensayo, no un modelo guardado"),
    ]
    if e["comparacion"]:
        filas.append(DE.fila("I", f"comparación de algoritmos (k-medias sin escalar, minilotes, Ward, enlaces, bisectivo), {etiqueta}",
                             "muestras del k-medias final", e["pre"], e["final"], motivo=f"{e['n_out']} outliers por consenso", particion=SIN_PARTICION,
                             nota=(f"ninguno se guarda: se comparan con el k-medias final (k = {e['k']}); la sección 3 no los repite. Entran también las 40 corridas "
                                   f"de A.2 (inicialización al azar contra k-means++)")))
for clave, etiqueta, mx, fr, rg in [("senasa", "solo Senasa", matrices["senasa"], frec, reglas), ("senasa_mq", "Senasa + MQ", matrices["senasa_mq"], frec2, reglas2)]:
    tv = tablas[clave]
    filas.append(DE.fila("I", f"reglas de asociación (FP-growth), {etiqueta}", "muestras con ítems (categorías de los análisis, familia, tag y subtag)", len(tv), len(mx),
                         motivo=f"{len(tv) - len(mx)} sin ningún ítem: no son transacciones (hay {int(tv['excluida'].sum())} excluidas por humedad + sólidos, que pierden los ítems de sus análisis)",
                         particion="sin partición: las reglas se buscan en todas las transacciones",
                         nota=(f"soporte mínimo {MIN_SOPORTE:.0%}, hasta {MAX_LEN} ítems, confianza ≥ {str(MIN_CONFIANZA).replace('.', ',')}: "
                               f"{len(fr)} conjuntos frecuentes y {len(rg)} reglas, {int(rg['motivo_trivial'].isna().sum())} no triviales")))
display(DE.guardar(filas, "trabajo_senasa_mq"))
""")

md("""
## 9 · Verificación
Que estén todas las salidas, con su forma y su huella (sha256), y que sean coherentes entre sí: el tag de cada muestra
en `clusters_*.csv` es el del modelo guardado, y el subtag el que dan los modelos del segundo nivel del joblib; las tablas de categorías tienen una fila por muestra de Senasa (o de
Senasa + MQ), y MQ encolumnado empieza con las 177 columnas de Senasa encolumnado.
""")

code(r"""
import hashlib
import joblib
import segundo_nivel

def huella(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]

salidas = {
    "excel/dataset_senasa_revision.xlsx": None, "excel/dataset_senasa_mq_revision.xlsx": None,
    "excel/MQ encolumnado.xlsx": None, "excel/MQ encolumnado - solo nuevas.xlsx": None,
    "tablas/clusters_senasa.csv": None, "tablas/clusters_senasa_mq.csv": None, "tablas/cortes_categorias.csv": None,
    "tablas/categorias_senasa.csv": None, "tablas/categorias_senasa_mq.csv": None,
    "tablas/declarado_vs_medido.csv": None, "excel/declarado_vs_medido.xlsx": None,
    "tablas/transacciones_senasa.parquet": None, "tablas/transacciones_senasa_mq.parquet": None,
    "modelos/agrupamiento_senasa_clase41.joblib": None, "modelos/agrupamiento_senasa_mq_clase41.joblib": None,
}
filas = []
for rel in salidas:
    p = R.RESULTADOS / rel
    assert p.exists(), f"falta {rel}"
    if p.suffix == ".csv":
        forma = pd.read_csv(p, low_memory=False).shape
    elif p.suffix == ".parquet":
        forma = pd.read_parquet(p).shape
    elif p.suffix == ".xlsx":
        forma = {h: d.shape for h, d in pd.read_excel(p, sheet_name=None).items()}
    else:
        forma = "-"
    filas.append({"salida": rel, "forma": str(forma), "bytes": p.stat().st_size, "sha256 (12)": huella(p)})
display(pd.DataFrame(filas))

for v, archivo in [("senasa", "agrupamiento_senasa_clase41.joblib"), ("senasa_mq", "agrupamiento_senasa_mq_clase41.joblib")]:
    cl = pd.read_csv(R.TABLAS / f"clusters_{v}.csv", encoding="utf-8-sig")
    m = joblib.load(R.MODELOS / archivo)
    con = cl.dropna(subset=["cluster"])
    assert (con["cluster"].astype(int).map(m["tags"]) == con["tag"]).all(), v
    assert cl.loc[cl["outlier"], "tag"].isna().all() and cl.loc[cl["outlier"], "subtag"].isna().all(), v
    assert con["subtag"].notna().all(), v
    q = cl[cl["rama"] == "queso (humedad y MG)"]            # la rama de queso: sin cluster, tag QUESO, subtag por humedad del CAA
    assert q["cluster"].isna().all() and (q["tag"] == "QUESO").all() and not q["outlier"].any(), v
    assert (q["subtag"].to_numpy() == segundo_nivel.subtag_queso(q["humedad"])).all(), v
    assert not set(q["Identificador"]) & set(con["Identificador"]), f"{v}: un queso en el k-medias y en la rama"
    assert len(q) + len(con) + int(cl["outlier"].sum()) == len(cl), v
    sn = m["segundo_nivel"]
    out_polvo = con["Identificador"].isin(sn.get("polvo", {}).get("outliers", [])).to_numpy()
    sub, crit = segundo_nivel.predecir(sn, con[COLUMNAS], con["tag"], out_polvo)
    assert (sub == con["subtag"].to_numpy()).all() and (crit == con["criterio_subtag"].to_numpy()).all(), v
    cat = pd.read_csv(R.TABLAS / f"categorias_{v}.csv", encoding="utf-8-sig", low_memory=False)
    assert cat["Identificador"].is_unique, v
    print(f"{v}: {len(con)} muestras con tag {sorted(m['tags'].values())} · subtags {sorted(set(con['subtag']) - set(con['tag']))} · "
          f"{int(cl['outlier'].sum())} outliers · {len(q)} quesos en su rama {q['subtag'].value_counts().to_dict()} · categorías {cat.shape}")
assert len(pd.read_csv(R.TABLAS / "categorias_senasa.csv", encoding="utf-8-sig", low_memory=False)) == len(S.crudo)
cols_senasa = list(pd.read_excel(R.SENASA, nrows=0).columns)
for d in (mq_todas, mq_nuevas):
    assert list(d.columns[:177]) == cols_senasa
sin = mq_nuevas["desc_familia"].eq("sin clasificar")
assert (mq_nuevas.loc[sin, "desc_familia_fuente"] == "sin clasificar").all() and not (mq_nuevas.loc[~sin, "desc_familia_fuente"] == "sin clasificar").any()
ce = cortes[cortes["estado"].fillna("").ne("")]
assert set(ce["familia"]) <= set(cortes_caa.FAMILIAS_CON_ESTADO) and set(ce["estado"]) <= set(cortes_caa.ESTADOS)
print("Verificación: todo en orden.")
""")

nb = nbf.v4.new_notebook()
nb["cells"] = c
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
if __name__ == "__main__":
    nbf.write(nb, SALIDA)
    print(f"Escrito: {SALIDA} ({len(c)} celdas)")
