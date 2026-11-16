"""Arma prediccion_tipo.ipynb (tipo de muestra a partir de lo medido: familia y subtipo).

    cd trabajo_senasa_mq
    ../.venv/bin/python build_prediccion_tipo.py
    ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace prediccion_tipo.ipynb

Al final escribe y abre app_prediccion.py (celdas de src/celdas_app.py). El código está en src/prediccion_tipo.py; el notebook solo lo llama y muestra. Tarda unos 4 minutos.
"""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import celdas_app  # noqa: E402

SALIDA = Path(__file__).resolve().parent / "prediccion_tipo.ipynb"
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: c.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# ¿Se puede predecir el tipo de una muestra por lo que se le midió?

**Datos.** Senasa (clasificación manual) + MQ nuevas (clasificación automática por texto o reglas).

**Fuera del entrenamiento** (se usan para evaluar):
- muestras de **calibración**: «control de laboratorio» o descripción con calibración, patrón, referencia,
  repetibilidad / reproducibilidad (la descripción se lee en memoria, no se muestra);
- «ensayo» y «pool»;
- las que **declaran algo que no coincide** con la humedad medida (`declarado_vs_medido`; «borde» sí entra);
- humedad + sólidos fuera de 100 ± 1 y las «sin clasificar».

**Variables.** Resultados numéricos (NaN si no se midieron) y qué análisis se pidieron.
**Modelo.** HistGradientBoosting en dos niveles: familia; después subtipo dentro de cada familia (en queso, la
consistencia del CAA). **Validación** cruzada en 5 pliegues agrupada por envío. Con probabilidad < 0,80 la muestra
queda «sin decidir».
""")

code(r"""
import sys
from pathlib import Path
SRC = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "rutas.py").exists()) / "src"
sys.path.insert(0, str(SRC))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import prediccion_tipo as PT

r = PT.armar(escribir=True)
t = r["datos"]
r["resumen"]
""")

md("## Métricas fuera de pliegue\nExactitud, F1 macro y, con el umbral, cuántas muestras se deciden y con qué exactitud.")
code('r["metricas"]')

md("## ¿Las probabilidades son creíbles?\nExactitud observada por rango de probabilidad (familia).")
code('PT.fiabilidad(r["oof_familia"])')

code(r"""
cm = r["confusion"]["familia"]
cmn = cm.div(cm.sum(1), axis=0)
fig, ax = plt.subplots(figsize=(9, 7.5))
ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(len(cm)), cm.columns, rotation=60, ha="right"); ax.set_yticks(range(len(cm)), cm.index)
for i in range(len(cm)):
    for j in range(len(cm)):
        if cmn.iat[i, j] >= .05:
            ax.text(j, i, f"{cmn.iat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if cmn.iat[i, j] > .5 else "k")
ax.set(xlabel="predicha", ylabel="real", title="Familia: matriz de confusión (fila = 100 %)")
plt.tight_layout(); plt.show()
""")

md("## Muestras de calibración\nNo entrenaron. Acierto de familia por origen y familia declarada.")
code(r"""
cal = r["calibracion"]
cal[cal["desc_familia"].ne(PT.SIN)].groupby(["origen", "desc_familia"]).agg(
    muestras=("acierta_familia", "size"), acierto=("acierta_familia", "mean"), prob_media=("prob_familia", "mean")).round(2)
""")

md("## Declarado que no coincide con lo medido\nEn quesos: consistencia declarada, la que corresponde a la humedad medida y la que predice el modelo.")
code(r"""
nc = r["no_coinciden"]
q = nc[nc["desc_familia"].eq("queso")]
pd.crosstab([q["subtipo"].rename("declarada"), q["clase_por_humedad"].rename("por humedad")], q["subtipo_pred"].rename("predicha"))
""")

md("## «Sin clasificar»: lo que propone el modelo\nSolo propuestas; no se cambia ninguna familia.")
code(r"""
sc = r["sin_clasificar"]
print(sc["decision"].value_counts().to_string())
sc[sc["decision"].eq("propuesta")]["familia_pred"].value_counts()
""")

md("""
## Datos usados en cada entrenamiento
Cuántas muestras entran en cada modelo de este cuaderno (familia y un modelo de subtipo por familia), cuáles se sacan y por qué,
y cómo se reparte lo que queda. No hay conjunto de validación propio: los hiperparámetros son fijos. La evaluación es la validación
cruzada agrupada por envío; «entrenamiento» y «prueba» son el tamaño medio de cada pliegue. Las muestras de calibración, las que
declaran algo que no coincide y las «sin clasificar» no entrenan: se predicen con el modelo final.
""")

code(r"""
import inspect
import datos_entrenamiento as DE
from datos_clase41 import TOL_HUM_SOL
from sklearn.model_selection import StratifiedGroupKFold

miles = lambda n: f"{int(n):,}".replace(",", ".")
N_PLIEGUES = inspect.signature(PT.validar).parameters["n_folds"].default
CX = PT.columnas_x(t)


def pliegues(objetivo, filtro):
    # Mismas filas y mismos pliegues que PT.validar: devuelve las filas y el tamaño medio de entrenamiento y de prueba.
    base = t[t["entrena"] & filtro & t[objetivo].notna()]
    base = base[base[objetivo].isin(PT._clases_validas(base[objetivo]))]
    cv = StratifiedGroupKFold(n_splits=N_PLIEGUES, shuffle=True, random_state=PT.SEMILLA)
    tam = [(len(a), len(b)) for a, b in cv.split(base[CX], base[objetivo].astype(str), base["Envío"].astype(str))]
    return base, round(np.mean([a for a, _ in tam])), round(np.mean([b for _, b in tam]))


PARTICION = f"validación cruzada de {N_PLIEGUES} pliegues agrupada por envío"
fuera = [("de calibración", t["calibracion"]),
         ("de ensayo, pool u otro tipo que no es producción", ~t["desc_tipo_muestra"].isin(["produccion"])),
         ("sin clasificar o sin familia", t["desc_familia"].eq(PT.SIN) | t["desc_familia"].isna()),
         (f"con humedad + sólidos fuera de 100 ± {TOL_HUM_SOL:g}", t["excluida_hum_sol"]),
         ("con lo declarado distinto de lo medido", t["declarado_no_coincide"]),
         ("sin ningún análisis", t["n_analisis"].eq(0))]
quedan, partes = pd.Series(True, index=t.index), []
for nombre, cond in fuera:
    partes.append(f"{nombre}: {miles((quedan & cond).sum())}")
    quedan &= ~cond
assert quedan.equals(t["entrena"])

base_f, tr_f, te_f = pliegues("desc_familia", pd.Series(True, index=t.index))
assert set(base_f.index) == set(r["oof_familia"].index)
partes.append(f"familias con menos de {PT.MIN_CLASE} muestras: {miles(quedan.sum() - len(base_f))}")
filas = [DE.fila(
    parte="V", modelo="familia: HistGradientBoosting", datos="muestras de Senasa y MQ nuevas",
    disponibles=len(t), usadas=len(base_f), motivo="; ".join(partes), entrenamiento=tr_f, prueba=te_f, particion=PARTICION,
    nota=(f"entrenamiento y prueba son el promedio por pliegue; el modelo final se reajusta con las {miles(len(base_f))} filas; "
          f"con probabilidad menor que {str(PT.UMBRAL_PROB).replace(".", ",")} la muestra queda sin decidir; sin entrenar se predicen "
          f"{miles(len(r['calibracion']))} de calibración, {miles(len(r['no_coinciden']))} con declarado que no coincide y "
          f"{miles(len(r['sin_clasificar']))} sin clasificar con análisis"))]

for fam, oof in r["oof_subtipo"].items():
    de_fam = t["desc_familia"].eq(fam)
    disp = (t["entrena"] & de_fam).sum()
    base_s, tr_s, te_s = pliegues("subtipo", de_fam)
    assert set(base_s.index) == set(oof.index)
    sin_sub = (t["entrena"] & de_fam & t["subtipo"].isna()).sum()
    filas.append(DE.fila(
        parte="V", modelo=f"subtipo · {fam}: HistGradientBoosting", datos=f"muestras de entrenamiento de {fam}",
        disponibles=disp, usadas=len(base_s),
        motivo=(f"sin subtipo declarado o fuera de las clases del mapa: {miles(sin_sub)}; "
                f"subtipos con menos de {PT.MIN_CLASE} muestras: {miles(disp - sin_sub - len(base_s))}"),
        entrenamiento=tr_s, prueba=te_s, particion=PARTICION,
        nota=f"entrenamiento y prueba son el promedio por pliegue; el modelo final se reajusta con las {miles(len(base_s))} filas"))

display(DE.guardar(filas, "prediccion_tipo"))
""")

md("""
## Lectura
- Con las mediciones solas la **familia** se acierta en ~8 de 10 muestras; con probabilidad ≥ 0,80 se decide ~2/3 de
  las muestras con ~95 % de exactitud. Las probabilidades entre 0,8 y 0,95 son algo optimistas; desde 0,95 son fiables.
- Se confunden familias con composición parecida y pocos análisis: base nutricional con agua / hisopado (se les pide
  poco), concentrado proteico con leche en polvo y suero, queso con helado / manteca.
- Los **subtipos** salen muy bien en leche en polvo (entera / descremada), concentrado proteico, manteca / crema; bien en
  queso (consistencia), DDL y leche fluida; peor en suero / permeado y en yogur de Senasa.
- En los quesos cuya consistencia declarada no coincide, el modelo sigue casi siempre a la **humedad medida**: es
  evidencia de que la consistencia declarada estaba mal, no la medición.
- Las de **calibración** se aciertan peor (son matrices de referencia, a veces con un solo análisis); por eso no deben
  entrar al entrenamiento.
- Archivo: `resultados/excel/prediccion_tipo.xlsx`; modelo: `resultados/modelos/prediccion_tipo.joblib`
  (`PT.predecir(joblib.load(...), X)` con las columnas de `PT._variables`).
""")

code(r"""
assert not (t["entrena"] & t["calibracion"]).any() and not (t["entrena"] & t["declarado_no_coincide"]).any()
assert (PT.R.EXCEL / "prediccion_tipo.xlsx").exists() and PT.SALIDA_MODELO.exists()
for k in ["calibracion", "no_coinciden", "sin_clasificar"]:
    assert "Descripción" not in r[k].columns
    assert r[k]["Cliente"].dropna().astype(str).str.match(r"^IND-\d+$").all()
print("Verificación: todo en orden.")
""")

c.extend(celdas_app.celdas())

nb = nbf.v4.new_notebook(); nb["cells"] = c
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
if __name__ == "__main__":
    nbf.write(nb, SALIDA); print(f"Escrito: {SALIDA} ({len(c)} celdas)")
