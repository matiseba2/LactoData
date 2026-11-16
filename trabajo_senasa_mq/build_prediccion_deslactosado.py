"""Arma prediccion_deslactosado.ipynb (cuánto esperar para que la leche UAT quede deslactosada).

    cd trabajo_senasa_mq
    ../.venv/bin/python build_prediccion_deslactosado.py
    ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace prediccion_deslactosado.ipynb

Al final escribe y abre app_prediccion.py (celdas de src/celdas_app.py). El código está en src/deslactosado_uat.py; el notebook solo lo llama y muestra.
"""
import sys
from pathlib import Path

import nbformat as nbf

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import celdas_app  # noqa: E402

SALIDA = Path(__file__).resolve().parent / "prediccion_deslactosado.ipynb"
c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: c.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# ¿Cuánto esperar para que la leche UAT quede deslactosada?

**Pregunta.** Con la lactosa del análisis original y del remuestreo, los días entre los dos y la temperatura media
de Rafaela (5 días previos a la recepción y el período entre análisis), ¿se puede predecir cuántos días esperar,
según la temperatura media de la semana, para que la lactosa baje de **0,1 g/100 g**?

**Datos.** Solo Senasa (MQ no mide lactosa): leche fluida UAT con lactosa medida. Lote remuestreado = mismo lote en
más de una fecha de recepción. «< 0,03» se trata como censurado (Tobit), no como 0.

**Modelo.** Primer orden: ln L₁ = ln L₀ + c − k(T)·días, con k(T) = exp(a + b·(T − 20)). El término c recoge la
baja que aparece ya al primer día: el lote se remuestrea porque el primer valor dio alto (selección), y esa baja no
es tiempo de espera. La espera se calcula solo con k(T): días = ln(L₀ / 0,1) / k(T). Se compara con el modelo sin
desfase, sin temperatura y «sin tiempo» (la baja no depende de los días).
""")

code(r"""
import sys
from pathlib import Path
SRC = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src" / "rutas.py").exists()) / "src"
sys.path.insert(0, str(SRC))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import deslactosado_uat as D

r = D.armar(escribir=True, n_boot=1000)
p, th = r["pares"], r["ajustes"]["desfase"]["theta"]
r["resumen"]
""")

md("## Validación dejando un lote afuera\nError en ln(lactosa) del remuestreo (sin los censurados) y acierto al decir si queda bajo 0,1.")
code('r["metricas"].round(3)')

md("## Todas las UAT: lactosa del análisis según la temperatura de los 5 días previos\nSin remuestreo: muestra que la temperatura sí importa en el conjunto, aunque en los 86 pares no alcance a medirse.")
code('r["transversal"]')

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
y = np.log(p["l1"] / p["l0"])
sc = ax[0].scatter(p["dias"] + np.random.default_rng(D.SEMILLA).uniform(-.15, .15, len(p)), y, c=p["t_exp"], cmap="coolwarm",
                   marker="o", edgecolor=np.where(p["ld1"], "k", "none"))
d = np.linspace(0, 8, 50)
for T in (12, 20, 26):
    ax[0].plot(d, th[2] - D.k_de(th, T) * d, label=f"{T} °C")
ax[0].axhline(0, color="grey", lw=.5)
ax[0].set(xlabel="días entre análisis", ylabel="ln(L remuestreo / L original)", title="Remuestreos (borde negro: < LD)")
ax[0].legend(); fig.colorbar(sc, ax=ax[0], label="T media del período (°C)")
T = np.linspace(8, 30, 50)
b = r["boot"]
for l0 in (0.3, 0.5, 0.7, 1.0):
    dd = D.dias_hasta(l0, T[:, None], b.T)
    ax[1].plot(T, D.dias_hasta(l0, T, th), label=f"L₀ = {l0}")
    ax[1].fill_between(T, *np.percentile(dd, [2.5, 97.5], axis=1), alpha=.12)
ax[1].set(xlabel="temperatura media de la semana (°C)", ylabel="días hasta 0,1 g/100 g", title="Espera predicha (IC 95 % bootstrap)")
ax[1].legend(); plt.tight_layout(); plt.show()
""")

md("## Tabla de espera\nDías hasta 0,1 g/100 g según la lactosa del primer análisis y la temperatura media de la semana.")
code('r["tabla"].pivot(index="lactosa_inicial", columns="t_media_semana", values="dias")')
code(r"""
# función para usar: días de espera para una lactosa medida y la temperatura media prevista de la semana
D.dias_hasta(0.6, 18, th).round(1)
""")

md("""
## Datos usados en cada entrenamiento
Cuántos datos entran en cada ajuste de este cuaderno. La unidad es el lote remuestreado (un par: primera y última recepción),
no la muestra. No hay conjunto de prueba aparte: con tan pocos lotes se ajusta con todos y la validación deja un lote afuera por vez.
""")

code(r"""
import datos_entrenamiento as DE

miles = lambda n: f"{int(n):,}".replace(",", ".")
muestras_uat = r["muestras"]
n_lotes = int((muestras_uat.dropna(subset=["lote_id"]).groupby("lote_id")["fecha"].nunique() > 1).sum())
n_boot, n_loo = len(r["boot"]), len(r["loo"])
n_cens = int(p["ld1"].sum())
un_lote = p.iloc[:1]
base_pares = dict(parte="VI", datos="lotes UAT remuestreados, primera y última recepción", disponibles=n_lotes, usadas=len(p),
                  motivo=f"lactosa del primer análisis bajo el límite de detección (sin punto de partida): {miles(n_lotes - len(p))}")

filas = [
    DE.fila(modelo="desfase: ln L1 = ln L0 + c − k(T)·días (con censura)", **base_pares,
            particion=f"sin partición: todos los lotes; intervalos por bootstrap de lotes ({miles(n_boot)} remuestreos)",
            nota=(f"{miles(n_cens)} de los {miles(len(p))} remuestreos están bajo el límite de detección (censurados); "
                  f"los lotes salen de {miles(len(muestras_uat))} muestras UAT de Senasa con lactosa medida (MQ no mide lactosa)")),
    DE.fila(modelo="comparación: sin desfase, sin temperatura y sin tiempo", **base_pares,
            particion="sin partición: mismos lotes que el modelo principal; se comparan por AIC"),
    DE.fila(modelo="validación dejando un lote afuera (los cuatro modelos)", **base_pares,
            entrenamiento=len(p.drop(index=un_lote.index)), prueba=len(un_lote),
            particion=f"un lote afuera por vez ({miles(n_loo)} ajustes)",
            nota="error en ln(lactosa) del remuestreo sin los censurados; entrenamiento y prueba son los de cada ajuste"),
    DE.fila(parte="VI", modelo="transversal: lactosa contra la temperatura de los 5 días previos (no es un modelo)",
            datos="muestras UAT de Senasa con lactosa medida", disponibles=len(muestras_uat), usadas=muestras_uat["t5"].notna().sum(),
            motivo=f"sin temperatura de los 5 días previos: {miles(muestras_uat['t5'].isna().sum())}",
            particion="sin partición", nota="Spearman y mediana por rango de temperatura; sin remuestreo"),
]
display(DE.guardar(filas, "prediccion_deslactosado"))
""")

md("""
## Lectura
- **El tiempo sí explica la baja**: cuanto más días entre análisis, más baja la lactosa (Spearman en el resumen) y el
  modelo con desfase es el mejor por AIC.
- **La temperatura, dentro de los pares, no alcanza a medirse**: el IC de b incluye 0. Los 86 pares son pocos, casi
  todos de 2020 a 2022, con 1 a 8 días y temperaturas de exposición parecidas dentro de cada estación.
- En el conjunto de las UAT la lactosa baja claramente con la temperatura previa (tabla transversal), así que el
  efecto existe; lo que falta es poder separarlo en la cinética.
- **Factibilidad**: la tabla sirve como orden de magnitud (≈ 5 a 9 días para una lactosa de 0,5 g/100 g, entre 30 y 10 °C), con
  intervalos anchos. Para un predictor operativo hacen falta remuestreos programados (por ejemplo a 0, 3, 7 y 14 días)
  y la fecha de elaboración, que hoy figura en 1 de las muestras UAT (3 en toda la leche fluida).
""")

code(r"""
assert len(p) == 86 and p["lote_id"].is_unique and (p["dias"] > 0).all()
assert r["muestras"].columns.tolist()[:3] == ["Identificador", "Cliente", "lote_id"]
assert r["muestras"]["Cliente"].dropna().astype(str).str.match(r"^IND-\d+$").all()
assert np.isfinite(r["boot"]).all()
print("Verificación: todo en orden.")
""")

c.extend(celdas_app.celdas())

nb = nbf.v4.new_notebook(); nb["cells"] = c
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
if __name__ == "__main__":
    nbf.write(nb, SALIDA); print(f"Escrito: {SALIDA} ({len(c)} celdas)")
