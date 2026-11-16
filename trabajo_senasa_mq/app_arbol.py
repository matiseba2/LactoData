# Trabajo Senasa + MQ · ¿qué producto es esta muestra? · aplicación Streamlit (Parte VII)
# Carga la cascada de árboles de decisión de V3 (familia → subtipo, src/arbol_decision.py) y clasifica una muestra
# por su composición. El modelo lo guarda arbol_decision.ipynb en resultados/modelos/arbol_decision.joblib.
# Para abrirla, desde la carpeta trabajo_senasa_mq:  ../.venv/bin/streamlit run app_arbol.py
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI / "src"))
import arbol_decision as AD  # noqa: E402

AZUL, GRIS, ROJO = "#0055a5", "#6b7280", "#c0392b"
NOMBRES = {"humedad": "Humedad (%)", "grasa": "Grasa (%)", "proteinas": "Proteínas (%)", "cenizas": "Cenizas (%)",
           "solidos_totales": "Sólidos totales (%)", "lactosa": "Lactosa (%)", "galactosa": "Galactosa (%)"}
coma = AD.coma

st.set_page_config(page_title="¿Qué producto es esta muestra?", layout="centered")
st.title("¿Qué producto es esta muestra?")
st.caption("Árbol de decisión en cascada sobre las muestras de Senasa y MQ: un árbol predice la familia y, "
           "dentro de la familia, otro árbol predice el subtipo. Solo con los análisis de laboratorio.")


@st.cache_resource
def cargar(ruta):
    return joblib.load(ruta)


if not AD.SALIDA_MODELO.exists():
    st.error(f"Falta {AD.SALIDA_MODELO.name}. Ejecutá antes arbol_decision.ipynb, que lo genera.")
    st.stop()
modelos = cargar(AD.SALIDA_MODELO)
PREDICTORAS = modelos["predictoras"]
arbol_familia = modelos["familia"]
SC = modelos["subtipo_sin_clasificar"]
UMBRAL = float(modelos["umbral_sin_clasificar"])

# ---------- 1 · Los análisis de la muestra ----------
st.subheader("Análisis de la muestra")
st.write("Cada análisis tiene su casilla «no medido»: si se marca, el valor queda vacío (NaN) y el árbol lo manda "
         "hacia el lado que aprendió para los faltantes. No se imputa nada.")
valores = {}
for col in PREDICTORAS:
    c1, c2 = st.columns([3, 1])
    no_medido = c2.checkbox("no medido", key=f"nm_{col}", value=pd.isna(modelos["medianas"][col]))
    inicial = modelos["medianas"][col]
    v = c1.number_input(NOMBRES.get(col, col), min_value=0.0, max_value=100.0, step=0.1,
                        value=0.0 if pd.isna(inicial) else float(round(inicial, 2)), disabled=no_medido, key=f"v_{col}")
    valores[col] = np.nan if no_medido else v
fila = pd.DataFrame([valores])[PREDICTORAS]
if fila.isna().all(axis=1).iloc[0]:
    st.warning("No hay ningún análisis medido: el árbol igual responde, pero solo con la regla para faltantes.")


# ---------- 2 · La predicción de los dos niveles ----------
def barras(probs, titulo, elegido):
    probs = probs.sort_values(ascending=False).head(8)
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.45 * len(probs)))
    ax.barh(probs.index, probs.values, color=[ROJO if k == elegido else GRIS if k == SC else AZUL for k in probs.index])
    for k, p in enumerate(probs.values):
        ax.text(min(p + 0.01, 0.9), k, coma(p), va="center")
    ax.set_xlim(0, 1); ax.set_xlabel("probabilidad"); ax.invert_yaxis(); ax.set_title(titulo)
    plt.tight_layout()
    return fig


st.subheader("Predicción")
p_familia = pd.Series(arbol_familia.predict_proba(fila)[0], index=arbol_familia.classes_)
familia = arbol_familia.predict(fila)[0]
entrada = modelos["subtipo"][familia]
if entrada["tipo"] == "árbol":
    p_subtipo = pd.Series(entrada["modelo"].predict_proba(fila)[0], index=entrada["modelo"].classes_)
elif entrada["tipo"] == "constante":
    p_subtipo = pd.Series(entrada["probabilidades"])
else:
    p_subtipo = None
hoja_dice = None if p_subtipo is None else p_subtipo.idxmax()          # lo que dice la hoja (puede ser SIN CLASIFICAR)
subtipo = AD.predecir_subtipo(entrada, fila, UMBRAL)[0][0]             # con el umbral: la predicción que se muestra
p_sc = float(p_subtipo[SC]) if p_subtipo is not None and SC in p_subtipo.index else None
reales = p_subtipo.drop(SC) if p_sc is not None else None

c1, c2 = st.columns(2)
c1.metric("Familia", familia, f"{coma(p_familia[familia] * 100, 1)} %", delta_color="off")
if subtipo is None:
    c2.metric("Subtipo", "—")
    c2.info(f"**La familia no tiene subtipos** ({entrada.get('motivo', 'la familia no tiene subtipos')}): "
            "la cascada termina en la familia.")
elif subtipo == SC:
    c2.metric("Subtipo", "sin clasificar", f"{coma(p_subtipo[subtipo] * 100, 1)} %", delta_color="off")
    c2.warning("**Sin clasificar: el modelo no asigna subtipo.** La muestra se parece más a las de la familia "
               "que se cargaron sin subtipo que a cualquier subtipo real.")
    if len(reales) and reales.max() > 0:
        c2.caption(f"Subtipo real más probable: {reales.idxmax()} ({coma(reales.max() * 100, 1)} %).")
else:
    c2.metric("Subtipo", subtipo, f"{coma(p_subtipo[subtipo] * 100, 1)} %", delta_color="off")
    if p_sc is not None:
        c2.caption(f"Probabilidad de SIN CLASIFICAR: {coma(p_sc * 100, 1)} % (por debajo del umbral, así que se muestra el subtipo real).")
    if entrada["tipo"] == "constante":
        c2.caption("Para esta familia no hay árbol de subtipo: se predice el más común.")
    if subtipo == "OTRO":
        c2.caption(f"«OTRO» agrupa los subtipos con menos de {modelos['minimo_subtipo']} muestras en la familia.")
if p_sc is not None:
    st.caption(f"Umbral de «sin clasificar»: {coma(UMBRAL * 100, 0)} %. Solo se muestra «sin clasificar» si la probabilidad "
               f"de SIN CLASIFICAR llega a ese valor; si no, se muestra el subtipo real más probable.")
st.pyplot(barras(p_familia, "Primer nivel: probabilidad de cada familia (rojo: la elegida)", familia))
if p_subtipo is not None:
    st.pyplot(barras(p_subtipo, f"Segundo nivel ({familia}): probabilidad de cada subtipo", subtipo))
    if SC in p_subtipo.index:
        st.caption(f"«{SC}» (gris si no es la elegida) no es un subtipo: reúne las muestras de la familia "
                   "que llegaron sin subtipo.")
else:
    st.write(f"Segundo nivel: «{familia}» no tiene subtipos, no hay probabilidades de subtipo.")
st.write("Los árboles responden en escalones: la probabilidad no cambia hasta que un valor cruza el corte de alguna "
         "pregunta, y entonces salta.")


# ---------- 3 · Por qué: la cadena de preguntas de cada árbol ----------
def explicar(arbol, que):
    pasos, hoja, cuentas = AD.camino(arbol, fila)
    st.markdown("\n".join(f"1. {p}" for p in pasos))
    st.write(f"Llega a la hoja {hoja}: de las **{cuentas.sum()}** muestras de entrenamiento que respondieron igual, "
             + ", ".join(f"**{v}** eran {k}" for k, v in cuentas.head(5).items())
             + f". {len(pasos)} preguntas, de {arbol.get_depth()} posibles en el árbol de {que}.")


st.subheader("Por qué")
pestanas = st.tabs(["árbol de familia", "árbol de subtipo"])
with pestanas[0]:
    explicar(arbol_familia, "familia")
with pestanas[1]:
    if entrada["tipo"] == "árbol":
        explicar(entrada["modelo"], f"subtipo de {familia}")
        if hoja_dice == SC and subtipo == SC:
            st.write(f"Esa hoja asigna «{SC}» y su probabilidad ({coma(p_sc * 100, 1)} %) llega al umbral "
                     f"({coma(UMBRAL * 100, 0)} %): el modelo no asigna subtipo.")
        elif hoja_dice == SC:
            st.write(f"Ojo: esa hoja asigna «{SC}», pero su probabilidad ({coma(p_sc * 100, 1)} %) no llega al "
                     f"umbral ({coma(UMBRAL * 100, 0)} %). Por eso la predicción mostrada es el subtipo real más probable, "
                     f"**{subtipo}**. El camino de preguntas es el del árbol; la decisión final la toma el umbral.")
    elif entrada["tipo"] == "constante":
        st.write(f"No hay árbol: en «{familia}» se predice siempre **{subtipo}**. Proporciones del entrenamiento: "
                 + ", ".join(f"{k} {coma(v * 100, 1)} %" for k, v in p_subtipo.items()) + ".")
    else:
        st.write(f"«{familia}»: {entrada.get('motivo', 'la familia no tiene subtipos')}. No hay segundo nivel.")

with st.expander("Ver la fila que reciben los modelos"):
    st.dataframe(fila)
