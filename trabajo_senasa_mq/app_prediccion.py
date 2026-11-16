# Trabajo Senasa + MQ · aplicación Streamlit con los dos modelos predictivos
# 1) Cuántos días esperar para que una leche UAT baje de 0,1 g/100 g de lactosa (src/deslactosado_uat.py).
# 2) Tipo de muestra (familia y subtipo) a partir de lo medido (src/prediccion_tipo.py).
# Los modelos los guardan los notebooks prediccion_deslactosado.ipynb y prediccion_tipo.ipynb en resultados/modelos/.
# Para abrirla, desde la carpeta trabajo_senasa_mq:  ../.venv/bin/streamlit run app_prediccion.py
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI / "src"))
import deslactosado_uat as D  # noqa: E402
import prediccion_tipo as PT  # noqa: E402

st.set_page_config(page_title="Predicciones Senasa + MQ", layout="centered")
st.title("Predicciones Senasa + MQ")
st.caption("Dos modelos ajustados en los notebooks del trabajo. Cargá los parámetros y la app devuelve el resultado.")


@st.cache_resource
def cargar(ruta):
    return joblib.load(ruta)


faltan = [p for p in (D.SALIDA_MODELO, PT.SALIDA_MODELO) if not p.exists()]
if faltan:
    st.error("Faltan modelos: " + ", ".join(p.name for p in faltan) + ". Ejecutá los notebooks prediccion_*.ipynb.")
    st.stop()

tab1, tab2 = st.tabs(["Deslactosado UAT", "Tipo de muestra"])

# ------------------------------------------------------------------ 1. deslactosado
with tab1:
    m = cargar(D.SALIDA_MODELO)
    th, boot, umbral = m["theta"], m["boot"], m["umbral"]
    st.subheader("¿Cuántos días esperar?")
    st.write(f"Modelo de primer orden ajustado con {m['n_lotes']} lotes UAT remuestreados de Senasa "
             f"({m['rango_dias'][0]} a {m['rango_dias'][1]} días entre análisis). Umbral de deslactosado: "
             f"**{umbral:.1f} g/100 g**.")
    c1, c2 = st.columns(2)
    l0 = c1.number_input("Lactosa medida (g/100 g)", 0.01, 3.0, 0.60, step=0.01, format="%.2f")
    t_sem = c2.slider("Temperatura media de la semana (°C)", 5.0, 35.0, 18.0, step=0.5)

    if l0 <= umbral:
        st.success(f"Ya está en {umbral:.1f} g/100 g o menos: no hace falta esperar.")
    else:
        dias = float(D.dias_hasta(l0, t_sem, th))
        dd = D.dias_hasta(l0, t_sem, boot.T)
        inf, sup = np.percentile(dd, [2.5, 97.5])
        k1, k2 = st.columns(2)
        k1.metric("Días de espera estimados", f"{dias:.1f}")
        k2.metric("Intervalo 95 % (bootstrap)", f"{inf:.1f} a {sup:.1f}")
        st.write(f"Tasa a esa temperatura: k = {float(D.k_de(th, t_sem)):.3f} por día "
                 f"(la lactosa baja ~{100 * (1 - np.exp(-float(D.k_de(th, t_sem)))):.0f} % por día).")
        if dias > m["rango_dias"][1]:
            st.warning(f"La espera supera los {m['rango_dias'][1]} días observados entre análisis: es una extrapolación.")
        lo, hi = m["rango_t"]
        if not lo - 2 <= t_sem <= hi + 2:
            st.warning(f"Temperatura fuera del rango observado ({lo:.0f} a {hi:.0f} °C): es una extrapolación.")

    st.info("El efecto de la temperatura dentro de los remuestreos no es significativo (el IC incluye cero): "
            "tomá el resultado como orden de magnitud, no como garantía.")

    T = np.linspace(5, 35, 61)
    fig, ax = plt.subplots(figsize=(8, 4))
    for v, col in [(0.3, "#9ecae1"), (0.5, "#4292c6"), (1.0, "#08519c")]:
        ax.plot(T, D.dias_hasta(v, T, th), color=col, lw=1, label=f"L₀ = {v}")
    if l0 > umbral:
        dT = D.dias_hasta(l0, T[:, None], boot.T)
        ax.fill_between(T, *np.percentile(dT, [2.5, 97.5], axis=1), color="#c0392b", alpha=0.15)
        ax.plot(T, D.dias_hasta(l0, T, th), color="#c0392b", lw=2, label=f"esta muestra (L₀ = {l0:.2f})")
        ax.scatter([t_sem], [D.dias_hasta(l0, t_sem, th)], s=200, marker="*", color="#c0392b", zorder=5)
    ax.axvspan(*m["rango_t"], color="grey", alpha=0.08, label="temperaturas observadas")
    ax.set(xlabel="temperatura media de la semana (°C)", ylabel=f"días hasta {umbral:.1f} g/100 g",
           title="Espera predicha según la temperatura")
    ax.legend(fontsize=8)
    st.pyplot(fig)

# ------------------------------------------------------------------ 2. tipo de muestra
with tab2:
    paq = cargar(PT.SALIDA_MODELO)
    st.subheader("¿Qué tipo de muestra es?")
    st.write("Cargá los análisis que se midieron; los que no, dejalos sin marcar (el modelo también usa *qué* "
             "análisis se pidieron). Con probabilidad menor a "
             f"{paq['umbral']:.2f} el resultado queda como «sin decidir».")

    CAMPOS = [  # (variable, etiqueta, valor por defecto, medida por defecto)
        ("humedad", "Humedad (g/100 g)", 3.5, True),
        ("grasa", "Materia grasa (g/100 g)", 26.0, True),
        ("proteinas", "Proteínas (g/100 g)", 25.0, True),
        ("cenizas", "Cenizas (g/100 g)", 6.0, False),
        ("grasa_es", "Materia grasa en extracto seco (g/100 g)", 45.0, False),
        ("cloruros", "Cloruros (NaCl g/100 g)", 1.5, False),
        ("acidez_naoh", "Acidez (ml NaOH 0,1 N / 10 g SNG)", 15.0, False),
        ("acidez_lactico", "Acidez (% ácido láctico)", 0.15, False),
        ("lactosa", "Lactosa (g/100 g)", 38.0, False),
        ("galactosa", "D-galactosa (g/100 g)", 0.1, False),
        ("azucares", "Azúcares totales (g/100 g)", 50.0, False),
        ("ph", "pH", 6.6, False),
        ("aw", "Actividad de agua", 0.85, False),
        ("almidon", "Almidón total (g/100 g)", 1.0, False),
        ("sodio", "Sodio (mg/100 g)", 400.0, False),
        ("calcio", "Calcio (mg/100 g)", 900.0, False),
        ("brix", "Grado Brix (°Bx)", 70.0, False),
        ("solubilidad", "Solubilidad (%)", 99.0, False),
    ]
    valores = {}
    c1, c2 = st.columns(2)
    for i, (var, etiqueta, defecto, medido) in enumerate(CAMPOS):
        col = c1 if i % 2 == 0 else c2
        if col.checkbox(etiqueta, value=medido, key=f"chk_{var}"):
            valores[var] = col.number_input(etiqueta, value=float(defecto), key=f"val_{var}", label_visibility="collapsed")

    if not valores:
        st.warning("Marcá al menos un análisis.")
        st.stop()

    fila = {v: np.nan for v in PT.VARIABLES}
    fila.update(valores)
    n_medidos = len(valores)
    if np.isfinite(fila["humedad"]):                         # igual que en el entrenamiento
        fila["solidos"] = 100 - fila["humedad"]
    fila["n_analisis"] = n_medidos
    for v in PT.VARIABLES:
        fila[f"pide_{v}"] = int(np.isfinite(fila[v]))
    X = pd.DataFrame([fila])[paq["columnas"]]
    if np.isfinite(fila["humedad"]) and not 0 <= fila["humedad"] <= 100:
        st.error("La humedad tiene que estar entre 0 y 100.")
        st.stop()

    res = PT.predecir(paq, X).iloc[0]
    k1, k2 = st.columns(2)
    k1.metric("Familia", res["familia_pred"], f"prob. {res['prob_familia']:.2f}", delta_color="off")
    if pd.notna(res["subtipo_pred"]):
        k2.metric("Subtipo", res["subtipo_pred"], f"prob. {res['prob_subtipo']:.2f}", delta_color="off")
    if res["decision"] == "sin decidir":
        st.warning("Probabilidad baja: el modelo no decide. Mirá las alternativas abajo o agregá análisis.")
    else:
        st.success("Propuesta del modelo con probabilidad suficiente.")

    mf = paq["modelos"]["familia"]
    pf = pd.Series(mf.predict_proba(X)[0], index=mf.classes_).sort_values(ascending=True).tail(6)
    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.barh(pf.index, pf.values, color=["#c0392b" if i == res["familia_pred"] else "#4292c6" for i in pf.index])
    ax2.axvline(paq["umbral"], color="black", lw=0.8, ls="--", label=f"umbral {paq['umbral']:.2f}")
    ax2.set(xlim=(0, 1), xlabel="probabilidad", title="Familias más probables")
    ax2.legend(fontsize=8, loc="lower right")
    st.pyplot(fig2)

    if res["familia_pred"] in paq["modelos"]:
        ms = paq["modelos"][res["familia_pred"]]
        ps = pd.Series(ms.predict_proba(X)[0], index=ms.classes_).sort_values(ascending=False)
        st.write("Subtipos dentro de la familia: " + " · ".join(f"{k} {v:.2f}" for k, v in ps.items()))

    st.caption("Validación agrupada por envío: familia 81 % de acierto en general, 95 % cuando la probabilidad es ≥ 0,80.")
    with st.expander("Ver la fila que recibe el modelo"):
        st.dataframe(X.T.rename(columns={0: "valor"}))
