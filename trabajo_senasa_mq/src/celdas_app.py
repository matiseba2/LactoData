"""Celdas para abrir una app de Streamlit desde un notebook, como en las clases (%%writefile + streamlit run).

La celda que lanza la app bloquea el kernel mientras la app está abierta, así que va detrás de ABRIR_APP = False:
el notebook se puede ejecutar entero con nbconvert y, para usarla, se cambia a True y se ejecuta esa celda.
Por defecto es app_prediccion.py (Partes V y VI); la Parte VII pasa app_arbol.py.
"""
from pathlib import Path

import nbformat as nbf

APP = Path(__file__).resolve().parents[1] / "app_prediccion.py"
QUE_HACE = ("(dos pestañas: días de espera para deslactosar una UAT y tipo de\n"
            "muestra según lo medido)")


def celdas(app=APP, que_hace=QUE_HACE):
    md = nbf.v4.new_markdown_cell
    code = nbf.v4.new_code_cell
    nombre = Path(app).name
    return [
        md(f"""## La aplicación, desde el cuaderno
La celda siguiente escribe `{nombre}` {que_hace}. La de después la abre en http://localhost:8501.

Streamlit no corre adentro de una celda: es un programa aparte, y mientras está abierto el kernel queda ocupado.
Para abrirla cambiá `ABRIR_APP = True` y ejecutá la celda; para cerrarla, interrumpí el kernel (■). También se abre
desde una terminal, en la carpeta `trabajo_senasa_mq`: `../.venv/bin/streamlit run {nombre}`."""),
        code(f"%%writefile {nombre}\n" + Path(app).read_text(encoding="utf-8").rstrip("\n")),
        code(f"""import sys
ABRIR_APP = False   # True para abrirla (bloquea el kernel hasta interrumpirlo)
if ABRIR_APP:
    print("Abrir http://localhost:8501 en el navegador. Para cortarla: interrumpir el kernel (■).")
    !"{{sys.executable}}" -m streamlit run {nombre} --server.headless true
else:
    print("App escrita en {nombre}. Para abrirla: ABRIR_APP = True y volver a ejecutar esta celda.")"""),
    ]
