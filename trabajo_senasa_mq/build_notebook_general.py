"""Arma `notebook_general.ipynb`: el cuaderno de entrega para la cátedra, con todo el trabajo en un solo archivo.

No tiene código propio de análisis: toma, en orden, las celdas de los ocho cuadernos que ya se generan con su build
(clustering, comparación numérico / categórico, regresión del ciclo, regresión logística, predicción del tipo de muestra,
predicción del deslactosado, árbol de decisión en cascada y leche cruda contra la temperatura), les baja un nivel a los
títulos para que cada cuaderno sea una «Parte», y agrega portada, índice y conclusiones generales. Después se ejecuta
entero en un solo kernel, así que también prueba que las ocho partes corren juntas.

Uso, desde trabajo_senasa_mq/ (antes, regenerar los ocho cuadernos con sus build_*.py):
    PYTHONIOENCODING=utf-8 ../.venv/Scripts/python.exe build_notebook_general.py
    ../.venv/Scripts/jupyter.exe nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=7200 notebook_general.ipynb
"""
import re
from pathlib import Path

import nbformat as nbf

AQUI = Path(__file__).resolve().parent
SALIDA = AQUI / "notebook_general.ipynb"

PARTES = [
    ("I", "trabajo_senasa_mq.ipynb", "Agrupamiento de las muestras de Senasa y MQ",
     "Revisión de los datos, k-medias con los cuatro análisis (humedad, grasa, proteínas y cenizas), segundo nivel por "
     "familia (dulce de leche, polvos, queso), categorías y reglas de asociación."),
    ("II", "comparacion_numerico_categorico.ipynb", "Clustering con números contra clustering con categorías",
     "Las mismas muestras de la Parte I, agrupadas con k-modas sobre los análisis pasados a categorías, y comparadas "
     "con el k-medias."),
    ("III", "regresion_ciclo_leche_cruda.ipynb", "Regresión lineal y polinómica: el ciclo anual de la leche cruda",
     "Proteína y grasa de la leche cruda en función del día del año y de la temperatura; elección del grado por "
     "validación temporal y comparación con otros algoritmos."),
    ("IV", "logistica_calidad_leche_cruda.ipynb", "Regresión logística: calidad de la leche cruda",
     "Probabilidad de que una muestra supere el umbral de células somáticas o de bacterias; curva ROC, calibración, "
     "matriz de confusión, lactosa y comparación con otros algoritmos."),
    ("V", "prediccion_tipo.ipynb", "¿Se puede predecir el tipo de una muestra por lo que se le midió?",
     "Senasa y MQ nuevas: con los resultados numéricos y los análisis pedidos, HistGradientBoosting en dos niveles (familia y "
     "subtipo), validación cruzada agrupada por envío, umbral de probabilidad, muestras de calibración y propuesta para las "
     "«sin clasificar»."),
    ("VI", "prediccion_deslactosado.ipynb", "¿Cuánto esperar para que la leche UAT quede deslactosada?",
     "Solo Senasa: modelo de primer orden con censura (Tobit) para la baja de la lactosa entre dos análisis del mismo lote, "
     "con la temperatura media de Rafaela; validación dejando un lote afuera, intervalos por bootstrap y tabla de espera."),
    ("VII", "arbol_decision.ipynb", "Árbol de decisión en cascada: familia y subtipo",
     "El trabajo de otro integrante del grupo: con solo la composición, un árbol de decisión predice la familia y otro, "
     "dentro de la familia, el subtipo. Tres versiones (solo arc, con la clase «sin clasificar», arc + MQ con umbral), "
     "búsqueda en cuadrícula con validación cruzada, ROC uno contra el resto y una aplicación."),
    ("VIII", "temperatura_leche_cruda.ipynb", "Leche cruda y temperatura de Rafaela",
     "La relación de la leche cruda con la temperatura, mirada directamente: cada parámetro en el tiempo, rectas por "
     "industria-día, linealidad, anomalías y desfase de una semana; células somáticas y recuento bacteriano."),
]

PORTADA = """# Calidad y composición de productos lácteos: agrupamiento y modelos supervisados
**Aprendizaje Automático y Grandes Datos · UNRaf · 2026**

Este cuaderno reúne el trabajo completo en ocho partes. Se entrega **ejecutado**: las tablas y los gráficos que se ven
son los de la última corrida, así que se puede leer sin volver a correrlo.

| parte | tema | algoritmos de la materia |
|---|---|---|
| [I](#parte-i) | agrupamiento de las muestras de laboratorio (Senasa y MQ) | k-medias (y minilotes, bisectivo), jerárquico (Ward, simple, promedio), mezcla de gaussianas, reglas de asociación (FP-growth) |
| [II](#parte-ii) | el mismo agrupamiento con categorías | k-modas, jerárquico con distancia de Hamming |
| [III](#parte-iii) | ciclo anual de la proteína y la grasa de la leche cruda | regresión lineal y polinómica; contra k-NN, árbol y SVM |
| [IV](#parte-iv) | probabilidad de que la leche cruda supere los umbrales de calidad | regresión logística; contra Bayes, k-NN, SVM y árbol |
| [V](#parte-v) | tipo de una muestra (familia y subtipo) según lo que se le midió | clasificación con árboles potenciados (HistGradientBoosting), validación cruzada agrupada |
| [VI](#parte-vi) | días de espera para que la leche UAT quede deslactosada | regresión de primer orden con censura (Tobit), validación dejando uno afuera, bootstrap |
| [VII](#parte-vii) | familia y subtipo de una muestra con una cascada de árboles (trabajo de otro integrante) | árbol de decisión (CART) con búsqueda en cuadrícula, validación cruzada estratificada, ROC uno contra el resto |
| [VIII](#parte-viii) | cuánto se parece la leche cruda a la temperatura: composición y recuentos | regresión lineal (mínimos cuadrados) |
| [—](#datos-por-entrenamiento) | cuántos datos se usaron en cada entrenamiento | |
| [—](#conclusiones-generales) | conclusiones generales | |

**Los datos.** Dos fuentes:
- Las muestras de laboratorio de productos lácteos (Senasa y el sistema MQ), con humedad, grasa, proteínas y cenizas.
- Las mediciones de leche cruda de tambo de 2019 a 2026: más de 600 000 filas, con células somáticas, bacterias,
  lactosa, grasa y proteína.

A eso se suma la temperatura diaria de Rafaela (NASA POWER, validada contra la estación del SMN).

**Confidencialidad.** Las empresas figuran con un código (`IND-nn`) y las marcas están codificadas: el cuaderno no
contiene nombres de clientes.

**Cómo se valida.**
- En las partes de agrupamiento no hay respuesta correcta. Los grupos se juzgan con la silueta, con su estabilidad y,
  al final, revelando la familia del producto, que el algoritmo no vio.
- En las partes sobre la leche cruda (III y IV) nunca se parte al azar, porque las muestras de un mismo día se parecen y
  eso filtraría el futuro. Se entrena hasta 2023 y se prueba en 2024–2026; además, se deja una industria afuera por vez.
- En la Parte V, validación cruzada agrupada por envío: las muestras de un mismo envío nunca quedan a ambos lados.
- En la Parte VI, que tiene pocos lotes, se deja un lote afuera por vez.
- En la Parte VII, como en el trabajo original, la prueba es un 20 % de las muestras al azar, estratificado por familia
  y subtipo. Para ver cuánto infla eso el resultado, se compara también con una validación agrupada por envío.
- La Parte VIII describe una asociación: las rectas usan todos los datos y se controlan con anomalías (lo que queda al
  restar la época del año).

**Para volver a correrlo.** Hace falta la carpeta del trabajo (`trabajo_senasa_mq/` con `src/`, `datos_entrada/` y los
datos anonimizados). Se ejecuta desde esa carpeta y tarda más de media hora.
"""

DATOS_MD = """<a id="datos-por-entrenamiento"></a>

# Datos usados en cada entrenamiento

Cada parte cierra con su propia tabla de datos usados; acá están todas juntas. Por cada modelo se indica:
- cuántas filas había disponibles;
- cuántas se excluyeron y por qué;
- cuántas se usaron y cómo se repartieron entre entrenamiento, validación y prueba.

En el agrupamiento no hay respuesta correcta y, por lo tanto, no hay conjunto de prueba: entrenamiento es igual a usadas.
Los números salen de esta misma corrida y quedan en `resultados/datos_entrenamiento/`.
"""

DATOS_CODIGO = """import datos_entrenamiento as DE
datos_por_modelo = DE.todas()
print(f"{len(datos_por_modelo)} modelos en {datos_por_modelo['parte'].nunique()} partes")
with pd.option_context("display.max_rows", 200, "display.max_colwidth", 80):
    display(datos_por_modelo)"""


def bajar_titulos(texto):
    """Baja un nivel cada título de markdown («# X» → «## X»), sin pasar de seis «#»."""
    return re.sub(r"^(#{1,5}) ", lambda m: "#" + m.group(1) + " ", texto, flags=re.M)


def referencias(texto):
    """Los cuadernos se citan entre sí por nombre de archivo: acá cada uno es una parte."""
    for parte, archivo, _, _ in PARTES:
        texto = texto.replace(f"`{archivo}`", f"la Parte {parte} (`{archivo}`)")
    return texto


def celdas_de(archivo):
    nb = nbf.read(AQUI / archivo, as_version=4)
    salida = []
    for c in nb.cells:
        if c.cell_type == "markdown":
            salida.append(nbf.v4.new_markdown_cell(referencias(bajar_titulos(c.source))))
        elif c.cell_type == "code":
            salida.append(nbf.v4.new_code_cell(c.source))
    return salida


def conclusiones():
    ruta = AQUI / "conclusiones_generales.md"
    return ruta.read_text(encoding="utf-8")


def main():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    celdas = [nbf.v4.new_markdown_cell(PORTADA)]
    for parte, archivo, titulo, resumen in PARTES:
        celdas.append(nbf.v4.new_markdown_cell(
            f'<a id="parte-{parte.lower()}"></a>\n\n# Parte {parte} · {titulo}\n\n{resumen}\n\n'
            f"*(Generada a partir de `{archivo}`.)*"))
        celdas += celdas_de(archivo)
    celdas.append(nbf.v4.new_markdown_cell(DATOS_MD))
    celdas.append(nbf.v4.new_code_cell(DATOS_CODIGO))
    celdas.append(nbf.v4.new_markdown_cell('<a id="conclusiones-generales"></a>\n\n' + conclusiones()))
    nb.cells = celdas
    nbf.validate(nb)
    nbf.write(nb, SALIDA)
    print(f"{SALIDA.name}: {len(celdas)} celdas "
          f"({sum(c.cell_type == 'code' for c in celdas)} de código)")


if __name__ == "__main__":
    main()
