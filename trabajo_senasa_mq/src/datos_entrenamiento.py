"""Cuántos datos entran en cada entrenamiento: una tabla con el mismo formato en todos los cuadernos.

Cada cuaderno arma su lista de filas con los números calculados en la propia corrida (nunca escritos a mano) y llama a
`guardar(filas, "<cuaderno>")`, que muestra la tabla y la escribe en `resultados/datos_entrenamiento/<cuaderno>.csv`.
`build_notebook_general.py` junta todos los CSV en la tabla de la portada del cuaderno general.

Cada fila es un modelo (o una etapa de un modelo) y lleva:
- parte: «I» a «VIII» (la del cuaderno general).
- modelo: qué se entrena, p. ej. «k-medias final, solo Senasa» o «logística M3, RCS > 400 mil».
- datos: de qué tabla salen las filas, p. ej. «muestras de Senasa con los 4 análisis».
- disponibles: filas de esa tabla antes de cualquier filtro propio del modelo.
- excluidas: disponibles − usadas.
- motivo de exclusión: por qué quedaron afuera (faltan análisis, outliers, industria descartada, ...).
- usadas: filas que entran al modelo.
- entrenamiento, validación, prueba: cómo se reparten las usadas (vacío si no corresponde; en el agrupamiento
  no hay prueba: entrenamiento = usadas).
- partición: cómo se separan (temporal hasta 2023 / 2024–26, validación cruzada de 5, sin partición, ...).
- nota: submuestras (p. ej. el k-NN entrena con 50 000 al azar) u otra aclaración.
"""
import pandas as pd

from rutas import RESULTADOS

CARPETA = RESULTADOS / "datos_entrenamiento"
COLUMNAS = ["parte", "modelo", "datos", "disponibles", "excluidas", "motivo de exclusión", "usadas",
            "entrenamiento", "validación", "prueba", "partición", "nota"]
ENTERAS = ["disponibles", "excluidas", "usadas", "entrenamiento", "validación", "prueba"]


def fila(parte, modelo, datos, disponibles, usadas, motivo="", entrenamiento=None, validacion=None, prueba=None,
         particion="sin partición", nota=""):
    """Una fila de la tabla. Si no se da `entrenamiento`, se toma igual a `usadas` (agrupamiento, sin prueba)."""
    return {"parte": parte, "modelo": modelo, "datos": datos, "disponibles": int(disponibles),
            "excluidas": int(disponibles) - int(usadas), "motivo de exclusión": motivo, "usadas": int(usadas),
            "entrenamiento": int(usadas if entrenamiento is None else entrenamiento),
            "validación": None if validacion is None else int(validacion),
            "prueba": None if prueba is None else int(prueba), "partición": particion, "nota": nota}


def tabla(filas):
    t = pd.DataFrame(filas, columns=COLUMNAS)
    for c in ENTERAS:
        t[c] = t[c].astype("Int64")
    return t


def guardar(filas, nombre):
    """Escribe resultados/datos_entrenamiento/<nombre>.csv y devuelve la tabla (para display)."""
    t = tabla(filas)
    malas = t[(t["entrenamiento"].fillna(0) + t["validación"].fillna(0) + t["prueba"].fillna(0)) > t["usadas"]]
    if len(malas):
        raise ValueError(f"entrenamiento + validación + prueba > usadas en: {list(malas['modelo'])}")
    CARPETA.mkdir(parents=True, exist_ok=True)
    t.to_csv(CARPETA / f"{nombre}.csv", index=False)
    return t


def todas():
    """Todas las tablas guardadas, en el orden de las partes."""
    archivos = sorted(CARPETA.glob("*.csv"))
    if not archivos:
        return tabla([])
    t = pd.concat([pd.read_csv(a) for a in archivos], ignore_index=True)
    orden = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8}
    return t.sort_values("parte", key=lambda s: s.map(orden), kind="stable").reset_index(drop=True)
