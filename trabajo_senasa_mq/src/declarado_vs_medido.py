"""Lo que declara la descripción contra la humedad medida.

Pedido del usuario (2026-11-09): la humedad declarada debería corresponder con los resultados. Se revisan
los dos ejes de la clasificación que afirman algo sobre la humedad:

  * consistencia del queso (desc_consistencia: baja / mediana / alta humedad) contra la clase del CAA
    art. 605 para la humedad medida. El catálogo junta «alta» y «muy alta», así que una humedad de 55 o
    más cuenta como «alta humedad». desc_consistencia_origen dice si la consistencia sale del rótulo
    (lo declara la descripción), de la variedad (se deduce del nombre: «REGGIANITO» → baja) o del formato.
  * estado (desc_estado: POLVO / LIQUIDO) contra los mismos umbrales que usan los cortes por estado
    (cortes_caa.HUM_POLVO_MAX y HUM_LIQUIDO_MIN).

Humedad medida = «Resultado Humedad» o, si falta, 100 − sólidos totales (los quesos de Senasa y todo MQ).
Entran Senasa encolumnado (clasificación manual) y las muestras nuevas de MQ encolumnado (automática).
«borde» = fuera de la clase declarada por 1 g/100 g o menos.

Escribe resultados/tablas/declarado_vs_medido.csv (todas las muestras comparables) y
resultados/excel/declarado_vs_medido.xlsx (resumen + las que no coinciden). No modifica nada más.
"""

import numpy as np
import pandas as pd

import rutas as R
from cortes_caa import HUM_LIQUIDO_MIN, HUM_POLVO_MAX
from datos_clase41 import a_numero, leer_senasa

SALIDA_CSV = R.TABLAS / "declarado_vs_medido.csv"
SALIDA_XLSX = R.EXCEL / "declarado_vs_medido.xlsx"
SALIDA_ANALISIS_CSV = R.TABLAS / "declarado_vs_medido_con_analisis.csv"
SALIDA_ANALISIS_XLSX = R.EXCEL / "declarado_vs_medido_con_analisis.xlsx"
MQ_NUEVAS = R.EXCEL / "MQ encolumnado - solo nuevas.xlsx"
BORDE = 1.0     # g/100 g

# clase declarada -> (desde, hasta) de humedad, con desde incluido y hasta excluido
RANGO_QUESO = {"baja humedad": (0.0, 36.0), "mediana humedad": (36.0, 46.0), "alta humedad": (46.0, np.inf)}
RANGO_ESTADO = {"POLVO": (0.0, HUM_POLVO_MAX + 1e-9), "LIQUIDO": (HUM_LIQUIDO_MIN, np.inf)}


def clase_queso(h):
    return pd.cut(h, [0, 36, 46, 55, np.inf], right=False,
                  labels=["baja humedad", "mediana humedad", "alta humedad", "muy alta humedad"]).astype(object)


def clase_estado(h):
    return pd.Series(np.where(h <= HUM_POLVO_MAX, "polvo (≤ 15)", np.where(h >= HUM_LIQUIDO_MIN, "líquido (≥ 25)", "intermedio (15 a 25)")),
                     index=h.index).where(h.notna())


def _comparar(d, eje, declarado, origen_declarado, rangos, clase):
    m = declarado.isin(list(rangos)) & d["humedad"].notna()
    x = d[m].copy()
    x["eje"] = eje
    x["declarado"] = declarado[m]
    x["origen_declarado"] = origen_declarado[m] if origen_declarado is not None else "catálogo"
    x["clase_medida"] = clase(x["humedad"])
    desde = x["declarado"].map(lambda k: rangos[k][0])
    hasta = x["declarado"].map(lambda k: rangos[k][1])
    x["distancia"] = np.where(x["humedad"] < desde, desde - x["humedad"], np.where(x["humedad"] >= hasta, x["humedad"] - hasta, 0.0)).round(2)
    x["coincide"] = x["distancia"] == 0
    x["grado"] = np.where(x["coincide"], "coincide", np.where(x["distancia"] <= BORDE, "borde", "no coincide"))
    return x


def armar(escribir=True):
    """Devuelve (tabla completa, resumen)."""
    S = leer_senasa()
    s = S.crudo
    sen = pd.DataFrame({"Identificador": s["Identificador"], "origen": "Senasa (manual)", "Cliente": s["Cliente"],
                        "Descripción": s["Descripción"], "familia": s["desc_familia"], "familia_fuente": "senasa (manual)",
                        "desc_consistencia": s["desc_consistencia"], "desc_consistencia_origen": s["desc_consistencia_origen"],
                        "desc_estado": s["desc_estado"], "humedad": s["humedad"].fillna(100 - s["solidos_senasa"]),
                        "humedad_origen": np.where(s["humedad"].notna(), "medida", "100 − sólidos")})
    sen = sen[~sen["Identificador"].isin(S.excluidas)]
    q = pd.read_excel(MQ_NUEVAS)
    mq = pd.DataFrame({"Identificador": q["Identificador"], "origen": "MQ nueva (automática)", "Cliente": q["Cliente"],
                       "Descripción": q["Descripción"], "familia": q["desc_familia"], "familia_fuente": q["desc_familia_fuente"],
                       "desc_consistencia": q["desc_consistencia"], "desc_consistencia_origen": q["desc_consistencia_origen"],
                       "desc_estado": q["desc_estado"], "humedad": a_numero(q["humedad_derivada_100_menos_solidos"]),
                       "humedad_origen": "100 − sólidos"})
    d = pd.concat([sen, mq], ignore_index=True)
    d["humedad"] = d["humedad"].where((d["humedad"] >= 0) & (d["humedad"] <= 100))
    es_queso = d["familia"].eq("queso")
    partes = [_comparar(d[es_queso], "consistencia del queso", d.loc[es_queso, "desc_consistencia"],
                        d.loc[es_queso, "desc_consistencia_origen"], RANGO_QUESO, clase_queso),
              _comparar(d, "estado", d["desc_estado"].astype("string").str.upper(), None, RANGO_ESTADO, clase_estado)]
    t = pd.concat(partes, ignore_index=True)
    t = t[["Identificador", "origen", "Cliente", "Descripción", "familia", "familia_fuente", "eje", "declarado",
           "origen_declarado", "humedad", "humedad_origen", "clase_medida", "distancia", "coincide", "grado"]]
    t["humedad"] = t["humedad"].round(2)
    g = t.groupby(["eje", "origen", "declarado"])
    resumen = pd.DataFrame({"muestras": g.size(), "coincide": g["coincide"].sum(),
                            "borde": g["grado"].apply(lambda x: (x == "borde").sum()),
                            "no coincide": g["grado"].apply(lambda x: (x == "no coincide").sum())}).reset_index()
    tot = resumen.groupby(["eje", "origen"], as_index=False)[["muestras", "coincide", "borde", "no coincide"]].sum().assign(declarado="TOTAL")
    resumen = pd.concat([resumen, tot], ignore_index=True).sort_values(["eje", "origen", "declarado"]).reset_index(drop=True)
    resumen["% coincide"] = (100 * resumen["coincide"] / resumen["muestras"]).round(1)
    resumen["% coincide o borde"] = (100 * (resumen["coincide"] + resumen["borde"]) / resumen["muestras"]).round(1)
    if escribir:
        R.preparar()
        t.to_csv(SALIDA_CSV, index=False, encoding="utf-8-sig")
        no = t[~t["coincide"]].sort_values(["eje", "origen", "distancia"], ascending=[True, True, False])
        por_origen = (t[t["eje"] == "consistencia del queso"].groupby(["origen", "origen_declarado", "declarado", "clase_medida"])
                      .size().rename("muestras").reset_index())
        with pd.ExcelWriter(SALIDA_XLSX) as w:
            resumen.to_excel(w, sheet_name="resumen", index=False)
            por_origen.to_excel(w, sheet_name="queso_por_origen", index=False)
            no.to_excel(w, sheet_name="no_coinciden", index=False)
        print(f"{len(t)} comparaciones ({int((~t['coincide']).sum())} no coinciden) -> {SALIDA_CSV.name}, {SALIDA_XLSX.name}")
    return t, resumen


def _valor(v):
    """Número si se puede leer como número («54,.34» -> 54.34); si no, el texto tal cual («< LD»)."""
    if pd.isna(v) or isinstance(v, (int, float, np.number)):
        return v
    x = str(v).strip().replace(",.", ".").replace(",", ".")
    try:
        return float(x)
    except ValueError:
        return str(v).strip()


def con_analisis(t=None, escribir=True):
    """Cada comparación (una fila por muestra y eje, como en declarado_vs_medido.xlsx) con los resultados de
    todos los análisis de la muestra. Solo columnas «Resultado …» con algún dato; sin las de técnica."""
    if t is None:
        t = armar(escribir=False)[0]
    s = leer_senasa().crudo
    q = pd.read_excel(MQ_NUEVAS)
    res = [c for c in q.columns if c.startswith("Resultado ")]
    res += [c for c in s.columns if c.startswith("Resultado ") and c not in res]
    fuentes = pd.concat([s.reindex(columns=["Identificador"] + res), q.reindex(columns=["Identificador"] + res)], ignore_index=True)
    fuentes = fuentes.drop_duplicates("Identificador")
    x = t.merge(fuentes, on="Identificador", how="left", validate="many_to_one")
    for c in res:
        x[c] = x[c].map(_valor)
    x = x.dropna(axis=1, how="all")
    if escribir:
        R.preparar()
        x.to_csv(SALIDA_ANALISIS_CSV, index=False, encoding="utf-8-sig")
        with pd.ExcelWriter(SALIDA_ANALISIS_XLSX) as w:
            x.to_excel(w, sheet_name="todas", index=False)
            x[~x["coincide"]].sort_values(["eje", "origen", "distancia"], ascending=[True, True, False]).to_excel(
                w, sheet_name="no_coinciden", index=False)
        print(f"{len(x)} filas × {x.shape[1]} columnas -> {SALIDA_ANALISIS_XLSX.name}, {SALIDA_ANALISIS_CSV.name}")
    return x


if __name__ == "__main__":
    tabla, resumen = armar()
    print(resumen.to_string(index=False))
    con_analisis(tabla)
