"""Arma resultados/tablas/cortes_categorias.csv: los cortes para pasar cada análisis numérico a categorías.

Ex build_cortes_categorias.py (original en _archivo/proyecto/). Se regenera con la sección 5 de
trabajo_senasa_mq.ipynb (`cortes_caa.armar()`) o con `python src/cortes_caa.py` desde trabajo_senasa_mq/.

Criterio «CAA por familia»: donde el Código Alimentario Argentino define clases, se usan esas (tabla
CAA más abajo, con artículo y URL). Donde no, terciles (percentiles 33 y 66) dentro de la familia, con
los datos de «Senasa encolumnado.xlsx». Si la familia tiene menos de 30 valores, cortes globales
(familia «*»). Explicación y decisiones abiertas: docs/cortes_categorias.md.

Terciles por estado (decisión del usuario, 2026-11-09): en las familias de FAMILIAS_CON_ESTADO el mismo
rótulo cubre un polvo y un líquido (o manteca y crema), con composición muy distinta, así que los terciles
se calculan por separado para cada estado (columna «estado»: POLVO, LIQUIDO o SOLIDO). El estado de cada
muestra sale de `estado_para_cortes`: la humedad medida si existe, si no el estado declarado. Si un estado
tiene menos de 30 valores en Senasa se suman las muestras nuevas de «MQ encolumnado.xlsx» (la columna
«fuente» lo dice); si tampoco alcanza, ese estado queda sin cortes.

Cómo se usa la tabla: para (analisis, familia, subtipo, estado) se toma la regla más específica que exista
(familia + subtipo, si no familia + estado, si no familia, si no «*»), y el valor cae en la fila con
desde <= valor < hasta. En una familia con cortes por estado, la muestra sin estado no se categoriza.

Lee «Senasa encolumnado.xlsx» y, solo para los estados con pocos valores, «MQ encolumnado.xlsx».
No modifica nada fuera de cortes_categorias.csv.
"""

import re

import numpy as np
import pandas as pd

import rutas as R

ENTRADA = R.SENASA
ENTRADA_MQ = R.EXCEL / "MQ encolumnado.xlsx"
COL_HUMEDAD_MQ = "humedad_derivada_100_menos_solidos"     # MQ mide sólidos; la humedad va en esta columna

# Familias donde el mismo rótulo junta dos estados de composición muy distinta.
FAMILIAS_CON_ESTADO = ["suero / permeado", "concentrado proteico", "yogur / bebida lactea", "base nutricional",
                       "manteca / crema"]
ESTADOS = ["POLVO", "LIQUIDO", "SOLIDO"]
HUM_POLVO_MAX = 15.0      # g/100 g: hasta acá es un polvo (los polvos declarados llegan a 15)
HUM_LIQUIDO_MIN = 25.0    # desde acá es un líquido o concentrado (los líquidos declarados arrancan en 25)
HUM_MANTECA_MAX = 30.0    # manteca / crema: por debajo es manteca (SOLIDO), por encima crema (LIQUIDO)
SALIDA = R.TABLAS / "cortes_categorias.csv"
MIN_N = 30
INF = np.inf

URL_VIII = "https://www.argentina.gob.ar/sites/default/files/capitulo_viii_lacteosactualiz_2025-10.pdf"
URL_XII = "https://www.argentina.gob.ar/sites/default/files/capitulo_xii_aguas_actualiz_2025-11.pdf"
URL_XVII = "https://www.argentina.gob.ar/sites/default/files/capitulo_xvii_dieteticosactualiz_2025-09_0.pdf"

# analisis -> (columnas de origen, base). Las dos galactosas son la misma determinación con otra técnica
# (nunca están las dos en la misma muestra; igual que en «Categorizacion de los Datos.ipynb»).
ANALISIS = {
    "humedad": (["Resultado Humedad g/ 100g (*)"], "g/100 g"),
    "materia_grasa": (["Resultado Materia Grasa g/ 100g (*)_x"], "g/100 g"),
    "proteinas": (["Resultado Proteínas g/ 100g (*)"], "g/100 g"),
    "cenizas": (["Resultado Cenizas g/ 100g"], "g/100 g"),
    "solidos_totales": (["Resultado Sólidos totales g/ 100g_x"], "g/100 g"),
    "lactosa": (["Resultado Lactosa g /100g (*)"], "g/100 g"),
    "lactosa_g100ml": (["Resultado Contenido de Lactosa % (g/ 100ml) (*)",
                        "Resultado Determinación del contenido de lactosa (g/ 100ml) (*)"], "g/100 ml"),
    "galactosa": (["Resultado D - Galactosa g/ 100g (*)", "Resultado Galactosa g/ 100g (*)"], "g/100 g"),
    "mg_extracto_seco": (["Resultado Materia grasa en extracto seco g/ 100g (*)_x"], "g/100 g de extracto seco"),
    "actividad_agua": (["Resultado Actividad de agua (% Aw) (*)"], "aw (0 a 1)"),
    "acido_sorbico": (["Resultado Acido Sórbico mg/ Kg (*)"], "mg/kg"),
    "carbohidratos_totales": (["Resultado Carbohidratos Totales g/ 100g (*)"], "g/100 g"),
    "azucares_totales": (["Resultado Azucares totales g/100gr (*) (fructosa/glucosa/sacarosa/maltosa/lactosa)_x",
                          "Resultado Azucares totales g/100gr (*) (fructosa/glucosa/sacarosa/maltosa/lactosa/galactosa)",
                          "Resultado Azucares totales g/100gr (*)"], "g/100 g"),
    "acidez_naoh": (["Resultado Acidez (ml NaOH 0,1 N/ 10 g sólidos no grasos) (*)"], "ml NaOH 0,1 N/10 g SNG"),
    "solidos_no_grasos": (["Resultado Sólidos No Grasos de leche % (*)"], "g/100 g"),
    "sulfitos": (["Resultado Sulfitos ppm SO₂ (*)"], "ppm SO2"),
    "grasas_saturadas": (["Resultado Grasas saturadas g% (*)"], "g/100 g"),
    "grasas_trans": (["Resultado Grasas trans g% (*)"], "g/100 g"),
    "sodio": (["Resultado Sodio mg /100g (*)"], "mg/100 g"),
    "fibra": (["Resultado Fibra Alimentaria Total g/ 100g (*)"], "g/100 g"),
    "plomo": (["Resultado Plomo ppm (*)"], "ppm"),
    "arsenico": (["Resultado Arsénico ppm (*)"], "ppm"),
}
# g/100 g por encima de 100 es imposible (p. ej. «4548» por 45,48): no entra al cálculo de terciles.
MAX_FISICO = {k: 100.0 for k, (_, b) in ANALISIS.items() if b.startswith("g/100")}
MAX_FISICO["actividad_agua"] = 1.0


def parse(v):
    """Coma decimal; valor antes de «+/-» (o de «=», «/»); «<» y «no detectado/no contiene» = 0; «,.» = «,»."""
    if pd.isna(v):
        return np.nan
    s = str(v).strip().replace(",.", ",")
    if re.search(r"&lt;|<|no detectado|no contiene", s, re.I):
        return 0.0
    s = re.split(r"\+/-|=|/", s)[0].strip()
    if "." in s and "," in s:          # «1.320,65»: el punto es separador de miles
        s = s.replace(".", "")
    s = s.replace(",", ".")
    m = re.match(r"^-?\d+(\.\d+)?", s)
    return float(m.group(0)) if m else np.nan


# ---------------------------------------------------------------- cortes del CAA (verificados en el PDF)
def caa(analisis, familia, subtipo, base, fuente, clases, nota="", estado=""):
    """clases: [(categoria, desde, hasta), ...] con desde incluido y hasta excluido."""
    return [dict(analisis=analisis, familia=familia, subtipo=subtipo, estado=estado, categoria=c, desde=a, hasta=b,
                 base=base, fuente=fuente, verificado="sí", nota=nota) for c, a, b in clases]


def estado_para_cortes(familia, declarado, subtipo, humedad):
    """Estado de cada muestra para elegir los cortes: POLVO, LIQUIDO, SOLIDO o "" (no se sabe).
    Cuatro Series con el mismo índice. Manda la humedad medida (o 100 − sólidos): hasta HUM_POLVO_MAX es POLVO,
    desde HUM_LIQUIDO_MIN es LIQUIDO; entre los dos, o sin humedad, vale el estado declarado (desc_estado).
    En manteca / crema los estados son SOLIDO (manteca) y LIQUIDO (crema): corte en HUM_MANTECA_MAX y, sin
    humedad, el subtipo."""
    familia = familia.fillna("")
    dec = declarado.astype("string").str.upper().where(lambda t: t.isin(["POLVO", "LIQUIDO"]), "").fillna("")
    sub = subtipo.astype("string").fillna("")
    h = pd.to_numeric(humedad, errors="coerce")
    est = pd.Series(np.where(h <= HUM_POLVO_MAX, "POLVO", np.where(h >= HUM_LIQUIDO_MIN, "LIQUIDO", "")), index=familia.index)
    est = est.where(est.ne(""), dec)
    mc = familia.eq("manteca / crema")
    est_mc = pd.Series(np.where(h < HUM_MANTECA_MAX, "SOLIDO", np.where(h >= HUM_MANTECA_MAX, "LIQUIDO",
                       np.where(sub.isin(SUB_CREMA), "LIQUIDO", np.where(sub.eq("MANTECA"), "SOLIDO", "")))), index=familia.index)
    return est.where(~mc, est_mc).astype(object)


# El laboratorio informa 2 decimales: un límite «hasta X inclusive» se escribe hasta X + 0,005.
CAA = []
CAA += caa("humedad", "queso", "", "g/100 g",
           f"CAA art. 605, inc. 2.b ({URL_VIII})",
           [("baja humedad", 0, 36.0), ("mediana humedad", 36.0, 46.0),
            ("alta humedad", 46.0, 55.0), ("muy alta humedad", 55.0, INF)],
           "CAA: hasta 35,9 / 36,0-45,9 / 46,0-54,9 / no menor a 55,0. Igual escala que desc_consistencia "
           "(proyecto/catalogo.py), que junta alta y muy alta.")
CAA += caa("mg_extracto_seco", "queso", "", "g/100 g de extracto seco",
           f"CAA art. 605, inc. 2.a ({URL_VIII})",
           [("descremado", 0, 10.0), ("magro", 10.0, 25.0), ("semigraso", 25.0, 45.0),
            ("graso", 45.0, 60.0), ("extra graso", 60.0, INF)],
           "CAA: <10,0 / 10,0-24,9 / 25,0-44,9 / 45,0-59,9 / no menos de 60 (extra graso o doble crema).")
CAA += caa("materia_grasa", "dulce de leche", "", "g/100 g",
           f"CAA art. 592, inc. 5.2 ({URL_VIII})",
           [("bajo el mínimo (<6)", 0, 6.0), ("dulce de leche", 6.0, 9.005), ("con crema", 9.005, INF)],
           "CAA: dulce de leche 6,0 a 9,0; con crema mayor de 9,0. Menos de 6 no es una clase del CAA: "
           "no cumple el requisito.")
CAA += caa("materia_grasa", "leche en polvo", "", "g/100 g",
           f"CAA art. 567, inc. 2.2.1 ({URL_VIII})",
           [("descremada", 0, 1.505), ("parcialmente descremada", 1.505, 26.0), ("entera", 26.0, INF)],
           "CAA: descremada <= 1,5; parcialmente descremada > 1,5 y < 26,0; entera >= 26,0 (% m/m). "
           "Mismos límites que proyecto/build_mq_subfamilia.py.")
CAA += caa("materia_grasa", "leche fluida", "", "g/100 g",
           f"CAA art. 560 bis (UAT) y art. 562 a/b ({URL_VIII})",
           [("descremada", 0, 0.6), ("parcialmente descremada", 0.6, 3.0), ("entera", 3.0, INF)],
           "CAA: descremada máx. 0,5; semidescremada 0,6 a 2,9; entera mín. 3,0, en % m/v (g/100 cm3). "
           "El hueco 0,5-0,6 se asigna a descremada. No aplica a leche concentrada.")
CAA += caa("materia_grasa", "yogur / bebida lactea", "", "g/100 g",
           f"CAA art. 576, inc. 2.a ({URL_VIII})",
           [("descremada", 0, 0.6), ("parcialmente descremada", 0.6, 3.0),
            ("entera", 3.0, 6.0), ("con crema", 6.0, INF)],
           "CAA (leches fermentadas): descremada máx. 0,5; parcialmente descremada 0,6-2,9; entera 3,0-5,9; "
           "con crema mín. 6,0, sobre la BASE LÁCTEA (no el producto con agregados). Hueco 0,5-0,6 a descremada.")
CAA += caa("materia_grasa", "helado", "", "g/100 g",
           f"CAA art. 1077, inc. 1-3 ({URL_XII})",
           [("helado de agua", 0, 1.5), ("helado de leche", 1.5, 6.0), ("helado de crema", 6.0, INF)],
           "CAA: agua máx. 1,5; de leche mín. 1,5; de crema mín. 6,0 (% p/p de materia grasa DE LECHE; "
           "el laboratorio mide grasa total).")
CAA += caa("materia_grasa", "manteca / crema", "CREMA; CREMA DE LECHE; DOBLE CREMA", "g/100 g",
           f"CAA art. 585, inc. 1 y 4 ({URL_VIII})",
           [("bajo el mínimo (<10)", 0, 10.0), ("crema liviana", 10.0, 20.0), ("crema", 20.0, 50.0),
            ("crema de alto tenor graso", 50.0, INF)],
           "CAA: bajo tenor graso/liviana/semicrema 10,0-19,9; crema 20,0-49,9; alto tenor graso mín. 50,0.")

# Lactosa (cap. XVII). «Sin lactosa» vale para cualquier alimento. «Bajo» y «reducido» son relativos al
# alimento corriente (5 % y 30 % de su lactosa): solo se aplican a leche fluida, con la lactosa de la leche
# cruda del proyecto (mediana 4,83 g/100 g, proyecto/diccionario_leche_cruda.xlsx, hoy en _archivo/) como referencia.
# El texto de la columna fuente conserva la ruta original para que la tabla sea idéntica a la anterior.
LAC_REF = 4.83
SIN_LAC = ("sin lactosa", 0, 0.105)
LAC_FLUIDA = [SIN_LAC, ("bajo en lactosa", 0.105, round(0.05 * LAC_REF, 4)),
              ("reducido en lactosa", round(0.05 * LAC_REF, 4), round(0.30 * LAC_REF, 4)),
              ("lactosa normal", round(0.30 * LAC_REF, 4), INF)]
for an, base in (("lactosa", "g/100 g"), ("lactosa_g100ml", "g/100 ml")):
    for r in caa(an, "leche fluida", "", base,
                 f"CAA art. 1372 y 1372 bis ({URL_XVII}); referencia: trabajo previo, "
                 f"proyecto/diccionario_leche_cruda.xlsx", LAC_FLUIDA,
                 f"Sin lactosa <= 0,1 (100 mg/100 g o 100 ml, verificado). Bajo <= 5 % y reducido <= 30 % de la "
                 f"lactosa del alimento corriente: el CAA no da el valor; se usa {LAC_REF} g/100 g (mediana leche "
                 f"cruda del proyecto) -> {0.05 * LAC_REF:.4f} y {0.30 * LAC_REF:.4f}."):
        r["verificado"] = "sí" if r["categoria"] == "sin lactosa" else "no"
        CAA.append(r)

# una crema sin subtipo en la descripción (estado LIQUIDO de manteca / crema) lleva las mismas clases
CAA += caa("materia_grasa", "manteca / crema", "", "g/100 g",
           f"CAA art. 585, inc. 1 y 4 ({URL_VIII})",
           [("bajo el mínimo (<10)", 0, 10.0), ("crema liviana", 10.0, 20.0), ("crema", 20.0, 50.0),
            ("crema de alto tenor graso", 50.0, INF)],
           "Igual que la fila por subtipo: en esta familia el estado LIQUIDO es crema.", estado="LIQUIDO")

CAA_CLAVES = {(r["analisis"], r["familia"], r["subtipo"]) for r in CAA if r["estado"] == ""}
CAA_CLAVES_ESTADO = {(r["analisis"], r["familia"], r["estado"]) for r in CAA if r["estado"] != ""}
SUB_CREMA = ["CREMA", "CREMA DE LECHE", "DOBLE CREMA"]


# ---------------------------------------------------------------- terciles
def terciles(x):
    """Cortes (p33, p66) o una regla para distribuciones con muchos ceros/empates."""
    x = np.sort(np.asarray(x, float))
    if (x == 0).mean() >= 1 / 3:
        pos = x[x > 0]
        if len(pos) < 2:
            return [("no detectado / 0", 0, 1e-6), ("detectado", 1e-6, INF)], "más de 1/3 son 0: 0 vs. > 0"
        med = float(np.percentile(pos, 50))
        return ([("no detectado / 0", 0, 1e-6), ("bajo", 1e-6, med), ("alto", med, INF)],
                f"más de 1/3 son 0 (< LD): clase propia; el resto se parte en la mediana de los > 0")
    p33, p66 = (float(np.percentile(x, q)) for q in (33, 66))
    if p33 == p66:
        return [("bajo", 0, p33), ("alto", p33, INF)], "p33 = p66 por empates: dos clases"
    return [("bajo", 0, p33), ("medio", p33, p66), ("alto", p66, INF)], ""


def _valores(d, cols, an):
    v = pd.concat([d[c].map(parse) for c in cols], axis=1).bfill(axis=1).iloc[:, 0]
    tope = MAX_FISICO.get(an)
    return v.where(v <= tope + 1e-9) if tope is not None else v


def _estado(d, humedad_extra=None):
    h = _valores(d, ANALISIS["humedad"][0], "humedad")
    h = h.fillna(100 - _valores(d, ANALISIS["solidos_totales"][0], "solidos_totales"))
    if humedad_extra is not None:
        h = h.fillna(d[humedad_extra].map(parse))
    return estado_para_cortes(d["desc_familia"], d["desc_estado"], d["desc_subtipo"], h.where((h >= 0) & (h <= 100)))


def armar():
    """Escribe cortes_categorias.csv y devuelve la tabla."""
    R.preparar()
    d = pd.read_excel(ENTRADA)
    est_d = _estado(d)
    # muestras nuevas de MQ: solo refuerzan los estados con menos de MIN_N valores en Senasa
    mq = est_mq = None
    if ENTRADA_MQ.exists():
        mq = pd.read_excel(ENTRADA_MQ)
        mq = mq[mq["en_catalogo_senasa"].eq("no")].reset_index(drop=True)
        est_mq = _estado(mq, COL_HUMEDAD_MQ)
    familias = sorted(d["desc_familia"].dropna().unique())
    filas = list(CAA)
    log, log_estado = [], []
    for an, (cols, base) in ANALISIS.items():
        v = _valores(d, cols, an)
        ok = v.notna()
        if not ok.any():
            continue
        v_mq = None
        if mq is not None:
            v_mq = _valores(mq, cols, an)
            if an == "humedad":
                v_mq = v_mq.fillna(mq[COL_HUMEDAD_MQ].map(parse)).where(lambda z: z <= 100)
        es_lac = an.startswith("lactosa")

        def emitir(x, fam, estado, fuente, nota_n):
            """Agrega las filas de terciles de x (y «sin lactosa» si corresponde)."""
            nonlocal filas
            if es_lac:
                filas += caa(an, fam, "", base, f"CAA art. 1372 bis ({URL_XVII})", [SIN_LAC],
                             "<= 0,1 g (100 mg) por 100 g o 100 ml", estado=estado)
            clases, nota = terciles(x)
            if es_lac:
                clases = [(c, max(a, SIN_LAC[2]), b) for c, a, b in clases]
            for c, a, b in clases:
                filas.append(dict(analisis=an, familia=fam, subtipo="", estado=estado, categoria=c, desde=a, hasta=b,
                                  base=base, fuente=fuente, verificado="no",
                                  nota=nota_n + (f"; {nota}" if nota else "")))

        globales_usadas = []
        for fam in familias:
            m = ok & (d["desc_familia"] == fam)
            if (an, fam, "") in CAA_CLAVES:
                continue
            if fam == "manteca / crema" and an == "materia_grasa":
                m = m & ~d["desc_subtipo"].isin(SUB_CREMA)   # la crema va por CAA (subtipo); la manteca, terciles
            por_estado = 0
            if fam in FAMILIAS_CON_ESTADO:
                for est in ESTADOS:
                    if (an, fam, est) in CAA_CLAVES_ESTADO:
                        por_estado += 1
                        continue
                    x = v[m & est_d.eq(est)]
                    if es_lac:
                        x = x[x >= SIN_LAC[2]]
                    n_sen, fuente = len(x), "terciles por estado (Senasa encolumnado.xlsx)"
                    if n_sen < MIN_N and v_mq is not None:
                        xm = v_mq[v_mq.notna() & mq["desc_familia"].eq(fam) & est_mq.eq(est)]
                        if es_lac:
                            xm = xm[xm >= SIN_LAC[2]]
                        if len(xm):
                            x = pd.concat([x, xm])
                            fuente = "terciles por estado (Senasa + MQ encolumnado.xlsx)"
                    if len(x) < MIN_N:
                        if len(x):
                            log_estado.append(f"{an} · {fam} · {est}: {len(x)} valores, sin cortes")
                        continue
                    emitir(x, fam, est, fuente, f"n = {len(x)}" + (f" ({n_sen} de Senasa)" if len(x) != n_sen else ""))
                    por_estado += 1
            if por_estado:           # la familia se corta por estado: no lleva terciles con los estados mezclados
                continue
            x = v[m]
            if es_lac:   # «sin lactosa» (CAA art. 1372 bis) primero; terciles sobre lo que tiene lactosa
                x = x[x >= SIN_LAC[2]]
            if not len(v[m]):
                continue
            if len(x) < MIN_N:
                globales_usadas.append(f"{fam} ({len(x)})")
                continue
            emitir(x, fam, "", "terciles (Senasa encolumnado.xlsx)", f"n = {len(x)}")
        # cortes globales para las familias con menos de 30 valores
        xg = v[ok]
        if es_lac:
            xg = xg[xg >= SIN_LAC[2]]
        if globales_usadas and len(xg) >= MIN_N:
            clases, nota = terciles(xg)
            if es_lac:
                clases = [(c, max(a, SIN_LAC[2]), b) for c, a, b in clases]
            for c, a, b in clases:
                filas.append(dict(analisis=an, familia="*", subtipo="", estado="", categoria=c, desde=a, hasta=b, base=base,
                                  fuente="terciles globales (Senasa encolumnado.xlsx)", verificado="no",
                                  nota=f"n = {len(xg)} (todas las familias); se usa en: " + ", ".join(globales_usadas)
                                       + (f"; {nota}" if nota else "")))
            if es_lac:
                filas += caa(an, "*", "", base, f"CAA art. 1372 bis ({URL_XVII})", [SIN_LAC],
                             "<= 0,1 g (100 mg) por 100 g o 100 ml")
        log.append((an, globales_usadas))

    t = pd.DataFrame(filas)
    t = t[["analisis", "familia", "subtipo", "estado", "categoria", "desde", "hasta", "base", "fuente", "verificado", "nota"]]
    t = t.drop_duplicates(subset=["analisis", "familia", "subtipo", "estado", "categoria"])
    for c in ("desde", "hasta"):
        t[c] = t[c].map(lambda z: "inf" if np.isinf(z) else format(z, ".6g"))
    orden = list(ANALISIS)
    t["_o"] = t["analisis"].map(orden.index)
    t["_d"] = t["desde"].replace("inf", "1e18").astype(float)
    t = t.sort_values(["_o", "familia", "subtipo", "estado", "_d"]).drop(columns=["_o", "_d"])
    t.to_csv(SALIDA, index=False, encoding="utf-8")
    print(f"{len(t)} filas -> {SALIDA}")
    for an, g in log:
        if g:
            print(f"  {an}: cortes globales para {', '.join(g)}")
    pe = t[t["estado"] != ""].drop_duplicates(["analisis", "familia", "estado"])
    print(f"cortes por estado: {len(pe)} juegos (análisis × familia × estado)")
    for linea in log_estado:
        print("  sin cortes por estado:", linea)
    return t


def asegurar():
    """Ruta de cortes_categorias.csv; lo arma si falta. Los cuadernos de clustering lo leen antes de la sección que lo
    regenera, así que en una carpeta recién descomprimida (sin resultados/) tiene que poder armarse solo."""
    if not SALIDA.exists():
        print(f"Falta {SALIDA.name}: se arma ahora con cortes_caa.armar().")
        armar()
    return SALIDA


main = armar

if __name__ == "__main__":
    armar()
