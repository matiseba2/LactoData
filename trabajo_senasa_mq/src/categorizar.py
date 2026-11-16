"""Pasa los análisis numéricos de las muestras a categorías e ítems para reglas de asociación (estilo clase 44).

Uso: sección 6 de trabajo_senasa_mq.ipynb (o `python src/categorizar.py` desde trabajo_senasa_mq/).

Salidas (en resultados/tablas/):
    categorias_senasa.csv, categorias_senasa_mq.csv            una fila por muestra: valores, cat_*, req_*, items
    transacciones_senasa.parquet, transacciones_senasa_mq.parquet   matriz one-hot (muestra × ítem, booleana)

Qué hace:
  - Lee con datos_clase41.py (correcciones, «,.», NO DETECTADO y «< LD» = 0, exclusión humedad + sólidos, unión
    Senasa + MQ con Senasa al mando). No repite esa lógica: solo agrega «No contiene» = 0 (4 celdas de grasas trans),
    como dice cortes_categorias.md.
  - Aplica cortes_categorias.csv con el juego más específico: familia + subtipo, si no familia, si no «*».
    Intervalos desde <= valor < hasta.
  - Devuelve cat_<analisis>, req_<requisito> (cumple / no cumple) y la lista de ítems «analisis=categoria»,
    «familia=…», «tag=…» y, cuando el segundo nivel le da uno, «subtag=…». Si una muestra tiene un valor y no tiene categoría, el motivo queda en «sin_categoria».

Las decisiones están en las constantes de abajo; se pueden cambiar sin tocar el resto.
No se hacen terciles por subtipo: los cortes son los de cortes_categorias.csv (por familia y, en las familias que
juntan polvo y líquido o manteca y crema, por estado: columna «estado», de cortes_caa.estado_para_cortes).
"""
import numpy as np
import pandas as pd

import rutas as R
from cortes_caa import estado_para_cortes
from datos_clase41 import SIN_FAMILIA, a_numero, leer_mq, leer_senasa, unir

RUTA_CORTES = R.TABLAS / "cortes_categorias.csv"

# ================================================================ decisiones (cambiables)
# Humedad del queso: True = las 4 clases del CAA (art. 605); False = 3 clases, «muy alta» se junta con «alta»
# (como desc_consistencia).
HUMEDAD_QUESO_4_CLASES = True
# Análisis que se categorizan (cat_*) pero NO entran como ítems: redundantes con la humedad (≈ 100 − humedad).
# desc_consistencia tampoco entra: se deja como columna.
NO_SON_ITEMS = ["solidos_totales"]
# Familias con menos de 30 valores: usan los cortes globales («*») de la tabla. False = quedan sin categoría.
USAR_CORTES_GLOBALES = True
# Muestras solo MQ (sin familia): True = cortes globales; False = sin categoría.
GLOBALES_PARA_SIN_FAMILIA = True
# Dulce de leche con grasa < 6: conserva su categoría («bajo el mínimo (<6)») y se marca en una columna.
MARCAR_DDL_GRASA_BAJO_MINIMO = True
# Lactosa «bajo / reducido / normal» (leche fluida): los cortes de la tabla usan la referencia 4,83 g/100 g
# (mediana de la leche cruda del proyecto). El CAA no da ese valor: se marca como no verificado.
LACTOSA_REFERENCIA = 4.83
CAT_LACTOSA_SIN_VERIFICAR = {"bajo en lactosa", "reducido en lactosa", "lactosa normal"}
# Subtipo como ítem («subtipo=…»). No pedido; apagado.
SUBTIPO_COMO_ITEM = False
# Subtag del agrupamiento (clusters_*.csv; segundo_nivel.py) como ítem aparte: «tag=DDL» se mantiene para todo el DDL
# (el primer nivel) y se suma «subtag=DDL_REPOSTERO», «subtag=POLVO_WPC»... solo donde el subtag es distinto del tag.
# Para las reglas, tag y subtag son el mismo «análisis»: tag ⇒ subtag (o al revés) es trivial («mismo análisis»).
SUBTAG_COMO_ITEM = True
# Subtags que no entran como ítem: el del queso ES la clase de humedad del CAA (los mismos cortes del art. 605), que ya
# está como «humedad=…»; sumarlo daría reglas humedad ⇒ subtag que repiten la definición. Queda en la columna del CSV.
SUBTAG_SIN_ITEM = ("QUESO_",)

# Requisitos del CAA que no son clases (cortes_categorias.md, decisión 3): ítem «req_<id>=cumple / no cumple».
# (id, analisis, familia, subtipos o None, "max" | "min", limite, fuente). El laboratorio informa 2 decimales:
# «hasta X» se evalúa como valor < X + 0,005, y «mínimo X» como valor >= X.
REQUISITOS = [
    ("ddl_humedad_max_30", "humedad", "dulce de leche", None, "max", 30.0, "CAA art. 592"),
    ("ddl_proteinas_min_5", "proteinas", "dulce de leche", None, "min", 5.0, "CAA art. 592"),
    ("ddl_cenizas_max_2", "cenizas", "dulce de leche", None, "max", 2.0, "CAA art. 592"),
    ("polvo_humedad_max_5", "humedad", "leche en polvo", None, "max", 5.0, "CAA art. 567"),
    ("manteca_grasa_min_82", "materia_grasa", "manteca / crema", ["MANTECA"], "min", 82.0, "CAA art. 596"),
    ("manteca_humedad_max_16", "humedad", "manteca / crema", ["MANTECA"], "max", 16.0, "CAA art. 596"),
]

# ================================================================ análisis y columnas de origen
# Mismas columnas que build_cortes_categorias.py (las dos galactosas, las dos lactosas en g/100 ml y los tres
# azúcares totales se juntan: nunca hay dos en la misma muestra).
ANALISIS = {
    "humedad": [], "materia_grasa": [], "proteinas": [], "cenizas": [], "solidos_totales": [],   # de datos_clase41
    "lactosa": ["Resultado Lactosa g /100g (*)"],
    "lactosa_g100ml": ["Resultado Contenido de Lactosa % (g/ 100ml) (*)",
                       "Resultado Determinación del contenido de lactosa (g/ 100ml) (*)"],
    "galactosa": ["Resultado D - Galactosa g/ 100g (*)", "Resultado Galactosa g/ 100g (*)"],
    "mg_extracto_seco": ["Resultado Materia grasa en extracto seco g/ 100g (*)_x"],
    "actividad_agua": ["Resultado Actividad de agua (% Aw) (*)"],
    "acido_sorbico": ["Resultado Acido Sórbico mg/ Kg (*)"],
    "carbohidratos_totales": ["Resultado Carbohidratos Totales g/ 100g (*)"],
    "azucares_totales": ["Resultado Azucares totales g/100gr (*) (fructosa/glucosa/sacarosa/maltosa/lactosa)_x",
                         "Resultado Azucares totales g/100gr (*) (fructosa/glucosa/sacarosa/maltosa/lactosa/galactosa)",
                         "Resultado Azucares totales g/100gr (*)"],
    "acidez_naoh": ["Resultado Acidez (ml NaOH 0,1 N/ 10 g sólidos no grasos) (*)"],
    "solidos_no_grasos": ["Resultado Sólidos No Grasos de leche % (*)"],
    "sulfitos": ["Resultado Sulfitos ppm SO₂ (*)"],
    "grasas_saturadas": ["Resultado Grasas saturadas g% (*)"],
    "grasas_trans": ["Resultado Grasas trans g% (*)"],
    "sodio": ["Resultado Sodio mg /100g (*)"],
    "fibra": ["Resultado Fibra Alimentaria Total g/ 100g (*)"],
    "plomo": ["Resultado Plomo ppm (*)"],
    "arsenico": ["Resultado Arsénico ppm (*)"],
}
ANALISIS_LISTA = list(ANALISIS)
# tope físico: g/100 g no puede pasar de 100 (p. ej. humedad «4548» por 45,48); aw no puede pasar de 1
TOPE = {a: 100.0 for a in ["humedad", "materia_grasa", "proteinas", "cenizas", "solidos_totales", "lactosa",
                           "lactosa_g100ml", "galactosa", "mg_extracto_seco", "carbohidratos_totales",
                           "azucares_totales", "solidos_no_grasos", "grasas_saturadas", "grasas_trans", "fibra"]}
TOPE["actividad_agua"] = 1.0
DE_DATOS_CLASE41 = {"humedad": "humedad", "materia_grasa": "grasa", "proteinas": "proteinas", "cenizas": "cenizas",
                    "solidos_totales": "solidos_senasa"}
MOTIVO_EXCLUIDA = "muestra excluida: humedad + sólidos fuera de 100 ± 1"


# ================================================================ cortes
def leer_cortes(ruta=RUTA_CORTES):
    if ruta == RUTA_CORTES:
        import cortes_caa
        cortes_caa.asegurar()
    c = pd.read_csv(ruta, dtype={"subtipo": str, "estado": str}, keep_default_na=False)
    if "estado" not in c:
        c["estado"] = ""
    c["desde"] = c["desde"].astype(float)
    c["hasta"] = c["hasta"].replace("inf", np.inf).astype(float)
    if not HUMEDAD_QUESO_4_CLASES:
        m = (c["analisis"] == "humedad") & (c["familia"] == "queso")
        alta = m & (c["categoria"] == "alta humedad"); muy = m & (c["categoria"] == "muy alta humedad")
        c.loc[alta, "hasta"] = np.inf
        c = c[~muy]
    return c.reset_index(drop=True)


def juego_de_cortes(cortes, analisis, familia, subtipo, estado=""):
    """Filas del juego más específico (familia + subtipo, familia + estado, familia, «*») y de dónde salió;
    (None, motivo) si no hay. Si la familia tiene cortes por estado, la muestra sin estado no se categoriza
    (no se la manda a terciles con los estados mezclados ni a los globales)."""
    c = cortes[cortes["analisis"] == analisis]
    f = c[c["familia"] == familia]
    if pd.notna(subtipo) and subtipo != "":
        s = f[f["subtipo"].map(lambda x: subtipo in [t.strip() for t in x.split(";")] if x else False)]
        if len(s):
            return s, "familia + subtipo"
    fe = f[f["estado"] != ""]
    if len(fe):
        if not estado:
            return None, "la familia tiene cortes por estado y la muestra no tiene estado (ni humedad medida ni declarado)"
        s = fe[(fe["estado"] == estado) & (fe["subtipo"] == "")]
        if len(s):
            return s, "familia + estado"
        return None, f"sin cortes de {analisis} para el estado {estado} de «{familia}» (menos de 30 valores)"
    s = f[f["subtipo"] == ""]
    if len(s):
        return s, "familia"
    g = c[c["familia"] == "*"]
    if not len(g):
        return None, f"sin cortes de {analisis} para la familia «{familia}» ni cortes globales"
    if familia == SIN_FAMILIA and not GLOBALES_PARA_SIN_FAMILIA:
        return None, "sin familia (solo MQ): cortes globales desactivados"
    if not USAR_CORTES_GLOBALES:
        return None, f"familia «{familia}» sin cortes propios: cortes globales desactivados"
    return g, "globales (*)"


# ================================================================ categorizar
def categorizar(tabla, cortes=None, excluidas=None):
    """tabla: una fila por muestra, con «familia», «subtipo» y una columna numérica por análisis (nombres de ANALISIS;
    las que falten se ignoran). Opcionales: «tag» y «subtag» (se agregan como ítems; el subtag solo si es distinto del
    tag y SUBTAG_COMO_ITEM). excluidas: índice booleano de muestras que no
    se categorizan (motivo MOTIVO_EXCLUIDA).

    Devuelve un DataFrame con el mismo índice y las columnas:
      cat_<analisis>, juego_<analisis> (de dónde salió el corte), req_<id>, ddl_grasa_bajo_minimo,
      lactosa_sin_verificar, sin_categoria (análisis: motivo; …) e items (lista).
    """
    cortes = leer_cortes() if cortes is None else cortes
    excluidas = pd.Series(False, index=tabla.index) if excluidas is None else excluidas.reindex(tabla.index).fillna(False)
    fam = tabla["familia"].fillna("sin clasificar")
    sub = tabla["subtipo"] if "subtipo" in tabla else pd.Series(np.nan, index=tabla.index)
    est = tabla["estado"].fillna("") if "estado" in tabla else pd.Series("", index=tabla.index)
    out = pd.DataFrame(index=tabla.index)
    motivos = {i: [] for i in tabla.index}
    presentes = [a for a in ANALISIS_LISTA if a in tabla]
    for an in presentes:
        v = tabla[an]
        cat = pd.Series(pd.NA, index=tabla.index, dtype="object"); juego = cat.copy()
        tiene = v.notna()
        for i in tabla.index[tiene & excluidas]:
            motivos[i].append(f"{an}: {MOTIVO_EXCLUIDA}")
        fuera = tiene & ~excluidas & (v > TOPE.get(an, np.inf) + 1e-9)
        neg = tiene & ~excluidas & (v < 0)
        for i in tabla.index[fuera]:
            motivos[i].append(f"{an}: valor {v[i]:g} fuera del rango físico (> {TOPE[an]:g})")
        for i in tabla.index[neg]:
            motivos[i].append(f"{an}: valor negativo {v[i]:g}")
        ok = tiene & ~excluidas & ~fuera & ~neg
        for (f, s, e), idx in tabla[ok].groupby([fam[ok], sub[ok].fillna(""), est[ok]], dropna=False).groups.items():
            filas, origen = juego_de_cortes(cortes, an, f, s, e)
            if filas is None:
                for i in idx:
                    motivos[i].append(f"{an}: {origen}")
                continue
            x = v[idx]
            asignada = pd.Series(pd.NA, index=idx, dtype="object")
            for r in filas.itertuples():
                asignada[(x >= r.desde) & (x < r.hasta)] = r.categoria
            cat[idx] = asignada; juego[idx] = np.where(asignada.notna(), origen, pd.NA)
            for i in idx[asignada.isna().values]:
                motivos[i].append(f"{an}: valor {v[i]:g} fuera de los cortes")
        out[f"cat_{an}"] = cat; out[f"juego_{an}"] = juego

    # requisitos del CAA sin clases
    for rid, an, f, subs, tipo, lim, _ in REQUISITOS:
        r = pd.Series(pd.NA, index=tabla.index, dtype="object")
        if an in tabla:
            m = (fam == f) & out[f"cat_{an}"].notna()
            if subs is not None:
                m &= sub.isin(subs)
            cumple = tabla[an] < lim + 0.005 if tipo == "max" else tabla[an] >= lim
            r[m] = np.where(cumple[m], "cumple", "no cumple")
        out[f"req_{rid}"] = r

    out["ddl_grasa_bajo_minimo"] = ((fam == "dulce de leche") & out["cat_materia_grasa"].isin(["bajo el mínimo (<6)"])
                                    ) if MARCAR_DDL_GRASA_BAJO_MINIMO and "cat_materia_grasa" in out else False
    lac = [c for c in ("cat_lactosa", "cat_lactosa_g100ml") if c in out]
    out["lactosa_sin_verificar"] = out[lac].isin(CAT_LACTOSA_SIN_VERIFICAR).any(axis=1) if lac else False
    out["sin_categoria"] = pd.Series({i: "; ".join(m) for i, m in motivos.items()}).replace("", pd.NA)

    # ítems
    cols_item = [f"cat_{a}" for a in presentes if a not in NO_SON_ITEMS] + [f"req_{r[0]}" for r in REQUISITOS]
    def items_de(i):
        it = [f"{c[4:] if c.startswith('cat_') else c}={out.at[i, c]}" for c in cols_item if pd.notna(out.at[i, c])]
        if not it:
            return []                     # sin análisis categorizado: no es una transacción
        it.append(f"familia={fam[i]}")
        if SUBTIPO_COMO_ITEM and pd.notna(sub[i]) and sub[i] != "":
            it.append(f"subtipo={sub[i]}")
        if "tag" in tabla and pd.notna(tabla.at[i, "tag"]):
            it.append(f"tag={tabla.at[i, 'tag']}")
            st = tabla.at[i, "subtag"] if "subtag" in tabla else np.nan
            if SUBTAG_COMO_ITEM and pd.notna(st) and st != tabla.at[i, "tag"] and not str(st).startswith(SUBTAG_SIN_ITEM):
                it.append(f"subtag={st}")
        return it
    out["items"] = [items_de(i) for i in tabla.index]
    return out


# ================================================================ lectura de las dos versiones
def _valores_senasa(S):
    d = S.crudo
    t = pd.DataFrame({"Identificador": d["Identificador"], "familia": d["desc_familia"], "subtipo": d["desc_subtipo"],
                      "producto": d["desc_producto"], "desc_consistencia": d["desc_consistencia"],
                      "desc_estado": d["desc_estado"]})
    for an, cols in ANALISIS.items():
        if an in DE_DATOS_CLASE41:
            t[an] = d[DE_DATOS_CLASE41[an]]
            continue
        partes = []
        for c in cols:
            texto = d[c].astype("string").str.strip()
            x = a_numero(d[c])
            x = x.where(~texto.str.lower().eq("no contiene").fillna(False), 0.0)   # «No contiene» = 0 (cortes_categorias.md)
            partes.append(x)
        t[an] = pd.concat(partes, axis=1).bfill(axis=1).iloc[:, 0]
    return t


def _tags(ruta, ids):
    cl = pd.read_csv(ruta, encoding="utf-8-sig")
    if "rama" not in cl:
        cl["rama"] = "k-medias"
    cl = cl[["Identificador", "tag", "subtag", "outlier", "rama"]]
    m = pd.DataFrame({"Identificador": ids}).merge(cl, on="Identificador", how="left")
    # los quesos de la rama propia llevan tag QUESO pero no pasaron por el k-medias de cuatro análisis
    estado = np.where(m["rama"].notna() & m["rama"].ne("k-medias"), "con tag (rama de queso)",
             np.where(m["tag"].notna(), "con tag",
                      np.where(m["outlier"].eq(True), "outlier (sin tag)",
                               "no agrupada (faltan análisis, suma > 102 o excluida)")))
    return m["tag"].values, m["subtag"].values, estado


def armar(version="senasa"):
    """version «senasa» o «senasa_mq». Devuelve la tabla completa (una fila por muestra)."""
    S = leer_senasa()
    t = _valores_senasa(S)
    if version == "senasa":
        t["origen"] = "Senasa"
        ruta_cl = R.TABLAS / "clusters_senasa.csv"
    else:
        _, mq = leer_mq()
        base = t.rename(columns={"familia": "desc_familia", "producto": "desc_producto", "materia_grasa": "grasa",
                                 "subtipo": "desc_subtipo"})
        t, _ = unir(base, mq, extra=("desc_subtipo", "desc_estado"))
        t = t.rename(columns={"desc_familia": "familia", "desc_producto": "producto", "grasa": "materia_grasa",
                              "desc_subtipo": "subtipo"})
        ruta_cl = R.TABLAS / "clusters_senasa_mq.csv"
    t = t.reset_index(drop=True)
    # estado para elegir los cortes: humedad medida (o 100 − sólidos) y, si no, el declarado
    hum = t["humedad"].fillna(100 - t["solidos_totales"]) if "solidos_totales" in t else t["humedad"]
    t["estado"] = estado_para_cortes(t["familia"], t["desc_estado"], t["subtipo"], hum.where((hum >= 0) & (hum <= 100)))
    t["tag"], t["subtag"], t["estado_cluster"] = _tags(ruta_cl, t["Identificador"])
    excl = t["Identificador"].isin(S.excluidas)
    cat = categorizar(t, excluidas=excl)
    salida = pd.concat([t[["Identificador", "familia", "subtipo", "estado", "producto", "origen", "tag", "subtag",
                           "estado_cluster", "desc_consistencia"] + ANALISIS_LISTA], cat], axis=1)
    salida["excluida"] = excl
    return salida


def matriz_transaccional(items, index=None):
    """Lista de listas de ítems → matriz one-hot booleana (TransactionEncoder, como en la clase 44).
    Las muestras sin ítems no son transacciones y no entran."""
    from mlxtend.preprocessing import TransactionEncoder
    items = pd.Series(list(items), index=index)
    items = items[items.map(len) > 0]
    te = TransactionEncoder()
    M = pd.DataFrame(te.fit(items.tolist()).transform(items.tolist()), columns=te.columns_, index=items.index)
    return M


# ================================================================ reglas: filtro de las triviales
def analisis_de_item(item):
    """«humedad=alto» → humedad; «req_ddl_humedad_max_30=cumple» → humedad; «familia=…» → familia; «tag=…» y
    «subtag=…» → tag (el subtag es el segundo nivel del mismo agrupamiento)."""
    clave = item.split("=", 1)[0]
    if clave == "subtag":
        return "tag"
    if clave.startswith("req_"):
        return next(r[1] for r in REQUISITOS if f"req_{r[0]}" == clave)
    return clave


# Análisis que son una cuenta con otros: una regla que junta el derivado con TODAS sus entradas repite la cuenta.
DERIVADOS = {"mg_extracto_seco": {"materia_grasa", "humedad"},
             "carbohidratos_totales": {"humedad", "materia_grasa", "proteinas", "cenizas"}}
LIFT_MIN = 1.2          # elevación ajustada mínima para que una regla no se considere trivial


def filtrar_reglas(reglas, M, lift_min=LIFT_MIN):
    """Agrega a las reglas de association_rules las columnas familias, lift_ajustado y motivo_trivial (vacío = útil).

    lift_ajustado: la elevación calculada solo sobre las muestras que pueden tener la regla: las de las familias donde
    aparecen todos sus ítems (si el consecuente es familia=X, donde aparecen los del antecedente) y que tienen medidos
    todos sus análisis (y tag, si la regla tiene un tag o un subtag). El lift global mezcla dos efectos que no dicen nada: que una categoría sea propia de una
    familia (las clases del CAA) y qué análisis se midieron en cada muestra (cenizas y proteínas van juntas en el panel).

    Triviales:
      - «mismo análisis»: un ítem del consecuente es del mismo análisis que uno del antecedente (p. ej. humedad=alto ⇒
        req_ddl_humedad_max_30=no cumple: repite la definición del corte; o subtag=DDL_HELADERO ⇒ tag=DDL);
      - «cuenta»: junta un análisis derivado con todas sus entradas (mg en extracto seco con grasa y humedad);
      - «define la familia»: el consecuente es familia=X y el antecedente tiene un ítem que solo aparece en X
        (p. ej. materia_grasa=helado de crema ⇒ familia=helado);
      - «familia redundante»: tiene familia=X junto con un ítem que solo aparece en X;
      - «cortes globales en la familia»: tiene familia=X y un análisis que en X se categoriza con los cortes globales
        (familias chicas y muestras solo MQ): la categoría no se compara con la de las otras familias;
      - «explicada por la familia o el panel»: lift_ajustado < lift_min.
    """
    cols_fam = [c for c in M.columns if c.startswith("familia=")]
    fam_de_fila = M[cols_fam].idxmax(axis=1).str[len("familia="):]
    familias_item = {c: frozenset(fam_de_fila[M[c]].unique()) for c in M.columns}
    an_col = {c: analisis_de_item(c) for c in M.columns}
    medido = {a: M[[c for c in M.columns if an_col[c] == a]].any(axis=1) for a in set(an_col.values())}
    c = leer_cortes()
    con_cortes = {a: set(g["familia"]) - {"*"} for a, g in c.groupby("analisis")}   # familias con cortes propios
    r = reglas.copy()
    filas = []
    for A, B, lift in zip(r["antecedents"], r["consequents"], r["lift"]):
        anA = {an_col[i] for i in A}; anB = {an_col[i] for i in B}
        posibles = frozenset.intersection(*(familias_item[i] for i in A | B))
        famB = [i[len("familia="):] for i in B if i.startswith("familia=")]
        fams = [i[len("familia="):] for i in A | B if i.startswith("familia=")]
        S = pd.Series(True, index=M.index)
        for a in anA | anB:
            S &= medido[a]
        if famB:   # la base son las familias donde puede aparecer el antecedente (sin contar sus ítems de familia)
            noF = [familias_item[i] for i in A if not i.startswith("familia=")]
            if noF:
                S &= fam_de_fila.isin(frozenset.intersection(*noF))
        else:
            S &= fam_de_fila.isin(posibles)
        nA = M.loc[S, list(A)].all(axis=1); nB = M.loc[S, list(B)].all(axis=1)
        lift_aj = (nA & nB).mean() / (nA.mean() * nB.mean()) if nA.any() and nB.any() else np.nan
        motivo = None
        if (anA & anB) - {"familia"}:      # tag y subtag cuentan como el mismo análisis
            motivo = "mismo análisis"
        elif any(d in anA | anB and ent <= anA | anB for d, ent in DERIVADOS.items()):
            motivo = "cuenta"
        elif famB and any(familias_item[i] == {famB[0]} for i in A):
            motivo = "define la familia"
        elif fams and any(familias_item[i] == {fams[0]} for i in A | B if not i.startswith("familia=")):
            motivo = "familia redundante"
        elif any(an_col[i] in con_cortes and f not in con_cortes[an_col[i]] for f in fams for i in A | B):
            motivo = "cortes globales en la familia"
        elif not lift_aj >= lift_min:
            motivo = "explicada por la familia o el panel"
        filas.append(("; ".join(sorted(posibles)) if len(posibles) <= 2 else f"{len(posibles)} familias", lift_aj, motivo))
    r["familias"], r["lift_ajustado"], r["motivo_trivial"] = zip(*filas) if filas else ([], [], [])
    return r


# ================================================================ script
COLUMNAS_CSV_FUERA = []      # las columnas juego_<analisis> (de dónde salió cada corte) quedan en el CSV


def guardar(version):
    """Arma la tabla de una versión, escribe categorias_<version>.csv y transacciones_<version>.parquet.
    Devuelve (tabla, matriz)."""
    t = armar(version)
    M = matriz_transaccional(t["items"], index=t["Identificador"])
    csv = t.drop(columns=COLUMNAS_CSV_FUERA).assign(items=t["items"].map(" | ".join))
    csv.to_csv(R.TABLAS / f"categorias_{version}.csv", index=False, encoding="utf-8-sig")
    M.to_parquet(R.TABLAS / f"transacciones_{version}.parquet")
    return t, M


def main():
    for version in ("senasa", "senasa_mq"):
        t, M = guardar(version)
        n_val = int(t[ANALISIS_LISTA].notna().sum().sum())
        n_cat = int(t[[f"cat_{a}" for a in ANALISIS_LISTA]].notna().sum().sum())
        print(f"{version}: {len(t)} muestras · {len(M)} transacciones · {M.shape[1]} ítems distintos · "
              f"{n_cat}/{n_val} valores categorizados ({100 * n_cat / n_val:.2f} %)")


if __name__ == "__main__":
    main()
