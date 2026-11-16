"""Excel de inconsistencias para revisión manual (ex exportar_dataset_senasa_mq.py; original en _archivo/proyecto/).

Agrupa por su cuenta, con la misma técnica que las secciones 3 y 4 de trabajo_senasa_mq.ipynb (lectura, unión con MQ,
filtro suma <= 102, outliers por consenso dentro del k-medias preliminar, k-medias final con nombres de grupo, tag y
subtag; con las funciones de mejoras_clustering.py), pero NO reproduce el agrupamiento del cuaderno único: parte de otra
población, que incluye los no lácteos y los quesos (el cuaderno los deja afuera de este agrupamiento), y por eso el K
que elige la silueta es otro (K = 6). Los grupos de este Excel sirven para detectar inconsistencias, no son los del
cuaderno. A eso se agregan las reglas de inconsistencia.

    exportar()        # sección 2 del notebook; o `python src/revision_excel.py` desde trabajo_senasa_mq/

Salidas (hojas dataset, inconsistencias, resumen, notas), en resultados/excel/:
  dataset_senasa_mq_revision.xlsx  Senasa + MQ, como el notebook
  dataset_senasa_revision.xlsx     solo Senasa, sin completar nada con MQ
Los textos de la hoja «notas» conservan las rutas de antes de la mudanza (proyecto/...), para que el Excel sea idéntico.
"""
import re

import numpy as np
import pandas as pd
from openpyxl.utils import get_column_letter
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

import mejoras_clustering as M
import rutas as R
from datos_clase41 import (COL_SOLIDOS, COLUMNAS, N_INIT_KMEDIAS, ORIGEN, SEMILLA, SIN_FAMILIA, TOL_HUM_SOL, a_numero, etiquetar,
                           leer_mq, leer_senasa, unir)

SALIDA = R.EXCEL / "dataset_senasa_mq_revision.xlsx"
SALIDA_SENASA = R.EXCEL / "dataset_senasa_revision.xlsx"
MQ_COL = {"humedad": "solidos", "grasa": "grasa", "proteinas": "proteina", "cenizas": "ceniza"}

# Tolerancias de las reglas (TOL_HUM_SOL = 1: humedad + sólidos totales de Senasa debe dar 100 ± 1, en datos_clase41.py)
TOL_ABS, TOL_REL = 0.5, 0.05        # Senasa vs MQ: difieren si |dif| > 0,5 g/100 g Y |dif| > 5 % del mayor
CERO = 1.0                          # "todo cerca de 0": todos los análisis disponibles (al menos 2) <= 1 g/100 g


def exportar():
    """Escribe los dos Excel de revisión en resultados/excel/."""
    R.preparar()
    # ---------------------------------------------------------------- 1. lectura y unión (datos_clase41.py, la misma que el notebook)
    S = leer_senasa()
    crudo, crudo_original, correcciones, EXCLUIDAS, CELDAS_TIPEO = S.crudo, S.crudo_original, S.correcciones, S.excluidas, S.celdas_tipeo
    CORREGIDAS = correcciones.groupby("Identificador").apply(
        lambda g: " | ".join(f"{r.campo}: {r.antes} → {r.despues if pd.notna(r.despues) else 'anulado'}" for r in g.itertuples()),
        include_groups=False)
    senasa = crudo[["Identificador", "desc_familia", "desc_producto"] + COLUMNAS].copy()
    extra_senasa = crudo[["Identificador", "Descripción", "solidos_senasa"] + COLUMNAS].rename(
        columns={c: f"{c}_senasa" for c in COLUMNAS})
    mq_crudo, mq = leer_mq()
    todas, _ = unir(senasa, mq)
    assert todas["Identificador"].is_unique
    todas["excluida_hum_solidos"] = todas["Identificador"].isin(EXCLUIDAS)

    # ---------------------------------------------------------------- 2. filtro, outliers y k-medias final
    todas["tiene_4_analisis"] = todas[COLUMNAS].notna().all(axis=1)
    todas["suma_4"] = todas[COLUMNAS].sum(axis=1, min_count=4)
    todas["supera_102"] = todas["tiene_4_analisis"] & (todas["suma_4"] > 102)
    completas = todas[~todas["excluida_hum_solidos"]].dropna(subset=COLUMNAS)
    muestras = completas[completas[COLUMNAS].sum(axis=1) <= 102].reset_index(drop=True)


    def escalar(df):
        imp = SimpleImputer(strategy="median").fit(df[COLUMNAS])
        esc = StandardScaler().fit(imp.transform(df[COLUMNAS]))
        return esc.transform(imp.transform(df[COLUMNAS])), esc


    def agrupar(muestras, sin_familia=SIN_FAMILIA):
        """Outliers por consenso dentro del k-medias preliminar y k-medias final, con la técnica de las secciones 3 y 4 del
        cuaderno (mismas funciones de mejoras_clustering.py) pero sobre la población de este Excel (con no lácteos y
        quesos), así que el resultado no coincide con el del cuaderno. Devuelve info por Identificador."""
        Z, _ = escalar(muestras)
        ks = list(range(2, 11))
        _, grupo_pre, k_pre = M.outliers_iqr(Z)                       # solo para el k-medias preliminar
        es_out, votos = M.outliers_consenso(muestras[COLUMNAS].to_numpy(), grupo_pre, pct=3, minimo=2, n_vecinos=50)
        metodos = votos[["factor local", "bosque", "elipse"]]
        pre_info = pd.DataFrame({"Identificador": muestras["Identificador"], "grupo_preliminar": grupo_pre,
                                 "votos_outlier": votos["votos"].to_numpy(),
                                 "metodos_outlier": metodos.apply(lambda f: ", ".join(c for c in metodos.columns if f[c]), axis=1).replace("", pd.NA).to_numpy(),
                                 "es_outlier": es_out})
        finales = muestras[~es_out].reset_index(drop=True)
        Z2, esc = escalar(finales)
        modelos = {k_: KMeans(n_clusters=k_, n_init=N_INIT_KMEDIAS, random_state=SEMILLA).fit(Z2) for k_ in ks}
        sil = [silhouette_score(Z2, modelos[k_].labels_) for k_ in ks]
        k = ks[int(np.argmax(sil))]
        grupos = modelos[k].labels_
        centros = pd.DataFrame(esc.inverse_transform(modelos[k].cluster_centers_), columns=COLUMNAS,
                               index=[f"grupo {j}" for j in range(k)]).round(2)
        nombres, tags, _ = etiquetar(centros)
        tag_de = np.array([tags[g] for g in grupos], dtype=object)
        ref = M.agregar_subtipo(finales[["Identificador", "desc_familia", "desc_producto"]], S)["subtipo_ref"]
        subtag_de, _ = M.subtags(finales[COLUMNAS].to_numpy(), tag_de, ref)
        coef = silhouette_samples(Z2, grupos)
        fam = finales["desc_familia"]; con = (fam != sin_familia).values
        moda = {j: (fam[con & (grupos == j)].mode()[0] if (con & (grupos == j)).any() else None) for j in range(k)}
        fin_info = pd.DataFrame({"Identificador": finales["Identificador"], "grupo_final_n": grupos,
                                 "grupo_final": [nombres[g] for g in grupos], "tag": tag_de, "subtag": subtag_de, "silueta": coef,
                                 "familia_mayoritaria_grupo": [moda[g] for g in grupos]})
        return pre_info, fin_info, k_pre, k


    pre_info, fin_info, K_PRE, K = agrupar(muestras)
    es_outlier = pre_info["es_outlier"].values
    finales = fin_info

    print(f"Con 4 análisis: {len(completas)} · tras filtro <=102: {len(muestras)} · K_PRE={K_PRE} · outliers: {int(es_outlier.sum())}"
          f" · quedan {len(finales)} · K={K}")

    # ---------------------------------------------------------------- 3. dataset completo
    mq_extra = pd.DataFrame({"Identificador": mq_crudo["codigo"]})
    for a, c in MQ_COL.items():
        usado = mq_crudo[c] if a != "proteinas" else mq_crudo[c].clip(lower=0)
        propio = mq_crudo[f"{c}_mq"]
        if a == "humedad":
            usado, propio = 100 - usado, 100 - propio
        mq_extra[f"{a}_mq_usado"] = usado
        mq_extra[f"{a}_mq_fuente"] = mq_crudo[f"{c}_fuente"]
        mq_extra[f"{a}_mq_medido"] = propio
    mq_extra["solidos_mq_usado"] = mq_crudo["solidos"]
    mq_extra["solidos_mq_medido"] = mq_crudo["solidos_mq"]
    mq_extra["revisar_mq"] = mq_crudo["revisar"]

    ds = (todas.merge(extra_senasa, on="Identificador", how="left")
          .merge(mq_extra, on="Identificador", how="left")
          .merge(pre_info, on="Identificador", how="left")
          .merge(fin_info, on="Identificador", how="left"))
    ds["en_senasa"] = ds["origen"] != "solo MQ"
    ds["en_mq"] = ds["Identificador"].isin(mq_crudo["codigo"])
    ds["es_outlier"] = ds["es_outlier"].astype("boolean")
    ds["en_agrupamiento_final"] = ds["grupo_final"].notna()
    ds["familia_vs_grupo_inconsistente"] = pd.Series(pd.NA, index=ds.index, dtype="boolean")
    fam_ok = ds["en_agrupamiento_final"] & (ds["desc_familia"] != SIN_FAMILIA)
    ds.loc[fam_ok, "familia_vs_grupo_inconsistente"] = (ds.loc[fam_ok, "desc_familia"] != ds.loc[fam_ok, "familia_mayoritaria_grupo"]).values


    def estado(r):
        if r["excluida_hum_solidos"]: return "descartada: humedad + sólidos fuera de 100 ± 1"
        if not r["tiene_4_analisis"]: return "incompleta (faltan análisis)"
        if r["supera_102"]: return "descartada: suma > 102"
        if r["es_outlier"] is True: return "descartada: outlier"
        return "en agrupamiento final"


    ds["estado"] = ds.apply(estado, axis=1)
    ds["correcciones"] = ds["Identificador"].map(CORREGIDAS)

    # ---------------------------------------------------------------- 4. reglas de inconsistencia
    filas = []


    def agregar(mask, regla, detalle_fn, valores_fn, df=None, destino=None):
        df = ds if df is None else df
        destino = filas if destino is None else destino
        for _, r in df[mask.fillna(False).astype(bool)].iterrows():
            destino.append({"Identificador": r["Identificador"], "desc_familia": r["desc_familia"], "origen": r["origen"],
                          "estado": r["estado"], "regla": regla, "detalle": detalle_fn(r), "valores implicados": valores_fn(r)})


    TEXTO_CLASIF = (crudo.set_index("Identificador")[["desc_subtipo", "desc_sabor"]].fillna("").astype(str)
                    .agg(" ".join, axis=1))                   # subtipo y sabor de la clasificación previa
    # Grupos sin captura (?:...): con grupos de captura, str.contains avisa con un UserWarning («has match groups»).
    FRUTAL = re.compile(r"\b(?:FRUTAL(?:ES)?|AL AGUA|DE AGUA|AGUA|SORBETES?|ICE(?! ?CREAM)|LIMON|LEMON|FRUTILLA|STRAWBERRY|RASPBERRY|FRAMBUESA|"
                        r"MARACUYA|PASSION FRUIT|NARANJA|ORANGE|DURAZNO|ANANA|MANGO|FRUTOS ROJOS)\b")


    def _texto(df):
        return (df["desc_producto"].fillna("").astype(str) + " " + df["Descripción"].fillna("").astype(str) + " "
                + df["Identificador"].map(TEXTO_CLASIF).fillna("")).str.upper()


    def es_lactosa_pura(df):
        # panificados / otros que son lactosa: todo cerca de 0 en los cuatro análisis es lo esperable
        return (df["desc_familia"] == "panificados / otros") & _texto(df).str.contains("LACTOSA")


    def es_helado_frutal(df):
        # helados frutales / al agua: son casi agua, todo cerca de 0 es lo esperable
        return (df["desc_familia"] == "helado") & _texto(df).str.contains(FRUTAL)


    def excepcion_cero(df):
        return es_lactosa_pura(df) | es_helado_frutal(df)


    def f(x):
        return "vacío" if pd.isna(x) else f"{x:.2f}".replace(".", ",")


    def cuatro(r):
        return " · ".join(f"{c}={f(r[c])}" for c in COLUMNAS)


    agregar(ds["supera_102"], "R01 suma de los cuatro > 102",
            lambda r: f"humedad + grasa + proteínas + cenizas = {f(r['suma_4'])} g/100 g (máximo admitido 102)", cuatro)

    for c in COLUMNAS:
        agregar((ds[c] < 0) | (ds[c] > 100), "R02 análisis negativo o > 100",
                lambda r, c=c: f"{c} final = {f(r[c])} fuera de [0, 100]",
                lambda r, c=c: f"{c} final={f(r[c])} · Senasa={f(r[f'{c}_senasa'])} · MQ usado={f(r[f'{c}_mq_usado'])}")

    hs = ds["humedad_senasa"] + ds["solidos_senasa"]
    ds["humedad_mas_solidos_senasa"] = hs
    agregar((hs - 100).abs() > TOL_HUM_SOL, "R03 Senasa: humedad + sólidos totales fuera de 100 ± 1",
            lambda r: f"humedad + sólidos = {f(r['humedad_mas_solidos_senasa'])} · excluida del análisis",
            lambda r: f"humedad Senasa={f(r['humedad_senasa'])} · sólidos Senasa={f(r['solidos_senasa'])}")

    for c in COLUMNAS:
        s, m = ds[f"{c}_senasa"], ds[f"{c}_mq_medido"]
        dif = (s - m).abs()
        ds[f"dif_{c}_senasa_mq"] = s - m
        agregar((dif > TOL_ABS) & (dif > TOL_REL * np.maximum(s.abs(), m.abs())), "R04 Senasa y MQ difieren",
                lambda r, c=c: f"{c}: Senasa {f(r[f'{c}_senasa'])} vs MQ medido {f(r[f'{c}_mq_medido'])} "
                               f"(dif {f(r[f'dif_{c}_senasa_mq'])}; tolerancia 0,5 g/100 g y 5 %)",
                lambda r, c=c: f"{c} Senasa={f(r[f'{c}_senasa'])} · MQ medido={f(r[f'{c}_mq_medido'])} · MQ usado={f(r[f'{c}_mq_usado'])}")

    agregar(ds["revisar_mq"].notna(), "R05 MQ con nota en revisar",
            lambda r: str(r["revisar_mq"]), cuatro)

    agregar(ds["es_outlier"] == True, "R06 outlier (k-medias preliminar)",
            lambda r: f"rara dentro del grupo preliminar {int(r['grupo_preliminar'])} para {int(r['votos_outlier'])} de 3 métodos "
                      f"({r['metodos_outlier']}; corte 3 %)", cuatro)

    agregar(ds["silueta"] < 0, "R07 silueta negativa",
            lambda r: f"silueta {r['silueta']:.3f} en el grupo «{r['grupo_final']}»: está más cerca de otro grupo", cuatro)

    agregar(ds["familia_vs_grupo_inconsistente"] == True, "R08 familia distinta de la mayoritaria de su grupo",
            lambda r: f"familia «{r['desc_familia']}» en grupo «{r['grupo_final']}», cuya familia mayoritaria es «{r['familia_mayoritaria_grupo']}»",
            lambda r: cuatro(r) + f" · producto={r['desc_producto']}")

    disp = ds[COLUMNAS].notna().sum(axis=1)
    casi_cero = (disp >= 2) & (ds[COLUMNAS].fillna(-np.inf).abs().le(CERO) | ds[COLUMNAS].isna()).all(axis=1) & ~excepcion_cero(ds)
    agregar(casi_cero, "R09 valores todos cerca de 0",
            lambda r: f"todos los análisis disponibles <= {CERO:g} g/100 g", cuatro)

    inc = pd.DataFrame(filas, columns=["Identificador", "desc_familia", "origen", "estado", "regla", "detalle", "valores implicados"])
    inc = inc.sort_values(["regla", "Identificador"], kind="stable").reset_index(drop=True)
    ds["n_reglas"] = ds["Identificador"].map(inc.groupby("Identificador")["regla"].nunique()).fillna(0).astype(int)
    ds["reglas"] = ds["Identificador"].map(inc.groupby("Identificador")["regla"].apply(lambda s: " | ".join(sorted(set(x[:3] for x in s)))))

    # ---------------------------------------------------------------- 5. hojas
    orden = (["Identificador", "desc_familia", "desc_producto", "Descripción", "origen", "estado", "en_senasa", "en_mq"]
             + COLUMNAS
             + [x for c in COLUMNAS for x in (f"{c}_senasa", f"{c}_mq_usado", f"{c}_mq_fuente", f"{c}_mq_medido", f"dif_{c}_senasa_mq")]
             + ["solidos_senasa", "solidos_mq_usado", "solidos_mq_medido", "humedad_mas_solidos_senasa", "revisar_mq",
                "tiene_4_analisis", "suma_4", "supera_102", "es_outlier", "grupo_preliminar", "votos_outlier",
                "metodos_outlier", "en_agrupamiento_final", "grupo_final", "tag", "subtag", "silueta", "familia_mayoritaria_grupo",
                "familia_vs_grupo_inconsistente", "excluida_hum_solidos", "correcciones", "n_reglas", "reglas"])
    dataset = ds[orden].copy()
    for c in dataset.select_dtypes("float").columns:
        dataset[c] = dataset[c].round(4)

    por_regla = inc.groupby("regla").agg(filas=("Identificador", "size"), identificadores=("Identificador", "nunique")).reset_index()
    por_familia = pd.crosstab(inc["desc_familia"], inc["regla"].str[:3], margins=True, margins_name="total").reset_index()
    por_familia = por_familia.rename(columns={"desc_familia": "familia"})
    conteos = pd.DataFrame([
        ("muestras unidas (Senasa + solo MQ)", len(ds)),
        ("  en Senasa", int(ds["en_senasa"].sum())), ("  solo MQ", int((ds["origen"] == "solo MQ").sum())),
        ("excluidas por humedad + sólidos de Senasa fuera de 100 ± 1", int(ds["excluida_hum_solidos"].sum())),
        ("con los cuatro análisis (sin las excluidas)", len(completas)),
        ("descartadas por suma > 102", int(ds["supera_102"].sum())),
        ("tras filtro suma <= 102", len(muestras)),
        (f"outliers (k-medias preliminar, k = {K_PRE})", int(es_outlier.sum())),
        (f"en agrupamiento final (k = {K})", len(finales)),
        ("muestras con al menos una inconsistencia", int((ds["n_reglas"] > 0).sum())),
    ], columns=["concepto", "cantidad"])
    grupos_tab = (fin_info.groupby("grupo_final").agg(muestras=("Identificador", "size"), silueta_media=("silueta", "mean"),
                                                       familia_mayoritaria=("familia_mayoritaria_grupo", "first"))
                  .round(3).reset_index())

    NOTAS = [
        ("General", "Fuente", "Reproduce proyecto/build_clase41_senasa.py (notebook clase41_agrupamiento_senasa.ipynb) sin ejecutarlo. "
         f"SEMILLA={SEMILLA}, n_init={N_INIT_KMEDIAS}. Una fila por Identificador en la hoja dataset: todas las muestras de Senasa más las que están solo en MQ."),
        ("Columna", "Identificador, desc_familia, desc_producto, Descripción", "De Senasa encolumnado.xlsx. Las muestras solo MQ tienen "
         f"desc_familia «{SIN_FAMILIA}» y sin producto ni descripción. Los clientes no se incluyen."),
        ("Columna", "origen", "Senasa: valores de Senasa (MQ pudo completar algún hueco pero sigue incompleta o ya estaba completa); "
         "Senasa completada con MQ: le faltaba algún análisis y con MQ quedó con los cuatro; solo MQ: no está en Senasa."),
        ("Columna", "estado", "incompleta (faltan análisis) / descartada: suma > 102 / descartada: outlier / en agrupamiento final."),
        ("Columna", "en_senasa, en_mq", "Si el Identificador está en Senasa encolumnado.xlsx / en muestras_ancha.csv (columna codigo)."),
        ("Columna", "humedad, grasa, proteinas, cenizas", "Valor final usado en el agrupamiento (g/100 g): el de Senasa; si Senasa no lo tiene, "
         "el de MQ (MQ nunca pisa a Senasa)."),
        ("Columna", "<análisis>_senasa", "Valor de Senasa parseado: coma decimal, se toma lo anterior a «+/-», «< LD» (o &lt;) = 0. "
         "Columnas: Resultado Humedad g/ 100g (*), Resultado Materia Grasa g/ 100g (*)_x, Resultado Proteínas g/ 100g (*), Resultado Cenizas g/ 100g."),
        ("Columna", "<análisis>_mq_usado", "Valor de la tabla ancha de MQ que usa el notebook: ceniza, grasa, proteina (con piso 0), y humedad = 100 − solidos. "
         "Ojo: la tabla ancha ya prioriza el catálogo; <análisis>_mq_fuente dice si el valor vino de 'catalogo' o de 'mq'."),
        ("Columna", "<análisis>_mq_medido", "Medición propia de MQ (columnas ceniza_mq, grasa_mq, proteina_mq, 100 − solidos_mq), sin el catálogo. "
         "Es la que se compara con Senasa en R04."),
        ("Columna", "dif_<análisis>_senasa_mq", "<análisis>_senasa − <análisis>_mq_medido."),
        ("Columna", "solidos_senasa, solidos_mq_usado, solidos_mq_medido", "Sólidos totales: Senasa (Resultado Sólidos totales g/ 100g_x), "
         "MQ tabla ancha (solidos) y MQ medición propia (solidos_mq)."),
        ("Columna", "humedad_mas_solidos_senasa", "humedad_senasa + solidos_senasa (debería dar ≈ 100)."),
        ("Columna", "revisar_mq", "Columna revisar de muestras_ancha.csv (notas del procesamiento de MQ)."),
        ("Columna", "tiene_4_analisis, suma_4, supera_102", "Si tiene los cuatro valores finales; su suma; si la suma > 102 (se descarta)."),
        ("Columna", "grupo_preliminar, votos_outlier, metodos_outlier, es_outlier",
         f"Solo para las {len(muestras)} con 4 análisis y suma <= 102. Imputar (mediana) + escalar (StandardScaler); k-medias con k de 2 a 10 "
         f"elegido por silueta (k = {K_PRE}). Dentro de cada grupo preliminar (de 30 muestras o más) se reescala y se calculan tres puntajes: "
         "factor local (LOF, 50 vecinos), bosque de aislamiento (200 árboles) y elipse (EllipticEnvelope); cada método marca el 3 % más raro "
         "del grupo; outlier si lo marcan al menos 2 de 3 (votos_outlier; metodos_outlier dice cuáles). Reemplaza a la regla Q3 + 1,5·IQR "
         "de la distancia al centroide, que sacaba casi todos los DDL heladeros."),
        ("Columna", "en_agrupamiento_final, grupo_final, tag, subtag, silueta", f"Sin outliers ({len(finales)} muestras): se reimputa y reescala, k-medias con k de 2 a 10 "
         f"por silueta (k = {K}); grupo_final es el nombre de la función nombrar sobre el centroide en g/100 g, tag el tag corto (el de "
         "clusters_senasa_mq.csv) y subtag el segundo nivel del DDL (DDL_HELADERO, DDL_FAMILIAR; el tag si no tiene); silueta = silhouette_samples."),
        ("Columna", "familia_mayoritaria_grupo, familia_vs_grupo_inconsistente", f"Moda de desc_familia en el grupo final, contando solo familia conocida "
         f"(distinta de «{SIN_FAMILIA}»; 'sin clasificar' cuenta como conocida, igual que en el notebook). Inconsistente si la familia conocida de la muestra no es esa moda; vacío para solo MQ o fuera del agrupamiento."),
        ("Columna", "excluida_hum_solidos", "Verdadero si Senasa tiene humedad y sólidos totales y su suma está fuera de 100 ± 1: queda fuera del análisis (también en el notebook)."),
        ("Columna", "correcciones", "Correcciones puntuales de correcciones_clase41.csv aplicadas a esa muestra (campo: antes → después)."),
        ("Columna", "n_reglas, reglas", "Cuántas reglas distintas dispara la muestra y sus códigos (R01..R09)."),
        ("Regla", "R01 suma de los cuatro > 102", "humedad + grasa + proteínas + cenizas (valores finales) > 102 g/100 g. Son las descartadas por el filtro."),
        ("Regla", "R02 análisis negativo o > 100", "Algún valor final < 0 o > 100 (una fila por análisis)."),
        ("Regla", "R03 Senasa: humedad + sólidos totales fuera de 100 ± 1", f"|humedad_senasa + solidos_senasa − 100| > {TOL_HUM_SOL:g}; solo si Senasa tiene ambos. Estas muestras quedan fuera del análisis."),
        ("Regla", "R04 Senasa y MQ difieren", f"Mismo Identificador con valor de Senasa y medición propia de MQ: difieren si |dif| > {TOL_ABS:g} g/100 g "
         f"y además |dif| > {TOL_REL:.0%} del mayor de los dos en valor absoluto (lo absoluto evita marcar diferencias chicas en valores bajos, "
         "lo relativo evita marcar ruido de método en valores altos como la humedad de líquidos). Una fila por análisis. "
         "Como Senasa manda, el valor final es el de Senasa: revisar cuál es el correcto."),
        ("Regla", "R05 MQ con nota en revisar", "La columna revisar de MQ no está vacía (negativos llevados a 0, valores llevados a 100, mediciones que no coinciden...)."),
        ("Regla", "R06 outlier (k-medias preliminar)", "es_outlier verdadero: rara dentro de su grupo preliminar para al menos 2 de 3 métodos (corte 3 %). Se excluye del agrupamiento final."),
        ("Regla", "R07 silueta negativa", "Coeficiente de silueta < 0 en el agrupamiento final: la muestra está más cerca, en promedio, de otro grupo."),
        ("Regla", "R08 familia distinta de la mayoritaria de su grupo", "familia_vs_grupo_inconsistente verdadero. Puede ser error de familia, de valores, "
         "o una familia que comparte composición con otra (p. ej. suero y concentrado magro): no siempre es un error."),
        ("Regla", "R09 valores todos cerca de 0", f"Al menos 2 análisis finales disponibles y todos <= {CERO:g} g/100 g (en valor absoluto): suele indicar carga en otra unidad o muestra vacía. "
         "Excepciones, donde es lo esperable: panificados / otros que son lactosa (LACTOSA en producto o descripción) y helados frutales / al agua."),
        ("Hoja", "inconsistencias", "Una fila por (Identificador, regla, y análisis cuando la regla es por análisis). estado indica si la muestra quedó en el agrupamiento."),
        ("Hoja", "resumen", "Conteos generales, filas e identificadores por regla, tabla familia × regla y tamaño/silueta por grupo final."),
    ]


    def cero_crudo(df):
        return (df[COLUMNAS].notna().sum(axis=1) >= 2) & (df[COLUMNAS].abs().le(CERO) | df[COLUMNAS].isna()).all(axis=1)


    def notas_correcciones(df):
        """Sección «Correcciones aplicadas» con cuántas filas toca cada una en el dataset del archivo."""
        ids = set(df["Identificador"])
        filas_ = []
        for r in correcciones.itertuples():
            despues = r.despues if pd.notna(r.despues) else "vacío (anulado, marcado como error de carga)"
            filas_.append(("Correcciones aplicadas", f"{r.Identificador} · {r.campo}",
                           f"{r.antes} → {despues}. Motivo: {r.motivo}. Filas: {int(r.Identificador in ids)}."))
        coma = sum(v[0] for v in CELDAS_TIPEO.values()); nd = sum(v[1] for v in CELDAS_TIPEO.values())
        filas_ += [
            ("Correcciones aplicadas", "Regla general «,.» → «,»", f"Tipeo de la coma decimal (p. ej. «54,.34» = 54,34). Celdas de Senasa resueltas: {coma}."),
            ("Correcciones aplicadas", "Regla general «NO DETECTADO» = 0", f"«NO DETECTADO» vale 0 por defecto (igual que «< LD»), en todas las celdas de resultado. Celdas de Senasa resueltas: {nd}."),
            ("Correcciones aplicadas", "Exclusión humedad + sólidos fuera de 100 ± 1",
             f"Quedan fuera del análisis (estado «descartada: humedad + sólidos...»). Filas: {int(df['Identificador'].isin(EXCLUIDAS).sum())}. "
             "Las que no tienen alguno de los dos análisis no se excluyen por esta regla."),
            ("Correcciones aplicadas", "Todo cerca de 0 en lactosa pura", f"No se marca como inconsistencia en panificados / otros que son lactosa. Filas exceptuadas: {int((es_lactosa_pura(df) & cero_crudo(df)).sum())}."),
            ("Correcciones aplicadas", "Todo cerca de 0 en helados frutales", "No se marca en familia helado cuando producto, descripción, subtipo o sabor indican frutal o al agua "
             "(FRUTAL, AL AGUA, DE AGUA, AGUA, SORBETE, ICE sin CREAM, o una fruta: LIMON/LEMON, FRUTILLA/STRAWBERRY, RASPBERRY/FRAMBUESA, MARACUYA/PASSION FRUIT, NARANJA/ORANGE, DURAZNO, ANANA, MANGO, FRUTOS ROJOS): "
             f"son casi agua. La clasificación previa no tiene subtipo para helados; las palabras salen de producto y sabor. Filas exceptuadas: {int((es_helado_frutal(df) & cero_crudo(df)).sum())}."),
            ("Correcciones aplicadas", "Regla descripción vs familia eliminada", "La heurística (antes R10/S06) no es compatible con la clasificación previa del proyecto: se quitó. "
             "Su único caso aceptado (CREMOSO DOBLE CREMA) pasó a la tabla de correcciones."),
            ("Correcciones aplicadas", "Tabla", "proyecto/correcciones_clase41.csv (Identificador, campo, antes, despues, motivo; despues vacío = anulado). La lee también "
             "build_clase41_senasa.py, así el notebook usa los mismos datos. Senasa encolumnado.xlsx y muestras_ancha.csv no se modifican."),
        ]
        return filas_


    notas = pd.DataFrame(NOTAS + notas_correcciones(ds), columns=["tipo", "nombre", "cómo se construyó"])



    def escribir(ruta, dataset, inc, bloques_resumen, notas):
        with pd.ExcelWriter(ruta, engine="openpyxl") as xw:
            dataset.to_excel(xw, sheet_name="dataset", index=False)
            inc.to_excel(xw, sheet_name="inconsistencias", index=False)
            fila = 0
            for titulo, tabla in bloques_resumen:
                pd.DataFrame({titulo: []}).to_excel(xw, sheet_name="resumen", index=False, startrow=fila)
                tabla.to_excel(xw, sheet_name="resumen", index=False, startrow=fila + 1)
                fila += len(tabla) + 4
            notas.to_excel(xw, sheet_name="notas", index=False)
            for nombre, df in [("dataset", dataset), ("inconsistencias", inc), ("notas", notas)]:
                ws = xw.sheets[nombre]
                ws.freeze_panes = "B2" if nombre == "dataset" else "A2"
                ws.auto_filter.ref = ws.dimensions
                for i, col in enumerate(df.columns, start=1):
                    largo = max([len(str(col))] + [len(str(v)) for v in df[col].head(500).tolist()])
                    ws.column_dimensions[get_column_letter(i)].width = min(max(10, largo + 2), 60 if nombre != "notas" else 110)
            ws = xw.sheets["resumen"]
            ws.freeze_panes = "A2"
            for i in range(1, 20):
                ws.column_dimensions[get_column_letter(i)].width = 14
            ws.column_dimensions["A"].width = 55


    escribir(SALIDA, dataset, inc, [("Conteos generales", conteos), ("Por regla", por_regla), ("Por grupo final", grupos_tab),
                                     ("Por familia y regla (filas de inconsistencias)", por_familia)], notas)
    print(f"Escrito: {SALIDA}")
    print(f"dataset: {len(dataset)} filas · inconsistencias: {len(inc)} filas")
    print(por_regla.to_string(index=False))


    # ================================================================ SOLO SENASA (sin completar nada con MQ)
    EXTRA_S = {"solidos_totales": COL_SOLIDOS, "lactosa": "Resultado Lactosa g /100g (*)",
               "galactosa": "Resultado Galactosa g/ 100g (*)", "d_galactosa": "Resultado D - Galactosa g/ 100g (*)"}
    s_ds = crudo[["Identificador", "Cliente", "Envío", "Descripción", "desc_familia", "desc_producto", "desc_subtipo",
                  "Fecha de Recepción"]].rename(columns={"Fecha de Recepción": "fecha_recepcion"}).copy()
    for c, col in ORIGEN.items():
        s_ds[c] = crudo[c]
        s_ds[f"{c}_texto"] = crudo_original[col]          # texto original, antes de las correcciones
        s_ds[f"{c}_corregido"] = crudo[col]
    for c, col in EXTRA_S.items():
        s_ds[c] = a_numero(crudo[col])
        s_ds[f"{c}_texto"] = crudo_original[col]
        s_ds[f"{c}_corregido"] = crudo[col]
    # las dos columnas de galactosa nunca coinciden en una misma fila: se unen
    assert not (s_ds["galactosa"].notna() & s_ds["d_galactosa"].notna()).any()
    s_ds["galactosa"] = s_ds["galactosa"].fillna(s_ds["d_galactosa"])
    s_ds["galactosa_texto"] = s_ds["galactosa_texto"].fillna(s_ds["d_galactosa_texto"])
    s_ds["galactosa_corregido"] = s_ds["galactosa_corregido"].fillna(s_ds["d_galactosa_corregido"])
    s_ds = s_ds.drop(columns=["d_galactosa", "d_galactosa_texto", "d_galactosa_corregido"])
    s_ds["origen"] = "Senasa"
    s_ds["correcciones"] = s_ds["Identificador"].map(CORREGIDAS)
    s_ds["excluida_hum_solidos"] = s_ds["Identificador"].isin(EXCLUIDAS)
    # valores anulados por error de carga (despues vacío en la tabla)
    ANULADOS = correcciones[correcciones["despues"].isna()]
    s_ds["valor_anulado"] = s_ds["Identificador"].map(ANULADOS.groupby("Identificador")["campo"].apply(
        lambda x: " | ".join(f"{v} (error de carga)" for v in x)))

    s_ds["tiene_4_analisis"] = s_ds[COLUMNAS].notna().all(axis=1)
    s_ds["suma_4"] = s_ds[COLUMNAS].sum(axis=1, min_count=4)
    s_ds["supera_102"] = s_ds["tiene_4_analisis"] & (s_ds["suma_4"] > 102)
    s_completas = s_ds[s_ds["tiene_4_analisis"] & ~s_ds["excluida_hum_solidos"]]
    s_muestras = s_completas[s_completas["suma_4"] <= 102].reset_index(drop=True)
    s_pre, s_fin, S_K_PRE, S_K = agrupar(s_muestras, sin_familia=None)
    s_ds = s_ds.merge(s_pre, on="Identificador", how="left").merge(s_fin, on="Identificador", how="left")
    s_ds["es_outlier"] = s_ds["es_outlier"].astype("boolean")
    s_ds["en_agrupamiento_final"] = s_ds["grupo_final"].notna()
    s_ds["familia_vs_grupo_inconsistente"] = pd.Series(pd.NA, index=s_ds.index, dtype="boolean")
    ok = s_ds["en_agrupamiento_final"]
    s_ds.loc[ok, "familia_vs_grupo_inconsistente"] = (s_ds.loc[ok, "desc_familia"] != s_ds.loc[ok, "familia_mayoritaria_grupo"]).values
    s_ds["estado"] = s_ds.apply(estado, axis=1)
    s_ds["humedad_mas_solidos"] = s_ds["humedad"] + s_ds["solidos_totales"]
    print(f"[solo Senasa] con 4 análisis: {len(s_completas)} · tras filtro <=102: {len(s_muestras)} · K_PRE={S_K_PRE} · "
          f"outliers: {int(s_pre['es_outlier'].sum())} · quedan {len(s_fin)} · K={S_K}")

    s_filas = []
    A = dict(df=s_ds, destino=s_filas)
    agregar(s_ds["supera_102"], "S01 suma de los cuatro > 102",
            lambda r: f"humedad + grasa + proteínas + cenizas = {f(r['suma_4'])} g/100 g (máximo admitido 102)", cuatro, **A)
    for c in COLUMNAS + list(k for k in EXTRA_S if k != "d_galactosa"):
        agregar((s_ds[c] < 0) | (s_ds[c] > 100), "S02 análisis negativo o > 100",
                lambda r, c=c: f"{c} = {f(r[c])} fuera de [0, 100]", lambda r, c=c: f"{c} texto original: {r[f'{c}_texto']}", **A)
    agregar((s_ds["humedad_mas_solidos"] - 100).abs() > TOL_HUM_SOL, "S03 humedad + sólidos totales fuera de 100 ± 1",
            lambda r: f"humedad + sólidos = {f(r['humedad_mas_solidos'])} · excluida del análisis",
            lambda r: f"humedad={f(r['humedad'])} ({r['humedad_texto']}) · sólidos={f(r['solidos_totales'])} ({r['solidos_totales_texto']})", **A)
    for c in COLUMNAS + ["solidos_totales", "lactosa", "galactosa"]:
        txt = s_ds[f"{c}_corregido"]                       # tras las correcciones (un anulado no cuenta como no parseable)
        agregar(txt.notna() & (txt.astype(str).str.strip() != "") & s_ds[c].isna(), "S04 valor que no se puede parsear",
                lambda r, c=c: f"{c}: el texto no se convierte a número", lambda r, c=c: f"{c} texto original: «{r[f'{c}_texto']}»", **A)
    s_disp = s_ds[COLUMNAS].notna().sum(axis=1)
    s_cero = (s_disp >= 2) & (s_ds[COLUMNAS].abs().le(CERO) | s_ds[COLUMNAS].isna()).all(axis=1) & ~excepcion_cero(s_ds)
    agregar(s_cero, "S05 valores todos cerca de 0", lambda r: f"todos los análisis disponibles <= {CERO:g} g/100 g",
            lambda r: " · ".join(f"{c}={r[f'{c}_texto']}" for c in COLUMNAS), **A)
    agregar(s_ds["valor_anulado"].notna(), "S10 valor anulado por error de carga",
            lambda r: f"{r['valor_anulado']}: se deja vacío (no se puede saber el valor correcto)",
            lambda r: f"lactosa texto original: {r['lactosa_texto']}", **A)
    agregar(s_ds["es_outlier"] == True, "S07 outlier (k-medias preliminar, solo Senasa)",
            lambda r: f"rara dentro del grupo preliminar {int(r['grupo_preliminar'])} para {int(r['votos_outlier'])} de 3 métodos "
                      f"({r['metodos_outlier']}; corte 3 %)", cuatro, **A)
    agregar(s_ds["silueta"] < 0, "S08 silueta negativa (solo Senasa)",
            lambda r: f"silueta {r['silueta']:.3f} en el grupo «{r['grupo_final']}»", cuatro, **A)
    agregar(s_ds["familia_vs_grupo_inconsistente"] == True, "S09 familia distinta de la mayoritaria de su grupo (solo Senasa)",
            lambda r: f"familia «{r['desc_familia']}» en grupo «{r['grupo_final']}», cuya familia mayoritaria es «{r['familia_mayoritaria_grupo']}»",
            lambda r: cuatro(r) + f" · producto={r['desc_producto']}", **A)

    s_inc = (pd.DataFrame(s_filas, columns=["Identificador", "desc_familia", "origen", "estado", "regla", "detalle", "valores implicados"])
             .drop(columns="origen").sort_values(["regla", "Identificador"], kind="stable").reset_index(drop=True))
    s_ds["n_reglas"] = s_ds["Identificador"].map(s_inc.groupby("Identificador")["regla"].nunique()).fillna(0).astype(int)
    s_ds["reglas"] = s_ds["Identificador"].map(s_inc.groupby("Identificador")["regla"].apply(lambda x: " | ".join(sorted(set(v[:3] for v in x)))))

    s_orden = (["Identificador", "Cliente", "Envío", "Descripción", "desc_familia", "desc_producto", "desc_subtipo", "fecha_recepcion", "estado"]
               + [x for c in COLUMNAS for x in (c, f"{c}_texto")]
               + [x for c in ["solidos_totales", "lactosa", "galactosa"] for x in (c, f"{c}_texto")]
               + ["humedad_mas_solidos", "tiene_4_analisis", "suma_4", "supera_102", "es_outlier", "grupo_preliminar",
                  "votos_outlier", "metodos_outlier", "en_agrupamiento_final", "grupo_final", "tag", "subtag", "silueta",
                  "familia_mayoritaria_grupo", "familia_vs_grupo_inconsistente", "excluida_hum_solidos", "valor_anulado",
                  "correcciones", "n_reglas", "reglas"])
    s_dataset = s_ds[s_orden].copy()
    for c in s_dataset.select_dtypes("float").columns:
        s_dataset[c] = s_dataset[c].round(4)

    s_por_regla = s_inc.groupby("regla").agg(filas=("Identificador", "size"), identificadores=("Identificador", "nunique")).reset_index()
    s_por_familia = (pd.crosstab(s_inc["desc_familia"], s_inc["regla"].str[:3], margins=True, margins_name="total")
                     .reset_index().rename(columns={"desc_familia": "familia"}))
    s_conteos = pd.DataFrame([
        ("muestras de Senasa", len(s_ds)),
        ("excluidas por humedad + sólidos fuera de 100 ± 1", int(s_ds["excluida_hum_solidos"].sum())),
        ("con los cuatro análisis (sin las excluidas)", len(s_completas)),
        ("descartadas por suma > 102", int(s_ds["supera_102"].sum())),
        ("tras filtro suma <= 102", len(s_muestras)),
        (f"outliers (k-medias preliminar, k = {S_K_PRE})", int(s_pre["es_outlier"].sum())),
        (f"en agrupamiento final (k = {S_K})", len(s_fin)),
        ("muestras con al menos una inconsistencia", int((s_ds["n_reglas"] > 0).sum())),
    ], columns=["concepto", "cantidad"])
    s_grupos = (s_fin.groupby("grupo_final").agg(muestras=("Identificador", "size"), silueta_media=("silueta", "mean"),
                                                 familia_mayoritaria=("familia_mayoritaria_grupo", "first")).round(3).reset_index())

    S_NOTAS = [
        ("General", "Fuente", "Solo proyecto/Senasa encolumnado.xlsx: todas sus filas, sin completar nada con MQ. Una fila por Identificador. "
         "Los clientes vienen ya codificados (IND-nn)."),
        ("Columna", "Identificador, Cliente, Envío, Descripción, desc_familia, desc_producto, desc_subtipo, fecha_recepcion",
         "Tal cual vienen en Senasa (fecha_recepcion = Fecha de Recepción)."),
        ("Columna", "humedad, grasa, proteinas, cenizas", "Parseados de Resultado Humedad g/ 100g (*), Resultado Materia Grasa g/ 100g (*)_x, "
         "Resultado Proteínas g/ 100g (*), Resultado Cenizas g/ 100g: coma decimal, se toma lo anterior a «+/-», «< LD» (o &lt;) = 0."),
        ("Columna", "<análisis>_texto", "Texto original de la celda de Senasa, sin tocar (antes de las correcciones). El valor numérico sale del texto ya corregido."),
        ("Columna", "solidos_totales, lactosa, galactosa", "Mismo parseo sobre Resultado Sólidos totales g/ 100g_x, Resultado Lactosa g /100g (*) y "
         "galactosa = Resultado Galactosa g/ 100g (*) o, si está vacía, Resultado D - Galactosa g/ 100g (*) (nunca están las dos en la misma fila). "
         "No se usan las lactosas en g/100 ml."),
        ("Columna", "humedad_mas_solidos", "humedad + solidos_totales (debería dar ≈ 100)."),
        ("Columna", "estado", "incompleta (faltan análisis) / descartada: suma > 102 / descartada: outlier / en agrupamiento final."),
        ("Columna", "tiene_4_analisis, suma_4, supera_102", "Si tiene los cuatro; su suma; si supera 102."),
        ("Columna", "grupo_preliminar, votos_outlier, metodos_outlier, es_outlier, grupo_final, tag, subtag, silueta",
         f"Se rehízo el procedimiento del notebook SOLO con Senasa ({len(s_muestras)} muestras con 4 análisis y suma <= 102): imputar + escalar, "
         f"k-medias preliminar k 2..10 por silueta (k = {S_K_PRE}), outlier si al menos 2 de 3 métodos (LOF 50 vecinos, IsolationForest 200 "
         "árboles, EllipticEnvelope; reescalando dentro del grupo) la ponen en el 3 % más raro de su grupo preliminar; "
         f"sin outliers, k-medias final por silueta (k = {S_K}), nombres con la misma función nombrar, tag y subtag del DDL (los de "
         f"clusters_senasa.csv, sección 3 del notebook). SEMILLA={SEMILLA}, n_init={N_INIT_KMEDIAS}. "
         "Los grupos NO son los del archivo combinado (que usa Senasa + MQ) y pueden diferir."),
        ("Columna", "familia_mayoritaria_grupo, familia_vs_grupo_inconsistente", "Moda de desc_familia en el grupo final (todas las familias cuentan) "
         "y si la de la muestra es distinta."),
        ("Columna", "excluida_hum_solidos", "Verdadero si humedad + sólidos totales está fuera de 100 ± 1 (con los dos presentes): fuera del análisis."),
        ("Columna", "valor_anulado", "Análisis anulado por error de carga según correcciones_clase41.csv (queda vacío)."),
        ("Columna", "correcciones", "Correcciones puntuales aplicadas a la muestra (campo: antes → después)."),
        ("Columna", "n_reglas, reglas", "Cuántas reglas distintas dispara y sus códigos (S01..S10, sin S06)."),
        ("Regla", "S01 suma de los cuatro > 102", "humedad + grasa + proteínas + cenizas > 102 g/100 g."),
        ("Regla", "S02 análisis negativo o > 100", "Alguno de humedad, grasa, proteínas, cenizas, sólidos totales, lactosa o galactosa < 0 o > 100."),
        ("Regla", "S03 humedad + sólidos totales fuera de 100 ± 1", f"|humedad + solidos_totales − 100| > {TOL_HUM_SOL:g}, si están ambos. Quedan fuera del análisis."),
        ("Regla", "S04 valor que no se puede parsear", "La celda (ya corregida) tiene texto pero el parseo da vacío. Una fila por análisis. "
         "Con las reglas generales «,.» → «,» y «NO DETECTADO» = 0 (por defecto) los casos conocidos quedan resueltos."),
        ("Regla", "S05 valores todos cerca de 0", f"Al menos 2 de los cuatro análisis disponibles y todos <= {CERO:g} g/100 g. "
         "Excepciones, donde es lo esperable: panificados / otros que son lactosa y helados frutales / al agua."),
        ("Regla", "S07 outlier (k-medias preliminar, solo Senasa)", "es_outlier verdadero en el agrupamiento rehecho solo con Senasa."),
        ("Regla", "S08 silueta negativa (solo Senasa)", "Coeficiente de silueta < 0 en el agrupamiento final solo Senasa."),
        ("Regla", "S09 familia distinta de la mayoritaria de su grupo (solo Senasa)", "familia_vs_grupo_inconsistente verdadero. No siempre es error: "
         "hay familias de composición parecida."),
        ("Regla", "S10 valor anulado por error de carga", "Valor imposible que no se puede corregir (p. ej. lactosa 124,3): queda vacío y marcado. "
         "La lactosa no entra en el agrupamiento."),
        ("Regla", "(no aplicables)", "No se incluyen las reglas que necesitan MQ: desacuerdo Senasa vs MQ y notas 'revisar' de MQ."),
    ]
    s_notas = pd.DataFrame(S_NOTAS + notas_correcciones(s_ds), columns=["tipo", "nombre", "cómo se construyó"])
    escribir(SALIDA_SENASA, s_dataset, s_inc, [("Conteos generales", s_conteos), ("Por regla", s_por_regla), ("Por grupo final", s_grupos),
                                                ("Por familia y regla (filas de inconsistencias)", s_por_familia)], s_notas)
    print(f"Escrito: {SALIDA_SENASA}")
    print(f"dataset: {len(s_dataset)} filas · inconsistencias: {len(s_inc)} filas")
    print(s_por_regla.to_string(index=False))


if __name__ == "__main__":
    exportar()
