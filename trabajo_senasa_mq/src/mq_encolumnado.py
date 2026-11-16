"""Dataset de las muestras MQ con el formato exacto de «Senasa encolumnado.xlsx».

Motivo: el modelo supervisado se entrena con «Senasa encolumnado.xlsx», que tiene pocos quesos con
análisis y muchos dulces de leche. MQ tiene muchos más quesos analizados; este script los pone en
el mismo formato para sumarlos al entrenamiento.

Ex build_mq_encolumnado.py (original en _archivo/proyecto/), pasado a una función:

    armar(regenerar=False)   # lee los dos xlsx ya generados de resultados/excel/ (sección 7, REGENERAR_MQ=False)
    armar(regenerar=True)    # los vuelve a armar desde datos_entrada/mq/ y los escribe (en `destino`, por defecto
                             # resultados/excel/; los reemplaza si existen)

Devuelve (todas, nuevas): los dos DataFrames.

Salidas:
    MQ encolumnado.xlsx                 todas las muestras de tipo MQ
    MQ encolumnado - solo nuevas.xlsx   solo las MQ que no están en Senasa encolumnado

Las muestras de tipo MO se dejan afuera: ya formaron parte del entrenamiento (pedido del usuario).

Columnas: las 177 de Senasa encolumnado, con el mismo nombre, orden y tipo; después los pares
Resultado/Técnica de los analitos de MQ que Senasa no tiene, `humedad_derivada_100_menos_solidos`,
`fecha_recepcion_origen` y, al final, `en_catalogo_senasa` (sí/no).

De dónde sale cada cosa
-----------------------
* Muestras que ya están en Senasa encolumnado (mismo Identificador): Cliente, Envío, Descripción,
  todas las desc_* y la Fecha de Recepción se copian de Senasa, así la clasificación es idéntica.
  Los resultados son los medidos en MQ.
* Muestras nuevas:
  - Cliente: el IND-nn de `datos_entrada/mq/muestras_ancha.csv` (codificado con la clave de
    clientes en su momento). Vacío si el crudo no trae cliente.
  - desc_familia, desc_tipo_muestra y desc_marca (MARCA-nn / IND-nn): el trabajo previo
    `datos_entrada/mq/mq_tipo_muestra.csv` (build_mq_tipo_muestra.py, archivado). Su
    "sin especificar" de familia se escribe "sin clasificar", que es el rótulo de Senasa.
    Las que quedan "sin clasificar" y tienen una descripción clara («CREMA CRUDA», «MANTECA»,
    «ARROZ CON LECHE»…) toman la familia de `src/reglas_descripcion.py`; las ambiguas («LECHE
    ENTERA» sin estado, «ALIMENTO», «FORMULA») quedan sin clasificar. `desc_familia_fuente`
    (última columna) dice de dónde salió la familia de cada fila.
  - desc_producto y los campos etiquetados de la descripción (lote, partida, vencimiento...): el
    extractor `extraer_metadatos` de `src/compartido/Filtrado_de_Datos_2_ordenado.ipynb` (se ejecuta la celda
    tal cual, no se copia). Se valida contra Senasa en las muestras compartidas y solo se
    llenan las columnas que coinciden al menos UMBRAL_ACUERDO; el resto queda vacío.
  - desc_subtipo, desc_estado, desc_grasa, desc_lactosa, desc_consistencia y _origen:
    `catalogo.normalizar_catalogo` (src/compartido/catalogo.py), el mismo que normaliza Senasa.
  - Las desc_* que salían del extractor original del catálogo, que no está en el proyecto
    (atributos, formato, sabor, claims, sku, codigo_interno, linea, sector, punto_toma...), quedan
    vacías. desc_exportacion queda False (no se puede reproducir; ver el informe).
  - Fecha de Recepción: la fecha estimada por número de muestra de muestras_ancha.csv
    (interpolada o extrapolada). `fecha_recepcion_origen` lo dice fila por fila.
* Resultados: solo lo medido en MQ (`*_fuente == "mq"` en muestras_ancha.csv); lo que la tabla
  ancha completaba desde el catálogo no se usa. Texto con coma decimal, como Senasa.
  La técnica es el método de MQ; en los cuatro analitos compartidos los nombres de método de MQ
  son los mismos textos que usa Senasa, así que la equivalencia es directa.
* Humedad: MQ mide sólidos totales, no humedad. «Resultado Humedad» queda vacío y 100 − sólidos
  va aparte, en `humedad_derivada_100_menos_solidos`.

Privacidad: la descripción cruda trae razones sociales y marcas. Se usa en memoria; lo que se
escribe pasa por `anonimizar.redactar` (vocabulario de producto + palabras que sobrevivieron a la
anonimización de Senasa encolumnado), con los nombres de cliente de los crudos tapados antes.
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import reglas_descripcion
import rutas as R

sys.path.insert(0, str(R.COMPARTIDO))
import anonimizar  # noqa: E402
import catalogo  # noqa: E402  (src/compartido/catalogo.py, ex proyecto/catalogo.py; no datos/catalogo.py del curso)

# anonimizar.py busca la lista privada de marcas en una ruta relativa al directorio de trabajo; se recarga desde
# rutas.MARCAS_PRIVADAS (si no existe, como en esta máquina, queda vacía igual que antes).
anonimizar.MARCAS_FRASE, anonimizar.MARCAS_TOKEN = anonimizar.cargar_marcas(str(R.MARCAS_PRIVADAS))

SENASA = R.SENASA
NOTEBOOK_EXTRACTOR = R.EXTRACTOR
CRUDOS = R.MQ_CRUDOS
ANCHA = R.MQ_ANCHA
TIPO = R.MQ_TIPO
NOMBRE = "MQ encolumnado.xlsx"
NOMBRE_NUEVAS = "MQ encolumnado - solo nuevas.xlsx"

ARCHIVOS = ["ceniza.csv", "grasa.csv", "grasa 1.csv", "prote.csv", "st.csv", "st 2.csv", "cloruro.csv"]
UMBRAL_ACUERDO = 0.75     # acuerdo mínimo con Senasa (muestras compartidas) para llenar una desc_*
SIN = "sin clasificar"    # rótulo de Senasa para familia desconocida

# analito de muestras_ancha -> nombre del análisis (sin "Resultado"/"Técnica").
# Los cuatro primeros son columnas de Senasa; el resto, columnas nuevas con el nombre de MQ.
COMPARTIDOS = {
    "ceniza": "Cenizas g/ 100g",
    "grasa": "Materia Grasa g/ 100g (*)_x",
    "proteina": "Proteínas g/ 100g (*)",
    "solidos": "Sólidos totales g/ 100g_x",
}
NUEVOS = {
    "cloruro_nacl_g100g": "Cloruros (NaCl g / 100g) (*)",
    "cloruro_mg100g": "Cloruros (mg/100g) (*)",
    "cloruro_mg100ml": "Cloruros mg/ 100ml (*)",
    "ceniza_acida": "Cenizas ácidas g / 100g (*)",
    "grasa_libre": "Materia grasa libre g/ 100g (*)",
    "caseina": "Caseína g/100g (*)",
    "proteina_base_seca": "Proteínas en base seca % (*)",
    "solidos_suspendidos_mgl": "Sólidos suspendidos mg/ L (*)",
    "solidos_disueltos_mgl": "Sólidos disueltos mg/ L (*)",
}

# campo de extraer_metadatos -> columna de Senasa
EXTRACTOR = {
    "lote": "desc_lote", "partida": "desc_partida", "acta": "desc_acta", "pallet": "desc_pallet",
    "orden_produccion": "desc_orden_produccion", "articulo": "desc_articulo", "codigo": "desc_codigo",
    "equipo": "desc_equipo", "big_bag": "desc_big_bag", "bolsa": "desc_bolsa", "hora": "desc_hora",
    "fecha_elaboracion": "desc_fecha_elab", "fecha_vencimiento": "desc_vencimiento", "fecha": "desc_fecha",
    "muestra": "desc_muestra_n", "presentacion": "desc_gramaje", "producto": "desc_producto",
}
EJES = ["desc_subtipo", "desc_estado", "desc_grasa", "desc_lactosa", "desc_consistencia", "desc_consistencia_origen"]


def sa(t):
    return "".join(c for c in unicodedata.normalize("NFKD", str(t)) if not unicodedata.combining(c))


def armar(regenerar=False, destino=None):
    """regenerar=False: lee los xlsx de resultados/excel/. regenerar=True: los arma y los escribe en destino."""
    destino = Path(destino) if destino else R.EXCEL
    SALIDA = destino / NOMBRE
    SALIDA_NUEVAS = destino / NOMBRE_NUEVAS
    if not regenerar:
        out = pd.read_excel(SALIDA)
        nuevas = pd.read_excel(SALIDA_NUEVAS)
        print(f"Leídos (sin regenerar): {SALIDA.name} {out.shape} · {SALIDA_NUEVAS.name} {nuevas.shape}")
        return out, nuevas
    destino.mkdir(parents=True, exist_ok=True)
    # --------------------------------------------------------------------------- 1. Senasa (formato)
    senasa = pd.read_excel(SENASA)
    COLS = list(senasa.columns)
    assert len(COLS) == 177, len(COLS)
    TIPOS = senasa.dtypes
    sen = senasa.set_index("Identificador")

    # --------------------------------------------------------------------------- 2. MQ
    ancha = pd.read_csv(ANCHA, dtype=str, encoding="utf-8-sig")
    n_mo = int((ancha["tipo"] == "MO").sum())
    ancha = ancha[ancha["tipo"] == "MQ"].set_index("codigo")
    tipo = pd.read_csv(TIPO, dtype=str, encoding="utf-8-sig", keep_default_na=False).set_index("codigo").reindex(ancha.index)
    assert tipo["desc_familia"].ne("").all()
    en_sen = ancha.index.isin(sen.index)
    print(f"muestras MQ: {len(ancha)} | en Senasa: {int(en_sen.sum())} | nuevas: {int((~en_sen).sum())} | MO excluidas: {n_mo}")

    # descripción cruda (en memoria; nunca se imprime) y nombres de cliente crudos, mismo criterio que
    # build_mq_tipo_muestra.py: solo procesado con valor, sin duplicados por id, la de mayor id
    partes = [pd.read_csv(CRUDOS / a, dtype=str, keep_default_na=False,
                          usecols=["id", "codigo", "estado", "resultado", "ingreso", "cliente"]) for a in ARCHIVOS]
    d = pd.concat(partes, ignore_index=True)
    d = d[(d["id"] != "id") & d["codigo"].str.startswith("MQ-")].drop_duplicates(subset="id")
    d = d[(d["estado"] == "procesado") & d["resultado"].str.match(r"^\s*-?\d")].copy()
    d["id"] = d["id"].astype(int)
    PREFIJO = re.compile(r"^\s*\d+\s*-\s*[A-Z]{2}-\d+-\d+\s*-\s*")
    d["desc"] = d["ingreso"].str.replace(PREFIJO, "", regex=True).str.strip()
    desc = d.sort_values("id").groupby("codigo")["desc"].last().reindex(ancha.index).fillna("")
    nombres_cliente = set(d["cliente"].str.strip()) - {""}
    del partes, d

    # --------------------------------------------------------------------------- 3. redacción
    # Sin la lista privada de marcas (no está en esta máquina) la regla de "usado por 4 clientes"
    # podría dejar pasar una marca. Se usa una regla más estricta: una palabra sin números queda solo
    # si es vocabulario de producto o si sobrevivió (>= 3 veces) a la anonimización ya auditada de
    # Senasa encolumnado (Descripción y desc_producto).
    pal = pd.Series([w for t in pd.concat([senasa["Descripción"], senasa["desc_producto"]]).dropna()
                     for w in set(re.split(r"[^A-Z0-9]+", sa(t).upper())) if len(w) > 2]).value_counts()
    USO = {w: 99 for w in pal[pal >= 3].index}



    def canonico(nombre):
        """Razón social sin la forma societaria: la regla de respaldo de codificar_clientes.canonico
        (la función completa necesita datos/clientes_equivalencias.csv, que no está en esta máquina)."""
        limpio = re.sub(r"\b(S\.?A\.?|S\.?R\.?L\.?|SRL|SA|LTDA|COOP\.?|SOCIEDAD ANONIMA|"
                        r"SOCIEDAD DE RESPONSABILIDAD LIMITADA)\b", " ", sa(nombre).upper())
        return re.sub(r"[^A-Z0-9 ]", " ", limpio).split("  ")[0].strip() or None


    tapar = set()
    for n in nombres_cliente:
        for v in (n, canonico(n)):
            if isinstance(v, str):
                v = sa(v).upper().strip()
                if len(v) >= 4 and not all(w in anonimizar.VOCABULARIO for w in re.split(r"\W+", v) if w):
                    tapar.add(v)
    PAT_CLIENTES = re.compile("|".join(anonimizar.patron_flexible(n) for n in sorted(tapar, key=len, reverse=True)),
                              re.IGNORECASE)


    def redactar(t):
        if not isinstance(t, str) or not t:
            return None
        r = anonimizar.redactar(PAT_CLIENTES.sub("·", t), USO)
        return r or None


    # --------------------------------------------------------------------------- 4. extractor del notebook
    nb = json.loads(NOTEBOOK_EXTRACTOR.read_text(encoding="utf-8"))
    celda = next("".join(c["source"]) for c in nb["cells"] if "def extraer_metadatos" in "".join(c["source"]))
    ns = {"pd": pd}
    exec(celda, ns)  # noqa: S102  (la celda del notebook, sin cambios)
    meta = pd.DataFrame([ns["extraer_metadatos"](v) for v in desc], index=desc.index)

    ETQ = [p for _, p in ns["ETIQUETAS"]]


    def producto(p):
        """Primer tramo, sin lo etiquetado ni la presentación (va a desc_gramaje), redactado y normalizado."""
        if not isinstance(p, str):
            return None
        for patron in ETQ:
            p = patron.sub(" ", p)
        r = redactar(p)
        if not r:
            return None
        r = catalogo.normalizar_texto(r.replace("·", " "))
        r = re.sub(r"\(\s*\)", " ", r)
        return re.sub(r"\s+", " ", r).strip(" -():.") or None


    def valor_meta(campo, v):
        if not isinstance(v, str) or not v.strip():
            return None
        if campo == "producto":
            return producto(v)
        if campo == "muestra":
            return float(v.lstrip("M"))
        if campo in ("articulo", "big_bag", "bolsa", "acta"):
            n = pd.to_numeric(re.sub(r"\D", "", v), errors="coerce")
            return None if pd.isna(n) else float(n)
        return redactar(v)


    desc_mq = pd.DataFrame(index=desc.index)
    for campo, col in EXTRACTOR.items():
        desc_mq[col] = [valor_meta(campo, v) for v in meta[campo]]

    # familia, tipo de muestra y marca: trabajo previo
    desc_mq["desc_familia"] = tipo["desc_familia"].replace("sin especificar", SIN)
    # descripción clara: solo llena las «sin clasificar» de las muestras nuevas (sobre el texto crudo, en memoria)
    fam_regla, regla = reglas_descripcion.aplicar(desc_mq["desc_familia"].where(~en_sen, "(senasa)"), desc)
    cambia = regla.notna()
    desc_mq.loc[cambia, "desc_familia"] = fam_regla[cambia]
    fuente_familia = pd.Series(np.where(en_sen, "senasa (manual)",
                               np.where(cambia, "regla por descripción: " + regla.fillna(""),
                               np.where(desc_mq["desc_familia"].eq(SIN), "sin clasificar", "modelo"))), index=desc.index)
    print(f"sin clasificar con descripción clara: {int(cambia.sum())} pasan a una familia "
          f"({regla[cambia].value_counts().to_dict()}); quedan sin clasificar: "
          f"{int((desc_mq['desc_familia'].eq(SIN) & ~en_sen).sum())}")
    desc_mq["desc_tipo_muestra"] = tipo["desc_tipo_muestra"]
    desc_mq["desc_marca"] = tipo["marca_codigo"].replace("", np.nan)

    # subtipo (mismas tablas que build_mq_tipo_muestra.py) y ejes con catalogo.normalizar_catalogo
    texto_norm = desc.map(catalogo.normalizar_texto)
    sub_regla = [catalogo.primer_match(t, catalogo.SUBTIPOS.get(f, []))[0]
                 for f, t in zip(desc_mq["desc_familia"], texto_norm)]
    nuevo_tipo = tipo["desc_subtipo"].replace("sin especificar", None)
    igual = ~en_sen & ~cambia          # donde la familia no cambió, el subtipo es el del trabajo previo
    assert (pd.Series(sub_regla, index=desc.index)[igual].fillna("") == nuevo_tipo[igual].fillna("")).all(), \
        "el subtipo no coincide con mq_tipo_muestra.csv"
    entrada = pd.DataFrame({"desc_familia": desc_mq["desc_familia"], "desc_producto": desc_mq["desc_producto"],
                            "desc_subtipo": sub_regla, "desc_formato": None, "desc_claims": None, "desc_sabor": None},
                           index=desc.index).astype(object)
    ejes = catalogo.normalizar_catalogo(entrada)
    for c in EJES:
        desc_mq[c] = ejes[c]

    # --------------------------------------------------------------------------- 5. validación contra Senasa
    def comparable(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, float):
            return f"{v:g}"
        v = re.sub(r"\s+", " ", str(v).replace("(·)", " ").replace("·", " ")).strip().upper()
        return re.sub(r"\.0$", "", v)


    comp = ancha.index[en_sen]
    acuerdo = {}
    for col in list(EXTRACTOR.values()) + EJES + ["desc_familia", "desc_tipo_muestra", "desc_marca"]:
        mio = desc_mq.loc[comp, col].map(comparable)
        ref = sen.loc[comp, col].map(comparable)
        alguno = (mio != "") | (ref != "")
        acuerdo[col] = {"senasa": int((ref != "").sum()), "mq": int((mio != "").sum()),
                        "acuerdo": round(float((mio == ref)[alguno].mean()), 3) if alguno.any() else np.nan}
    acuerdo = pd.DataFrame(acuerdo).T
    # familia, tipo y marca vienen de mq_tipo_muestra, que en las compartidas copió la clasificación de
    # Senasa: su acuerdo ahí es 1 por construcción y no valida nada (se informa la CV del trabajo previo).
    validables = list(EXTRACTOR.values()) + EJES
    llenar = [c for c in validables if acuerdo.loc[c, "acuerdo"] >= UMBRAL_ACUERDO]
    vaciar = [c for c in validables if c not in llenar]
    print(f"\nacuerdo con Senasa en las {len(comp)} muestras compartidas (se llena si >= {UMBRAL_ACUERDO}):")
    print(acuerdo.loc[validables].to_string())
    print("se llenan:", llenar)
    print("quedan vacías (no reproducibles):", vaciar)
    for c in vaciar:
        desc_mq[c] = None

    # --------------------------------------------------------------------------- 6. armado
    def fmt(v):
        if pd.isna(v):
            return None
        v = float(v)
        s = f"{v:.2f}" if abs(round(v, 2) - v) < 1e-9 else f"{v:.4f}".rstrip("0")
        return s.replace(".", ",")


    out = pd.DataFrame(index=ancha.index, columns=COLS, dtype=object)
    out["Identificador"] = ancha.index
    out["Envío"] = "Q-" + ancha["numero"].str.zfill(5)
    out["Cliente"] = ancha["cliente"]
    out["Descripción"] = desc.map(redactar)
    for c in COLS:
        if c.startswith("desc_") and c in desc_mq.columns:
            out[c] = desc_mq[c]
    out["desc_exportacion"] = False
    fechas = pd.to_datetime(ancha["fecha_estimada"])
    out["Fecha de Recepción"] = fechas.dt.strftime("%d-%m-%Y")
    origen_fecha = ancha["fecha_metodo"].map(lambda m: f"estimada ({m})")

    # compartidas: identificación, descripción y desc_* tal como están en Senasa
    cols_sen = ["Cliente", "Envío", "Descripción", "Fecha de Recepción"] + [c for c in COLS if c.startswith("desc_")]
    out.loc[comp, cols_sen] = sen.loc[comp, cols_sen].values
    origen_fecha[comp] = "senasa"

    # resultados medidos en MQ
    num = {}
    for analito in list(COMPARTIDOS) + list(NUEVOS):
        v = pd.to_numeric(ancha[analito], errors="coerce").where(ancha[f"{analito}_fuente"].eq("mq"))
        num[analito] = v
        nombre = COMPARTIDOS.get(analito) or NUEVOS[analito]
        out[f"Resultado {nombre}"] = v.map(fmt)
        out[f"Técnica {nombre}"] = ancha[f"{analito}_metodo"].where(v.notna())
    extra = []
    for analito, nombre in NUEVOS.items():
        extra += [f"Resultado {nombre}", f"Técnica {nombre}"]
    out["humedad_derivada_100_menos_solidos"] = (100 - num["solidos"]).map(fmt)
    out["fecha_recepcion_origen"] = origen_fecha
    out["en_catalogo_senasa"] = np.where(en_sen, "sí", "no")
    out["desc_familia_fuente"] = fuente_familia
    FINAL = COLS + extra + ["humedad_derivada_100_menos_solidos", "fecha_recepcion_origen", "en_catalogo_senasa",
                            "desc_familia_fuente"]
    out = out[FINAL].reset_index(drop=True)

    # tipos de Senasa
    for c in COLS:
        t = TIPOS[c]
        if t == bool:
            out[c] = out[c].astype(bool)
        elif np.issubdtype(t, np.floating):
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
        else:
            out[c] = out[c].astype(object).where(out[c].notna(), None)
    # openpyxl escribe como fórmula un texto que empieza con "="
    for c in out.columns[out.dtypes == object]:
        out[c] = out[c].map(lambda v: " " + v if isinstance(v, str) and v.startswith("=") else v)

    # técnicas: ¿el texto de MQ es uno de los que Senasa usa para ese análisis?
    print("\ntécnicas de MQ que son exactamente una técnica de Senasa para el mismo análisis:")
    for analito, nombre in COMPARTIDOS.items():
        t_mq = out[f"Técnica {nombre}"].dropna()
        ok = t_mq.isin(set(senasa[f"Técnica {nombre}"].dropna()))
        print(f"  {nombre:32} {int(ok.sum()):5} de {len(t_mq):5}  sin equivalente: {sorted(t_mq[~ok].unique())}")

    # --------------------------------------------------------------------------- 7. escritura y controles
    anonimizar.escribir_xlsx(out, SALIDA)
    nuevas = out[out["en_catalogo_senasa"] == "no"].reset_index(drop=True)
    anonimizar.escribir_xlsx(nuevas, SALIDA_NUEVAS)

    for ruta, n in ((SALIDA, len(out)), (SALIDA_NUEVAS, len(nuevas))):
        leido = pd.read_excel(ruta)
        assert list(leido.columns[:177]) == COLS, ruta
        assert len(leido) == n
        assert not leido["Identificador"].str.startswith("MO").any()
        assert leido["Cliente"].dropna().str.fullmatch(r"IND-\d+").all()
        assert leido["desc_marca"].dropna().str.fullmatch(r"(IND|MARCA)-\d+").all()
        print(f"\n{ruta.name}: {leido.shape[0]} filas x {leido.shape[1]} columnas; primeras 177 = Senasa encolumnado")

    ctrl = pd.read_excel(SALIDA, dtype=str).set_index("Identificador").loc["MQ-45752-02"]
    esperado = {"Resultado Materia Grasa g/ 100g (*)_x": "24,43", "Resultado Proteínas g/ 100g (*)": "24,39",
                "Resultado Sólidos totales g/ 100g_x": "54,82", "Resultado Cloruros (NaCl g / 100g) (*)": "1,29",
                "desc_familia": "queso"}
    for c, v in esperado.items():
        assert ctrl[c] == v, (c, ctrl[c])
    assert pd.isna(ctrl["Resultado Humedad g/ 100g (*)"])
    print("control MQ-45752-02:", {c: ctrl[c] for c in esperado}, "| subtipo:", ctrl["desc_subtipo"],
          "| humedad derivada:", ctrl["humedad_derivada_100_menos_solidos"])

    # --------------------------------------------------------------------------- 8. balance por familia
    RES_SEN = [c for c in COLS if c.startswith("Resultado ")]
    PANEL_SEN = ["Resultado Materia Grasa g/ 100g (*)_x", "Resultado Proteínas g/ 100g (*)", "Resultado Cenizas g/ 100g"]


    def balance(df, res_cols, humedad):
        con = df[res_cols].notna().any(axis=1)
        hgp = df[PANEL_SEN[:2]].notna().all(axis=1) & humedad          # sin cenizas
        panel = hgp & df[PANEL_SEN[2]].notna()
        g = df["desc_familia"]
        return pd.DataFrame({"n": con.groupby(g).sum(), "hgp": hgp.groupby(g).sum(), "panel": panel.groupby(g).sum()})


    b_sen = balance(senasa, RES_SEN, senasa["Resultado Humedad g/ 100g (*)"].notna()
                    | senasa["Resultado Sólidos totales g/ 100g_x"].notna())
    b_mq = balance(nuevas, RES_SEN + [c for c in extra if c.startswith("Resultado")],
                   nuevas["Resultado Sólidos totales g/ 100g_x"].notna())
    tabla = b_sen.join(b_mq, how="outer", lsuffix="_senasa", rsuffix="_mq_nuevas").fillna(0).astype(int)
    tabla["n_total"] = tabla["n_senasa"] + tabla["n_mq_nuevas"]
    tabla["hgp_total"] = tabla["hgp_senasa"] + tabla["hgp_mq_nuevas"]
    tabla["panel_total"] = tabla["panel_senasa"] + tabla["panel_mq_nuevas"]
    tabla = tabla[[f"{c}_{o}" for o in ("senasa", "mq_nuevas", "total") for c in ("n", "hgp", "panel")]]
    tabla = tabla.sort_values("n_total", ascending=False)
    tabla.loc["TOTAL"] = tabla.sum()
    print("\nbalance por familia (n = muestras con >= 1 análisis; hgp = humedad o sólidos + grasa + proteína; "
          "panel = hgp + cenizas):")
    print(tabla.to_string())
    for f in ("queso", "dulce de leche"):
        for c in ("n", "hgp", "panel"):
            s, t = tabla.loc[f, f"{c}_senasa"] / tabla.loc["TOTAL", f"{c}_senasa"], tabla.loc[f, f"{c}_total"] / tabla.loc["TOTAL", f"{c}_total"]
            print(f"  {f:15} {c:5}: {s:5.1%} del total en Senasa -> {t:5.1%} con MQ nuevas")
    print(f"\nMO excluidas: {n_mo}")
    return out, nuevas


if __name__ == "__main__":
    armar(regenerar=True)
