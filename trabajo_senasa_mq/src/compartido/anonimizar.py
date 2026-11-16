"""Anonimización: códigos de cliente y redacción de rótulos, en un solo lugar.

Es la fuente única para todo el proyecto. El notebook lo importa y los datasets exportados se
procesan con `python anonimizar.py --aplicar`, así no puede pasar que la lógica del notebook y
la de los archivos diverjan.

Dos capas, porque una sola no alcanza:

1. **Código de cliente.** La columna de cliente pasa a `IND-nn` según `datos/clientes_clave.csv`,
   que genera `codificar_clientes.py` y es el único archivo que no se comparte.
2. **Redacción del rótulo.** El 7,5% de las descripciones trae la marca adentro del texto
   ("DULCE DE LECHE ALFAJORERO <marca>"), así que codificar la columna de cliente no basta.
   Se conserva el vocabulario de producto y se tapa el resto con `·`.

La redacción se aplica a lo que se exporta o se muestra. El análisis usa el rótulo completo,
porque de él dependen la segmentación y los efectos fijos de producto.

Uso:
    python anonimizar.py             # informa el estado de cada archivo
    python anonimizar.py --aplicar   # reescribe los datasets anonimizados
"""

import argparse
import glob
import re
import unicodedata
from pathlib import Path

import pandas as pd

CLAVE = "datos/clientes_clave.csv"

# Vocabulario que se conserva: términos de producto y de las reglas de clasificación.
VOCABULARIO = set("""
PARMESANO PARMIGIANO REGGIANITO REGGIANO SARDO ROMANO SBRINZ GRANA GOYA PROVOLONE
GOUDA DANBO PATEGRAS PATEGAS TYBO PRATO HOLANDA GRUYERE GRUYERITO EDAM COLONIA CHEDDAR
MAR PLATA SAINT PAULIN PROVOLETA CASCARA COLORADA CRIOLLO FONTINA
CREMOSO CUARTIROLO PORT SALUT MASCARPONE BRIE CAMEMBERT PETIT SUISSE MOZZARELLA ITALIANO
MEDITERRANEO NAPOLITANO CUATRO MEZCLA
QUEIJO MUSSARELA LEITE MANTEIGA PARMESAO REQUEIJAO IOGURTE NATA MOZARELLA MUZZARELLA
DIETAS COM RESTRICAO PARA
QUESO QUESOS QUESILLO RICOTA DURO SEMIDURO BLANDO PROCESADO FUNDIDO UNTABLE
HUMEDAD ALTA MEDIANA BAJA PASTA HILADA HEBRAS BARRA HORMA PORCIONES TROZADO FETAS FETEAR
SANDWICH SANWICH RALLADO BLOCK BLOQUE MASA PRENSADO REDONDO AZUL CREMA
DULCE LECHE CLASICO FAMILIAR TRADICIONAL COLONIAL REPOSTERO HELADERO ALFAJORERO PANADERO
MAQUINA UAT UHT CRUDA FLUIDA ENTERA DESCREMADA PARCIALMENTE DESLACTOSADA
DESNATADA MODIFICADA MATERNIZADA INFANCIA CONCENTRADA CHOCOLATADA SABORIZADA VAINILLA
LACTOSA SIN REDUCIDA LACTASA POLVO INSTANTANEA SUERO LACTOSUERO WPC WPI PERMEADO
DESMINERALIZADO CONCENTRADO PROTEINA PROTEINAS MANTECA MANTEQUILLA YOGUR BASE
LIBRE GLUTEN LIGHT DIETETICO GRASO REDUCIDO CONTENIDO POOL MUESTRA LOTE
DE DEL LA EL LOS LAS EN CON POR UN UNA KG GR ML Y A O E U L V N T HS AL SOBRE ENTRE HASTA
""".split())

# Ampliación revisada sobre el catálogo encolumnado. La regla de "término de al menos cuatro
# clientes" tapaba 996 palabras distintas de desc_producto (10.591 veces): UNTAR, CASEINATO,
# los sabores de helado en inglés, los envases... El rótulo quedaba como "QUESO CHEDDAR FUNDIDO
# PARA · CON · DE JAMON". Se revisó la lista palabra por palabra y acá van las que son
# vocabulario genérico de producto, proceso, envase o sabor. Lo que podía ser nombre de
# empresa, línea comercial, persona o lugar quedó afuera y se sigue tapando.
VOCABULARIO |= set("""
HELADO SABOR SABORES QUESO YOGUR
VIDA UTIL PARTIDA OP EQUIPO CODIGO VTO ELAB FECHA ENVIO PEDIDO ORDEN PRODUCCION TURNO TANQUE
SILO SECTOR ACTA PALLET CUNETE TIPO LARGA REDUCIDA
ADEREZO UNTAR AGREGADO AGREGADA AGREGADOS CASEINATO CASEINATOS CASEINA SODIO ACIDA ACIDO FOLICO
QUESERIA CALIBRACION REFERENCIA PATRON ARROZ BEBIDA PREPARAR PREPARACION PREPARADO PREPARADA
OLEOMARGARINA OLEO MARGARINA MALTODEXTRINA ALIMENTICIA ALIMENTICIO ALIMENTO ALIMENTOS
ALIMENTARIO LACTEOS LACTEAS LACTEO LACTEA LACTO ENJUAGUE EVAPORADOR HOMOGENEIZADOR CAMARA
HISOPADO HISOPO ACEITE ACEITES LINEA PASTEURIZADO PASTEURIZADA HOMOGENEIZADA HOMOGENEIZADO
HOMOGENEAS HOMO HOMOG TABLETA TABLETAS ESTUCHE PINTA BALDE POTE PALITO PALETA BOMBON CONO
CONOS PORCION VASITO SACHET FRASCO BIDON TAMBOR BOTELLA BOTELLITA CAJA BOLSA BOLSAS BAG BOX
PACK ENVASE ENVASADO GRANEL CALOSTRAL CALOSTRO PREMIUM PREMIUN CONITO PASTELERO EXTRA DOBLE
CALIDAD DESNATAR DESNATADO DESNATADORA BEBIBLE BATIDO RED UF NANO NANOFILTRADO MICROFILTRADO
MPC PROD USO EXCLUSIVO INDUSTRIAL GRANIZADO GRANIZADA AMERICANA PRODUCTO CARNICO CARNICOS
HEBRA FAT FILLED POSTRE LIMON ANALIZADO ANALIZAR ANALIZADA GIRASOL BEBE PROT CONCENTRADAS
INSTANTANEO INSTANTANEAS INSNTANTANEA INSTANTENEA CASERA SEMIDESCREMADO SEMIDESCREMADA
SEMIDESCREMDA BLANDA SEMIDURA DURA ALMIDON LIQUIDA LIQUIDO MIX INSUMO TERMINADO CUAJO ARTICULO
NEGRO BLANCO MENTA COCO ACIDIFICACION MORTADELA SUSTITUTO COMPOSTO COMPUESTO ZINC HIERRO
CALCIO REENGRASADO HILADO FORMULA FORMULADA FORMULADO TACC PREFRITO PREFRITA PREFRITOS PREFRITAS
ETAPA PRODUCIDO UNID UNIDAD CORAZON FRACCIONADO MAIZ FRAMBUESA ADICION INTENSO COB COBERTURA
INFANTIL LACTANTES MANJAR CROCANTE PICADO NUGGETS PRE FIN CHOCOTORTA ENZIMA ENZIMAS TILSIT
JALAPENO GRASAS GRASA LOMO FUNDIR MAGRO MAGRA PAN SPRAY MARACUYA CRUDO CRUDOS PREMEZCLA ENTRADA
SALIDA PATAGONICO PATAGONICA KOSHER SUPER MILANESA SALAMIN SALAME SOLUCION ALFAJOR ENDULZADO
GLUCOSA CARAMELOS CARAMELO SAMPLE PEPATO PARRILLERO FETEADO FETEADA BOVINA CHORIZO CHORIZOS
FRESCO FRESCA FRESCOS REGULAR ELABORAR DESHIDRATADO CARBOHIDRATOS PIEL SALCHICHON AMARGO HARINA
SALSA ADULTO REGG HINCHADO PRIMA REFRIGERADO CALORIAS HORNEADO HORNEADAS HORNO FINO EMPAREDADOS
ANANA FORMADO REBOZADO REBOZADOR FILET HAMBURGUESA HAMBURGUESAS GALLETITAS PIZZA NORMAL PLAN
MONITOREO CONC FORTIFICADO FORTIFICADA PREGELATINIZADO LAB SENASA VNTO VENCIMIENTO OITO OCHO
ESPANOLA FOOD SANTA FE
GAUDA DAMBO MOZZA MOZZRELLA CHEEDAR CHEESE CHEES ICECREAM DESERT DIARY PRATTO PARMESADO
REGGIANITTO REGGINIATO REGIANITO PROVOLETAS EMMENTHAL EMMENTAL ROMANITO REPOSERO REPSOTERO
RESPOTERO FAMIIAR FAMILEAR FAMILILAR FLIAR FAMILIA HELADOF HELDAERO HELDERO SANDWICHERO
MAGDALENERO PROUDCCION CRUDSO
DESSERT FROZEN FROZZEN FROZER DAIRY ICE CREAM CREAMY COOKIES COOKIE FLAVORED FLAVRED WITH WHIT
WHITOUT CHUNKS CHUNK CARAMEL MILK LEMON STRAWBERRY STRABERRY RASPBERRY RASPERRY BANANA SPLIT
COCONUT COCNUT DARK WHITE TIRAMISU MINT MINTS CHIP GELATO CHOCO CHOCOLATE CHOCOLATES NOCCIOLA
HAZELNUT BARS BAR ORANGE PASSION FRUIT FRUITS VANILLA ZABAIONE ZABAGLIONE ZAMBAYON SAMBAYON
PISTACHIO PISTACCIO PISTACHO PISTACHOS CHOCOLATY BISCUITS DOUBLE PEAR BLUEBERRIES SYRUP COCOA
SWIRLS SWILRS CHICKEN COOKED COFFE CAKE BROWNIE PANCAKE NUTS PECAN RAISINS RUMRAISINS WHISKEY
SALTED SUGAR FREE PROTEIN LACTOSE YOGURT SWISS AND ON SOY SOYA SOJA FOLLOW PWD STD DIABETES MAX
FULLY POWDER
DURAZNO MANZANA MANGO PERA MEMBRILLO MIEL CAFE MOCA MATCHA JENGIBRE ALBAHACA CIBOULETTE ROMERO
PIMIENTA PIMIENTAS PIMENTON MORRONES ACEITUNAS HIERBAS AVENA CEBADA CHIA MANDIOCA POLENTA FECULA
AZUCARES SACAROSA JARABE MALTA MALTEADAS OBLEA GRISINES FACTURAS MADALENAS TORTA TARTA BUDIN
CROISSANT CROSTATA CANNOLI CANNOLIS BIZCOCHUELO TOSTADA TOSTADAS PEBETES MIGA EMPANADA CHIPA
PANIFICADOS REPOSTERIA DULCERIA BONDIOLA PANCETA PECHUGA MUSLO PATA HUESO MEDALLON PATTY FUET
SOPRESATTA SPIANATTA CANTIMPALO VACA OVEJA TERNERO CABRA ARROLLADO NAPOLITANA MEDITERRANEA
MEDITERRRANEO GRIEGO VEGANO VEGETAL VEGETALES INTEGRALES ESCAMAS BASTONCITOS COPITOS RODAJA
REBANADAS TROZADA DESMENUZADA CORTADO MARINADAS GRILLADO PLANCHA COCIDA COCIDOS COCINADA
CONGELADOS FREEZER HELADERA SECADO SECO SOLIDA SOLIDOS SALADA SALMUERA ESPECIAL ESPECIALES ESTILO
ARTESANAL ARTESANO ORIGINAL CLASICA ECONOMICO LIVIANA LIGTH ZERO SEMI SEMIGRASO FILTRADA FILTRO
OSMOSIS SOBRENADANTE PRECIPITADO CRISTALIZADO CRISTALIZADOR CRISTAL RECUPERADOR TUNEL CISTERNA
CAMION CANAL POZO EFLUENTE RESIDUO PLANTA ESTABLECIMIENTO MUESTREO MUESTRAS TANDA TIRADA RECORTE
REMITO PRECINTO CALIBRE PESO NETO GRAMOS VALOR COMPOSICION PROPORCIONES TOTALES MINERALES
HEMOGLOBINA HPLC UFC AGAR AZIDIOL HIDROLIZADA HIDROLIZADAS HIDROLIZADO HIDROLIZAR DESLACTOSAR
DESLACTOSA FERMENTACION FERMENTO DERIVADO ESTANDARIZADO RECONSTITUIDA CONDENSADA CONDENSADO
EVAPORADA MODIFICADO TERMOESTABLE POTENCIADO PROTEICO SEGUNDA PRIMERA PRIMER PRIMEROS SECUNDARIO
PREVIO ANTES DESPUES ABAJO IZQ DER CONTINUA FINALIZAR INGRESAR PARTIR SIMULAR TRATAR TRATADA
TIEMPO DIAS ANOS ETC OTRAS SOLO MUY BUENA LLENA VACANTE DISCO VIDRIO CHICLE CHECK COUNT CALCULO
VERACIDAD OBSERV OBSER IDENTIF PRODUCIBILIDAD INCERTIDUMBRERE COMPETENCIA CLIENTE TERCEROS
MERCADO LOCAL REGIONAL COMERCIAL INSCRIPTO RNPA
ENE FEB MAR ABR MAY JUN JUL AGO SEP OCT NOV DIC ENERO FEBRERO MARZO ABRIL MAYO JUNIO JULIO
AGOSTO SEPTIEMBRE OCTUBRE NOVIEMBRE DICIEMBRE
""".split())

MARCAS_PRIVADAS = "datos/marcas_privadas.txt"


def cargar_marcas(ruta=MARCAS_PRIVADAS):
    """Marcas que se tapan siempre, tengan o no pinta de vocabulario.

    Se leen de un archivo aparte y no se escriben en el código, porque el código se comparte
    y una lista de marcas es tan identificatoria como la propia clave de clientes. Tiene
    prioridad sobre todo lo demás: una marca usada por muchos clientes igual se tapa.
    """
    if not Path(ruta).exists():
        print(f"AVISO: falta {ruta}; sin lista negra de marcas.")
        return [], set()
    lineas = [l.strip().upper() for l in Path(ruta).read_text(encoding="utf-8").splitlines()
              if l.strip() and not l.strip().startswith("#")]
    # Las frases se buscan sobre el texto tal cual viene, así que hay que conservar la grafía
    # con acentos además de la versión sin ellos: si solo se guardara la normalizada,
    # "<marca>" no coincidiría y sus partes sueltas pasarían el filtro una por una.
    frases = {l for l in lineas if " " in l} | {sin_acentos(l) for l in lineas if " " in l}
    tokens = {sin_acentos(l) for l in lineas if " " not in l}
    # Primero las largas: si no, "<marca>" consumiría el final de "<marca>".
    return sorted(frases, key=len, reverse=True), tokens

# Columnas de cliente y de rótulo, según cómo las nombra cada fuente.
COL_CLIENTE = ["Cliente", "cliente"]
COL_ROTULO = ["Descripcion", "Descripción", "TipoMuestra", "tipo_muestra", "variantes",
              "variables_principales", "tipo"]


def sin_acentos(texto):
    return "".join(c for c in unicodedata.normalize("NFKD", str(texto))
                   if not unicodedata.combining(c))


MARCAS_FRASE, MARCAS_TOKEN = cargar_marcas()

ACENTOS = {"A": "AÁÀÂÃ", "E": "EÉÈÊ", "I": "IÍÌÎ", "O": "OÓÒÔÕ", "U": "UÚÙÛÜ", "N": "NÑ", "C": "CÇ"}


def patron_flexible(frase):
    """Regex de una frase donde cada letra también acepta su forma acentuada.

    La lista privada guarda "<marca>" y el rótulo dice "<marca>": ni la
    grafía guardada ni su versión sin acentos coinciden con el texto, y la marca pasaba.
    """
    partes = []
    for c in sin_acentos(frase).upper():
        partes.append(f"[{ACENTOS[c]}]" if c in ACENTOS else re.escape(c))
    return r"\b" + "".join(partes) + r"\b"


def uso_por_cliente(rotulos, clientes):
    """Cuántos clientes distintos usan cada término. Un término de muchos no es marca de nadie."""
    uso = {}
    for rotulo, cli in zip(rotulos.fillna(""), clientes.fillna("?")):
        for palabra in re.split(r"[^A-Z0-9]+", sin_acentos(rotulo).upper()):
            if len(palabra) > 2:
                uso.setdefault(palabra, set()).add(cli)
    return {k: len(v) for k, v in uso.items()}


def redactar(texto, uso=None, minimo_clientes=4, solo_marcas=False):
    """Tapa de un rótulo todo lo que no sea vocabulario de producto.

    Con `solo_marcas=True` tapa únicamente la lista negra: es para columnas de códigos de lote,
    fechas o países, donde el vocabulario de producto no aplica y la regla de uso lo borraría
    todo. Sigue siendo lista negra, así que una marca desconocida se le escapa.
    """
    if not isinstance(texto, str):
        return texto
    uso = uso or {}

    # Primero las marcas de varias palabras: sus partes sueltas ("LAS", "TRES") pasarían
    # el filtro de vocabulario una por una.
    for frase in MARCAS_FRASE:
        texto = re.sub(patron_flexible(frase), "·", texto, flags=re.IGNORECASE)

    salida = []
    trozos = re.split(r"(\W+)", texto)
    for i, palabra in enumerate(trozos):
        # Separadores (" - ", " + ", "/") se conservan: no son palabras que tapar. Si no, el
        # " - " del rótulo quedaba como "·" y el "+" con que el catálogo une subtipos también.
        if not any(c.isalnum() for c in palabra):
            salida.append(palabra)
            continue
        clave = sin_acentos(palabra).upper()
        antes = trozos[i - 1] if i else ""
        despues = trozos[i + 1] if i + 1 < len(trozos) else ""
        siguiente = trozos[i + 2] if i + 2 < len(trozos) else ""
        # Sigla de un código: "LL/21/371" es un número de muestra. Sólo si lo que sigue a la
        # barra es un número: "<marca>/<marca>" también lleva barra y son dos marcas.
        anterior = trozos[i - 2] if i >= 2 else ""
        en_codigo = ((despues.startswith("/") and siguiente[:1].isdigit())
                     or (antes.endswith("/") and anterior[:1].isdigit()))
        prohibida = clave in MARCAS_TOKEN
        # "DIA" es una marca, pero "VIDA UTIL: DIA 3" es un día. Seguido de número, es el día.
        es_dia = clave == "DIA" and siguiente[:1].isdigit()
        # Un número o un código de lote (B314, L16/10/26, 35%) no identifica a nadie y sin él
        # el rótulo queda "QUESO MOZZARELLA · Lote" con el lote en otra columna.
        permitida = not prohibida and (solo_marcas or en_codigo or es_dia or any(c.isdigit() for c in clave)
                                       or clave in VOCABULARIO
                                       or uso.get(clave, 0) >= minimo_clientes)
        salida.append(palabra if permitida else "·")
    texto = re.sub(r"(·[\s·]*)+", "· ", "".join(salida))
    # "(· )" cuando lo tapado era todo lo que había entre paréntesis: se cierra como "(·)".
    # El punto se conserva también al principio y al final: "· SABOR DULCE DE LECHE" dice que
    # antes había una marca; "SABOR DULCE DE LECHE" a secas parecería un rótulo incompleto.
    return re.sub(r"· \)", "·)", texto).strip(" -")


def sin_marcas(df, *columnas, uso=None, solo_marcas=False):
    """Copia del DataFrame con los rótulos redactados, para mostrar o exportar."""
    salida = df.copy()
    for col in columnas:
        if col in salida.columns:
            salida[col] = salida[col].map(lambda x: redactar(x, uso, solo_marcas=solo_marcas))
    return salida


def cargar_mapa():
    clave = pd.read_csv(CLAVE)
    return dict(zip(clave.nombre_original, clave.codigo))


CLAVE_MARCAS = "datos/marcas_clave.csv"


def cargar_mapa_marcas():
    """marca -> código. Las marcas que son un cliente llevan su mismo IND-nn."""
    if not Path(CLAVE_MARCAS).exists():
        return {}
    clave = pd.read_csv(CLAVE_MARCAS)
    return dict(zip(clave.marca, clave.codigo))


CATALOGO = "Senasa encolumnado.xlsx"
CATALOGO_ORIGINAL = "datos/Senasa encolumnado_con_nombres.xlsx"

# Columnas del catálogo que traen códigos, fechas o países, no vocabulario de producto. En ellas
# sólo se tapa la lista negra; en el resto de las desc_* rige la regla de vocabulario.
CATALOGO_COLS_CODIGO = {
    "desc_lote", "desc_partida", "desc_muestra_n", "desc_sku", "desc_codigo",
    "desc_codigo_interno", "desc_fecha_elab", "desc_fecha", "desc_vencimiento", "desc_hora",
    "desc_gramaje", "desc_porcentaje", "desc_porcion", "desc_orden_produccion", "desc_big_bag",
    "desc_pallet", "desc_bolsa", "desc_acta", "desc_pais_destino",
}


def escribir_xlsx(df, ruta):
    """Escribe con el ancho de cada columna según su contenido y la fila de títulos fija."""
    from openpyxl.utils import get_column_letter

    with pd.ExcelWriter(ruta, engine="openpyxl") as w:
        df.to_excel(w, index=False)
        hoja = w.sheets["Sheet1"]
        for i, col in enumerate(df.columns, 1):
            largo = df[col].astype(str).str.len().quantile(0.95) if len(df) else 0
            hoja.column_dimensions[get_column_letter(i)].width = min(max(len(col), largo) + 2, 60)
        hoja.freeze_panes = "A2"


from catalogo import limpiar_lista  # noqa: E402  (mismo criterio para listas desc_*)
from nombre_informe import (anonimizar_columna_archivo, anonimizar_palabras,  # noqa: E402
                             cargar_terminos)


def anonimizar_catalogo(ruta, mapa, mapa_marcas, uso, aplicar):
    """El catálogo encolumnado: se regenera entero desde el original con nombres.

    A diferencia de los CSV, acá la marca viene en una columna propia (`desc_marca`) y se cuela
    en cualquiera de las otras 35 columnas desc_* (<marca> apareció como subtipo; <marca>, como
    lote). Por eso se tratan todas, no sólo el rótulo. Antes de redactar se normaliza con
    `catalogo.normalizar_catalogo` (portugués, errores de tipeo, consistencia de los quesos).
    """
    from catalogo import normalizar_catalogo

    origen = Path(CATALOGO_ORIGINAL) if Path(CATALOGO_ORIGINAL).exists() else ruta
    df = pd.read_excel(origen)
    sin_cli = int((~df.Cliente.astype(str).str.match(r"IND-", na=False)).sum())
    marcas = df.desc_marca.dropna()
    sin_marca = int((~marcas.astype(str).str.match(r"(IND|MARCA)-", na=False)).sum())

    if aplicar:
        df = normalizar_catalogo(df)
        df["Cliente"] = df.Cliente.map(mapa).fillna(df.Cliente)
        df["desc_marca"] = df.desc_marca.map(
            lambda m: mapa_marcas.get(sin_acentos(m).upper().strip(), m) if isinstance(m, str) else m)
        # desc_marca ya está codificada; las dos de consistencia las genera catalogo.py;
        # desc_familia y desc_tipo_muestra son vocabularios cerrados de 15 y 4 valores, sin
        # marcas, y la regla de vocabulario los mutilaba ("base nutricional" -> "base").
        texto = [c for c in df.columns if c.startswith("desc_") and df[c].dtype == object
                 and c not in ("desc_marca", "desc_consistencia", "desc_consistencia_origen",
                               "desc_estado", "desc_grasa", "desc_lactosa",
                               "desc_familia", "desc_tipo_muestra")]
        libres = ["Descripción"] + [c for c in texto if c not in CATALOGO_COLS_CODIGO]
        codigos = [c for c in texto if c in CATALOGO_COLS_CODIGO]
        df = sin_marcas(df, *libres, uso=uso)
        df = sin_marcas(df, *codigos, uso=uso, solo_marcas=True)
        # Las desc_* son listas unidas con " + ". Si lo tapado era un elemento entero, queda
        # "+ GRASO + MEDIANA HUMEDAD" o "+ · +": se saca el elemento vacío; la marca ya está en
        # desc_marca. Una lista que queda vacía pasa a sin dato.
        for c in [c for c in texto if c != "Descripción"]:
            df[c] = df[c].map(limpiar_lista)
        # Un cliente puede colarse en una columna de códigos (desc_lote: "CUÑETE CARTON <cliente>"),
        # donde lo único que se aplica es la lista negra de marcas. Se codifica por palabra completa.
        terminos = cargar_terminos_cliente()
        for c in ["Descripción"] + texto:
            df[c] = df[c].map(lambda v: anonimizar_palabras(v, terminos))
        # Un texto que empieza con "=" lo escribe openpyxl como fórmula y Excel lo rechaza al
        # abrir ("T=4HS" con la T tapada quedaba "=4HS"). Un espacio adelante lo evita.
        for c in df.columns[df.dtypes == object]:
            df[c] = df[c].map(lambda v: " " + v if isinstance(v, str) and v.startswith("=") else v)
        try:
            escribir_xlsx(df, ruta)
            estado = "reescrito desde el original" if origen != ruta else "reescrito"
        except PermissionError:
            estado = "BLOQUEADO - sigue con datos reales"
    else:
        estado = "leído"
    return {"archivo": str(ruta), "filas": len(df), "col_cliente": "Cliente",
            "sin_codificar": sin_cli + sin_marca, "rotulos": "Descripción + 35 desc_*",
            "estado": estado}


CLAVE_ALIAS = "datos/clientes_alias.csv"
COL_ARCHIVO = "archivo"


def cargar_terminos_cliente():
    """Términos de la clave (y alias) para anonimizar nombres de archivo."""
    return cargar_terminos(CLAVE, CLAVE_ALIAS)


def anonimizar_archivo(ruta, mapa, uso, aplicar, terminos=None):
    df = pd.read_csv(ruta, low_memory=False)
    col_cli = next((c for c in COL_CLIENTE if c in df.columns), None)
    cols_rot = [c for c in COL_ROTULO if c in df.columns]

    pendiente_cli = 0
    if col_cli:
        pendiente_cli = int((~df[col_cli].astype(str).str.match(r"IND-", na=False)).sum())
        if aplicar and pendiente_cli:
            df[col_cli] = df[col_cli].map(mapa).fillna("IND-XX")
    if aplicar and cols_rot:
        df = sin_marcas(df, *cols_rot, uso=uso)
    # El nombre del informe (leche cruda) puede traer el cliente: mismo criterio que `cliente`.
    if aplicar and terminos and COL_ARCHIVO in df.columns:
        df[COL_ARCHIVO] = anonimizar_columna_archivo(df[COL_ARCHIVO], terminos)

    estado = "leído"
    if aplicar:
        try:
            df.to_csv(ruta, index=False, encoding="utf-8-sig")
            estado = "reescrito"
        except PermissionError:
            # Abierto en Excel. Un archivo trabado no puede abortar la corrida entera, pero
            # tampoco puede pasar por anonimizado: se avisa fuerte.
            estado = "BLOQUEADO - sigue con datos reales"
    return {"archivo": str(ruta), "filas": len(df), "col_cliente": col_cli or "-",
            "sin_codificar": pendiente_cli, "rotulos": ", ".join(cols_rot) or "-",
            "estado": estado}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aplicar", action="store_true")
    ap.add_argument("--solo-archivo", action="store_true",
                    help="solo anonimiza la columna `archivo` de datos/leche_cruda.csv (no toca "
                         "el resto de los datasets ni pide las planillas originales)")
    args = ap.parse_args()

    if args.solo_archivo:
        fila = anonimizar_archivo(Path("datos/leche_cruda.csv"), cargar_mapa(), {}, args.aplicar,
                                  cargar_terminos_cliente())
        print(fila["archivo"], fila["estado"])
        return

    mapa = cargar_mapa()
    terminos = cargar_terminos_cliente()

    # El uso por término se mide sobre las planillas originales y no sobre los CSV del
    # proyecto: esos ya pueden estar redactados de una corrida anterior, y entonces el
    # conteo se haría sobre puntos en vez de palabras.
    crudas = [pd.read_excel(p, header=0, usecols=[0, 3]) for p in sorted(glob.glob("datos/Reporte Filtro*.xlsx"))]
    base = pd.concat(crudas, ignore_index=True)
    base.columns = ["Cliente", "Descripcion"]
    uso = uso_por_cliente(base.Descripcion, base.Cliente)
    print(f"vocabulario medido sobre {len(base)} filas originales: "
          f"{sum(1 for v in uso.values() if v >= 4)} términos de uso compartido\n")

    objetivos = (["analisis_segmentado.csv", "analisis_consolidado.csv", "datos/leche_cruda.csv"]
                 + sorted(glob.glob("datos_filtrados/*.csv"))
                 + sorted(glob.glob("datasets/*.csv")))

    filas = [anonimizar_archivo(Path(r), mapa, uso, args.aplicar, terminos)
             for r in objetivos if Path(r).exists()]
    if Path(CATALOGO).exists() or Path(CATALOGO_ORIGINAL).exists():
        filas.append(anonimizar_catalogo(Path(CATALOGO), mapa,
                                         cargar_mapa_marcas(), uso, args.aplicar))
    reporte = pd.DataFrame(filas)
    pd.set_option("display.width", 200)
    print(reporte.to_string(index=False))
    if not args.aplicar:
        print(f"\n{reporte.sin_codificar.sum()} filas con cliente sin codificar. "
              f"Correr con --aplicar para reescribirlas.")
    else:
        trabados = reporte[reporte.estado.str.startswith("BLOQUEADO")]
        if len(trabados):
            print(f"\nATENCIÓN: {len(trabados)} archivos no se pudieron reescribir porque están "
                  f"abiertos en otro programa. NO están anonimizados:")
            for a in trabados.archivo:
                print(f"   {a}")
            print("Cerralos y volvé a correr.")
        else:
            print(f"\n{len(reporte)} archivos anonimizados.")


if __name__ == "__main__":
    main()
