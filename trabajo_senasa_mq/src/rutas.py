"""Todas las rutas del trabajo, relativas a RAIZ (la carpeta trabajo_senasa_mq/).

RAIZ sale de la ubicación de este archivo (src/rutas.py → su carpeta padre). Para usar otra copia
de los datos, definir la variable de entorno TRABAJO_SENASA_MQ_RAIZ con la carpeta raíz.
Ningún otro módulo arma rutas por su cuenta: todos importan de acá.
"""
import os
from pathlib import Path

RAIZ = Path(os.environ.get("TRABAJO_SENASA_MQ_RAIZ") or Path(__file__).resolve().parents[1]).resolve()

# entradas (no se modifican)
ENTRADA = RAIZ / "datos_entrada"
SENASA = ENTRADA / "Senasa encolumnado.xlsx"
CORRECCIONES = ENTRADA / "correcciones_clase41.csv"
MQ = ENTRADA / "mq"
MQ_ANCHA = MQ / "muestras_ancha.csv"
MQ_TIPO = MQ / "mq_tipo_muestra.csv"
MQ_CRUDOS = MQ / "crudos"
LECHE_CRUDA = RAIZ.parent / "datos" / "leche_cruda.csv"      # leche cruda, industrias ya anonimizadas (IND-nn)
TEMPERATURA = ENTRADA / "temperatura_rafaela.csv"             # NASA POWER T2M diaria de Rafaela
CLIMA = ENTRADA / "clima_rafaela.csv"                         # NASA POWER diaria (descargar_clima.py); no la lee ningún cuaderno de la entrega

# código
SRC = RAIZ / "src"
COMPARTIDO = SRC / "compartido"
EXTRACTOR = COMPARTIDO / "Filtrado_de_Datos_2_ordenado.ipynb"

# resultados (se regeneran con el notebook)
RESULTADOS = RAIZ / "resultados"
EXCEL = RESULTADOS / "excel"
TABLAS = RESULTADOS / "tablas"
MODELOS = RESULTADOS / "modelos"

DOCS = RAIZ / "docs"

# Lista privada de marcas de anonimizar.py: no está en esta máquina (la ignora Syncthing). Por defecto
# se busca donde la espera .stignore (datos/ de la carpeta compartida, al lado de trabajo_senasa_mq/).
MARCAS_PRIVADAS = Path(os.environ.get("TRABAJO_SENASA_MQ_MARCAS") or RAIZ.parent / "datos" / "marcas_privadas.txt")


def preparar():
    """Crea las carpetas de resultados si faltan."""
    for d in (EXCEL, TABLAS, MODELOS):
        d.mkdir(parents=True, exist_ok=True)
