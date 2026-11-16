"""Celdas de las secciones 1, 3 y 4 de trabajo_senasa_mq.ipynb: la clase 41 (agrupamiento) aplicada a Senasa.

Ex build_clase41_senasa.py (original en _archivo/proyecto/), que armaba dos cuadernos separados
(clase41_agrupamiento_senasa.ipynb y clase41_agrupamiento_senasa_mq.ipynb, también archivados). Acá devuelve listas de
celdas que build_notebook.py junta en el notebook único:

    celdas_lectura()                     sección 1: lectura y correcciones (común a las dos versiones)
    celdas("senasa", didactico=False)    sección 3: solo Senasa (lo necesario para elegir k, tags y guardar)
    celdas("senasa_mq", didactico=True)  sección 4: Senasa + MQ, con todo el contenido didáctico de la clase 41

La lectura, las correcciones y la unión salen de datos_clase41.py (las comparte con revision_excel.py). Los outliers por
consenso y las métricas sin sesgo de desbalance salen de mejoras_clustering.py (probados en
prueba_mejoras_clustering.ipynb y aprobados); el segundo nivel por familia (subtags del DDL, de los polvos y del queso),
de segundo_nivel.py (lo que aprobó cada prueba por familia).
Los textos con números fijos (TEXTOS) se revisan contra la salida de cada versión después de ejecutar.
"""
import re

import nbformat as nbf

import celdas_queso

VERSIONES = {
    "senasa": dict(con_mq=False, numero=3, csv="clusters_senasa.csv", joblib="agrupamiento_senasa_clase41.joblib",
                   titulo="Senasa", otro="sección 4 (la misma clase con Senasa + MQ, con todos los algoritmos)"),
    "senasa_mq": dict(con_mq=True, numero=4, csv="clusters_senasa_mq.csv", joblib="agrupamiento_senasa_mq_clase41.joblib",
                      titulo="Senasa + MQ", otro="sección 3 (la misma clase solo con Senasa)"),
}


def bajar(s):
    """Baja un nivel los títulos markdown (las secciones del notebook único son «##»)."""
    return re.sub(r"(?m)^(#+) ", r"#\1 ", s)


# Textos con números: revisados contra la salida ejecutada de cada versión.
TEXTOS = {
    "senasa": dict(
        queso_nosup="""
**Respuesta.** Sí hay estructura, pero no es la del dominio y el número de grupos no es robusto. De los 1627 quesos de
Senasa, 43 salen como outliers y se agrupan 1584 (humedad y MGES estandarizadas).

- **Qué encuentra.** k-medias elige k = 5 (silueta 0,52; silueta macro 0,53; Davies-Bouldin 0,63) y la mezcla de
  gaussianas k = 7 por BIC (silueta 0,43). Son grupos más compactos que los del corte del CAA (silueta 0,31, Davies-Bouldin
  1,23): la composición no es una nube uniforme.
- **Coincidencia con el dominio.** Contra la clase de humedad del CAA el Rand ajustado (ARI) es 0,40 (k-medias) y 0,46
  (mezcla); contra la clase de MGES, 0,37 y 0,29. Los extremos se reconocen: un grupo de 135 quesos es casi puro de muy alta
  humedad (135 de 136) y otro de 87, puro de baja humedad; en el medio no hay un corte natural: dos grupos mezclan
  mediana y alta humedad (530 y 104; 140 y 253), porque el límite de 46 g/100 g del CAA cae dentro de una nube continua.
  Contra la **consistencia del rótulo** el agrupamiento da ARI 0,51 y 0,54, y la **regla del CAA aplicada directo, 0,76**:
  el corte por humedad coincide con el rótulo mejor que cualquier agrupamiento sin tag. Contra el **tipo** (variedad) pasa
  al revés: 0,46 y 0,52 contra 0,26 de la regla, porque los grupos separan también por grasa.
- **Estabilidad.** Con k fijo, otras 10 semillas dan la misma partición (ARI medio 1,00 en k-medias y 0,99 en la mezcla),
  y el k que elige la silueta es 5 en las 10 semillas (la mezcla elige 7 en las 10).

Conclusión: sin tag previo se separan grupos reproducibles en la composición, pero son una partición del continuo y
no una clasificación por tipo de queso; por eso el subtag de la rama sale de la regla del CAA, que además coincide mejor
con el rótulo.
""",
        histogramas="""
### Las cuatro columnas, una por una
Histogramas. La humedad separa los polvos (menos de 10 g/100 g) del dulce de leche (cerca de 28), con unas pocas
muestras de alta humedad (hasta 78); en Senasa casi no hay quesos ni líquidos con el panel completo. La proteína va de 2
a más de 90 y las cenizas de 0,7 a 8: unidades y rangos muy distintos.
""",
        lectura_b="""
**Lectura.** A diferencia de los socios del gimnasio (silueta 0,17 a 0,21, estructura débil), acá la silueta es alta
(0,80 ya con k = 2, sin outliers): hay grupos de verdad, porque los productos tienen composiciones muy distintas. El
máximo está en **k = 5** (0,854), con k = 3 y 4 muy cerca (0,84 y 0,85); con k = 6 cae a 0,44 y sigue entre 0,43 y 0,45
hasta k = 10: no hay un sexto grupo natural. (Con pocos reinicios el k que elige el k-medias preliminar, y con él el k final, cambiaba con la
semilla; con `n_init` = 50 no.) Ojo:
el dulce de leche es el 81 % de las muestras y forma una nube muy compacta, lo que
infla el promedio de la silueta. El diagrama de B.2 lo muestra grupo por grupo, y B.5 da las métricas que no dependen
del tamaño de cada grupo.
""",
        conclusiones="""
## G · Qué quedó
- Solo Senasa, con las correcciones de la revisión (`correcciones_clase41.csv`, «,.» → «,», NO DETECTADO = 0) y sin
  las 8 muestras con humedad + sólidos incoherentes. Sin las familias que no son lácteas (1727 muestras, 27 con los
  cuatro análisis: los cárnicos caían en ALTA_HUMEDAD): 2660 muestras con los cuatro análisis, ninguna suma más de 102.
- Outliers **dentro de cada tipo de producto**, por consenso (k-medias preliminar con k = 5; LOF, bosque de aislamiento y
  elipse dentro de cada grupo, corte 3 %, 2 de 3 votos): 63 (2,4 %), sobre todo dulces de leche (41) y sueros (19).
  Quedan 2597. La regla anterior (Q3 + 1,5·IQR) quitaría 229 (8,6 %) y se llevaría 140 de los 176 heladeros y los 11
  reducidos en grasa; el consenso conserva 152 heladeros y 3 reducidos en grasa.
- La silueta elige **k = 5** (0,854), con estructura fuerte; los mismos cinco tags con la regla anterior de outliers. Los
  cinco grupos y su tag: **DDL** (2116, dulce de leche), **POLVO_MAGRO** (335, concentrados y sueros), **POLVO_GRASO** (87,
  leche en polvo), **PROT_ALTA** (49, concentrados de más de 80 % de proteína) y **LIQUIDO** (10, humedad media 71,5:
  dulces de leche, helados y postres muy húmedos, no líquidos de verdad). **Ya no hay un sexto grupo (POLVO_GRASO_2)**: con
  k = 6 la silueta cae a 0,44 (B.1); los sueros reengrasados, que ese grupo separaba, quedan dentro de POLVO_GRASO y el
  segundo nivel los reconoce por regla (abajo). Sin los cárnicos ya no aparece ALTA_HUMEDAD; con solo Senasa no
  aparece QUESO en el k-medias: los quesos van a su rama propia (más abajo). PROT_ALTA es el grupo más débil (silueta media 0,62).
- Sin el sesgo del DDL (B.5): silueta macro 0,760, peor grupo 0,62, Davies-Bouldin 0,33, Calinski-Harabasz 7437. La
  silueta baja respecto de la regla anterior (0,869 y macro 0,791) porque ahora quedan muestras legítimas del borde de
  cada grupo; sobre las muestras que no quita ninguno de los dos filtros, los dos modelos dan lo mismo (0,876 y macro
  0,799).
- Segundo nivel por familia. **DDL** (logística entrenada con 1521 muestras de subtipo conocido): decide en 1632 de
  2116: **DDL_FAMILIAR** 854, **DDL_REPOSTERO** 621 y **DDL_HELADERO** 157; 484 quedan como DDL (452 sin probabilidad
  suficiente y 32 novedades). A las 595 sin subtipo les propone 246 familiar, 130 repostero y 7 heladero (212 sin
  decidir), lo mismo que `prueba_subtipo_ddl.ipynb`. **Polvo** (471 muestras con humedad ≤ 15; 8 outliers de la rama;
  k-medias k = 8 en base seca, silueta 0,73): por regla, **POLVO_LECHE_DESCREMADA** 29 (28 leche descremada y 1
  parcialmente descremada: todas las descremadas) y **POLVO_SUERO_REENGRASADO** 8 (2 rotuladas suero reengrasado y 6
  rotuladas suero); por k-medias, **POLVO_WPC** 275 (tres clusters de 99 a 100 % WPC; el de 45 % de proteína en
  base seca, de 29 muestras, es casi todo de IND-70: un concentrado), **POLVO_LECHE_ENTERA** 72, **POLVO_CASEINATO** 39,
  **POLVO_SUERO_PERMEADO** 32 (suero desmineralizado, suero y permeado: nombrado por la familia) y **POLVO_CONCENTRADO** 5
  (caseína). Un cluster de 7 sin tipo (grasa 16,4 y proteína 19,8 en base seca) y 4 outliers de la rama quedan con su tag. **Queso** (rama propia,
  fuera del k-medias: ninguno de los 4006 quesos de Senasa tiene los cuatro análisis, pero 1627 tienen humedad y grasa): 1627
  muestras, QUESO_SEMIDURO 793, QUESO_PASTA_BLANDA 358, QUESO_DURO 325 y QUESO_MUY_ALTA_HUMEDAD 151. El agrupamiento sin
  tag previo de esos quesos está en la sección anterior (k-medias k = 5, mezcla k = 7; ARI 0,40 y 0,46 contra la humedad del
  CAA).
- Al revelar la familia, los grupos se leen como tipos de producto; no coinciden del todo porque algunas familias
  comparten composición (suero en polvo y concentrados magros).
- En el trabajo anterior (los dos cuadernos separados) esta versión también corría los demás algoritmos: escalar casi no
  cambiaba la partición (Rand ajustado 0,98); k-medias, Ward, los enlaces completo y promedio euclidianos y los minilotes
  llegaban a la misma partición (1,00); el bisectivo que parte el grupo más grande cortaba el dulce de leche (silueta
  0,19). En el notebook único esa comparación se hace solo en la sección 4, con Senasa + MQ.
- Cada muestra queda con su `tag`, su `subtag` y su `criterio_subtag` en `resultados/tablas/clusters_senasa.csv` (2660
  del k-medias, 63 de ellas outliers sin tag, y 1627 de la rama de queso, con `rama = queso (humedad y MG)` y sin cluster),
  y el modelo con el mapeo cluster → tag y los modelos del segundo nivel en
  `resultados/modelos/agrupamiento_senasa_clase41.joblib`.
""",
    ),
    "senasa_mq": dict(
        queso_nosup="""
**Respuesta.** Sí hay estructura, pero no es la del dominio y el número de grupos es poco robusto. De los 2891 quesos de
la rama, 81 salen como outliers y se agrupan 2810 (humedad y MGES estandarizadas).

- **Qué encuentra.** k-medias elige k = 7 (silueta 0,49; silueta macro 0,54; Davies-Bouldin 0,63) y la mezcla de
  gaussianas k = 8 por BIC (silueta 0,43). Son grupos más compactos que los del corte del CAA (silueta 0,22,
  Davies-Bouldin 1,70).
- **Coincidencia con el dominio.** Contra la clase de humedad del CAA el Rand ajustado (ARI) es 0,31 (k-medias) y 0,38
  (mezcla); contra la clase de MGES, 0,29 y 0,18 (NMI de humedad 0,50 y 0,56). Los extremos se reconocen (un grupo de 101
  quesos puros de baja humedad y dos de 177 y 138 casi puros de muy alta humedad); el medio no tiene corte natural: los
  grupos de humedad intermedia mezclan mediana y alta humedad (641 y 320; 370 y 365). Contra la **consistencia del rótulo**
  el agrupamiento da ARI 0,31 y 0,37 y **la regla del CAA aplicada directo, 0,63**: el corte por humedad coincide con el
  rótulo mejor que cualquier agrupamiento sin tag. Contra el **tipo** (variedad) pasa al revés: 0,28 y 0,36 contra 0,19 de
  la regla, porque los grupos separan también por grasa.
- **Estabilidad.** Con k fijo y otras 10 semillas, el ARI contra la partición base es 1,00 en promedio en k-medias (y el k
  que elige la silueta es 7 en las 10 semillas) y 0,75 en la mezcla (de 0,51 a 0,99; el k que elige el BIC es 8 en 9 semillas
  y 7 en una). Con solo Senasa la mezcla era mucho más estable (ARI 0,99): al sumar los quesos de MQ la humedad y la grasa se
  solapan más y la partición de la mezcla deja de ser única; la del k-medias se mantiene.

Conclusión: sin tag previo se separan grupos en la composición, pero son una partición del continuo (7 a 8 zonas, y la
de la mezcla cambia con la semilla) y no una clasificación por tipo de queso; por eso el subtag de la rama sale de la regla del CAA, que
además coincide mejor con el rótulo.
""",
        histogramas="""
### Las cuatro columnas, una por una
Histogramas. La humedad tiene tres montones: los polvos (menos de 10 g/100 g), el dulce de leche (cerca de 28) y,
con los datos de MQ, los líquidos (leche fluida y suero, más de 80); los quesos ya no están: van a su rama propia. La
proteína va de 0 a más de 90 y las cenizas de 0 a 21: unidades y rangos muy distintos.
""",
        lectura_b="""
**Lectura.** A diferencia de los socios del gimnasio (silueta 0,17 a 0,21, estructura débil), acá la silueta es alta
(0,81 o más de k = 4 a 7, ya sin outliers): hay grupos de verdad, porque los productos tienen composiciones muy
distintas. Con solo Senasa también sale k = 5 (ver la sección 3); al sumar MQ entran líquidos (leche fluida, suero), que
forman un grupo propio, y la silueta es máxima en **k = 5** (0,835). Sin los quesos, que van a su rama, ya no hay un grupo
entre el dulce de leche y los polvos (en una versión anterior, con los quesos adentro, el k había sido 6, con un grupo QUESO
de 87 muestras). La curva es casi
plana entre 4 y 7 (0,81 a 0,83) y cae a 0,58 con k = 8, así que k = 5 es una elección por criterio, no una verdad. Ojo: el
dulce de leche es más de la mitad de las muestras y forma una nube muy compacta, lo que infla el promedio de la silueta. El
diagrama de B.2 lo muestra grupo por grupo, y B.5 da las métricas que no dependen del tamaño de cada grupo.
""",
        conclusiones="""
## G · Qué quedó
- Se sumaron Senasa y MQ con la regla del proyecto (Senasa manda; MQ completa huecos y aporta las muestras que no están
  en Senasa). MQ sale de `MQ encolumnado.xlsx` (misma estructura que Senasa; solo los códigos MQ, sin las 1839 MO de la tabla
  ancha, que ya son de Senasa), con las correcciones de la revisión, sin las 8 muestras con humedad + sólidos incoherentes,
  sin las familias que no son lácteas (68 con los cuatro análisis) y **sin los quesos confirmados por el catálogo, que van
  a su rama** (5915 quesos, 75 con los cuatro análisis). Quedan 3940 muestras con los cuatro análisis y 3937 tras
  descartar las 3 que suman más de 102.
- Outliers **dentro de cada tipo de producto**, por consenso (k-medias preliminar con k = 5; LOF, bosque de aislamiento y
  elipse dentro de cada grupo, corte 3 %, 2 de 3 votos): 101 (2,6 %), 57 solo MQ, 35 de Senasa y 9 de Senasa completadas
  con MQ; por familia, 29 sueros, 21 sin clasificar y 21 «queso» (solo MQ que el modelo llama queso y el rótulo no confirma:
  son raras para su grupo). Quedan 3836. La regla anterior (Q3 + 1,5·IQR) quitaría 362 (9,2 %), entre ellos 141 de
  los 176 heladeros: el consenso conserva 172 heladeros y los 11 reducidos en grasa.
- La silueta elige **k = 5** (0,835). Los cinco grupos y su tag: **DDL** (2226), **POLVO_GRASO** (644, leche en polvo),
  **POLVO_MAGRO** (446, concentrados y sueros), **LIQUIDO** (454, suero y leche fluida) y **PROT_ALTA** (66). **Ya no hay
  QUESO en el k-medias**: el tag sale solo de la rama de queso. POLVO_MAGRO y PROT_ALTA son los grupos más débiles
  (silueta media 0,63 y 0,60).
- Sin el sesgo del DDL (B.5): silueta macro 0,754, peor grupo 0,60, Davies-Bouldin 0,37, Calinski-Harabasz 17860.
  Contra la regla anterior (0,862 y macro 0,779) baja porque quedan muestras legítimas del borde; sobre las muestras que
  no quita ninguno de los dos filtros, los dos modelos dan lo mismo (0,864 y macro 0,784).
- Escalar es la decisión correcta, aunque acá la partición no cambia sin escalar (Rand ajustado 1,00).
- k-medias, minilotes y Ward llegan a particiones casi iguales (Rand ajustado 0,999 a 1). El enlace simple encadena y
  deja grupos de 1 a 5 muestras (0,64), el promedio manhattan también se aparta (0,73), y el bisectivo que parte el
  grupo **más grande** corta el dulce de leche en pedazos (0,49).
- Al revelar la familia (la de MQ encolumnado), los grupos se leen como tipos de producto (Rand ajustado contra la familia
  0,92); no coinciden del todo porque algunas familias comparten composición (suero en polvo y concentrados magros; suero
  líquido y leche fluida).
- Las muestras **solo MQ** (893 de las 3836) llevan la familia que les asigna un modelo (`mq_tipo_muestra.csv`, no revisada a
  mano): 508 caen en POLVO_GRASO y 228 en LIQUIDO. De las 1128 solo MQ que el modelo llama queso y el rótulo no confirma
  (697 con MG y ST), ninguna entra a la rama de queso; 44 tenían los cuatro análisis y entraron al k-medias, 21 salieron como
  outliers y las otras 23 cayeron en LIQUIDO (22) y POLVO_MAGRO (1).
- Segundo nivel por familia. **DDL** (logística entrenada con 1601 muestras de subtipo conocido): decide en 1699 de
  2226: **DDL_FAMILIAR** 876, **DDL_REPOSTERO** 646 y **DDL_HELADERO** 177; 527 quedan como DDL (496 sin probabilidad
  suficiente y 31 novedades). Entre las etiquetadas acierta 584 de 789 familiares, 460 de 637 reposteros y 165 de 175
  heladeros (es el ajuste sobre ellas mismas: la exactitud honesta, 0,86 en clientes no vistos, es la de
  `prueba_subtipo_ddl.ipynb`). **Polvo** (1154 muestras con humedad ≤ 15; 16 outliers de la rama; k-medias k = 7 en base seca, silueta
  0,77): por regla, **POLVO_LECHE_DESCREMADA** 73 (72 leches descremadas y 1 parcialmente descremada) y
  **POLVO_SUERO_REENGRASADO** 15 (4 rotuladas suero reengrasado, 8 rotuladas suero y 3 sin tipo); por k-medias,
  **POLVO_LECHE_ENTERA** 620, **POLVO_WPC** 310 (284 WPC; 15 rotuladas solo «suero», con proteína de WPC, y 8 sin tipo),
  **POLVO_SUERO_DESMINERALIZADO** 70 y **POLVO_CASEINATO** 40; un cluster de 18 sin subtag (78 % de proteína en base seca,
  sobre todo caseína) y los outliers de la rama quedan con su tag (PROT_ALTA 26), y 2 muestras con humedad > 15, que no son
  polvo, con el suyo (POLVO_GRASO 2). El LIQUIDO (454) no tiene segundo nivel.
- **Queso** (rama propia, fuera del k-medias; por la humedad del CAA): 2891 quesos (1627 de Senasa, 49 de Senasa completados
  con MQ y 1215 solo MQ con el rótulo que lo confirma): QUESO_SEMIDURO 1195, QUESO_PASTA_BLANDA 693, QUESO_DURO 539 y
  QUESO_MUY_ALTA_HUMEDAD 464. Contra la consistencia del rótulo: 491 de los 539 duros son de baja humedad, 997 de los 1195
  semiduros de mediana, 606 de los 693 de pasta blanda de alta; los muy altos de humedad son sobre todo untables (166) y
  procesados (162). La pregunta «¿se separan sin tag previo?» está arriba: sí, pero en 7 a 8 zonas (las de la mezcla cambian
  con la semilla) y sin coincidir con el CAA (ARI 0,31 a 0,38).
- Cada muestra queda con su `tag`, su `subtag` y su `criterio_subtag` en `resultados/tablas/clusters_senasa_mq.csv` (3937
  del k-medias, 101 de ellas outliers sin tag, y 2891 de la rama de queso, con `rama = queso (humedad y MG)` y sin cluster),
  y el modelo con el mapeo cluster → tag y los modelos del segundo nivel en
  `resultados/modelos/agrupamiento_senasa_mq_clase41.joblib`.
""",
    ),
}


LECTURA_MD = """
## 1 · Lectura y correcciones

`Senasa encolumnado.xlsx` (en `datos_entrada/`) tiene una fila por muestra y una columna `Resultado ...` por análisis.
Los resultados vienen como texto: `28,45`, `29,35 +/- 0,09`, `&lt; 0,03`. Se leen así: coma decimal, se toma el valor y
no la incertidumbre, y los «< LD» (menores al límite de detección) se toman como 0.

**Qué columnas entran en el agrupamiento.** Las cuatro de la composición centesimal que se miden **juntas**: humedad,
materia grasa, proteínas y cenizas. Los sólidos totales no entran porque son 100 − humedad (sería contar dos veces la
misma variable). La lactosa tampoco, porque casi nunca se mide junto con las otras cuatro. Entra una muestra solo si
tiene los cuatro valores, y se descartan las que suman más de 102 g/100 g (error de carga).

**Correcciones** (decididas al revisar los Excel de la sección 2). La lectura y las correcciones están en un solo
lugar, `src/datos_clase41.py`, que usan las secciones 2, 3, 4 y 6:
- las puntuales, en `datos_entrada/correcciones_clase41.csv` (Identificador, campo, antes, después, motivo);
  `Senasa encolumnado.xlsx` no se toca;
- dos reglas generales de lectura: «,.» es un tipeo de «,», y «NO DETECTADO» vale 0 por defecto (como «< LD»);
- quedan **fuera del análisis** las muestras de Senasa cuya humedad + sólidos totales no da 100 ± 1 (si falta alguno de
  los dos, no se excluyen por esto).
"""

LECTURA_CODE = r"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering, BisectingKMeans
from sklearn.metrics import silhouette_score, silhouette_samples, adjusted_rand_score

from datos_clase41 import COLUMNAS, FAMILIAS_NO_LACTEAS, N_INIT_KMEDIAS, SEMILLA, SIN_FAMILIA, SUMA_MAX, TAGS, leer_senasa, leer_mq, unir, etiquetar
import mejoras_clustering as M      # outliers por consenso y métricas (probados en prueba_mejoras_clustering.ipynb)
import segundo_nivel as N           # subtags por familia: DDL, polvos y queso (las pruebas por familia)
import clustering_quesos as Q       # la rama de queso: los quesos que confirma el catálogo, fuera del k-medias
import warnings
warnings.filterwarnings("ignore", message=".*covariance matrix.*full rank.*")   # la elipse en grupos chicos y casi planos
warnings.filterwarnings("ignore", message=".*Determinant has increased.*")

AZUL, AMBAR, GRIS, ROJO, VERDE, VIOLETA = "#0055a5", "#f0a800", "#6b7280", "#c0392b", "#2e8b57", "#7c3aed"
COLORES = [AZUL, ROJO, VERDE, AMBAR, VIOLETA, "#0e7490", "#be185d", "#4d7c0f", "#92400e", "#334155"]
pd.set_option("display.width", 140)

S = leer_senasa()                    # Senasa corregido y parseado (datos_clase41.py)
senasa = S.crudo[["Identificador", "desc_familia", "desc_producto", "desc_tipo_muestra"] + COLUMNAS].copy()
coma = sum(v_[0] for v_ in S.celdas_tipeo.values()); no_det = sum(v_[1] for v_ in S.celdas_tipeo.values())
print(f"Senasa: {len(senasa)} muestras · {int(senasa[COLUMNAS].notna().all(axis=1).sum())} con los cuatro análisis")
print(f"Reglas generales: «,.» → «,» en {coma} celdas · NO DETECTADO = 0 en {no_det} celdas · "
      f"{len(S.excluidas)} muestras excluidas por humedad + sólidos fuera de 100 ± 1")
print("Correcciones puntuales (correcciones_clase41.csv):")
S.correcciones
"""


def celdas_lectura():
    return [nbf.v4.new_markdown_cell(LECTURA_MD.strip("\n")), nbf.v4.new_code_cell(LECTURA_CODE.strip("\n"))]


def celdas(clave, didactico=True):
    """Celdas de una versión. didactico=False deja fuera A, B.3, B.4, C, D, E y las figuras de lectura de F."""
    v = VERSIONES[clave]; t = TEXTOS[clave]; con_mq = v["con_mq"]
    c = []
    md = lambda s: c.append(nbf.v4.new_markdown_cell(bajar(s.strip("\n"))))
    code = lambda s: c.append(nbf.v4.new_code_cell(s.strip("\n")))
    md_d = lambda s: md(s) if didactico else None
    code_d = lambda s: code(s) if didactico else None

    datos_intro = ("se aplican **los mismos algoritmos** a `Senasa encolumnado.xlsx`, sumándole las muestras de MQ "
                   "(`resultados/excel/MQ encolumnado.xlsx`, la misma estructura que Senasa)." if con_mq else
                   "se aplican los algoritmos a `Senasa encolumnado.xlsx`, **solo con los datos de Senasa** (sin MQ).")
    partes = ("**A · k-medias y k-means++** · **B · cuántos grupos** (codo, silueta, minilotes, escalar o no) "
              "· **C · aglomerativo** (Ward, enlaces, métricas, dendrograma) · **D · bisectivo** · **E · comparativo** "
              "· **B.5 · métricas sin el sesgo del grupo más grande** · **F · usarlo** (con el segundo nivel por familia) · **G · qué quedó**." if didactico else
              "**muestras y outliers** · **B.1 · cuántos grupos** (codo y silueta) · **B.2 · el modelo y sus tags** "
              "· **B.5 · métricas sin el sesgo del grupo más grande** · **F · qué es cada grupo, segundo nivel y guardar** "
              "· **G · qué quedó**. El resto de la clase (A, B.3, B.4, C, D, E) "
              "está en la sección 4.")
    c.append(nbf.v4.new_markdown_cell(f"""## {v["numero"]} · Agrupamiento (clase 41): {v["titulo"]}

Los dos cuadernos de la clase 41 (`Clase_41_agrupamiento_simple` y `Clase_41_agrupamiento_particional_y_jerarquico`)
agrupan socios de un gimnasio. Acá {datos_intro} Versión hermana: {v["otro"]}.

Pregunta: sin decirle al algoritmo qué producto es cada muestra, **¿hay grupos de muestras de composición parecida?
¿cuántos, y cómo son?** La familia del producto (dulce de leche, leche en polvo...), cuando se conoce, cumple el papel que tenía el *plan*
en el gimnasio: **no entra** en el agrupamiento; se guarda aparte y se compara al final. A cada grupo final se le da un
**tag** corto (DDL, POLVO_GRASO, QUESO...) que sale de su centroide, no del número de cluster. Dentro del DDL, de los
polvos y del queso, un segundo nivel por familia le da a una parte de las muestras un **subtag** (DDL_REPOSTERO,
POLVO_WPC, QUESO_SEMIDURO...).

Partes: {partes}"""))

    if con_mq:
        md("""
### Sumar los datos MQ (`resultados/excel/MQ encolumnado.xlsx`)
MQ se lee con **la misma estructura que Senasa**: de `MQ encolumnado.xlsx` (sección 7), que tiene las 177 columnas de
`Senasa encolumnado.xlsx` (texto con coma decimal; la humedad en `humedad_derivada_100_menos_solidos`, 100 − sólidos
totales) más algunas propias de MQ. Así los dos orígenes se leen, se corrigen y se filtran igual. Entran **solo los
códigos MQ** (15821): las 1839 muestras de tipo MO de `muestras_ancha.csv` quedan afuera (decisión de Matías) porque ya
forman parte de Senasa; las que tienen código en Senasa entran por Senasa como cualquier muestra de Senasa (1511 de 1839;
las otras 328 no están en Senasa y no se agregan). Se suman con la regla del proyecto (la de `build_datos_mq.py`, hoy en
`_archivo/proyecto/`): **el catálogo Senasa manda**. En una muestra que está en Senasa, MQ solo completa un análisis que
Senasa no tiene; nunca pisa un valor. Las muestras que están solo en MQ se agregan enteras, **con su `desc_familia`,
`desc_producto` y `desc_tipo_muestra` de MQ encolumnado**, y se les aplican las mismas reglas que a las de Senasa: las
familias que no son lácteas (cárnicos, panificados / otros, agua / hisopado y base nutricional: `FAMILIAS_NO_LACTEAS`) no
entran, y solo entra como queso lo que confirma el catálogo (rama de queso, más abajo). La familia de las solo MQ la
asigna un modelo (`mq_tipo_muestra.csv`) y no está revisada a mano; las que el modelo no clasifica quedan en
«sin clasificar», que no es lo mismo que «sin familia»: no se sabe que no sean lácteas y siguen al k-medias. La familia no
entra en el agrupamiento: se guarda aparte y se usa al comparar. MQ encolumnado tiene huecos donde la tabla ancha tiene
valor (se verifica en `datos_clase41.leer_mq`; donde las dos tienen valor, coinciden): esas muestras no completan los
cuatro análisis.
""")
        code(r"""
mq_crudo, mq = leer_mq()
todas, completado = unir(senasa, mq)
n_inicial = len(todas)                                     # todas las muestras de Senasa y de MQ, antes de cualquier filtro
todas = todas[~todas["Identificador"].isin(S.excluidas)]   # humedad + sólidos de Senasa incoherentes
no_lactea = todas["desc_familia"].isin(FAMILIAS_NO_LACTEAS)
print(f"Fuera por no ser lácteas ({', '.join(FAMILIAS_NO_LACTEAS)}): {int(no_lactea.sum())} muestras, "
      f"{int(todas.loc[no_lactea, COLUMNAS].notna().all(axis=1).sum())} con los cuatro análisis")
todas = todas[~no_lactea]
rama, confirmados, sin_confirmar, conteo_q = Q.rama_queso(S, True)   # los quesos que confirma el catálogo van a su rama
a_rama = todas["Identificador"].isin(confirmados)
print(f"Quesos confirmados por el catálogo: {len(confirmados)} · de las muestras de acá salen {int(a_rama.sum())} "
      f"({int(todas.loc[a_rama, COLUMNAS].notna().all(axis=1).sum())} con los cuatro análisis): no pasan por el k-medias")
todas = todas[~a_rama]
completas = todas.dropna(subset=COLUMNAS)
muestras = completas[completas[COLUMNAS].sum(axis=1) <= SUMA_MAX].reset_index(drop=True)
familia = muestras["desc_familia"]                  # se guarda aparte: NO entra en el agrupamiento
conocida = (familia != SIN_FAMILIA).values
print(f"MQ: {len(mq)} muestras · {int(mq['Identificador'].isin(senasa['Identificador']).sum())} también están en Senasa · "
      f"{int((todas['origen'] == 'solo MQ').sum())} solo en MQ")
print("Valores de Senasa completados con MQ, por análisis:", completado)
print(f"Con los cuatro análisis: {len(completas)} · descartadas por sumar más de {SUMA_MAX}: {len(completas) - len(muestras)} · quedan {len(muestras)}")
print("Por origen:", muestras["origen"].value_counts().to_dict())
print("Familias (no entran):", familia.value_counts().to_dict())
muestras[COLUMNAS].describe().round(2).T
""")
    else:
        md("""
### Las muestras que entran
Solo Senasa: no se completa nada con MQ. Se quitan las excluidas por humedad + sólidos y las familias que no son
lácteas (cárnicos, panificados / otros, agua / hisopado y base nutricional: `FAMILIAS_NO_LACTEAS`) y los quesos, que van
a su rama propia (más abajo); se piden los cuatro análisis y se descartan las que suman más de 102. Todas tienen familia
conocida.
""")
        code(r"""
n_inicial = len(senasa)                                    # todas las muestras de Senasa, antes de cualquier filtro
todas = senasa[~senasa["Identificador"].isin(S.excluidas)].assign(origen="Senasa")
no_lactea = todas["desc_familia"].isin(FAMILIAS_NO_LACTEAS)
print(f"Fuera por no ser lácteas ({', '.join(FAMILIAS_NO_LACTEAS)}): {int(no_lactea.sum())} muestras, "
      f"{int(todas.loc[no_lactea, COLUMNAS].notna().all(axis=1).sum())} con los cuatro análisis")
todas = todas[~no_lactea]
rama, confirmados, sin_confirmar, conteo_q = Q.rama_queso(S, False)  # los quesos (desc_familia = queso) van a su rama
a_rama = todas["Identificador"].isin(confirmados)
print(f"Quesos de Senasa: {len(confirmados)} · salen de acá {int(a_rama.sum())} ({int(todas.loc[a_rama, COLUMNAS].notna().all(axis=1).sum())} con los cuatro análisis): no pasan por el k-medias")
todas = todas[~a_rama]
completas = todas.dropna(subset=COLUMNAS)
muestras = completas[completas[COLUMNAS].sum(axis=1) <= SUMA_MAX].reset_index(drop=True)
familia = muestras["desc_familia"]                  # se guarda aparte: NO entra en el agrupamiento
conocida = (familia != SIN_FAMILIA).values          # en esta versión, todas
print(f"Con los cuatro análisis: {len(completas)} · descartadas por sumar más de {SUMA_MAX}: {len(completas) - len(muestras)} · quedan {len(muestras)}")
print("Familias (no entran):", familia.value_counts().to_dict())
muestras[COLUMNAS].describe().round(2).T
""")

    md_d(t["histogramas"])

    code_d(r"""
fig, ejes = plt.subplots(1, 4, figsize=(16, 3.6))
for ax, col in zip(ejes, COLUMNAS):
    ax.hist(muestras[col], bins=40, color=AZUL, alpha=0.85); ax.set_title(f"{col} (g/100 g)", fontsize=10)
fig.suptitle("Cuatro columnas de composición: rangos muy distintos, sin escalar la proteína decidiría sola")
plt.tight_layout(); plt.show()
""")

    md("""
### Imputar y escalar
No hay faltantes (se pidieron los cuatro análisis), pero el imputador queda en el flujo como en la clase, para que una
muestra nueva con un hueco también se pueda ubicar. Después se **escala**: media 0 y desvío 1 por columna. `Z` es lo
que ven los algoritmos.
""")

    code(r"""
imputador = SimpleImputer(strategy="median").fit(muestras[COLUMNAS])
escalador = StandardScaler().fit(imputador.transform(muestras[COLUMNAS]))
Z = escalador.transform(imputador.transform(muestras[COLUMNAS]))
print("Desvío de cada columna antes de escalar:", {c_: round(float(muestras[c_].std()), 2) for c_ in COLUMNAS})
print(f"Z: {Z.shape[0]} muestras × {Z.shape[1]} columnas · media ≈ {Z.mean(axis=0).round(2).tolist()} · desvío ≈ {Z.std(axis=0).round(2).tolist()}")
""")

    md(f"""
### Quitar los outliers
Un filtro global por columna (IQR o z-score sobre todas las muestras) no sirve acá: el dulce de leche es la mayoría, y
un filtro así marcaría como «raros» a todos los polvos{" y líquidos" if con_mq else ""}, que son grupos reales. Un outlier es una muestra
rara **para su tipo de producto**. Como la familia no se puede usar{" (y las muestras de MQ no la tienen)" if con_mq else ""}, el tipo sale de
un **k-medias preliminar**, con k elegido por silueta, igual que en B.1. Dentro de cada grupo se decide por **consenso**
(clase 46; `M.outliers_consenso`, probado en `prueba_mejoras_clustering.ipynb`):

1. k-medias preliminar sobre `Z`; cada muestra, a su grupo.
2. Dentro de cada grupo (de 30 muestras o más) se reescala con las muestras del grupo y se calculan tres puntajes, donde
   más bajo es más raro: el **factor local** (`LocalOutlierFactor`, 50 vecinos), el **bosque de aislamiento**
   (`IsolationForest`, 200 árboles) y la **elipse** (`EllipticEnvelope`).
3. Cada método marca el **3 %** de menor puntaje del grupo; es outlier lo que marcan **al menos 2 de los 3**.

Antes se usaba otra regla: distancia al centroide del grupo preliminar mayor que **Q3 + 1,5 · IQR** de las distancias
del grupo. Se reemplazó porque sacaba variantes legítimas del dulce de leche: el heladero (menos humedad, más cenizas)
queda lejos del centroide del DDL, y la regla IQR se llevaba casi todos. Abajo se comparan las dos reglas.

Se quitan, se vuelve a imputar y escalar con las que quedan, y **todo lo que sigue usa el conjunto sin outliers**.
""")

    por_origen = 'print("Outliers por origen:", muestras.loc[es_outlier, "origen"].value_counts().to_dict())\n' if con_mq else ""
    quedan = ('print(f"Quedan {len(muestras)} muestras · por origen: {muestras[\'origen\'].value_counts().to_dict()}")' if con_mq
              else 'print(f"Quedan {len(muestras)} muestras")')
    code(r"""
PCT_OUTLIER = 3                       # corte de cada método (% más raro del grupo); 2 de 3 votos
X_pre, Z_pre = muestras[COLUMNAS].to_numpy(), Z          # todas las que entran, antes de quitar (para B.5)
out_iqr, grupo_pre, K_PRE = M.outliers_iqr(Z)            # k-medias preliminar (k por silueta) y la regla IQR anterior
es_outlier, votos = M.outliers_consenso(X_pre, grupo_pre, pct=PCT_OUTLIER, minimo=2, n_vecinos=50)
ref = M.agregar_subtipo(muestras, S)                     # subtipo conocido y descripciones: NO entran, solo para comparar
muestras["subtipo_ref"] = ref["subtipo_ref"].to_numpy()
especiales = M.especiales_ddl(ref)                       # DDL repostero, heladero, reducido en grasa (por la descripción)

resumen = pd.DataFrame({"grupo preliminar": grupo_pre, "outlier": es_outlier, "regla IQR (antes)": out_iqr})
print(f"k-medias preliminar con k = {K_PRE} · outliers por consenso ({PCT_OUTLIER} %, 2 de 3): {int(es_outlier.sum())} de {len(Z)} "
      f"({100 * es_outlier.mean():.1f} %) · con la regla IQR de antes serían {int(out_iqr.sum())} ({100 * out_iqr.mean():.1f} %)")
display(resumen.groupby("grupo preliminar").agg(muestras=("outlier", "size"), outliers=("outlier", "sum"),
                                                **{"con IQR (antes)": ("regla IQR (antes)", "sum")}).assign(
    **{"% outliers": lambda t: (100 * t["outliers"] / t["muestras"]).round(1)}))
print("Cuántas marca cada método y cuántas votos juntan:", votos[["factor local", "bosque", "elipse"]].sum().to_dict(),
      "· votos:", votos["votos"].value_counts().sort_index().to_dict())
print("Consenso contra la regla IQR de antes:")
display(pd.crosstab(pd.Series(out_iqr, name="regla IQR (antes)"), pd.Series(es_outlier, name="consenso (ahora)"), margins=True))
conservados = pd.DataFrame({n: {"muestras": int(especiales[n].sum()), "se conservan (consenso)": int((especiales[n] & ~es_outlier).sum()),
                                "se conservaban (IQR)": int((especiales[n] & ~out_iqr).sum())} for n in especiales.columns}).T
print("Dulces de leche especiales (por la descripción, que no entra en el agrupamiento):"); display(conservados)
""" + por_origen + r"""print("Outliers por familia:", muestras.loc[es_outlier, "desc_familia"].value_counts().to_dict())
print("Algunos outliers (g/100 g), los de más votos:")
display(muestras.loc[es_outlier, ["Identificador", "desc_familia", "origen"] + COLUMNAS].assign(votos=votos.loc[es_outlier, "votos"].to_numpy())
        .sort_values("votos", ascending=False).head(12))

fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5))
for ax, (x, y) in [(a, ("humedad", "proteinas")), (b, ("grasa", "cenizas"))]:
    ax.scatter(muestras.loc[~es_outlier, x], muestras.loc[~es_outlier, y], s=6, alpha=0.4, color=GRIS, label="se quedan")
    solo_iqr = out_iqr & ~es_outlier
    ax.scatter(muestras.loc[solo_iqr, x], muestras.loc[solo_iqr, y], s=10, color=AMBAR, alpha=0.6, label="solo los quitaba la regla IQR (se quedan)")
    ax.scatter(muestras.loc[es_outlier, x], muestras.loc[es_outlier, y], s=18, color=ROJO, label="outliers")
    ax.set_xlabel(f"{x} (g/100 g)"); ax.set_ylabel(f"{y} (g/100 g)")
a.legend()
fig.suptitle(f"Outliers: raros dentro de su grupo preliminar para al menos 2 de 3 métodos (corte {PCT_OUTLIER} %)")
plt.tight_layout(); plt.show()

# se guardan todas (con la marca de outlier) para el csv del final; desde acá, sin outliers
muestras_pre = muestras.assign(outlier=es_outlier)
muestras = muestras[~es_outlier].reset_index(drop=True)
familia = muestras["desc_familia"]; conocida = (familia != SIN_FAMILIA).values
imputador = SimpleImputer(strategy="median").fit(muestras[COLUMNAS])
escalador = StandardScaler().fit(imputador.transform(muestras[COLUMNAS]))
Z = escalador.transform(imputador.transform(muestras[COLUMNAS]))
""" + quedan)

    md_d("""
## A · Centroides: k-medias y k-means++
### A.1 · Asignar y mover, a mano
El algoritmo de la clase en cuatro líneas, sobre dos columnas (humedad y proteínas, escaladas) para poder dibujarlo:
k centroides iniciales; cada muestra al centroide más cercano; cada centroide a la media de su grupo; se repite. Los
centroides arrancan a propósito en las muestras de menor humedad, para ver cómo se mueven. La inercia baja en cada paso.
""")

    code_d(r"""
i_h, i_p = COLUMNAS.index("humedad"), COLUMNAS.index("proteinas")
dos = Z[:, [i_h, i_p]]

def asignar(X, centroides):
    return np.argmin(np.linalg.norm(X[:, None, :] - centroides[None, :, :], axis=2), axis=1)
def mover(X, etiquetas, k, anteriores):
    return np.array([X[etiquetas == j].mean(axis=0) if (etiquetas == j).any() else anteriores[j] for j in range(k)])
def inercia(X, centroides, etiquetas):
    return float(np.sum((X - centroides[etiquetas]) ** 2))

k = 4
centroides = dos[np.argsort(dos[:, 0])[:k]]                 # todos en la humedad más baja, a propósito
fig, ejes = plt.subplots(2, 4, figsize=(17, 7)); ejes = ejes.ravel()
for paso in range(8):
    etiquetas = asignar(dos, centroides)
    ax = ejes[paso]; ax.scatter(dos[:, 0], dos[:, 1], c=[COLORES[e] for e in etiquetas], s=6, alpha=0.6)
    ax.scatter(centroides[:, 0], centroides[:, 1], marker="X", s=140, c="black", edgecolor="white", zorder=5)
    ax.set_title(f"paso {paso}: inercia {inercia(dos, centroides, etiquetas):.0f}", fontsize=10)
    ax.set_xlabel("humedad (escalada)", fontsize=8); ax.set_ylabel("proteínas (escaladas)", fontsize=8)
    nuevos = mover(dos, etiquetas, k, centroides)
    if np.allclose(nuevos, centroides):
        for ax_ in ejes[paso + 1:]: ax_.axis("off")
        break
    centroides = nuevos
fig.suptitle("k-medias a mano sobre humedad y proteínas: asignar, mover, repetir", y=1.01)
plt.tight_layout(); plt.show()
""")

    md_d("""
### A.2 · Converge, pero adónde: inicialización al azar contra k-means++
Cuarenta corridas de `KMeans(init="random", n_init=1)` y cuarenta con `init="k-means++"` sobre las cuatro columnas,
con k = 4: dónde termina cada una. Por eso la biblioteca corre `n_init` veces y se queda con la menor inercia, y por eso
k-means++ (centroides iniciales elegidos con probabilidad proporcional a D(x)²) es el defecto.
""")

    code_d(r"""
in_azar = np.array([KMeans(n_clusters=4, init="random", n_init=1, random_state=s).fit(Z).inertia_ for s in range(40)])
in_pp = np.array([KMeans(n_clusters=4, init="k-means++", n_init=1, random_state=s).fit(Z).inertia_ for s in range(40)])
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist([in_azar, in_pp], bins=25, color=[GRIS, AZUL], label=["init al azar", "k-means++"])
ax.set_xlabel("inercia final (n_init = 1)"); ax.set_ylabel("corridas"); ax.legend()
ax.set_title("40 corridas de cada una: dónde termina k-medias según dónde empieza")
plt.show()
mejor = min(in_azar.min(), in_pp.min())
print(f"Al azar: inercia entre {in_azar.min():.0f} y {in_azar.max():.0f} · {int((in_azar > mejor * 1.01).sum())} de 40 terminan más de 1 % arriba de la mejor")
print(f"k-means++: inercia entre {in_pp.min():.0f} y {in_pp.max():.0f} · {int((in_pp > mejor * 1.01).sum())} de 40 terminan más de 1 % arriba de la mejor")
""")

    md("""
## B · Cuántos grupos
### B.1 · El codo y la silueta
Para k de 2 a 10: la **inercia** (baja siempre; se busca el codo) y la **silueta** promedio, (b − a)/max(a, b). Criterio
fijado antes de mirar: **k con la silueta máxima**.
""")

    code(r"""
ks = list(range(2, 11))
modelos = {k_: KMeans(n_clusters=k_, n_init=N_INIT_KMEDIAS, random_state=SEMILLA).fit(Z) for k_ in ks}
inercias = [modelos[k_].inertia_ for k_ in ks]; siluetas = [silhouette_score(Z, modelos[k_].labels_) for k_ in ks]
K = ks[int(np.argmax(siluetas))]
fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4))
a.plot(ks, inercias, marker="o", color=AZUL, lw=2); a.set_xlabel("k"); a.set_ylabel("inercia"); a.set_title("La inercia baja siempre: se busca el codo")
b.plot(ks, siluetas, marker="o", color=AMBAR, lw=2); b.set_xlabel("k"); b.set_ylabel("silueta promedio"); b.set_title(f"La silueta: máximo en k = {K}")
for k_, s_ in zip(ks, siluetas): b.text(k_, s_ + 0.01, f"{s_:.2f}", ha="center", fontsize=9)
b.margins(y=0.3); plt.tight_layout(); plt.show()
print("Inercia por k:", {k_: round(float(v)) for k_, v in zip(ks, inercias)})
print("Silueta por k:", {k_: round(float(s_), 3) for k_, s_ in zip(ks, siluetas)})
print(f"Se elige k = {K}.")
""")

    md(t["lectura_b"])

    md("""
### B.2 · El modelo elegido, sus tags y su diagrama de silueta
Los centroides se devuelven a g/100 g para leerlos. Cada grupo recibe un **tag** con las reglas sobre su centroide
(`etiquetar` en `datos_clase41.py`): humedad > 70 → LIQUIDO; entre 35 y 65 con grasa > 15 → QUESO (se evalúa antes que alta humedad); > 50 → ALTA_HUMEDAD; > 15 → DDL;
si no es un polvo: proteína > 70 → PROT_ALTA; grasa > 15 → POLVO_GRASO; si no, POLVO_MAGRO. El número de cluster cambia
entre corridas; el tag no. Si dos grupos caen en la misma regla se desambiguan con `_2`, `_3`, ordenados por humedad.
El diagrama: un coeficiente por muestra, ordenados por grupo; la altura es cuántas muestras tiene, el ancho qué tan adentro
están, la línea roja el promedio.
""")

    code(r"""
km = modelos[K]; grupos = km.labels_
centros = pd.DataFrame(escalador.inverse_transform(km.cluster_centers_), columns=COLUMNAS, index=range(K)).round(2)
nombres, tags, repetidos = etiquetar(centros)
if repetidos:
    print(f"AVISO: más de un grupo cae en {repetidos}; se desambiguan con sufijo _2, _3 (por humedad del centroide).")
tag_de = np.array([tags[g] for g in grupos])
centros.insert(0, "muestras", np.bincount(grupos)); centros.insert(0, "tag", [tags[j] for j in range(K)])
centros.index.name = "cluster"
print("Centroides en g/100 g:"); display(centros)

def diagrama_silueta(ax, X, etiquetas, titulo, rotulos):
    valores = silhouette_samples(X, etiquetas); y0 = 10
    for j in range(etiquetas.max() + 1):
        v = np.sort(valores[etiquetas == j]); ax.fill_betweenx(np.arange(y0, y0 + len(v)), 0, v, color=COLORES[j], alpha=0.8)
        ax.text(-0.29, y0 + len(v) / 2, rotulos[j], fontsize=8); y0 += len(v) + 20
    ax.axvline(valores.mean(), color=ROJO, ls="--", label=f"promedio {valores.mean():.2f}")
    ax.set_xlim(-0.3, 1); ax.set_yticks([]); ax.set_xlabel("coeficiente de silueta"); ax.legend(fontsize=8); ax.set_title(titulo, fontsize=10)
    return valores

fig, ax = plt.subplots(figsize=(9, 5))
coef = diagrama_silueta(ax, Z, grupos, f"Diagrama de silueta, k-medias con k = {K}", tags)
plt.show()
print("Silueta media por tag:", pd.Series(coef).groupby(tag_de).mean().round(2).to_dict())
print(f"Muestras con coeficiente negativo: {int((coef < 0).sum())} de {len(Z)}")
""")

    md("""
### B.5 · Métricas sin el sesgo del grupo más grande
La silueta promedio pesa cada **muestra** igual, así que la domina el DDL, que es la mayoría y forma una nube compacta.
Junto a ella (clase 43, `M.metricas`): la **silueta macro** (el promedio de las siluetas medias de cada grupo: cada
**grupo** pesa igual), la del **peor grupo**, **Davies-Bouldin** (cuánto se pisan los grupos con su vecino más
parecido; menor es mejor) y **Calinski-Harabasz** (dispersión entre grupos sobre dispersión dentro; mayor es mejor).

Para comparar el filtro de outliers nuevo con el de antes no alcanza con la silueta de cada uno sobre sus propias
muestras: el que quita más saca justo lo que está entre grupos y sale favorecido. Por eso se rehace el k-medias final
con cada filtro (`M.variante`) y se mide también sobre las **muestras comunes** (las que no quita ninguno de los dos), en
la misma escala.
""")

    code(r"""
print("Métricas del modelo elegido (sus muestras):")
display(pd.DataFrame([M.metricas(Z, grupos)]).round(3))
comunes = ~(out_iqr | es_outlier)
filtros = {}
for nombre_f, quita in [("regla IQR (antes)", out_iqr), (f"consenso {PCT_OUTLIER} % (ahora)", es_outlier)]:
    r_f, filtros[nombre_f], _ = M.variante(X_pre, quita, Z_pre, comunes)
    for e in especiales.columns:
        filtros[nombre_f][f"conserva {e}"] = int((especiales[e].to_numpy() & ~quita).sum())
assert r_f["K"] == K and (r_f["tag_de"] == tag_de).all()     # el k-medias de M.variante es el mismo de B.2
comparacion_filtros = pd.DataFrame(filtros).T.infer_objects()
print(f"Los dos filtros, cada uno con su k-medias final · muestras comunes: {int(comunes.sum())}")
display(comparacion_filtros.style.format(precision=3).format({"% quitado": "{:.1f}", "Calinski-Harabasz": "{:.0f}"}))
""")

    md_d("""
### B.3 · Más rápido: Elkan y minilotes
Las dos variantes de la clase: `algorithm="elkan"` (usa la desigualdad triangular para calcular menos distancias) y
`MiniBatchKMeans` (lotes en vez de todo el conjunto). Con unas pocas miles de muestras y cuatro columnas, todo tarda
milisegundos; lo que importa es si la inercia empeora y si los grupos cambian.
""")

    code_d(r"""
filas = []
for nombre, modelo in [("k-medias (lloyd)", KMeans(n_clusters=K, n_init=N_INIT_KMEDIAS, algorithm="lloyd", random_state=SEMILLA)),
                       ("k-medias (elkan)", KMeans(n_clusters=K, n_init=N_INIT_KMEDIAS, algorithm="elkan", random_state=SEMILLA)),
                       ("minilotes (256)", MiniBatchKMeans(n_clusters=K, n_init=N_INIT_KMEDIAS, batch_size=256, random_state=SEMILLA))]:
    t0 = time.perf_counter(); modelo.fit(Z); ms = (time.perf_counter() - t0) * 1000
    filas.append({"variante": nombre, "ms": ms, "inercia": modelo.inertia_, "silueta": silhouette_score(Z, modelo.labels_),
                  "Rand ajustado contra k-medias": adjusted_rand_score(grupos, modelo.labels_)})
display(pd.DataFrame(filas).round(3))
""")

    md_d("""
### B.4 · Sin escalar
El mismo k-medias sobre las columnas en g/100 g. La proteína y la humedad (los desvíos más grandes) dominan la
distancia; las cenizas casi no cuentan. En los socios del gimnasio esto cambiaba todo (la cuota decidía sola). Acá los
grupos están tan separados en humedad y proteína que la partición cambia poco: se mide con el Rand ajustado.
""")

    code_d(r"""
X_crudo = imputador.transform(muestras[COLUMNAS])
km_crudo = KMeans(n_clusters=K, n_init=N_INIT_KMEDIAS, random_state=SEMILLA).fit(X_crudo)
print(f"Rand ajustado entre sin escalar y escalado: {adjusted_rand_score(grupos, km_crudo.labels_):.2f}")
print("Sin escalar contra escalado (tag):"); display(pd.crosstab(pd.Series(km_crudo.labels_, name="sin escalar"), pd.Series(tag_de, name="escalado")))
print("Centroides sin escalar (g/100 g):"); display(pd.DataFrame(km_crudo.cluster_centers_, columns=COLUMNAS).round(2))
""")

    md_d("""
## C · Fusionar: el aglomerativo
### C.1 · Ward
`AgglomerativeClustering(n_clusters=K, linkage="ward")`: cada muestra empieza sola y se fusionan de a dos las que menos
aumentan la varianza, hasta que quedan K. Se compara con k-medias: tamaños, tabla cruzada, Rand ajustado, silueta.
""")

    code_d(r"""
ward = AgglomerativeClustering(n_clusters=K, linkage="ward").fit(Z)
print(f"Ward: tamaños {np.bincount(ward.labels_).tolist()} · silueta {silhouette_score(Z, ward.labels_):.3f} "
      f"(k-medias {silhouette_score(Z, grupos):.3f}) · Rand ajustado contra k-medias {adjusted_rand_score(grupos, ward.labels_):.2f}")
display(pd.crosstab(pd.Series(ward.labels_, name="Ward"), pd.Series(tag_de, name="k-medias (tag)")))
""")

    md_d("""
### C.2 · Los enlaces y las métricas
Simple, completo y promedio, con distancia euclidiana, Manhattan y coseno (Ward solo admite la euclidiana). El enlace
decide qué es «cercano» entre dos grupos; el simple encadena vecino a vecino y suele dejar un grupo gigante. La silueta
se calcula con la misma métrica, así que no es comparable entre métricas.
""")

    code_d(r"""
filas = []
for enlace in ["single", "complete", "average"]:
    for metrica in ["euclidean", "manhattan", "cosine"]:
        et = AgglomerativeClustering(n_clusters=K, linkage=enlace, metric=metrica).fit_predict(Z)
        filas.append({"enlace": enlace, "métrica": metrica, "tamaños": sorted(np.bincount(et).tolist(), reverse=True),
                      "silueta (misma métrica)": silhouette_score(Z, et, metric=metrica),
                      "Rand ajustado contra k-medias": adjusted_rand_score(grupos, et)})
filas.append({"enlace": "ward", "métrica": "euclidean", "tamaños": sorted(np.bincount(ward.labels_).tolist(), reverse=True),
              "silueta (misma métrica)": silhouette_score(Z, ward.labels_), "Rand ajustado contra k-medias": adjusted_rand_score(grupos, ward.labels_)})
display(pd.DataFrame(filas).round(3))
""")

    md_d("""
### C.3 · El dendrograma
Con miles de muestras el árbol no se lee. Se toman 40 al azar **estratificadas por grupo de k-medias** (para que
aparezcan los grupos chicos) y se dibuja el dendrograma de Ward: la altura de cada unión es la distancia a la que se
fusionaron; se corta donde la rama vertical es más larga. Las hojas llevan el tag de k-medias y la familia, que el
algoritmo no vio.
""")

    code_d(r"""
from scipy.cluster.hierarchy import linkage, dendrogram
rng = np.random.default_rng(SEMILLA)
por_grupo = max(2, 40 // K)
muestra = np.concatenate([rng.choice(np.where(grupos == j)[0], min(por_grupo, int((grupos == j).sum())), replace=False) for j in range(K)])
enlaces = linkage(Z[muestra], method="ward")
fig, ax = plt.subplots(figsize=(15, 5))
dendrogram(enlaces, labels=[f"{tag_de[i]} · {familia[i][:16]}" for i in muestra], leaf_font_size=7, color_threshold=enlaces[-(K - 1), 2], ax=ax)
ax.axhline(enlaces[-(K - 1), 2] * 1.02, color=ROJO, ls="--", label=f"corte: {K} grupos"); ax.set_ylabel("distancia de fusión (Ward)"); ax.legend()
ax.set_title(f"Dendrograma de {len(muestra)} muestras (Ward); las hojas dicen el tag y la familia, que no entró")
plt.tight_layout(); plt.show()
saltos = np.diff(enlaces[:, 2]); k_sugerido = len(enlaces) + 1 - (int(np.argmax(saltos)) + 1)
print(f"Últimas cinco fusiones a distancias {enlaces[-5:, 2].round(1).tolist()} · el salto más grande deja {k_sugerido} grupos")
""")

    md_d("""
## D · Dividir: k-medias bisectivo
`BisectingKMeans`: parte un grupo en dos, y otra vez, hasta K. Las dos estrategias para elegir qué grupo partir: el más
grande o el de mayor inercia. Primero se ven las divisiones sucesivas sobre humedad y proteínas.
""")

    code_d(r"""
fig, ejes = plt.subplots(1, 4, figsize=(18, 4.2))
for ax, kb in zip(ejes, [2, 3, 4, 5]):
    bk = BisectingKMeans(n_clusters=kb, bisecting_strategy="biggest_inertia", random_state=SEMILLA).fit(Z)
    ax.scatter(muestras["humedad"], muestras["proteinas"], c=[COLORES[e] for e in bk.labels_], s=6, alpha=0.6)
    ax.set_title(f"{kb - 1} división(es): {kb} grupos", fontsize=10); ax.set_xlabel("humedad"); ax.set_ylabel("proteínas")
fig.suptitle("Bisectivo (mayor inercia): cada paso parte un grupo en dos; los demás no se tocan", y=1.02)
plt.tight_layout(); plt.show()
filas = []
for estrategia in ["largest_cluster", "biggest_inertia"]:
    bk = BisectingKMeans(n_clusters=K, bisecting_strategy=estrategia, random_state=SEMILLA).fit(Z)
    filas.append({"estrategia": estrategia, "tamaños": np.bincount(bk.labels_).tolist(), "inercia": bk.inertia_,
                  "silueta": silhouette_score(Z, bk.labels_), "Rand ajustado contra k-medias": adjusted_rand_score(grupos, bk.labels_)})
filas.append({"estrategia": "k-medias (referencia)", "tamaños": np.bincount(grupos).tolist(), "inercia": km.inertia_,
              "silueta": silhouette_score(Z, grupos), "Rand ajustado contra k-medias": 1.0})
display(pd.DataFrame(filas).round(3))
""")

    md_d("""
## E · El comparativo: cuál para qué
Los cuatro algoritmos con K grupos: silueta, inercia, tiempo, tamaños y coincidencia con k-medias. Y, recién ahora,
la **familia** (que ninguno vio): Rand ajustado contra la familia.
""")

    code_d(r"""
filas, particiones = [], {}
for nombre, modelo in [("k-medias", KMeans(n_clusters=K, n_init=N_INIT_KMEDIAS, random_state=SEMILLA)),
                       ("minilotes", MiniBatchKMeans(n_clusters=K, n_init=N_INIT_KMEDIAS, random_state=SEMILLA)),
                       ("aglomerativo (Ward)", AgglomerativeClustering(n_clusters=K, linkage="ward")),
                       ("bisectivo", BisectingKMeans(n_clusters=K, bisecting_strategy="biggest_inertia", random_state=SEMILLA))]:
    t0 = time.perf_counter(); et = modelo.fit_predict(Z); ms = (time.perf_counter() - t0) * 1000
    particiones[nombre] = et
    filas.append({"algoritmo": nombre, "silueta": silhouette_score(Z, et), "inercia": getattr(modelo, "inertia_", np.nan), "ms": ms,
                  "tamaños": sorted(np.bincount(et).tolist(), reverse=True), "Rand ajustado contra k-medias": adjusted_rand_score(grupos, et),
                  "Rand ajustado contra la familia (conocida)": adjusted_rand_score(familia[conocida], et[conocida])})
display(pd.DataFrame(filas).round(3))

fig, ejes = plt.subplots(1, 4, figsize=(18, 4.2))
for ax, (nombre, et) in zip(ejes, particiones.items()):
    ax.scatter(muestras["humedad"], muestras["proteinas"], c=[COLORES[e] for e in et], s=6, alpha=0.6)
    ax.set_title(nombre, fontsize=10); ax.set_xlabel("humedad"); ax.set_ylabel("proteínas")
fig.suptitle(f"Los cuatro algoritmos con {K} grupos, vistos en humedad y proteínas", y=1.02)
plt.tight_layout(); plt.show()
""")

    md("""
## F · Usarlo
### Qué es cada grupo: cluster → tag → nombre → centroide → familia mayoritaria
El centroide en g/100 g, cuántas muestras, la familia más común y la tabla cruzada con la familia. El tag y el nombre
salen de las reglas sobre el centroide (B.2), no de la familia.
""")

    pct_mq = ('descripcion["% solo MQ"] = [round(100 * (muestras.loc[grupos == j, "origen"] == "solo MQ").mean()) for j in range(K)]\n'
              if con_mq else "")
    origen_tab = ('print("Grupos contra el origen del dato:"); display(pd.crosstab(pd.Series(tag_de, name="tag"), muestras["origen"]))\n'
                  if con_mq else "")
    code(r"""
descripcion = centros.copy(); descripcion.insert(1, "nombre", [nombres[j] for j in range(K)])
descripcion["silueta media"] = [round(float(coef[grupos == j].mean()), 2) for j in range(K)]
fam_c = familia[conocida]; gr_c = grupos[conocida]
descripcion["familia mayoritaria"] = [fam_c[gr_c == j].mode()[0] if (gr_c == j).any() else "-" for j in range(K)]
descripcion["% de esa familia"] = [round(100 * (fam_c[gr_c == j] == fam_c[gr_c == j].mode()[0]).mean()) if (gr_c == j).any() else 0 for j in range(K)]
""" + pct_mq + r"""display(descripcion)
tabla = pd.crosstab(pd.Series(tag_de, name="tag"), familia)
display(tabla.loc[:, tabla.sum().sort_values(ascending=False).index])
""" + origen_tab + r"""
fig, ejes = plt.subplots(1, K, figsize=(3.6 * K, 3.8), sharey=True)
for j, ax in enumerate(np.atleast_1d(ejes)):
    ax.bar(COLUMNAS, km.cluster_centers_[j], color=COLORES[j]); ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"{tags[j]}\n{nombres[j]} ({np.bincount(grupos)[j]})", fontsize=9); ax.tick_params(axis="x", rotation=45, labelsize=8)
np.atleast_1d(ejes)[0].set_ylabel("centroide (escalado)")
fig.suptitle("Cada grupo contra la muestra promedio: arriba de 0 es «más que el promedio»", y=1.03)
plt.tight_layout(); plt.show()

fig, ejes = plt.subplots(1, 4, figsize=(17, 3.8))
for ax, col in zip(ejes, COLUMNAS):
    ax.boxplot([muestras.loc[grupos == j, col] for j in range(K)], tick_labels=[tags[j] for j in range(K)], showfliers=False)
    ax.set_title(f"{col} por grupo", fontsize=10); ax.tick_params(axis="x", rotation=45, labelsize=8)
plt.tight_layout(); plt.show()
""")

    md_d("""
### Los grupos en dos pares de columnas
Todas las muestras con el color de su grupo (calculado con las cuatro columnas) y su tag.
""")

    code_d(r"""
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 5))
for j in range(K):
    sel = grupos == j
    a.scatter(muestras.loc[sel, "humedad"], muestras.loc[sel, "proteinas"], s=8, alpha=0.5, color=COLORES[j], label=f"{tags[j]} ({sel.sum()})")
    b.scatter(muestras.loc[sel, "grasa"], muestras.loc[sel, "cenizas"], s=8, alpha=0.5, color=COLORES[j])
a.set_xlabel("humedad (g/100 g)"); a.set_ylabel("proteínas (g/100 g)"); a.legend(fontsize=8)
b.set_xlabel("grasa (g/100 g)"); b.set_ylabel("cenizas (g/100 g)")
fig.suptitle(f"Los {K} grupos de k-medias, vistos en dos pares de columnas")
plt.tight_layout(); plt.show()
""")

    md_d("""
### Una muestra nueva
Se arma la fila, se imputa y escala con lo aprendido; `predict` da el grupo (y su tag) y `transform` la distancia a cada
centroide. Dos ejemplos: un dulce de leche típico y una leche en polvo entera.
""")

    code_d(r"""
nuevas = pd.DataFrame([{"humedad": 28.0, "grasa": 7.0, "proteinas": 6.5, "cenizas": 1.6},
                       {"humedad": 3.0, "grasa": 26.0, "proteinas": 25.0, "cenizas": 5.8}], index=["dulce de leche típico", "leche en polvo entera"])[COLUMNAS]
z_nuevas = escalador.transform(imputador.transform(nuevas))
for (etiqueta, fila), z in zip(nuevas.iterrows(), z_nuevas):
    g = int(km.predict(z[None, :])[0]); d = km.transform(z[None, :])[0]
    print(f"{etiqueta} {fila.to_dict()} → {tags[g]} ({nombres[g]}) · distancias: { {tags[j]: round(float(v), 2) for j, v in enumerate(d)} }")
""")

    md_d("""
### Diez muestras, con su tag y su silueta
Diez muestras elegidas para cubrir el rango de la silueta: de la que está bien adentro de su grupo a la que está sobre
un límite (o del lado equivocado). Es la tabla para discutir: ¿la muestra mal ubicada es un error de carga, un producto
raro, o un grupo que falta?
""")

    code_d(r"""
objetivos = np.linspace(coef.max(), coef.min(), 10); elegidos, usados = [], set()
for objetivo in objetivos:
    for candidato in np.argsort(np.abs(coef - objetivo), kind="stable"):
        if candidato not in usados: elegidos.append(candidato); usados.add(candidato); break
elegidos = np.array(elegidos)
explicado = muestras.loc[elegidos, ["Identificador", "desc_familia", "desc_producto"] + COLUMNAS].copy()
explicado["tag"] = tag_de[elegidos]
explicado["silueta"] = coef[elegidos].round(3)
explicado.style.format(precision=2).background_gradient(subset=["silueta"], cmap="RdYlGn", vmin=-0.3, vmax=1)
""")

    md("""
### Segundo nivel por familia: subtags del DDL, de los polvos y del queso
El primer nivel separa tipos de producto; adentro de tres de ellos hay variantes que importan. Cada rama usa el método
que aprobó su prueba por familia (README, «Pruebas por familia»; código en `src/segundo_nivel.py`). Las ramas van por el
**tag**, no por la familia del rótulo: así el subtag queda siempre dentro de su tag y se puede predecir para una muestra
nueva con los cuatro análisis.

| rama | muestras | método | subtags |
|---|---|---|---|
| DDL | tag DDL | **regresión logística** multinomial sobre los cuatro análisis, entrenada con el subtipo conocido (`desc_subtipo`, o el de `desc_producto` o `mq_tipo_muestra.csv`); decide si la probabilidad es ≥ 0,70 y la muestra no es novedad (Mahalanobis robusta o LOF). Exactitud balanceada 0,86 en clientes no vistos (`prueba_subtipo_ddl.ipynb`) | DDL_FAMILIAR, DDL_REPOSTERO, DDL_HELADERO; «DDL» si no decide |
| polvo | tags POLVO_GRASO, POLVO_MAGRO y PROT_ALTA con humedad ≤ 15 | primero dos **reglas de composición en base seca**: suero reengrasado (grasa ≥ 15 y proteína ≤ 12) y leche descremada (grasa ≤ 2,5, cenizas ≥ 7 y proteína 30 a 40); el resto, outliers por consenso dentro de la rama y **k-medias** (k por silueta) en grasa, proteína y cenizas **en base seca**; un cluster lleva subtag si su tipo del rótulo tiene pureza ≥ 70 % y cobertura ≥ 50 % o, si no, su familia pureza ≥ 70 %, contado sin las muestras de las reglas (`prueba_clustering_polvos.ipynb`) | POLVO_SUERO_REENGRASADO, POLVO_LECHE_DESCREMADA, POLVO_LECHE_ENTERA, POLVO_WPC, POLVO_CASEINATO, POLVO_SUERO_...; el tag si no |
| queso | los quesos que confirma el catálogo (rama propia, fuera del k-medias; abajo) | **clase de humedad del CAA** (art. 605): < 36 duro, 36 a 46 semiduro, 46 a 55 pasta blanda, ≥ 55 muy alta humedad. Coincide con la consistencia del rótulo mejor que cualquier agrupamiento (`prueba_clustering_quesos.ipynb`) | QUESO_DURO, QUESO_SEMIDURO, QUESO_PASTA_BLANDA, QUESO_MUY_ALTA_HUMEDAD |

Las dos reglas del polvo existen porque el k-medias no separa esos productos: escala la proteína con todos los polvos
(de 3 a 91 g/100 g de extracto seco), y a esa escala el suero reengrasado queda con la leche entera o con los sueros
magros, y la leche descremada (proteína 35, cenizas 8) con un concentrado de suero de 46 % de proteína y las mismas
cenizas: ningún k de 6 a 14 le da un cluster puro. Por proteína sola la leche descremada (34 a 38,5) y el WPC 35 (35 a
39) se pisan; los separan las cenizas (8,1 contra 6,2) y la grasa (1,1 contra 3,4). Los umbrales salen de los rótulos y
se verifican abajo (subtag × tipo del rótulo).

Los demás tags quedan sin subtag (su subtag es el tag). El queso no depende del k-medias: sale de su rama. El subtipo o tipo del rótulo se usa para entrenar (DDL) y para
nombrar los clusters (polvo), **nunca como entrada**. Reemplaza al segundo nivel anterior (mezcla de gaussianas dentro del
DDL, que daba DDL_HELADERO y DDL_FAMILIAR y nunca el repostero). Los subtags se juzgan por pureza y cobertura, no con la
silueta: son un segundo nivel dentro de un grupo, no grupos del primero.
""")

    code(r"""
subtag_de, criterio_de, modelos_subtag = N.ajustar(muestras, tag_de, S)
print("De dónde sale el subtag:", pd.Series(criterio_de).value_counts().to_dict())
print("Muestras por subtag:", pd.Series(subtag_de).value_counts().to_dict())
base_ = N.tag_base(tag_de)          # sin el sufijo _2, _3: los clusters repetidos van a la misma rama
ddl_ = base_ == "DDL"
m_ = modelos_subtag["DDL"]
print(f"DDL · logística entrenada con {m_['etiquetadas']} muestras con subtipo {m_['por_clase']}; "
      f"decide en {int((criterio_de[ddl_] == N.C_DDL).sum())} de {int(ddl_.sum())}")
print("DDL · subtag (filas) contra el subtipo conocido (columnas); en las que tienen subtipo es el ajuste sobre ellas mismas:")
display(pd.crosstab(subtag_de[ddl_], muestras.loc[ddl_, "subtipo_ref"].fillna("sin subtipo").to_numpy(),
                    rownames=["subtag"], colnames=["subtipo conocido"], margins=True, margins_name="total"))
if "polvo" in modelos_subtag:
    p_ = modelos_subtag["polvo"]
    print(f"Polvo · {p_['n']} muestras en la rama, {p_['n_outliers']} outliers de la rama · k-medias con k = {p_['k']} "
          f"(silueta {p_['siluetas'][p_['k']]:.3f}) · por cluster (base seca, g/100 g de extracto seco):")
    display(p_["descripcion"])
    pol_ = N.es_polvo(tag_de)
    print("Polvo · tag (filas) × subtag (columnas):")
    display(pd.crosstab(tag_de[pol_], subtag_de[pol_], rownames=["tag"], colnames=["subtag"]))
    tipo_ = N._referencias_polvo(muestras.loc[pol_, "Identificador"], S)["tipo"].fillna("sin tipo").to_numpy()
    print("Polvo · subtag (filas) × tipo del rótulo (columnas); el tipo no entra, sirve para verificar las reglas y los nombres:")
    t_ = pd.crosstab(subtag_de[pol_], tipo_, rownames=["subtag"], colnames=["tipo del rótulo"])
    display(t_.loc[:, t_.sum().sort_values(ascending=False).index])
if "queso" in modelos_subtag:
    q_ = base_ == "QUESO"
    print("Queso · subtag × familia del rótulo:")
    display(pd.crosstab(subtag_de[q_], muestras.loc[q_, "desc_familia"].to_numpy(), rownames=["subtag"], colnames=["familia"],
                        margins=True, margins_name="total"))
""")

    c.extend(celdas_queso.celdas(con_mq, t["queso_nosup"]))

    md(f"""
### Guardar: el tag y el subtag de cada muestra y el agrupamiento
`{v["csv"]}`: una fila por muestra que pasó el filtro (incluidos los outliers, que quedan sin cluster, tag ni subtag),
con Identificador, familia, origen, los cuatro análisis, cluster, tag, subtag (el tag cuando no hay subtag),
criterio_subtag (de qué rama y regla sale), silueta, outlier, votos_outlier (cuántos de los tres métodos la marcan) y, en
el DDL, las tres probabilidades de la logística (p_familiar, p_heladero, p_repostero). Y `{v["joblib"]}`: imputar,
escalar y k-medias en un `Pipeline`, con el mapeo cluster → tag, los nombres, los centroides, el filtro de outliers usado
y los modelos del segundo nivel (`segundo_nivel`: la logística y los detectores de novedad del DDL; el escalador, el
k-medias, el mapeo cluster → subtag y los outliers de la rama del polvo; los cortes de humedad del queso).
`N.predecir` los usa.
""")

    code(r"""
import joblib
from sklearn.pipeline import Pipeline
muestras["cluster"] = grupos; muestras["tag"] = tag_de; muestras["subtag"] = subtag_de; muestras["criterio_subtag"] = criterio_de
muestras["silueta"] = coef
prob_ = N.probabilidades_ddl(modelos_subtag, muestras.loc[N.tag_base(tag_de) == "DDL", COLUMNAS]).round(4)
for c_ in prob_:
    muestras[c_] = np.nan
    muestras.loc[N.tag_base(tag_de) == "DDL", c_] = prob_[c_].to_numpy()
salida = muestras_pre[["Identificador", "desc_familia", "origen"] + COLUMNAS + ["outlier"]].rename(columns={"desc_familia": "familia"})
salida["votos_outlier"] = votos["votos"].to_numpy()
salida = salida.merge(muestras[["Identificador", "cluster", "tag", "subtag", "criterio_subtag", "silueta"] + list(prob_)],
                      on="Identificador", how="left")
salida["cluster"] = salida["cluster"].astype("Int64"); salida["silueta"] = salida["silueta"].round(4)
salida["rama"] = "k-medias"
filas_queso = pd.DataFrame({"Identificador": rama["Identificador"], "familia": "queso", "origen": rama["origen"], "humedad": rama["humedad"],
                            "grasa": rama["mg"], "proteinas": rama["proteina"], "cenizas": rama["cenizas"], "tag": "QUESO",
                            "subtag": rama["subtag"], "criterio_subtag": rama["criterio_subtag"], "outlier": False,
                            "rama": "queso (humedad y MG)"})
salida = pd.concat([salida, filas_queso], ignore_index=True)       # los quesos de la rama: sin cluster ni silueta
salida = salida[["Identificador", "familia", "origen", "rama"] + COLUMNAS + ["cluster", "tag", "subtag", "criterio_subtag", "silueta",
                                                                           "outlier", "votos_outlier"] + list(prob_)]
salida.to_csv(R.TABLAS / "__CSV__", index=False, encoding="utf-8-sig")
print(f"Guardado: resultados/tablas/__CSV__ ({len(salida)} filas: {int((salida['rama'] == 'k-medias').sum())} del k-medias y {len(filas_queso)} de la rama de queso; "
      f"{int(salida['tag'].notna().sum())} con tag, {int(salida['outlier'].sum())} outliers)")
print("tag:", salida["tag"].value_counts(dropna=False).to_dict())
print("subtag:", salida["subtag"].value_counts(dropna=False).to_dict())
print("criterio_subtag:", salida["criterio_subtag"].value_counts(dropna=False).to_dict())

agrupamiento = Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler()),
                         ("kmedias", KMeans(n_clusters=K, n_init=N_INIT_KMEDIAS, random_state=SEMILLA))]).fit(muestras[COLUMNAS])
assert (agrupamiento.predict(muestras[COLUMNAS]) == grupos).all()
out_polvo_ = muestras["Identificador"].isin(modelos_subtag.get("polvo", {}).get("outliers", [])).to_numpy()
assert (N.predecir(modelos_subtag, muestras[COLUMNAS], tag_de, out_polvo_)[0] == subtag_de).all()
joblib.dump({"modelo": agrupamiento, "tags": tags, "nombres": nombres, "columnas": COLUMNAS, "centros": centros,
             "outliers": {"método": "consenso (LOF 50 vecinos, IsolationForest 200, EllipticEnvelope) dentro del k-medias preliminar",
                          "k_preliminar": K_PRE, "pct": PCT_OUTLIER, "minimo_votos": 2},
             "segundo_nivel": N.para_guardar(modelos_subtag),
             "rama_queso": {"criterio": "familia queso del catálogo (Senasa) o rótulo con variedad (solo MQ); humedad = 100 − ST; subtag por clase de humedad del CAA",
                            "cortes_humedad": N.CORTES_QUESO, "muestras": int(len(rama))}},
            R.MODELOS / "__JOBLIB__")
print("Guardado: resultados/modelos/__JOBLIB__ · cluster → tag:", tags, "· ramas del segundo nivel:", list(modelos_subtag),
      "· polvo, cluster → subtag:", modelos_subtag.get("polvo", {}).get("mapa"))
""".replace("__CSV__", v["csv"]).replace("__JOBLIB__", v["joblib"]))

    md("""
### Cuántos datos entran en cada entrenamiento
La celda siguiente guarda los conteos de esta versión (cuántas muestras hay antes de los filtros, cuántas saca cada filtro,
cuántas entran a cada modelo); el apartado «Datos usados en cada entrenamiento», al final del cuaderno, los junta en una
tabla. No calcula nada nuevo.
""")

    code(r"""
ENTRENOS = globals().get("ENTRENOS", {})
n_nolact, n_quesos = int(no_lactea.sum()), int(a_rama.sum())
_ddl, _pol = modelos_subtag["DDL"], modelos_subtag["polvo"]
ENTRENOS["__CLAVE__"] = dict(
    inicial=n_inicial, n_hs=n_inicial - len(todas) - n_nolact - n_quesos, n_nolact=n_nolact, n_quesos=n_quesos,
    n_sin4=len(todas) - len(completas), n_suma=len(completas) - len(muestras_pre),
    pre=len(muestras_pre), k_pre=int(K_PRE), n_out=int(es_outlier.sum()), final=len(muestras), k=int(K), silueta=float(coef.mean()),
    ddl_n=int(ddl_.sum()), ddl_lab=int(_ddl["etiquetadas"]), ddl_clases=dict(_ddl["por_clase"]),
    ddl_decide=int((criterio_de[ddl_] == N.C_DDL).sum()),
    pol_tag=int(pol_.sum()), pol_rama=int(_pol["n"]), pol_out=int(_pol["n_outliers"]), pol_k=int(_pol["k"]),
    pol_sil=float(_pol["siluetas"][_pol["k"]]),
    q_leidos=int(conteo_q["quesos leídos"]), q_mgst=int(conteo_q["con MG y ST"]), q_rama=len(rama),
    q_sin_confirmar=int(conteo_q.get("solo MQ de familia queso sin confirmar por el rótulo (no entran)", 0)),
    q_out=int(ns["n_outliers"]), q_base=len(ns["base"]), q_k_km=int(ns["k_km"]), q_k_gm=int(ns["k_gm"]),
    comparacion=__DID__)
print({k_: v_ for k_, v_ in ENTRENOS["__CLAVE__"].items()})
""".replace("__CLAVE__", clave).replace("__DID__", str(bool(didactico))))

    md(t["conclusiones"])

    return c
