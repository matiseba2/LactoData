# trabajo_senasa_mq

El trabajo actual sobre Senasa + MQ en una sola carpeta, con **un solo notebook**: `trabajo_senasa_mq.ipynb`.
Reemplaza lo que estaba repartido en `proyecto/` y `datos/` (mudanza del 2026-11-08; lo viejo está en `../_archivo/`).

## Qué hay en cada carpeta

| ruta | qué es |
|---|---|
| `trabajo_senasa_mq.ipynb` | el notebook único. **No se edita a mano**: lo genera `build_notebook.py`. |
| `build_notebook.py` | arma el notebook con las celdas de `src/`. |
| `notebook_general.ipynb` | **el cuaderno de entrega para la cátedra**, ejecutado y con salidas. Contiene portada, ocho partes (I–IV: el notebook único, `comparacion_numerico_categorico`, `regresion_ciclo_leche_cruda` y `logistica_calidad_leche_cruda`; V: `prediccion_tipo`; VI: `prediccion_deslactosado`; VII: `arbol_decision`; VIII: `temperatura_leche_cruda`, con los títulos un nivel más abajo) y las conclusiones generales. No se edita a mano: lo genera `build_notebook_general.py` desde los ocho .ipynb ya regenerados, más `conclusiones_generales.md`. Se ejecuta entero en un kernel (`--ExecutePreprocessor.timeout=7200`). |
| `datos_entrada/` | entradas; no se modifican. `Senasa encolumnado.xlsx` (catálogo anonimizado), `correcciones_clase41.csv` (correcciones puntuales decididas al revisar los Excel). |
| `datos_entrada/mq/` | MQ: `muestras_ancha.csv` (una fila por muestra, ex `datos/mq_procesado/`; el agrupamiento ya no la lee para los valores: sale de `MQ encolumnado.xlsx`), `mq_tipo_muestra.csv` (familia, tipo y marca, ex `datos/mq_procesado/tipo_muestra/`) y `crudos/` (los siete CSV de `datos/datos mq/`; solo los usa la sección 7). |
| `src/rutas.py` | todas las rutas, relativas a la carpeta del trabajo (`RAIZ`; se cambia con la variable de entorno `TRABAJO_SENASA_MQ_RAIZ`). |
| `src/datos_clase41.py` | lectura de Senasa y MQ, correcciones, unión (Senasa manda) y tags de los grupos. También las dos constantes de azar: `SEMILLA = 42` (pedido de la cátedra; se usa siempre a través de la constante, nunca un número suelto) y `N_INIT_KMEDIAS = 50` (reinicios de todo k-medias). Con 10 reinicios, el k-medias preliminar del consenso de outliers caía en mínimos locales y el k final de solo Senasa dependía de la semilla (6 con la 2026, 4 con la 42); con 50, ocho semillas probadas dan K = 5. |
| `src/datos_entrenamiento.py` | formato único de la tabla «Datos usados en cada entrenamiento» (`fila`, `guardar`): por cada modelo, cuántas filas hay disponibles, cuántas se excluyen y por qué, cuántas se usan y cómo se reparten en entrenamiento, validación y prueba. Cada cuaderno arma sus filas con los números de su propia corrida. |
| `src/revision_excel.py` | Excel de inconsistencias (ex `exportar_dataset_senasa_mq.py`). |
| `src/celdas_clase41.py` | celdas de las secciones 1, 3 y 4 (ex `build_clase41_senasa.py`). |
| `src/mejoras_clustering.py` | outliers por consenso y métricas sin sesgo de desbalance (probados en `prueba_mejoras_clustering.ipynb`; los usan las secciones 2, 3 y 4). Conserva la mezcla de gaussianas del DDL (`subtags`), que ya no usa el notebook único (la usan las dos pruebas del DDL para comparar). |
| `src/segundo_nivel.py` | segundo nivel por familia de las secciones 3 y 4: subtags del DDL (regresión logística), de los polvos (k-medias en base seca) y del queso (humedad del CAA); `ajustar`, `predecir`. |
| `src/clustering_quesos.py`, `src/clustering_polvos.py`, `src/clustering_todas_hg.py` | agrupamiento propio de los quesos y de los polvos, y la prueba de humedad + grasa sobre todas las familias (pruebas por familia, más abajo); `segundo_nivel.py` usa el de los polvos. |
| `src/cortes_caa.py` | tabla de cortes para categorizar (ex `build_cortes_categorias.py`). |
| `src/categorizar.py`, `src/celdas_asociacion.py` | categorías, ítems y reglas de asociación (ex `build_categorias_asociacion.py`). |
| `src/mq_encolumnado.py` | MQ con el formato de Senasa encolumnado (ex `build_mq_encolumnado.py`, ahora la función `armar(regenerar)`). |
| `src/compartido/` | módulos que se usan sin cambios: `catalogo.py` (el de `proyecto/`, **no** `datos/catalogo.py` del curso), `anonimizar.py`, `nombre_informe.py` y `Filtrado_de_Datos_2_ordenado.ipynb` (la sección 7 ejecuta su celda `extraer_metadatos`). |
| `resultados/excel/` | `dataset_senasa_revision.xlsx`, `dataset_senasa_mq_revision.xlsx`, `MQ encolumnado.xlsx`, `MQ encolumnado - solo nuevas.xlsx`. |
| `resultados/tablas/` | `clusters_*.csv` (con `tag`, `subtag`, `criterio_subtag`, `outlier`, `votos_outlier` y, en el DDL, `p_familiar` / `p_heladero` / `p_repostero`), `cortes_categorias.csv`, `categorias_*.csv`, `transacciones_*.parquet`. |
| `resultados/modelos/` | `agrupamiento_senasa_clase41.joblib`, `agrupamiento_senasa_mq_clase41.joblib` (pipeline, mapeo cluster → tag, centroides, filtro de outliers y modelos del segundo nivel, clave `segundo_nivel`); `arbol_decision.joblib` (la cascada de árboles de V3 de la Parte VII, que usa `app_arbol.py`). |
| `resultados/datos_entrenamiento/` | un CSV por cuaderno (`trabajo_senasa_mq.csv`, `comparacion_numerico_categorico.csv`, `regresion_ciclo_leche_cruda.csv`, `logistica_calidad_leche_cruda.csv`, `prediccion_tipo.csv`, `prediccion_deslactosado.csv`, `arbol_decision.csv`, `temperatura_leche_cruda.csv`) con la tabla «Datos usados en cada entrenamiento»; `build_notebook_general.py` los junta en la portada del cuaderno general. |
| `docs/cortes_categorias.md` | criterios de los cortes y decisiones abiertas. |
| `docs/manifiesto_movimientos.csv` | registro de la mudanza: cada archivo movido, copiado o creado, con bytes y sha256. |
| `docs/mudanza.py`, `docs/deshacer.py` | el script que hizo la mudanza y el que la deshace. |

La carpeta de resultados se llama `resultados` y no `salidas` a propósito: `.stignore` ignora `salidas` en cualquier
nivel y no se sincronizaría.

## El notebook

0 rutas y parámetros · 1 lectura y correcciones · 2 Excel de inconsistencias (solo Senasa; Senasa + MQ) · 3 agrupamiento
solo Senasa · 4 agrupamiento Senasa + MQ, con todo el contenido de la clase 41 (codo, silueta, Ward, enlaces,
dendrograma, bisectivo, comparativo, tags) · 5 cortes CAA · 6 categorías y reglas de asociación · 7 MQ encolumnado ·
8 datos usados en cada entrenamiento (`resultados/datos_entrenamiento/trabajo_senasa_mq.csv`) · 9 verificación.

Las secciones van en orden: la 6 usa los `clusters_*.csv` de la 3 y la 4 y el `cortes_categorias.csv` de la 5.

### Agrupamiento (secciones 3 y 4)

0. **Solo productos lácteos.** No entran las familias `FAMILIAS_NO_LACTEAS` de `src/datos_clase41.py`: cárnicos
   (embutidos y queso de cerdo, que con su humedad ≈ 62 y proteína ≈ 18 caían en el tag QUESO), panificados / otros
   (incluye la lactosa alimenticia), agua / hisopado y base nutricional (fórmulas de soja, de caseinatos, sustitutos
   lácteos). Las solo MQ llevan la familia de `MQ encolumnado.xlsx` (la asigna un modelo, no está revisada a mano) y se
   les aplican las mismas reglas; las «sin clasificar» no son «sin familia»: siguen al k-medias.
0b. **Fuente de MQ.** MQ se lee con la misma estructura que Senasa: de `resultados/excel/MQ encolumnado.xlsx` (177
   columnas de Senasa encolumnado más las propias de MQ; humedad = `humedad_derivada_100_menos_solidos`), con
   `desc_familia`, `desc_producto` y `desc_tipo_muestra` para las solo MQ (`datos_clase41.leer_mq`). Senasa manda; MQ solo
   completa huecos. **Las 1839 muestras de tipo MO de `muestras_ancha.csv` no entran por MQ** (decisión de Matías): ya
   forman parte de Senasa; 1511 tienen código en Senasa (entran por Senasa) y 328 no (no se agregan; 90 de ellas tienen
   los cuatro análisis). Donde las dos fuentes tienen valor coinciden, pero MQ encolumnado tiene huecos que la tabla ancha
   no (humedad 940, grasa 923, proteína 808, cenizas 669 filas): esas muestras ya no completan los cuatro análisis.
   `desc_condicion` sale como número decimal (float; 3 valores: 50, 60 y 70) en MQ encolumnado y como texto («24 HS») en Senasa: no se corrigió.
0c. **Queso, rama propia.** Los quesos que confirma el catálogo (Senasa: `desc_familia = queso`; solo MQ: además el
   rótulo nombra una variedad) no pasan por el k-medias, porque casi ninguno tiene proteína ni cenizas. Van a una rama
   con humedad (100 − ST) y materia grasa, con tag QUESO y subtag por la clase de humedad del CAA (art. 605), en las dos
   versiones (`clustering_quesos.rama_queso`, `segundo_nivel.subtag_queso`). Cada muestra tiene una sola asignación: los
   quesos confirmados se sacan del k-medias antes de elegir k. Las solo MQ que el modelo llama queso y el rótulo no
   confirma no entran a la rama y siguen en el k-medias si tienen los cuatro análisis. Se guardan en `clusters_*.csv`
   con `rama = queso (humedad y MG)` y sin cluster. En cada versión, el notebook también agrupa esos quesos **sin tag
   previo** (k-medias y mezcla de gaussianas sobre humedad y MGES) y lo compara con el CAA y el rótulo.
1. **Outliers por consenso** dentro de cada grupo de un k-medias preliminar (k por silueta): se reescala dentro del
   grupo y se corren LOF (50 vecinos), IsolationForest (200 árboles) y EllipticEnvelope; cada uno marca el 3 % más raro
   y es outlier lo que marcan al menos 2 de 3. Reemplazó a la regla Q3 + 1,5·IQR de la distancia al centroide, que
   sacaba casi todos los DDL heladeros.
2. **k-medias final** con k por silueta y **tags** por el centroide (`etiquetar`).
3. **Métricas** junto a la silueta global: silueta macro, peor grupo, Davies-Bouldin y Calinski-Harabasz, y la
   comparación de los dos filtros sobre las muestras que no quita ninguno.
4. **Segundo nivel por familia** (`src/segundo_nivel.py`; lo que aprobó cada prueba por familia, más abajo). Las ramas
   van por el tag, así el subtag está siempre dentro de su tag y se predice con los cuatro análisis:
   - **DDL** (tag DDL): regresión logística multinomial entrenada con el subtipo conocido; decide con probabilidad ≥ 0,70
     si la muestra no es novedad. `DDL_FAMILIAR`, `DDL_REPOSTERO`, `DDL_HELADERO`; si no decide, `DDL`.
   - **Polvo** (tags POLVO_GRASO, POLVO_MAGRO y PROT_ALTA con humedad ≤ 15): primero dos **reglas de composición en
     base seca** (`REGLAS_POLVO`), porque el k-medias no separa esos productos: `POLVO_SUERO_REENGRASADO` (suero con
     oleomargarina: grasa ≥ 15 y proteína ≤ 12) y `POLVO_LECHE_DESCREMADA` (grasa ≤ 2,5, cenizas ≥ 7 y proteína 30 a
     40; por proteína sola se pisa con el WPC 35, la separan las cenizas, 8,0 contra 6,2, y la grasa, 1,1 contra 3,4).
     El resto: outliers por consenso dentro de la rama y k-medias (k por silueta) en grasa, proteína y cenizas en base
     seca; un cluster lleva subtag si coincide con un tipo del rótulo (o, si no, con una familia), contado sin las
     muestras de las reglas. Los clusters sin tipo y los outliers de la rama quedan con su tag (salvo que los tome una
     regla).
   - **Queso** (rama propia, fuera del k-medias; ver 0c): la clase de humedad del CAA (art. 605). El subtag no entra como
     ítem en la sección 6 (repetiría `humedad=`).
   - Las ramas van por el tag base (`tag_base`): si dos clusters caen en la misma regla, los dos van a la misma rama.
     Con la semilla 42 y 50 reinicios no aparece ningún caso (ya no hay `POLVO_GRASO_2`: sale solo con el corte de
     outliers de 2 % de la prueba de mejoras), pero la regla queda por si vuelve.
   - Los demás tags (LIQUIDO, y ALTA_HUMEDAD si aparece) no tienen segundo nivel: subtag = tag. La columna `criterio_subtag` dice de
     qué rama y regla sale cada subtag.

   Reemplazó a la mezcla de gaussianas dentro del DDL, que daba `DDL_HELADERO` y `DDL_FAMILIAR` y nunca el repostero
   (exactitud balanceada 0,53 contra 0,86 de la logística en clientes no vistos).

| | solo Senasa: IQR (antes) | solo Senasa: consenso | Senasa + MQ: IQR (antes) | Senasa + MQ: consenso |
|---|---|---|---|---|
| outliers | 229 (8,6 %) | 63 (2,4 %) | 362 (9,2 %) | 101 (2,6 %) |
| DDL heladeros que se conservan (de 176) | 36 | 152 | 35 | 172 |
| k y tags | 5: DDL, LIQUIDO, POLVO_GRASO, POLVO_MAGRO, PROT_ALTA | 5, los mismos | 5 | 5: DDL, POLVO_GRASO, POLVO_MAGRO, LIQUIDO, PROT_ALTA (sin QUESO) |
| silueta global / macro | 0,869 / 0,791 | 0,854 / 0,760 | 0,862 / 0,779 | 0,835 / 0,754 |
| Davies-Bouldin / Calinski-Harabasz | 0,32 / 9846 | 0,33 / 7437 | 0,36 / 23252 | 0,37 / 17860 |
| DDL: FAMILIAR / REPOSTERO / HELADERO / sin decidir | | 854 / 621 / 157 / 484 | | 876 / 646 / 177 / 527 |
| polvo, por regla: LECHE_DESCREMADA / SUERO_REENGRASADO | | 29 / 8 | | 73 / 15 |
| polvo, por k-medias (k = 8 en solo Senasa, 7 con MQ): LECHE_ENTERA / WPC / CASEINATO / SUERO_* / otros / con su tag | | 72 / 275 / 39 / 32 / 5 / 11 | | 620 / 310 / 40 / 70 / 0 / 28 |
| queso (rama propia, fuera del k-medias): DURO / SEMIDURO / PASTA_BLANDA / MUY_ALTA_HUMEDAD | | 325 / 793 / 358 / 151 | | 539 / 1195 / 693 / 464 |

Con la semilla 42 y 50 reinicios, las dos versiones dan **k = 5** (el k-medias preliminar del consenso también). Con
10 reinicios el k de solo Senasa dependía de la semilla (6 con la 2026, 4 con la 42), y por eso se subió `N_INIT_KMEDIAS`.
Ya no hay un sexto grupo ni `POLVO_GRASO_2`: la leche en polvo entera queda en `POLVO_GRASO`. Los dulces de
leche y postres muy húmedos (10) forman LIQUIDO: 5 DDL, 2 postres, 2 helados y 1 yogur.

El efecto de sacar las familias no lácteas depende de con qué se compare. En el Excel de inconsistencias (todas las
familias, 2687 muestras en solo Senasa y 4083 con MQ) el k sale **6 en las dos versiones**; solo con lácteos, **5 en las
dos**. Es decir: el k de 5 ya no se atribuye a «desaparece ALTA_HUMEDAD» ni a un grupo de sueros reengrasados, como se
decía con la semilla 2026; lo que muestran las salidas es que las familias no lácteas suman un grupo más.

Solo Senasa no cambia en el k-medias: ningún queso de Senasa trae los cuatro análisis, así que sacarlos no mueve nada. En
Senasa + MQ se pasa de 4083 muestras con los cuatro análisis a 4015 sin las no lácteas, a 3940 sin los quesos
confirmados (que van a su rama) y a 3937 tras sacar 3 más; 101 salen como outliers por consenso. Los quesos sin
confirmar por el rótulo (1128 solo MQ con familia queso del modelo, 697 con materia grasa y sólidos totales; 44 con los
cuatro análisis) siguen en el k-medias: 21 salen como outliers y las otras 23 reciben LIQUIDO (22) y POLVO_MAGRO (1).
Ningún grupo del k-medias lleva el tag QUESO.

La silueta baja respecto del IQR porque ahora quedan muestras legítimas del borde de cada grupo: sobre las muestras que no
quita ninguno de los dos filtros, la silueta es 0,876 global y 0,799 macro en solo Senasa, y 0,864 y 0,784 con MQ.
En la sección 6, cada muestra lleva el ítem `tag=` (igual que antes) y, si tiene subtag, además `subtag=`; los quesos de
la rama entran con `tag=QUESO`.

`comparacion_numerico_categorico.ipynb` (lo arma `build_comparacion_num_cat.py`, con `src/clustering_categorico.py`) lee
los `clusters_*.csv` y compara el k-medias con un k-modas sobre las categorías; se vuelve a ejecutar después del
notebook único. El k del k-modas está **fijo en 5** (`K_KMODAS` en `src/clustering_categorico.py`) en las dos versiones y las
dos variantes: con 4 variables categóricas hay solo 53 (Senasa) y 61 (Senasa + MQ) perfiles distintos y la silueta de
Hamming sube sola con k (el máximo cae en el borde del rango, hasta k = 14), así que no sirve para elegirlo; k = 5 es el
mismo k que el del k-medias numérico en las dos poblaciones y el codo del costo cae en 5 (solo Senasa; en Senasa + MQ con
cortes globales cae en 6). El k = 6 anterior ya no coincide con ningún k numérico y queda como sensibilidad
(`K_SENSIBILIDAD`): con k = 6 el acuerdo con el numérico es menor (Rand ajustado 0,07 y 0,25 contra 0,12 y 0,33 con k = 5).
Con k = 5 el k-modas parte el dulce de leche en 4 de sus 5 grupos en solo Senasa y en 2 de 5 con MQ, y los heladeros no
tienen un grupo propio; el grupo donde caen cambia con la semilla. Se usan 1000 reinicios (`N_INIT`) y el notebook mide la
estabilidad entre semillas. Los gráficos para decidir salen de `src/graficos_decision.py` (`resultados/graficos_decision/`,
archivos `*_k5`; sin `POLVO_GRASO_2`, que ya no existe).

### Pruebas por familia (cuadernos aparte)

Cuadernos de prueba: no cambian el notebook único ni `resultados/tablas/`. Lo que aprobaron ya está integrado en el
segundo nivel de las secciones 3 y 4 (`src/segundo_nivel.py`). Cada uno lo arma su `build_prueba_*.py` (no
editar el .ipynb a mano) y escribe solo en su carpeta `resultados/prueba_*/`. Los del DDL leen los `clusters_*.csv`,
así que se ejecutan después del notebook único.

| cuaderno (lo arma) | pregunta | veredicto |
|---|---|---|
| `prueba_mejoras_clustering.ipynb` (`build_prueba_mejoras.py`) | outliers por consenso, métricas y subtags del DDL | aprobado: es lo que usan las secciones 3 y 4 (ver arriba) |
| `prueba_clustering_quesos.ipynb` (`build_prueba_quesos.py`, `src/clustering_quesos.py`) | agrupar los quesos de la rama (los que confirma el catálogo, con MG y ST: 2891 en Senasa + MQ; humedad × MGES; NaCl, proteína y cenizas de MQ en un nivel extendido) | hay estructura, pero es la de la humedad: el **corte de humedad del CAA** (art. 605) coincide con la consistencia del rótulo mejor que cualquier agrupamiento (ARI 0,63 contra 0,37 de la mezcla y 0,31 de k-medias) y los subtags del agrupamiento cambian con la semilla (ARI 0,26 a 0,86). Es lo que ya hace la rama del notebook único. Propuesta: `QUESO_DURO` / `_SEMIDURO` / `_PASTA_BLANDA` / `_MUY_ALTA_HUMEDAD` por regla, en una rama propia (columna `subtag_propuesto` de `resultados/prueba_quesos/quesos_clusters.csv`). El nivel extendido no mejora de forma pareja (baja con NaCl, sube con proteína sola; 359 muestras de MQ) |
| `prueba_ddl_repostero.ipynb` (`build_prueba_ddl_repostero.py`) | ¿puede haber un subtag `DDL_REPOSTERO`? | sí, pero por **regla supervisada**, no por la mezcla: dentro del DDL y fuera del heladero, humedad ≤ 27,6 y cenizas ≤ 2,0 → repostero (pureza 82,6 %, cobertura 78,9 %, solo Senasa). La señal es casi toda de IND-51; en IND-11 el repostero tiene la composición del familiar. Solo escribe PNG |
| `prueba_subtipo_ddl.ipynb` (`build_prueba_subtipo_ddl.py`) | proponer familiar / repostero / heladero a los DDL sin subtipo | **regresión logística** multinomial sobre los cuatro análisis, con umbral 0,70 y marca de novedad: exactitud balanceada 0,86 en clientes no vistos (la regla, 0,84; la mezcla actual, 0,53). Propone 246 familiar, 130 repostero, 7 heladero y deja 212 sin decidir de 595 (solo Senasa; en Senasa + MQ, 246 / 147 / 8 y 224 de 625). CSV con las tres probabilidades en `resultados/prueba_subtipo_ddl/` |
| `prueba_clustering_polvos.ipynb` (`build_prueba_polvos.py`, `src/clustering_polvos.py`) | agrupar los polvos (leche en polvo, suero / permeado, concentrado proteico; humedad ≤ 15) en grasa, proteína y cenizas en base seca | sí: **k-medias** (k = 8, silueta 0,78; 1152 polvos) acuerda con el tipo del rótulo mejor que los tags generales (ARI 0,76 contra 0,70) y es casi idéntico entre semillas. Subtags `POLVO_LECHE_ENTERA` (605, = POLVO_GRASO), `POLVO_WPC` (267), `POLVO_SUERO_PERMEADO` (85) y `POLVO_CASEINATO` (41); `POLVO_CONCENTRADO` ya no sale. La leche descremada no sale: tiene casi la misma proteína que el WPC 35 en base seca (37,7 contra 37,0 y 37,4, con rangos que se pisan) y la separan las cenizas (8,0 contra 5,3 a 6,4) y la grasa (2,6 contra 3,5 a 3,8); el suero reengrasado queda como outlier. En el notebook único las dos van por regla de composición antes del k-medias. Para la leche, el corte de grasa del CAA coincide con el rótulo en 95,8 %. Sin nivel extendido: la lactosa informada es el resto por diferencia. Columna `subtag_propuesto` de `resultados/prueba_polvos/polvos_clusters.csv` |
| `prueba_clustering_todas_hg.ipynb` (`build_prueba_todas_hg.py`, `src/clustering_todas_hg.py`) | ¿un agrupamiento sin rótulos con solo humedad y grasa, sobre todas las muestras que las tienen (4982 de 14612 en Senasa; 9407 de 26634 con MQ; la leche fluida casi no entra: 21 de 2121 y 346 de 2951), separa las familias? Variantes g/100 g y base seca; k-medias, mezcla y HDBSCAN | **separa clases físicas, no familias; justifica las ramas, pero no por igual**. K-medias k = 6 (silueta): ARI 0,93 / AMI 0,87 contra la familia en Senasa y 0,79 / 0,74 con MQ, pero el ARI balanceado por familia es 0,69 y 0,36 (DDL y queso son el 86 % y el 61 %); ningún k de 2 a 12 pasa de AMI balanceado 0,81 y 0,69, y la base seca no mejora. Estable entre semillas (ARI 0,97 y 0,86 en k-medias; la mezcla, menos). **H + G aísla solo** el DDL (pureza 99,8 a 100 %), el queso (un cluster de pureza 99,6 a 100 % con el 82 al 87 % de los quesos), la manteca / crema y la leche en polvo entera; **se pisan** el suero con el concentrado proteico (solapamiento 91 en Senasa) y con la leche en polvo (39), y los líquidos (leche fluida con yogur 96; suero con yogur 62); el helado se dispersa (recall supervisado 41 %; el 43 % de los helados cae en queso). Con rótulos, H + G llega a exactitud 0,99 y 0,96 (cota optimista), o sea que la información alcanza y el agrupamiento sin rótulos pierde líquidos y polvos magros. Proteína y cenizas aportan para suero contra leche en polvo y concentrado (exactitud balanceada, con rótulos: suero contra leche en polvo 0,90 a 0,995; concentrado contra suero 0,94 a 0,997; concentrado contra leche en polvo 0,97 a 0,995), no para polvos contra DDL (1,00 con H + G). Veredicto: ramas de **humedad + grasa** para queso, manteca / crema y el aislamiento del DDL; rama de **polvos con grasa, proteína y cenizas en base seca**; líquidos y helado necesitan otras variables y hoy casi no hay datos. CSV y gráficos en `resultados/prueba_todas_hg/` |

## Modelos supervisados sobre la leche cruda

Lista de algoritmos de la cátedra (todos supervisados). Leen `../datos/leche_cruda.csv` y
`datos_entrada/temperatura_rafaela.csv` (`rutas.LECHE_CRUDA`, `rutas.TEMPERATURA`); validación temporal (entrenan hasta
2023 y prueban 2024–26) y por industria dejada afuera. El árbol de decisión lo hace otro integrante del grupo.

| cuaderno (lo arma) | pregunta | veredicto |
|---|---|---|
| `regresion_ciclo_leche_cruda.ipynb` (`build_regresion_ciclo_leche.py`, `src/regresion_leche.py`) | regresión lineal y polinómica: proteína y grasa según el ciclo anual, la anomalía de temperatura y la industria (477 512 muestras de 9 industrias, 2019–26) | el ciclo baja el RMSE del promedio semanal de prueba 28 % en proteína y 26 % en grasa contra la media global (por muestra, solo 10 % y 7 %: la variación entre tambos domina). Elegido: Fourier con 2 armónicos + anomalía de 7 días (los polinomios dejan un salto dic→ene de ~0,05 en proteína). Pico de proteína el 25 may (amplitud 0,21 g/100 g) y de grasa el 17 may (0,33). +1 °C de anomalía: −0,010 g/100 g en las dos; el término cuadrático no aporta. El ciclo mejora a 9 de 9 industrias dejadas afuera (en proteína, 8 de 9 cuando además se predicen 2024–26). Sobreajuste didáctico con 80 muestras: mínimo de prueba en grado 4 en proteína y 3 en grasa. CSV y gráficos en `resultados/regresion_ciclo_leche/` |
| `logistica_calidad_leche_cruda.ipynb` (`build_logistica_calidad_leche.py`, `src/logistica_leche.py`) | regresión logística: probabilidad de RCS > 400 mil cél/mL (53,6 % de positivos) y RBT > 100 mil UFC/mL (22,8 %) según estación, anomalía de temperatura, industria y (aparte) composición | **la estación explica, el calor no**. Prueba 2024–26: AUC 0,69 en RCS y 0,58 en RBT (referencia por industria 0,57 y 0,54); calibración aceptable (ECE 0,05 y 0,025). OR por +1 °C de anomalía 0,990 [0,981; 1,000] y 0,992 [0,984; 0,999]: sin evidencia de que el calor suba el riesgo; la asociación con la temperatura absoluta (OR 1,03) es estacionalidad. Lactosa: OR 0,70 por +0,1 g/100 g en RCS (señal de mastitis), sube el AUC a 0,72. Con una industria afuera, RCS generaliza algo y RBT casi nada. RBT de IND-59 descartado desde 2024-05 (cambió a UFC/mL y queda censurado en 100 000). CSV y gráficos en `resultados/logistica_calidad_leche/` |
| `temperatura_leche_cruda.ipynb` (`build_temperatura_leche_cruda.py`; Parte VIII del cuaderno general) | análisis exploratorio de la leche cruda contra la temperatura de Rafaela: cada parámetro y la temperatura por semana, dispersión industria-día contra la temperatura (grasa, proteína, sólidos totales y RCS y RBT en escala log con el umbral de la Parte IV), linealidad y ventanas de 1 a 90 días, anomalías y desfase de una semana. Lee `archivo` solo para armar el identificador `informe` y lo descarta | composición: más calor, leche más pobre (r −0,37 / −0,49 / −0,45 por industria-día), lineal, y sobrevive en parte al calendario (−0,27 / −0,48 / −0,41 en anomalías semanales). RCS r 0,23 y RBT 0,11 (×1,11 y ×1,13 cada 10 °C), positivos en las diez industrias; en anomalías semanales RCS cae a 0,02 (es la estación, como en la Parte IV) y RBT queda en 0,11 (en el borde). Mismas reglas de RBT/RCS de IND-59 que la logística. Gráficos en `resultados/temperatura_leche_cruda/` (~2 min) |

Ambos cuadernos cierran con una sección «¿Por qué este algoritmo y no otro?» que compara el modelo elegido con Naive Bayes, k-NN, SVM (lineal; en regresión también con núcleo RBF) y un árbol (solo como referencia; softmax con dos clases es la logística), con las mismas variables y la misma partición temporal: en la logística, AUC/log-loss/Brier/ECE y una industria afuera (`resultados/logistica_calidad_leche/comparacion_algoritmos.csv`, `05_comparacion_algoritmos.png`); en la regresión, RMSE por muestra y semanal (`resultados/regresion_ciclo_leche/comparacion_algoritmos.csv`, `fig5_comparacion_algoritmos.png`). Las diferencias son chicas. En la logística, el k-NN supera al elegido por 0,004 de AUC en RCS (0,696 contra 0,691) pero calibra peor (ECE 0,069 contra 0,050); en RBT queda 0,011 por debajo, y el SVM lineal empata. En la regresión hay un empate técnico: el SVR con núcleo RBF baja 0,7 % el RMSE semanal en proteína (0,0873 contra 0,0879 de Fourier) y el k-NN 0,3 % en grasa (0,1490 contra 0,1495). Se queda la familia lineal por los odds ratio y coeficientes con intervalo, y por la calibración.

### Predicción sobre Senasa y MQ

Tres cuadernos de predicción que no usan la leche cruda: son las Partes V, VI y VII del cuaderno general. Cada uno lo
arma su `build_*.py` y trae la tabla «Datos usados en cada entrenamiento».

| cuaderno (lo arma) | pregunta | veredicto |
|---|---|---|
| `prediccion_tipo.ipynb` (`build_prediccion_tipo.py`, `src/prediccion_tipo.py`; Parte V del cuaderno general) | ¿se puede predecir el tipo (familia y subtipo) de una muestra de Senasa o MQ nueva por lo que se le midió? 27 241 muestras de entrenamiento sobre 30 089; las de calibración, ensayo, pool, sin clasificar y con declarado que no coincide quedan afuera | **sí para la familia, con límites**: HistGradientBoosting en dos niveles, validación cruzada de 5 pliegues agrupada por envío; exactitud de familia 0,79 (F1 macro 0,62); con probabilidad ≥ 0,8 se decide el 64,8 % con exactitud 0,95. Subtipo: de 0,74 (suero / permeado) a 0,98 (leche en polvo, concentrado proteico). En calibración la familia se acierta 0,65 (0,92 con prob ≥ 0,8). No es validación temporal: es por envío. |
| `prediccion_deslactosado.ipynb` (`build_prediccion_deslactosado.py`, `src/deslactosado_uat.py`; Parte VI del cuaderno general) | ¿cuántos días hay que esperar para que la lactosa de la leche UAT baje de 0,1 g/100 g, según la lactosa inicial y la temperatura media de Rafaela? 86 lotes remuestreados de 1176 muestras UAT de Senasa (MQ no mide lactosa) | **orden de magnitud, no predictor operativo**: modelo de primer orden con censura; el tiempo sí explica la baja (Spearman −0,37) y el modelo con desfase es el mejor por AIC (236,3 contra 243,8 a 245,4), pero el efecto de la temperatura dentro de los pares no se separa (IC de b de −0,016 a 0,112); en las 1176 UAT la lactosa sí baja con la temperatura previa (Spearman −0,34). Con L₀ = 0,5 g/100 g, de 9,4 días a 10 °C a 5,4 a 30 °C, con intervalos anchos. |
| `arbol_decision.ipynb` (`build_arbol_decision.py`, `src/arbol_decision.py`; Parte VII del cuaderno general; app `app_arbol.py`) | el trabajo de otro integrante (cascada familia → subtipo con árboles de decisión, versiones V1 solo arc, V2 con SIN CLASIFICAR, V3 arc + MQ con umbral 0,85), llevado al proyecto con semilla 42 y rutas de `src/rutas.py`. Solo la composición (7 análisis, faltantes como NaN); prueba 20 % de arc al azar, estratificada | **familia sí, subtipo a medias**: exactitud de familia 0,893 / 0,892 / 0,889 (V1 / V2 / V3; el original daba 0,885 / 0,889 / 0,896 con semilla 2026), F1 macro 0,67 a 0,70, AUC macro 0,90 a 0,92. En V3, familia y subtipo correctos en el 65 % de las muestras con subtipo real. Agrupando por envío la familia baja 1,1 puntos. El modelo (`resultados/modelos/arbol_decision.joblib`) lo regenera el cuaderno. |

## Cómo regenerar

Desde esta carpeta, con el entorno del repositorio:

```bash
cd trabajo_senasa_mq
../.venv/bin/python build_notebook.py
../.venv/bin/jupyter nbconvert --to notebook --execute --inplace trabajo_senasa_mq.ipynb
```

Eso reescribe todo `resultados/` salvo los dos Excel de MQ encolumnado, que por defecto solo se leen. Para rearmarlos,
poner `REGENERAR_MQ = True` en la sección 0 (o, sin el notebook,
`../.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); import mq_encolumnado; mq_encolumnado.armar(True)"`).
Cada módulo de `src/` también corre solo (`../.venv/bin/python src/cortes_caa.py`, etc.).

Privacidad: `anonimizar.py` tapa marcas con la lista privada `marcas_privadas.txt`, que no está en esta máquina. Se busca
en `../datos/marcas_privadas.txt` (donde la espera `.stignore`) o en la ruta de la variable `TRABAJO_SENASA_MQ_MARCAS`.

## Cómo deshacer la mudanza

```bash
cd ~/IA
.venv/bin/python trabajo_senasa_mq/docs/deshacer.py            # dry run: muestra lo que haría
.venv/bin/python trabajo_senasa_mq/docs/deshacer.py --aplicar  # lo hace
```

Recorre el manifiesto al revés y devuelve cada archivo a su lugar en `proyecto/` o `datos/`, después de comprobar su
sha256. Lo creado en la mudanza (los módulos adaptados de `src/`, este README, el notebook) y los resultados que se
regeneraron pasan a `_archivo/_deshacer_<fecha>/`. No borra ni sobrescribe nada: si algo cambió o su lugar está ocupado,
lo deja y lo informa.

## Lo que no se tocó

`datos/{__init__.py, catalogo.py, sinteticos.py, verificar_sustitutos.py, manifiesto.json, cache/}` (el paquete del
curso) y `datos/CLIENT.txt` siguen en `datos/`. El resto de `proyecto/` y `datos/` está en `../_archivo/proyecto/` y
`../_archivo/datos/`, con la misma ruta de antes. Lo archivado puede no ejecutarse desde ahí (sus rutas apuntan a
`proyecto/` y `datos/`).

Los textos de las salidas que citan rutas (la hoja `notas` de los Excel de revisión, la columna `fuente` de
`cortes_categorias.csv`) conservan las rutas de antes de la mudanza (`proyecto/...`), para que las salidas sean idénticas
a las anteriores.
