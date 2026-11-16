# Cortes para categorizar los análisis («CAA por familia»)

Tabla: `proyecto/cortes_categorias.csv` (268 filas). Se regenera con
`.venv/bin/python proyecto/build_cortes_categorias.py` (solo lee `proyecto/Senasa encolumnado.xlsx`).
El objetivo es convertir cada resultado numérico en un ítem para reglas de asociación (por ejemplo `humedad=alta humedad`
o `materia_grasa=entera`), como en la clase 44.

## Cómo se lee la tabla

- Cada fila es una categoría: `desde <= valor < hasta` (`inf` = sin tope). Un límite que el CAA da como «hasta X
  inclusive» se escribe `X + 0,005`, porque el laboratorio informa 2 decimales (por ejemplo, dulce de leche hasta 9,0 →
  `9.005`; sin lactosa hasta 0,1 → `0.105`).
- Para una muestra se elige el juego de filas más específico para su análisis: familia y subtipo, si no familia y
  estado, si no la familia, y si no `*` (los cortes globales). `subtipo` puede tener varios valores de `desc_subtipo` separados por `;`.
- `verificado = sí`: el valor se leyó en el PDF oficial del CAA. `no`: terciles o valores derivados.
  `nota` explica cada regla y da el `n` de los terciles.
- Lectura de los resultados: coma decimal; el valor antes de `+/-`; `&lt;`, `<`, «NO DETECTADO» y «No contiene» = 0;
  `,.` se lee como `,`; con punto y coma juntos (`1.320,65`) el punto es separador de miles. Las dos columnas de
  galactosa (D-galactosa y galactosa, según la técnica) se juntan, igual que en `Categorizacion de los Datos.ipynb`. Los
  valores en g/100 g por encima de 100 (por ejemplo, humedad `4548` en vez de 45,48) no entran a los terciles.

## Criterio 1: clases del CAA

| Análisis | Familia | Clases (desde incluido, hasta excluido) | Fuente |
|---|---|---|---|
| humedad | queso | baja < 36 · mediana 36–46 · alta 46–55 · muy alta ≥ 55 | CAA art. 605, inc. 2.b, [cap. VIII][viii] |
| mg en extracto seco | queso | descremado < 10 · magro 10–25 · semigraso 25–45 · graso 45–60 · extra graso ≥ 60 | CAA art. 605, inc. 2.a, [cap. VIII][viii] |
| materia grasa | dulce de leche | < 6 (bajo el mínimo) · dulce de leche 6,0–9,0 · con crema > 9,0 | CAA art. 592, inc. 5.2, [cap. VIII][viii] |
| materia grasa | leche en polvo | descremada ≤ 1,5 · parcialmente descremada 1,5–26 · entera ≥ 26 | CAA art. 567 (RTM GMC 07/18), 2.2.1, [cap. VIII][viii] |
| materia grasa | leche fluida | descremada < 0,6 · parcialmente descremada 0,6–3,0 · entera ≥ 3,0 | CAA art. 560 bis (UAT) y art. 562 a/b, [cap. VIII][viii] |
| materia grasa | yogur / bebida láctea | descremada < 0,6 · parcialmente descremada 0,6–3,0 · entera 3,0–6,0 · con crema ≥ 6,0 | CAA art. 576, inc. 2.a, [cap. VIII][viii] |
| materia grasa | helado | de agua < 1,5 · de leche 1,5–6 · de crema ≥ 6 | CAA art. 1077, inc. 1–3, [cap. XII][xii] |
| materia grasa | manteca / crema, subtipos de crema | < 10 · liviana 10–20 · crema 20–50 · alto tenor graso ≥ 50 | CAA art. 585, inc. 1 y 4, [cap. VIII][viii] |
| lactosa | todas | sin lactosa ≤ 0,1 (100 mg por 100 g o 100 ml); el resto, por terciles | CAA art. 1372 bis, [cap. XVII][xvii] |
| lactosa (g/100 g y g/100 ml) | leche fluida | sin lactosa ≤ 0,1 · bajo ≤ 0,2415 · reducido ≤ 1,449 · normal | CAA art. 1372 (bajo ≤ 5 %, reducido ≤ 30 % de la lactosa del alimento corriente), [cap. XVII][xvii]; referencia 4,83 g/100 g: trabajo previo, `diccionario_leche_cruda.xlsx`. **Sin verificar**: el CAA no da el valor de referencia |

Los capítulos son las versiones vigentes enlazadas desde [argentina.gob.ar/anmat/codigoalimentario](https://www.argentina.gob.ar/anmat/codigoalimentario)
(cap. VIII actualizado a 10/2025, cap. XII a 11/2025, cap. XVII a 06/2025).

Cuántas muestras caen en cada clase del CAA:

- **Humedad del queso** (n = 1.955): baja 385, mediana 979, alta 419, muy alta 172. Coincide con `desc_consistencia`
  en el 93 % de los quesos de baja, mediana o alta humedad (el «alta humedad» del catálogo junta alta y muy alta). Los
  procesados son casi todos de muy alta humedad.
- **Materia grasa en extracto seco del queso** (n = 1.629): magro 30, semigraso 591, graso 1.000, extra graso 8.
- **Materia grasa del dulce de leche** (n = 3.336): menos de 6: 525; dulce de leche: 2.734; con crema: 77.
  Las de menos de 6 son de todos los subtipos (familiar 199, heladero 136, repostero 89, sin subtipo 101).
- **Leche en polvo** (n = 100): descremada 29, parcialmente descremada 2, entera 69. **Helado** (n = 125): de leche 11,
  de crema 114. **Leche fluida** (n = 22) y **yogur** (n = 21): pocas muestras con grasa medida.
- **Crema**: ninguna muestra de crema tiene grasa medida. Las filas quedan para cuando lleguen.
- **Lactosa de la leche fluida** (n = 1.930): sin lactosa 456, bajo en lactosa 253, reducido 670, normal 551.

Los límites del CAA que son solo un requisito, sin clases, **no** se usan como cortes. Por ejemplo: humedad máxima 30 y
proteínas mínimas 5 en el dulce de leche (art. 592); humedad máxima 5 en la leche en polvo (art. 567); manteca con
grasa mínima 82 y humedad máxima 16 (art. 596); proteína mínima de los sueros y WPC (art. 582 bis y siguientes). Esos
análisis van por terciles (ver la decisión 3).

## Criterio 2: terciles dentro de la familia

Donde el CAA no define clases: `bajo` < p33 ≤ `medio` < p66 ≤ `alto`, con los valores de la familia en Senasa. Hace
falta un mínimo de 30 valores. Hay dos casos especiales, anotados en `nota`:

- **Más de un tercio de ceros** (resultados menores al límite de detección): una clase `no detectado / 0`, y el resto
  se parte en la mediana de los positivos (`bajo` / `alto`). Si quedan menos de 2 positivos, `detectado`. Pasa con la
  galactosa del helado, los azúcares de queso, cárnicos y panificados, la fibra, los sulfitos, el plomo y el arsénico.
- **Lactosa**: primero la clase `sin lactosa` del CAA. Los terciles se calculan solo sobre los valores mayores a 0,1.

**Terciles por estado (columna `estado`)**: en suero / permeado, concentrado proteico, yogur / bebida láctea, base
nutricional y manteca / crema los terciles se calculan aparte para `POLVO`, `LIQUIDO` y `SOLIDO`, porque la composición
del polvo y la del líquido no se pueden comparar (decisión del 2026-11-09).

- El estado de la muestra sale de la humedad medida: ≤ 15 es polvo y ≥ 25 es líquido. En manteca / crema, < 30 es
  `SOLIDO` (manteca) y ≥ 30 es `LIQUIDO` (crema); sin humedad, decide el subtipo. Si no hay humedad se usa
  `desc_estado`. Entre 15 y 25 sin estado declarado, la muestra queda sin estado.
- Si Senasa tiene menos de 30 valores de un estado, se suman las muestras nuevas de MQ (`nota` lo dice). Con menos de
  30 aun así, ese estado no tiene cortes.
- Una muestra de estas familias sin estado, o de un estado sin cortes, no se categoriza en ese análisis: no baja a los
  terciles de la familia entera ni a los globales.
- La crema (manteca / crema, `LIQUIDO`) usa las clases de materia grasa del CAA.

**Cortes globales (`familia = *`)**: se usan para las familias con menos de 30 valores. Son terciles con todas las
familias juntas. Se usan en:

- humedad y sólidos totales: leche fluida (22), helado (14), yogur (18), cárnicos, panificados, base nutricional
- proteínas y cenizas: yogur, arroz con leche, cárnicos, panificados, base nutricional
- galactosa: concentrado proteico (26), agua / hisopado, base nutricional, manteca / crema
- materia grasa en extracto seco: dulce de leche (4)
- lactosa: manteca / crema, agua / hisopado, panificados, arroz con leche
- azúcares totales: leche en polvo (25) y leche fluida (17), entre otras

La lista completa está en la columna `nota` de cada fila `*`.

## Trabajo previo que se respetó

- `proyecto/catalogo.py`: la consistencia del queso por humedad (< 36 / 36–46 / ≥ 46). Es la misma escala que la del
  art. 605, que además separa la muy alta humedad (≥ 55).
- `proyecto/build_mq_subfamilia.py`: leche en polvo descremada ≤ 1,5 y entera ≥ 26. Coincide con el art. 567.
- `Categorizacion de los Datos.ipynb`: el parseo de valores con `<` y la unión de las dos columnas de galactosa.
- `diccionario_leche_cruda.xlsx`: la lactosa típica de la leche (mediana 4,83 g/100 g).

En `proyecto/Categorizacion de los Datos.ipynb`, `docs/` y `.scratch/` no había cortes bajo/medio/alto definidos.

## Decisiones para confirmar

1. **Humedad del queso: 4 clases (CAA) o 3** (como `desc_consistencia`, sin «muy alta»). Además, `humedad` y
   `desc_consistencia` dicen casi lo mismo, igual que `solidos_totales` (≈ 100 − humedad). En las reglas de asociación
   eso produce reglas triviales: conviene dejar uno solo de los tres.
2. **Bases que no coinciden exactamente**: en la leche fluida el CAA usa % m/v y el laboratorio informa g/100 g. En
   el yogur, el CAA clasifica la grasa de la base láctea, no del producto con agregados, y las bebidas lácteas
   (art. 578) no tienen esas clases. En el helado, el CAA habla de grasa de leche y el laboratorio mide grasa total.
   ¿Se aplican igual? La leche concentrada no entra en las clases del art. 560 bis.
3. **Requisitos sin clases** (dulce de leche: humedad ≤ 30, proteínas ≥ 5, cenizas ≤ 2; leche en polvo: humedad ≤ 5;
   manteca: grasa ≥ 82, humedad ≤ 16): ¿agregar un ítem `cumple / no cumple` además de los terciles?
4. **Bajo y reducido en lactosa**: el CAA los define en relación con el alimento corriente. Se tomó como referencia
   la leche cruda (4,83 g/100 g). ¿Está bien, o se usa otro valor (el de la leche UAT entera, por ejemplo)? Fuera de la
   leche fluida solo se aplica «sin lactosa».
5. **Cortes globales para las familias chicas**: mezclan productos muy distintos (la humedad de la leche fluida sale
   siempre «alto»). Alternativa: dejar esas familias sin categoría.
6. **Terciles por subtipo**: hoy son por familia. En el dulce de leche, el repostero tiene menos humedad y puede ocupar
   solo el tercil bajo. En el suero, el concentrado proteico y la leche en polvo, la lactosa y las proteínas mezclan
   productos distintos (WPC y caseinatos; entera, descremada y modificada). ¿Pasarlos a familia y subtipo cuando
   haya 30 muestras?
7. **Dulce de leche con menos de 6 g de grasa** (525 muestras): no es una clase del CAA, es «bajo el mínimo».
   ¿Se deja como ítem o se revisan antes como posibles errores de carga?
8. **Analitos con pocos datos** (sórbico, sulfitos, sodio, grasas saturadas y trans, fibra, plomo, arsénico): casi
   todos son de dulce de leche. ¿Entran a las reglas o se dejan afuera?

[viii]: https://www.argentina.gob.ar/sites/default/files/capitulo_viii_lacteosactualiz_2025-10.pdf
[xii]: https://www.argentina.gob.ar/sites/default/files/capitulo_xii_aguas_actualiz_2025-11.pdf
[xvii]: https://www.argentina.gob.ar/sites/default/files/capitulo_xvii_dieteticosactualiz_2025-09_0.pdf
