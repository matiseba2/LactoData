# Conclusiones generales

Cada parte cierra con sus propias conclusiones y sus números; acá van las ideas principales del trabajo.

**Parte I · Agrupamiento (no supervisado).**
- Con humedad, grasa, proteínas y cenizas, el k-medias encuentra **tipos de producto**:
  - en solo Senasa, 5 grupos con silueta 0,854;
  - en Senasa + MQ, también 5 grupos, con silueta 0,835.
- Los cinco son los mismos en las dos poblaciones: dulce de leche, polvo graso, polvo magro, proteína alta y líquido.
  Con 10 arranques, el k-medias preliminar del consenso de outliers caía en mínimos locales y el número de grupos cambiaba con la semilla; con 50
  arranques, ocho semillas dan los mismos cinco grupos.
- Cuando se revela la familia de cada muestra, que el algoritmo no vio, los grupos coinciden con ella (Rand ajustado
  0,97 y 0,92).
- Quitar los outliers **dentro de cada tipo de producto y por consenso** de tres métodos descarta el 2,4 % y el 2,6 % de
  las muestras. La regla clásica del rango intercuartil descartaba el 8,6 % y el 9,2 %, y con ellas casi todos los dulces de leche
  heladeros, que son legítimos.
- Dentro de cada familia hace falta un segundo nivel:
  - subtipos del dulce de leche;
  - clases de polvo, en base seca;
  - el queso, que va por su propia rama con la humedad del Código Alimentario.

**Parte II · Números contra categorías.**
- Pasar los análisis a categorías y agrupar con k-modas (k fijo en 5, el mismo k que el numérico) **no reemplaza** al
  k-medias:
  - con cuatro variables en tres niveles hay solo 53 o 61 perfiles distintos;
  - por eso la silueta de Hamming sube sola con k y no sirve para elegirlo;
  - en solo Senasa, 4 de sus 5 grupos son dulce de leche casi puro (parten el dulce de leche) y uno mezcla los polvos
    con dulce de leche; en Senasa + MQ solo 2 de 5, y los otros mezclan polvos, líquidos y dulce de leche;
  - el acuerdo con el numérico es bajo (Rand ajustado 0,12 y 0,33; con k = 6 era 0,07 y 0,25);
  - los heladeros no tienen un grupo propio y el grupo donde caen cambia con la semilla.
- Las categorías sirven para describir y para las reglas de asociación, no para encontrar tipos de producto.

**Parte III · Regresión (supervisado): el ciclo anual de la leche cruda.**
- El día del año explica una parte chica de cada muestra: baja el error un 10 % en proteína y un 7 % en grasa.
- En cambio, explica bien el **promedio semanal**: baja el error un 28 % en proteína y un 26 % en grasa, en los años
  2024–2026, que el modelo no vio.
- La proteína pica a fines de mayo y la grasa a mediados de mayo; las dos tocan fondo en diciembre.
- Se eligió un ciclo con senos y cosenos de dos armónicos (Fourier):
  - el polinomio del día del año necesita grado 3 y aun así tiene un salto entre el 31 de diciembre y el 1 de enero;
  - la regresión lineal de grado 1 no capta el ciclo (mejora menos del 2 %);
  - k-NN, árbol y SVR llegan a errores parecidos: el SVR con núcleo RBF queda apenas mejor en proteína (0,0873 contra
    0,0879 de RMSE semanal) y el k-NN en grasa (0,1490 contra 0,1495). Es un empate técnico, y ellos no dan una forma
    que se pueda leer.
- Un grado más de calor que lo normal para la fecha baja la proteína y la grasa en apenas 0,01 g/100 g.

**Parte IV · Regresión logística (supervisado): calidad de la leche cruda.**
- La probabilidad de superar las 400 000 células somáticas por mL se predice razonablemente: AUC 0,69, contra 0,57 de
  la industria sola, con probabilidades calibradas.
- La de superar las 100 000 bacterias por mL apenas mejora a la industria sola: AUC 0,58.
- Lo que explica el riesgo es la **estación**, no el calor: con la fecha en el modelo, cada grado de anomalía tiene un
  odds ratio de 0,99, que no difiere de 1.
- La **lactosa** baja es la mejor señal de células somáticas altas, por la mastitis: odds ratio de 0,70 por cada
  +0,1 g/100 g.
- Con el corte de Youden, el modelo detecta 3 de cada 4 muestras con células altas, a cambio de dar falsa alarma en la
  mitad de las buenas.
- Contra Bayes ingenuo, k-NN, SVM lineal y un árbol, con las mismas variables:
  - el SVM lineal empata con la logística en los dos umbrales;
  - en células somáticas, el k-NN le saca 0,004 de AUC pero calibra peor (ECE 0,069 contra 0,050); en bacterias
    queda 0,011 por debajo;
  - Bayes calibra mucho peor;
  - la logística se elige porque es la única que da odds ratios con intervalos, que es lo que responde la pregunta del
    calor, y porque rinde igual dejando una industria afuera.

**Parte V · Predicción del tipo de producto (supervisado).**
- Con la composición y el texto de la descripción, un modelo de árboles potenciados acierta la familia en el 79 % de las
  muestras (F1 macro 0,62), validando por envío para que muestras hermanas no se repartan entre entrenamiento y prueba.
- Cuando la probabilidad supera 0,8 decide sola el 65 % de las muestras, con un acierto del 95 %; el resto queda para
  revisión manual.

**Parte VI · Lactosa residual de la leche deslactosada (supervisado).**
- Con solo 86 lotes medidos dos veces, la caída de la lactosa da un orden de magnitud. La lactosa baja con los días
  (Spearman −0,37), pero el intervalo del efecto de la temperatura incluye el cero.
- Sirve como guía para el control de la enzima, no como predictor operativo.

**Parte VII · Árbol de decisión en cascada (supervisado, trabajo de otro integrante).**
- Con solo la composición, un árbol acierta la familia en el 89 % de una prueba al azar del 20 % (F1 macro 0,67,
  AUC macro 0,90). Agrupando por envío baja apenas un punto.
- El subtipo es lo difícil: la cascada completa acierta familia y subtipo en el 65 % de las muestras con subtipo real.
  Va bien en manteca, leche fluida y dulce de leche, y mal en queso, suero y leche en polvo.
- La clase «sin clasificar» con umbral de 0,85 (V3) evita que el modelo se refugie en ella (19 % → 6 % de las reales).
  Sumar las muestras de MQ no mejora la familia, pero sí el subtipo de concentrados y sueros.

**Parte VIII · Leche cruda y temperatura de Rafaela (asociación).**
- Con más calor, la leche es más pobre: r de −0,37 en grasa, −0,49 en proteína y −0,45 en sólidos totales por
  industria-día, en las diez industrias y de forma lineal. Restando el calendario se mantiene en parte (r −0,27 a
  −0,48 por semana).
- Los recuentos suben con la temperatura en crudo, pero en anomalías semanales las células somáticas caen a r 0,02:
  es la estación, como en la Parte IV. En el recuento bacteriano queda una señal chica (r 0,11), en el borde.

**Límites.**
- Las muestras de laboratorio no son un muestreo al azar de la producción: reflejan lo que cada cliente decidió
  analizar.
- Una industria cambió la unidad del recuento bacteriano en 2024 y esos valores se descartaron.
- Las Partes V y VII responden casi la misma pregunta con muestras, filtros y validación distintos. Sus exactitudes
  (79 % y 89 %) no se comparan directamente.
