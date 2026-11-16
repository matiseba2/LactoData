# LactoData

Calidad y composición de productos lácteos · UNRaf 2026.

## Qué hay

- `trabajo_senasa_mq/notebook_general.ipynb`: el cuaderno general, ya ejecutado (8 partes, con todas las salidas).
  GitHub lo muestra directamente.
- `trabajo_senasa_mq/app_prediccion.py`: app de Streamlit con el deslactosado UAT (¿cuántos días esperar?) y el
  tipo de muestra.
- `trabajo_senasa_mq/app_arbol.py`: app de Streamlit con el árbol de decisión en cascada (familia y subtipo de una
  muestra a partir de su composición).
- `trabajo_senasa_mq/resultados/modelos/`: los modelos ya entrenados que usan las apps.
- `datos/` y `_archivo/`: datos de leche cruda que lee el cuaderno (tienen que quedar al lado de
  `trabajo_senasa_mq/`).

Las empresas figuran como IND-nn y las marcas están codificadas. Semilla 42 en todo el trabajo.

## Apps en Streamlit Community Cloud

En <https://share.streamlit.io/deploy>, una app por vez:

| Campo | Valor |
|---|---|
| Repository | `matiseba2/LactoData` |
| Branch | `main` |
| Main file path | `trabajo_senasa_mq/app_prediccion.py` (o `trabajo_senasa_mq/app_arbol.py`) |
| Advanced settings → Python version | 3.12 |

Las dependencias salen de `requirements.txt` (solo lo que usan las apps, con las versiones con que se guardaron
los modelos).

Para correrlas en la computadora, desde `trabajo_senasa_mq/`:

```
pip install -r ../requirements.txt
streamlit run app_prediccion.py
streamlit run app_arbol.py
```

## Volver a ejecutar el cuaderno

1. Python 3.12 y `pip install -r requirements-notebook.txt`.
2. Desde `trabajo_senasa_mq/`:
   `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebook_general.ipynb`
   (tarda unos 30 minutos; los resultados se escriben en `trabajo_senasa_mq/resultados/`).

No se incluyen los archivos crudos del LIMS (`datos_entrada/mq/crudos/`) porque no están anonimizados; solo hacen
falta para regenerar MQ encolumnado, que ya viene armado en `trabajo_senasa_mq/resultados/excel/`.
