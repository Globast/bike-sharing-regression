# 0.1 Guía de reproducción

## 1. Requisitos de software
Instalación de dependencias desde requirements.txt
Ejecutar en la raíz del repositorio:

pip install -r requirements.txt

Las versiones concretas incluidas en `requirements.txt` son las siguientes.

    jupyter-book==1.0.2
    myst-nb>=1.1.0
    jupyterlab>=4.2
    pandas>=2.0
    numpy>=1.26
    matplotlib>=3.8
    seaborn>=0.13
    scikit-learn>=1.4
    statsmodels>=0.14
    scipy>=1.11
    kaggle>=1.6.17
    ghp-import>=2.1.0



## 2. Descarga y organización de los datos

El conjunto de datos horario proviene del repositorio UCI Machine Learning Repository [@fanaee2014]. El siguiente bloque crea la estructura mínima de carpetas (`data/`, `notebooks/`, `figures/`, `models/`) y descarga automáticamente el archivo `hour.csv` en la carpeta `data/`.

```{code-cell} python
'''
Script de inicialización del proyecto.

1. Crea las carpetas base si no existen.
2. Descarga el archivo hour.csv desde UCI.
3. Fija una semilla global de NumPy para asegurar reproducibilidad.
'''
import os
from pathlib import Path
import numpy as np
import requests

# 1. Estructura de carpetas
BASE_DIR = Path(".")
for sub in ["data", "notebooks", "figures", "models"]:
    (BASE_DIR / sub).mkdir(parents=True, exist_ok=True)

# 2. Descarga del dataset si no existe
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/hour.csv"
DATA_PATH = BASE_DIR / "data" / "hour.csv"

if not DATA_PATH.exists():
    response = requests.get(DATA_URL, timeout=30)
    response.raise_for_status()
    DATA_PATH.write_bytes(response.content)
    print(f"Dataset descargado en: {DATA_PATH}")
else:
    print(f"Dataset ya existe en: {DATA_PATH}")

# 3. Semilla global para reproducibilidad
np.random.seed(42)
print("Semilla global fijada en 42.")
```

## 3. Estructura de carpetas del repositorio

La estructura lógica del repositorio es la siguiente:

- `data/`: contiene el archivo `hour.csv` original y, opcionalmente, versiones procesadas.
- `notebooks/`: incluye todos los capítulos del libro en formato MyST-NB (`.md`).
- `figures/`: almacena recursos gráficos estáticos, por ejemplo `bike.png` usado como portada.
- `models/`: guarda artefactos de modelos (por ejemplo, parámetros estimados o serializaciones).

```{code-cell} python
# Ejemplo: listar el contenido de la carpeta data
from pathlib import Path

data_path = Path("data")
print(list(data_path.glob("*.csv")))
```

## 4. Orden sugerido de ejecución de notebooks

El orden recomendado coincide con la numeración de los capítulos:

1. `notebooks/intro_contexto.md`
2. `notebooks/ingenieria_temporal.md`
3. `notebooks/exploratorio_temporal.md`
4. `notebooks/mco_matricial.md`
5. `notebooks/inferencia_temporal.md`
6. `notebooks/diagnostico_supuestos.md`
7. `notebooks/metodos_robustos.md`
8. `notebooks/validacion_modelo.md`
9. `notebooks/monte_carlo.md`
10. `notebooks/interpretacion_operativa.md`
11. `notebooks/conclusiones_extensiones.md`
12. `notebooks/referencias.md`

## 5. Ejecución del análisis de forma autónoma

Puede ejecutar los notebooks de dos maneras principales:

1. Desde un servidor interactivo (por ejemplo, `jupyter lab`) abriendo cada archivo `.md`.
2. Construyendo el libro estático con Jupyter Book:

```{code-cell} python
# Construir el libro (ejecutar en la raíz del proyecto)
# !jupyter-book build .
```

En ambos casos, asegúrese de que el kernel de Python apunta al mismo entorno virtual en el que instaló las dependencias.
