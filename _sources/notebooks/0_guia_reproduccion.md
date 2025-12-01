# 0.1 Guía de reproducción

### 0.1.1. Requisitos de software
Instalación de dependencias desde `requirements.txt`
Ejecutar en la raíz del repositorio:

pip install -r requirements.txt

Las versiones concretas incluidas en `requirements.txt` son las siguientes:

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


### 0.1.2. Estructura de carpetas del repositorio

La estructura lógica del repositorio es la siguiente:

- `data/`: contiene el archivo `day.csv` original , versiones procesadas.
- `notebooks/`: incluye todos los capítulos del libro en formato MyST-NB (`.md`).
    - 1_intro_contexto
        title: "1. Introducción y contexto de movilidad urbana"
      2_ingenieria_temporal
        title: "2. Ingeniería de variables y preprocesamiento temporales"
      3_exploratorio_temporal
        title: "3. Análisis exploratorio de series temporales"
      4_mco_matricial
        title: "4. Matricial de MCO y diagnostico de supuestos"
      5_metodos_robustos
        title: "5. Métodos robustos para datos temporales"
      6_validacion_modelo
        title: "6. Validación y selección del modelo"
      7_monte_carlo
        title: "7. Simulación Monte Carlo"
      8_interpretacion_operativa
        title: "8. Interpretación operativa y recomendaciones"
      9_conclusiones_extensiones
        title: "9. Conclusiones y extensiones futuras"
      10_referencias
        title: "10. Referencias"

- `figures/`: almacena recursos gráficos estáticos, por ejemplo `bike.png` usado como portada.
- `models/`: guarda artefactos de modelos (por ejemplo, parámetros estimados o serializaciones).

El orden recomendado de ejecución coincide con la numeración de los capítulos:

### 0.1.3. Ejecución del análisis de forma autónoma

Puede ejecutar los notebooks de dos maneras principales:

1. Desde un servidor interactivo (por ejemplo, `jupyter lab`) abriendo cada archivo `.md`.
2. Construyendo el libro estático con Jupyter Book:

```{code-cell} python
# Construir el libro (ejecutar en la raíz del proyecto)
# jupyter-book build .
```

En ambos casos, asegúrese de que el kernel de Python apunta al mismo entorno virtual en el que instaló las dependencias.
