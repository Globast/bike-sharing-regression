---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---


# 1. Ingenieria de Variables y Procesamiento Temporal
En esta etapa se prepararan los datos para el modelado, asegurando consistencia y calidad

## 1.1. Importacion de librerias

```{code-cell} ipython3
# Librerías científicas básicas
import numpy as np
import pandas as pd
 
# Visualización
import matplotlib.pyplot as plt
import seaborn as sns
 
# Modelos estadísticos
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
 
# Machine Learning
from sklearn.model_selection import train_test_split
```

## 1.2. Lectura de datos

```{code-cell} ipython3
data = pd.read_csv('../data/hour.csv', sep =",")
data['dteday'] = pd.to_datetime(data['dteday'])
data
```

## 1.3. Descripcion de variables del dataset
Se muestra por variable, su tipo, su rango, y la interpretaciona utilizar en el analisis

| Variable       | Tipo                             | Rango / Valores         | Descripción                                                                  |
| -------------- | -------------------------------- | ----------------------- | ---------------------------------------------------------------------------- |
| **instant**    | Numérica (entero)                | 1 – 17379               | Índice del registro. No tiene interpretación analítica directa.              |
| **dteday**     | Temporal (fecha)                 | 2011-01-01 – 2012-12-31 | Fecha del registro horario.                                                  |
| **season**     | Categórica (1–4)                 | {1, 2, 3, 4}            | Estación: 1=invierno, 2=primavera, 3=verano, 4=otoño.                        |
| **yr**         | Categórica (binaria)             | {0, 1}                  | Año: 0=2011, 1=2012.                                                         |
| **mnth**       | Categórica                       | 1 – 12                  | Mes del año.                                                                 |
| **hr**         | Categórica              | 0 – 23                  | Hora del día.                                                                |
| **holiday**    | Categórica (binaria)             | {0, 1}                  | 1 si es día festivo.                                                         |
| **weekday**    | Categórica                       | 0 – 6                   | Día de la semana (0 = domingo).                                              |
| **workingday** | Categórica (binaria)             | {0, 1}                  | 1 si es día laboral (no festivo ni fin de semana).                           |
| **weathersit** | Categórica (1–4)                 | {1, 2, 3, 4}            | Estado del clima: 1=despejado, 2=nublado, 3=lluvia ligera, 4=clima severo.   |
| **temp**       | Numérica (continua, normalizada) | 0.02 – 1.00             | Temperatura normalizada.                                            |
| **atemp**      | Numérica (continua, normalizada) | 0.00 – 1.00             | Sensación térmica normalizada.                                      |
| **hum**        | Numérica (continua, normalizada) | 0.00 – 1.00             | Humedad relativa normalizada por 100%.                                       |
| **windspeed**  | Numérica (continua, normalizada) | 0.00 – ~0.85            | Velocidad del viento normalizada.                                |
| **casual**     | Numérica (entero)                | 0 – 367                 | Usuarios casuales (no registrados).                                          |
| **registered** | Numérica (entero)                | 0 – 886                 | Usuarios registrados.                                                        |
| **cnt**        | Numérica (entero)                | 1 – 977                 | Total de bicicletas alquiladas (casual + registered). **Variable objetivo**. |

## 1.4. Exploracion de datos faltantes 
Se revisa cada variable para observar sus faltantes, en caso de existir, se imputaran utilizando la mediana

```{code-cell} ipython3
# Vemos por variable si hay algun faltante
print(data.isna().sum())
# los imputamos usando la mediana en caso de haber
data = data.fillna(data.median(numeric_only=True))

```

## 1.5. Conteo de outliers 
Se realiza solo conteo de outliers, no se corrigen debido a que son condiciones ambientals las cuales influyen dentro del modelo y estan fuera del control humano, es decir, son validas.

```{code-cell} ipython3
## Columnas numericas no categoricas
num_cols = ["temp", "hum", "windspeed", "cnt"]
outlier_counts = {}

for col in num_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    ## valores fuera de rango
    n_outliers = ((data[col] < lower) | (data[col] > upper)).sum()
    outlier_counts[col] = n_outliers

outlier_counts

```

# 1.6. Creacion, Transformacion y adecuacion de variables

1. Se crean variables categoricas a partir de momentos como Horas pico, fin de semana
2. Transformaciones SEN y COS, esto permite que variables temporales que naturalmente son circulares (hora, día, mes)
sean representadas en un espacio donde puntos cercanos evitan saltos artificiales y permitiendo al modelo aprender patrones periódicos
4. Conversion d variables categoricas a binarias


```{code-cell} ipython3
# Hora pico
def hora_pico(h):
    return 1 if (7 <= h <= 9) or (17 <= h <= 19) else 0

data["peak_hour"] = data["hr"].apply(hora_pico)

# conversion d evariables ciclicas
data["hr_sin"]  = np.sin(2*np.pi*data["hr"]/24)
data["hr_cos"]  = np.cos(2*np.pi*data["hr"]/24)

data["month_sin"] = np.sin(2*np.pi*data["mnth"]/12)
data["month_cos"] = np.cos(2*np.pi*data["mnth"]/12)

data["weekday_sin"] = np.sin(2*np.pi*data["weekday"]/7)
data["weekday_cos"] = np.cos(2*np.pi*data["weekday"]/7)

# Conversion de variables categoricas
cat_feats = ["season", "weathersit", "workingday", "holiday"]

# Nos aseguramos de que sean categóricas
for c in cat_feats:
    data[c] = data[c].astype("category")

# Crear dummies
dummies = pd.get_dummies(data[cat_feats], drop_first=True, prefix=cat_feats)

# inclusion de variables dummies al dataset
df = pd.concat([data.drop(columns=cat_feats), dummies], axis=1)

# Eliminar columnas redudnantes
df = df.astype({col: int for col in data.select_dtypes("bool").columns})
df.drop(columns=['instant', 'hr','dteday', 'mnth', 'weekday','casual', 'registered','atemp'], inplace=True)


# Pprint de datos
print("Columnas finales:")
print(df.columns.tolist()[:20])
df.to_csv('hour_clean.csv', sep=";",index = False) ## Guardamos las variables listas para empezar a correr los modelos
df
```


