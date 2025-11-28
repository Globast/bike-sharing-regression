---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---
# 2. Ingeniería de variables y preprocesamiento temporales

En este capítulo construimos variables derivadas que capturan patrones de tendencia, estacionalidad y se prepararan los datos para el modelado, asegurando consistencia y calidad.

## 2.1. Importacion de librerias

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Opciones de visualización de pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

from pathlib import Path
data = pd.read_csv(Path('../data/hour.csv'), sep =",")
data['dteday'] = pd.to_datetime(data['dteday'])
data
```

## 2.2. Descripcion de variables del dataset
Se muestra por variable, su tipo, su rango, y la interpretaciona utilizar en el analisis

| Variable       | Tipo                             | Rango / Valores         | Interpretación / Comentarios                                              |
| -------------- | -------------------------------- | ----------------------- | ------------------------------------------------------------------------- |
| **instant**    | Numérica (entero)                | 1 – 17379               | Índice de observación, representa el número de registro. No tiene interpretación analítica directa.            |
| **dteday**     | Fecha                            | 2011-01-01 a 2012-12-31 | Fecha del registro diario. Permite análisis temporal y estacional.       |
| **season**     | Categórica (1 a 4)               | 1: Invierno, 2: Primavera, 3: Verano, 4: Otoño | Estación del año (climatológica), influye en la demanda de bicicletas.   |
| **yr**         | Categórica (0,1)                 | 0: 2011, 1: 2012        | Año del registro. Permite observar cambios entre años (tendencias).      |
| **mnth**       | Categórica (1 a 12)              | 1 – 12                  | Mes del año. También permite capturar patrones estacionales mensuales.   |
| **hr**         | Numérica (entero)                | 0 – 23                  | Hora del día. Fundamental para capturar variaciones intradía.            |
| **holiday**    | Categórica (0,1)                 | 0: No festivo, 1: Festivo | Indica si el día es festivo. Afecta el patrón de demanda.              |
| **weekday**    | Categórica (0 a 6)               | 0: Domingo – 6: Sábado  | Día de la semana. Permite distinguir entre días laborales y fines de semana. |
| **workingday** | Categórica (0,1)                 | 0: No laboral, 1: Laboral | Día laboral (excluye fines de semana y festivos). Afecta los patrones de movilidad. |
| **weathersit** | Categórica (1 a 4)               | 1: Despejado – 4: Condiciones extremas | Condición climática general. La demanda depende fuertemente del clima. |
| **temp**       | Numérica (continua)              | 0 – 1 (normalizada)     | Temperatura normalizada. Representa la sensación térmica.                |
| **atemp**      | Numérica (continua)              | 0 – 1 (normalizada)     | Temperatura percibida (sensación). Similar a `temp` pero ajustada.       |
| **hum**        | Numérica (continua)              | 0 – 1 (normalizada)     | Humedad relativa normalizada. Puede afectar el confort al usar bicicleta. |
| **windspeed**  | Numérica (continua)              | 0 – 1 (normalizada)     | Velocidad del viento normalizada. Vientos fuertes reducen el uso de bicicleta. |
| **casual**     | Numérica (entero)                | 0 – 367                 | Número de usuarios casuales (no registrados) en la hora.                 |
| **registered** | Numérica (entero)                | 0 – 886                 | Número de usuarios registrados en la hora.                               |
| **cnt**        | Numérica (entero, variable objetivo) | 1 – 977             | Total de bicicletas rentadas por hora (`casual + registered`).           |

## 2.3. Exploracion de datos faltantes 
Se revisa cada variable para observar su cantidad de datos faltantes y posterior imputacion

```{code-cell} ipython3
# Vemos por variable si hay algun faltante
print(data.isna().sum())
# los imputamos usando la mediana en caso de haber
data = data.fillna(data.median(numeric_only=True))
```
En este dataset no hay valores faltantes en ninguna variable. Sin embargo hay *datos no observados*. Se observan "huecos de tiempo". En el dataset hay unas cuantas horas menos, del orden de ~0,8 % del total. Es posible que el sistema pudo estar apagado, problemas de registro, posible limpieza previa, etc. En el modelo de regresión que usamos, simplemente se ignoran y el análisis se hace sobre las horas observadas.
## 2.4. Conteo de outliers 
Se realiza solo conteo de outliers, no se corrigen dada su interpretacion y valor al modelo

```{code-cell} ipython3
## Columnas numericas no categoricas
num_cols = ["temp", "atemp", "hum", "windspeed", "cnt"]
outlier_counts = {}

for col in num_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    
    outliers = data[(data[col] < lower_fence) | (data[col] > upper_fence)]
    outlier_counts[col] = outliers.shape[0]

outlier_counts
```
En el dataset, cnt tiene: percentil 95 ≈ 563, percentil 99 ≈ 782, máximo = 977. Son horas de demanda muy alta, pero plausibles (horas pico en días favorables). No parecen errores de registro, sino casos extremos reales. **Lo razonable es No eliminarlos. Estabilizando con log(cnt+1): se reduce la influencia relativa de los valores extremos, los residuos suelen ser más estables y el ajuste es más robusto, a costa de que la interpretación de los coeficientes pase al dominio logarítmico (efectos porcentuales, no absolutos).**

## 2.5 Creacion, Transformacion y adecuacion de variables

1. Se crean variables de hora pico de la noche y mañana: La demanda tiene picos claros asociados a entrada/salida de trabajo/estudio. 
2. Se transforman variables ciclicas (hora, mes, día de la semana) usando seno y coseno. 
3. Se crean variables dummies para las categorías relevantes
4. Se eliminan columnas redundantes para el modelo

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


