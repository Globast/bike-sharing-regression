# 1. Introducción y contexto de movilidad urbana

## 1.1 Contexto del problema

Los sistemas de bicicletas compartidas constituyen una fuente rica de datos sobre patrones de movilidad urbana. Cada alquiler funciona como un sensor que registra la interacción entre clima, calendario, infraestructura y comportamiento de los usuarios [@fanaee2014; @hyndman2018].

En este proyecto utilizaremos el conjunto de datos de *Bike Sharing* para modelar la demanda horaria de bicicletas, con el propósito de:

- Entender los factores que explican la variación en el número de alquileres.
- Construir un modelo de regresión lineal interpretable.
- Evaluar la capacidad predictiva del modelo bajo un esquema de validación temporal.

## 1.2 Descripción del conjunto de datos

El archivo `data/hour.csv` contiene observaciones horarias con información sobre:

- Fecha y hora (`dteday`, `hr`).
- Condiciones climáticas (`temp`, `atemp`, `hum`, `windspeed`, `weathersit`).
- Variables de calendario (`season`, `yr`, `mnth`, `holiday`, `weekday`, `workingday`).
- Conteos de bicicletas: usuarios registrados, casuales y totales (`registered`, `casual`, `cnt`).

```{code-cell} python
import numpy as np
import pandas as pd

# Semilla global para reproducibilidad en todo el libro
np.random.seed(42)

data_path = "data/hour.csv"
df = pd.read_csv(data_path)

df.head()
```

## 1.3 Objetivo general

Nuestro objetivo general es ajustar y analizar un modelo de regresión que explique el conteo total de bicicletas (`cnt`) a partir de variables temporales y climáticas, respetando la estructura de serie temporal de los datos y utilizando técnicas de validación cruzada temporal descritas en capítulos posteriores.
