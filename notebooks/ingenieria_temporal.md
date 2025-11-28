# 2. Ingeniería de variables y preprocesamiento temporales

En este capítulo construimos variables derivadas que capturan patrones de tendencia, estacionalidad y efectos de calendario, así como lags útiles para el modelado de series temporales.

## 2.1 Conversión de fechas y orden temporal

```{code-cell} python
import pandas as pd

df = pd.read_csv("data/hour.csv")
df["dteday"] = pd.to_datetime(df["dteday"])

# Orden temporal explícito
df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)

df[["dteday", "hr"]].head()
```

## 2.2 Variables de calendario enriquecidas

```{code-cell} python
# Día de la semana en formato texto y fin de semana
df["weekday_name"] = df["dteday"].dt.day_name()
df["is_weekend"] = df["weekday"].isin([0, 6]).astype(int)

# Hora del día como categoría ordenada
df["hr_cat"] = pd.Categorical(df["hr"], ordered=True)

df[["dteday", "hr", "weekday", "weekday_name", "is_weekend"]].head()
```

## 2.3 Lags y promedios móviles

```{code-cell} python
# Lag de una hora en el conteo total
df["cnt_lag1"] = df["cnt"].shift(1)

# Promedio móvil de 24 horas
df["cnt_roll24"] = df["cnt"].rolling(window=24, min_periods=1).mean()

df[["dteday", "hr", "cnt", "cnt_lag1", "cnt_roll24"]].head(30)
```

Estas nuevas variables temporales se utilizarán posteriormente en el modelo de regresión y en el análisis de diagnóstico.
