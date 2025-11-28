# 10. Interpretación operativa y recomendaciones

En este capítulo se traducen los resultados estadísticos a recomendaciones operativas para la planificación de sistemas de bicicletas compartidas.

## 10.1 Sensibilidad de la demanda a variables climáticas

```{code-cell} python
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("data/hour.csv")
features = ["temp", "atemp", "hum", "windspeed", "hr", "workingday"]

X = sm.add_constant(df[features])
y = df["cnt"]

model = sm.OLS(y, X)
results = model.fit()

results.params
```

A partir de los coeficientes estimados se pueden derivar elasticidades aproximadas y escenarios de demanda bajo distintos perfiles de clima y hora del día.

## 10.2 Implicaciones para la operación del sistema

Los resultados pueden ayudar a:

- Dimensionar la flota de bicicletas y puntos de anclaje.
- Diseñar estrategias de reubicación en horas pico.
- Ajustar tarifas o incentivos en función de la demanda esperada.
