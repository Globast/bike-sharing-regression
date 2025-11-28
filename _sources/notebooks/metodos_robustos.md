# 7. Métodos robustos para datos temporales

Dado que los datos horarios presentan autocorrelación y posibles cambios de varianza, complementamos el análisis con métodos robustos.

## 7.1 Errores estándar robustos HAC

```{code-cell} python
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("data/hour.csv")
features = ["temp", "atemp", "hum", "windspeed", "hr", "workingday"]

X = sm.add_constant(df[features])
y = df["cnt"]

model = sm.OLS(y, X)
results = model.fit()
results_hac = results.get_robustcov_results(cov_type="HAC", maxlags=24)

results_hac.summary().tables[1]
```

## 7.2 Modelos alternativos

En esta sección se discuten posibles extensiones:

- Modelos con términos autorregresivos explícitos.
- Transformaciones de la respuesta (por ejemplo, raíz cuadrada o logaritmo).
- Penalización de coeficientes (ridge, lasso) respetando la estructura temporal mediante validación cruzada temporal.
