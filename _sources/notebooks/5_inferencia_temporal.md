# 5. Inferencia estadística en datos temporales

En este capítulo discutimos la inferencia sobre los coeficientes del modelo de regresión en presencia de dependencia temporal.

## 5.1 Errores estándar clásicos y robustos

```{code-cell} python
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv("data/hour.csv")
features = ["temp", "atemp", "hum", "windspeed", "hr", "workingday"]

X = sm.add_constant(df[features])
y = df["cnt"]

model = sm.OLS(y, X)
results = model.fit()

# Resumen con errores estándar clásicos
print(results.summary().tables[1])
```

```{code-cell} python
# Errores estándar robustos tipo Newey-West (HAC) para dependencia temporal
results_hac = results.get_robustcov_results(cov_type="HAC", maxlags=24)
results_hac.summary().tables[1]
```

Compararemos los intervalos de confianza obtenidos con supuestos clásicos frente a los robustos para evaluar la sensibilidad de la inferencia.
