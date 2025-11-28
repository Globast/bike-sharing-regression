# 6. Diagnóstico de supuestos del modelo

Aquí analizamos si el modelo lineal cumple con los supuestos de linealidad, homocedasticidad, normalidad y ausencia de autocorrelación fuerte en los residuos.

## 6.1 Residuos versus valores ajustados

```{code-cell} python
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

df = pd.read_csv("data/hour.csv")
features = ["temp", "atemp", "hum", "windspeed", "hr", "workingday"]

X = sm.add_constant(df[features])
y = df["cnt"]

model = sm.OLS(y, X)
results = model.fit()

fitted = results.fittedvalues
residuals = results.resid

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(fitted, residuals, alpha=0.3)
ax.axhline(0, color="black", linewidth=1)
ax.set_xlabel("Valores ajustados")
ax.set_ylabel("Residuos")
ax.set_title("Residuos vs valores ajustados")

plt.tight_layout()
plt.savefig("../data/resid_vs_fitted.png")
plt.show()
```

```{figure} ../data/resid_vs_fitted.png
:name: fig:resid_vs_fitted

Gráfico de residuos frente a valores ajustados.
```

## 6.2 Autocorrelación de residuos

```{code-cell} python
from statsmodels.graphics.tsaplots import plot_acf

fig, ax = plt.subplots(figsize=(6, 4))
plot_acf(residuals, ax=ax, lags=48)
ax.set_title("ACF de los residuos")

plt.tight_layout()
plt.savefig("../data/acf_residuals.png")
plt.show()
```

```{figure} ../data/acf_residuals.png
:name: fig:acf_residuals

Función de autocorrelación de los residuos del modelo.
```

El análisis de {numref}`fig:resid_vs_fitted` y {numref}`fig:acf_residuals` guía la elección de métodos robustos y esquemas de validación adecuados en los capítulos siguientes.
