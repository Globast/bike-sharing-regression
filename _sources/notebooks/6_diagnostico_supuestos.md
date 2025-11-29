---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---
# 6. Diagnóstico de supuestos del modelo
En esta seccion, a partir del modelo anterior, se revisara elñ cumplimiento de la teoria sobre los errores, para entender si el modelo es valido o no

### 6.1. Revision de residuos (Normalidad, Homocedasticidad e independencia)

Se parte de la inspeccion visual de residuos
- Disribucion de residuos
- Q-Q Plot de residuos
- Densiad de residuales

```{code-cell} ipython3
# Librerías científicas básicas
import numpy as np
import pandas as pd
import scipy.stats as stats

# Visualización
import matplotlib.pyplot as plt
import seaborn as sns

# Modelos estadísticos
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.api import OLS, add_constant
 
# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
```

```{code-cell} ipython3
df = pd.read_csv("../data/day_clean.csv", sep =";")
```

```{code-cell} ipython3
## Generacion de resiudals

fitted_values = ols_model.fittedvalues
residuals = ols_model.resid

## Particion de grafica
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
 
# 1. Histograma de resiudal
sns.histplot(residuals, kde=True, bins=30, color="skyblue", ax=axes[0,0])
axes[0,0].set_title("Histograma de resiudal")
axes[0,0].set_xlabel("resiudals")
axes[0,0].set_ylabel("Frecuencia")
 
# 2. Q-Q plot
stats.probplot(residuals, dist="norm", plot=axes[0,1])
axes[0,1].set_title("Q-Q plot de resiudal")
 
# 3. resiudals vs valores ajustados
axes[1,0].scatter(fitted_values, residuals, alpha=0.6, color="purple")
axes[1,0].axhline(y=0, color="red", linestyle="--")
axes[1,0].set_title("resiudal vs Valores ajustados")
axes[1,0].set_xlabel("Valoresiudals ajustados")
axes[1,0].set_ylabel("resiudals")

# 4. Densidad de resiudal
sns.kdeplot(residuals, color="green", fill=True, ax=axes[1,1])
axes[1,1].set_title("Densidad de resiudals")
axes[1,1].set_xlabel("resiudals")
plt.tight_layout()

plt.show()
 
```

Se observa que los residuos forman patrones y que no siguen una distribucion normal, lo cual validaremos con las pruebas estadisticas:
1. Normalidad de residuos (Shapiro-Wilk)
2. Homocedasticidad (Breusch-Pagan)
3. Homocedasticidad (White test)

```{code-cell} ipython3
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro

# 1. Normalidad de residuos (Shapiro-Wilk)
shapiro_test = shapiro(residuals)
print("Shapiro-Wilk Test:")
print("Statistic =", shapiro_test.statistic, "p-value =", shapiro_test.pvalue)
 
# 2. Homocedasticidad (Breusch-Pagan)
bp_test = het_breuschpagan(residuals, ols_model.model.exog)
print("\nBreusch-Pagan Test:")
print("LM stat =", bp_test[0], "p-value =", bp_test[1])
print("F stat =", bp_test[2], "p-value =", bp_test[3])
 
# 3. Homocedasticidad (White test)
white_test = het_white(residuals, ols_model.model.exog)
print("\nWhite Test:")
print("LM stat =", white_test[0], "p-value =", white_test[1])
print("F stat =", white_test[2], "p-value =", white_test[3])
 
# 4. Independencia de residuos (Durbin-Watson)
dw_stat = durbin_watson(residuals)
print("\nDurbin-Watson Test:")
print("Statistic =", dw_stat)
```

## 6.2 Resultados validacion supuestos

**Shapiro-Wilk Test**
Estadístico = 0.9745, p-value = 9.16e-08
El p-value es extremadamente bajo, por lo tanto rechazamos la hipótesis nula de normalidad.
Conclusión: los residuos no siguen una distribución normal. Esto afecta la validez de los intervalos de confianza y los tests de significancia clásicos.

**Breusch-Pagan Test**
Estadístico LM = 42.44, p-value = 6.26e-06
El p-value es muy bajo, indicando heterocedasticidad.
Conclusión: los residuos no son homocedásticos (la varianza no es constante), lo que puede sesgar los errores estándar del modelo.

F-stat BP Test
F = 4.53, p-value = 3.73e-06
El resultado confirma nuevamente la presencia de heterocedasticidad.

**White Test**
Estadístico LM = 114.47, p-value = 8.92e-07
Confirma la presencia de heterocedasticidad en los residuos.
F = 2.60, p-value = 8.27e-08
Indica patrones no lineales en la varianza de los residuos.
Conclusión: se confirma heterocedasticidad.

**Durbin-Watson Test**
Estadístico = 1.253
Valores cercanos a 2 indican ausencia de autocorrelación; valores más bajos indican autocorrelación positiva.
Conclusión: el estadístico 1.25 sugiere autocorrelación positiva en los residuos.
 
 

```{code-cell} ipython3
import pickle

# Guardaremos el modelo en formato .pkl para su posterior utilizacion
with open("../models/ols_model.pkl", "wb") as f:
    pickle.dump(ols_model, f)
```
