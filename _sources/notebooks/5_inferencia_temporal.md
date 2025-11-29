---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---
# 5. Inferencia estadística en datos temporales
En este capítulo discutimos la inferencia sobre los coeficientes del modelo de regresión en presencia de dependencia temporal.

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

df = pd.read_csv("../data/day_clean.csv", sep =";")
N = len(df)
cut = int(N * 0.7)

train = df.iloc[:cut].copy()
test  = df.iloc[cut:].copy()
```
## 5.2 Calculo de Coeficientes Regresion

```{code-cell} ipython3
X = X.astype(float) ## Matriz X
X_np = train_ols[['yr', 'temp', 'hum', 'windspeed', 'season_2', 'season_3', 'season_4',
       'weathersit_2', 'weathersit_3', 'holiday_1']].to_numpy().astype(float)
X_np = np.column_stack([np.ones(X_np.shape[0]), X_np])  # columna de unos
# 4. Aplicar ecuación normal
XtX = X_np.T @ X_np
XtY = X_np.T @ y
beta_manual = np.linalg.pinv(XtX) @ XtY
print("Coeficientes (ecuación normal):", beta_manual.flatten())
```

Ahora utilizaremos OLS de Statmodels, para la generacion de los coeficientes de la regresion

```{code-cell} ipython3
# 6. Ajustar OLS con statsmodels
X_sm = add_constant(train_ols[['yr', 'temp', 'hum', 'windspeed', 'season_2', 'season_3', 'season_4',
       'weathersit_2', 'weathersit_3', 'holiday_1']].astype(float))
ols_model = OLS(y, X_sm).fit()
beta_ols = ols_model.params
print("===============")
print("Coeficientes (OLS statsmodels):\n", beta_ols)
```

```{code-cell} ipython3
# 6. Comparar en tabla
coef_table = pd.DataFrame({
    "Variable": ["Intercepto"] + ['yr', 'temp', 'hum', 'windspeed', 'season_2', 'season_3', 'season_4',
       'weathersit_2', 'weathersit_3', 'holiday_1'],
    "Beta (Ecuación Normal)": beta_manual.flatten(),
    "Beta (OLS Statsmodels)": beta_ols.values
})

print("\nTabla comparativa de coeficientes:")
coef_table
```

## 5.3. Interpretación de coeficientes OLS – Alquiler de bicicletas
### 5.3.1. Tabla de coeficientes

| Variable         | Coeficiente                        | Interpretación económica/urbana                                                                                                                 |
| ---------------- | ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| **const**        | **1675.78**                        | Nivel base de demanda cuando todas las variables están en su valor de referencia. Representa la demanda estructural del sistema de bicicletas.  |
| **yr**           | **1971.33**                        | La demanda aumenta fuertemente con los años, reflejando la expansión del sistema, mayor adopción del servicio y crecimiento poblacional/urbano. |
| **temp**         | **5113.45**                        | Temperaturas más altas elevan de manera sustancial la demanda. El clima cálido hace propicio el uso de bicileta.                                |
| **hum**          | **–1240.47**                       | La humedad elevada reduce la demanda; suele asociarse con sensación térmica incómoda y probabilidad de lluvias.                                 |
| **windspeed**    | **–2770.39**                       | Vientos fuertes disminuyen drásticamente el uso de bicicletas debido a incomodidad y mayor esfuerzo físico.                                     |
| **season_2**     | **1045.70**                        | En la estación 2 (primavera/verano) la demanda aumenta considerablemente, condiciones ideales para montar bicicleta.                            |
| **season_3**     | **658.25**                         | En la estación 3 (otoño), la demanda también sube, aunque menos que en verano. Clima aún favorable.                                             |
| **season_4**     | **1424.82**                        | En la estación 4 (invierno), la demanda crece respecto a la base. Puede reflejar hábitos locales donde el invierno no reduce el uso.            |
| **weathersit_2** | **–374.83**                        | Clima nublado reduce moderadamente el uso de bicicletas frente a días despejados.                                                               |
| **weathersit_3** | **–1668.39**                       | Lluvia ligera tiene un impacto muy fuerte, reduciendo de forma notable la demanda.                                                              |
| **holiday_1**    | **–527.96**                        | En días festivos la demanda disminuye, posiblemente por reducción en viajes laborales y de estudio.                                             |


## Síntesis económica y urbana
 
El modelo confirma que la **demanda de bicicletas está determinada por factores climáticos, temporales y estacionales**.   
- **Clima**: la temperatura impulsa fuertemente la demanda, mientras que la humedad, el viento y la lluvia la reducen.  
- **Tiempo**: las horas pico y los días laborales son los principales motores de la movilidad en bicicleta.  
- **Estacionalidad**: la demanda aumenta en primavera, verano y otoño, mostrando patrones de uso ligados al clima.  
- **Tendencia temporal**: el crecimiento interanual refleja una mayor adopción del sistema de bicicletas en la ciudad.  
- **Festivos**: la caída en días festivos sugiere que la movilidad laboral es el principal determinante de la demanda, con el ocio como complemento.  

En términos de planificación urbana, esto implica que el sistema de alquiler debe considerar tanto la **variabilidad climática** como los **patrones de movilidad laboral y estacional**, reforzando la disponibilidad en horas pico y temporadas favorables, y anticipando caídas en días festivos o de clima adverso.
 

```{code-cell} ipython3
print(ols_model.summary())
```

Extraemos weathersit_4 ya que no es estadisticamene significativo