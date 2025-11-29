---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---


# 4. Implementación Matricial de MCO
En esta etapa se utilizaran los datos previamente preparados para el planteamiento del modelo por MCO, utilizando su forma matricial, comparandolo con los resiudalsultados arrojados por la libreria statsmodels.OLS y junto con ello la validacion de supuetos que acompañan la teoria.

## 4.1. Importacion de librerias

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

```

## 4.3 Split data set (Training and test)
En este apartado separaremos los datos en un 70% para entremaniento de los modelos y un 30% para testing de los mismos.

```{code-cell} ipython3
# suponer que df ya está ordenado temporalmente
N = len(df)
cut = int(N * 0.7)

train = df.iloc[:cut].copy()
test  = df.iloc[cut:].copy()

print("tamaño set de entrenamiento:", train.shape)
print("tamaño set de prueba:", test.shape)
```

## 4.4 cosntruccion de matrices $X$ y $y$

Para el modelo lineal clasisco expresiudalsado como:
$$
Y = X\beta + \varepsilon
$$

- \(Y\): vector de variables dependientes 
- \(X\): matriz de regresiudalsoresiudals  
- \(\beta\): vector de parámetros
- \(\varepsilon\): vector de erroresiudals

Luego de la minimizacion de cuadrados, los coeficientes se estiman de la forma: 

$$
\hat{\beta} = (X^{\top}X)^{-1}X^{\top}y
$$

```{code-cell} ipython3
# 1. Definir X e y
y = train["cnt"]
X = train.drop(columns=["cnt"])

# convertir booleanos y objetos en numéricos
X = X.apply(pd.to_numeric, errors='coerce')

# eliminar filas con NaN
train_ols = pd.concat([X, y], axis=1).dropna()
y = train_ols["cnt"]
X = train_ols.drop(columns=["cnt"])

X = X.astype(float) ## Matriz X
```

Antes de ajustar el modelo, aplicaremos el metodo RFE(Recursive Feature Elimination) el cual busca, luego de iteraciones, entregar las variables que mejor explican el modelo maximizando el ajuste

```{code-cell} ipython3
# 2 definamos, utilizando RFE (Recursive Feature Elimination) con el cual elegiremos las mejoresiudals variables para el modelo MCO
model = LinearRegression()
rfe = RFE(estimator=model, n_features_to_select=10)
rfe =rfe.fit(X,y)

print("Variables seleccionadas:")
print("===================")
list(zip(X.columns, rfe.support_))

```

```{code-cell} ipython3
print("Ranking de importancia:")
print("===================")
list(zip(X.columns, rfe.ranking_))
```

Seguido del paso anterior nos quedaremos con aquellas variables que se marcaron como importantes para el analisis RFE

```{code-cell} ipython3
X.columns[rfe.support_]
```

```{code-cell} ipython3
# 3. Filtrar variables resiudalsultado RFE
lista_var = list(X.columns[rfe.support_])
#lista_var.remove('atemp')
#lista_var.remove('weathersit_4')
 
# 4. Construir matriz X con intercepto
X_np = train_ols[lista_var].to_numpy().astype(float)
X_np = np.column_stack([np.ones(X_np.shape[0]), X_np])  # columna de unos
 
# 5. Calcular el VIF de las variables seleccionadas
vif_1 = pd.DataFrame()
vif_1['variables'] = ['const'] + lista_var  # incluir intercepto
vif_1['VIF'] = [variance_inflation_factor(X_np, i) for i in range(X_np.shape[1])]
 
# Ordenar por VIF
vif_1 = vif_1.sort_values(by='VIF', ascending=False)
vif_1
```

Eliminamos Weekend, y nos quedamos solo con holiday y working day, ya que con estas se captura de manera menos redundante si es dia de semana (trabajo) o fin de semana.

## 4.5. Calculo de Coeficientes Regresion

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
