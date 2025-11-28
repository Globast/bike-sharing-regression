# 4. Implementación matricial de MCO

En este capítulo implementamos la estimación de mínimos cuadrados ordinarios (MCO) para un modelo lineal que relaciona el conteo total de bicicletas con un conjunto de variables explicativas.

## 4.1 Formulación matricial

Denotemos por \(\mathbf{y}\) el vector columna de respuestas (conteos) y por \(\mathbf{X}\) la matriz de diseño que incluye un término constante y las covariables seleccionadas. El estimador de MCO para los coeficientes \(\hat{\boldsymbol{\beta}}\) se define como

```{math}
:label: eq:ols_matrix
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}.
```

La ecuación {eq}`eq:ols_matrix` será el núcleo de nuestra implementación en NumPy.

## 4.2 Selección de variables y construcción de la matriz de diseño

```{code-cell} python
import numpy as np
import pandas as pd

np.random.seed(42)

df = pd.read_csv("data/hour.csv")
df["dteday"] = pd.to_datetime(df["dteday"])
df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)

features = ["temp", "atemp", "hum", "windspeed", "hr", "workingday"]
X = df[features].to_numpy()
y = df["cnt"].to_numpy()

# Agregar columna de unos (intercepto)
X_design = np.column_stack([np.ones(X.shape[0]), X])

X_design.shape, y.shape
```

## 4.3 Implementación directa de MCO

```{code-cell} python
# Implementación de MCO usando la ecuación matricial
XtX = X_design.T @ X_design
XtX_inv = np.linalg.inv(XtX)
Xt_y = X_design.T @ y

beta_hat = XtX_inv @ Xt_y
beta_hat
```

## 4.4 Comparación con una librería estándar

```{code-cell} python
import statsmodels.api as sm

X_sm = sm.add_constant(df[features])
model = sm.OLS(y, X_sm)
results = model.fit()

print(results.summary().tables[1])
```

En capítulos posteriores utilizaremos estos coeficientes para la inferencia estadística y la validación del modelo.
