# 8. Validación y selección del modelo

Para respetar la naturaleza temporal de los datos, utilizamos esquemas de validación cruzada específicos para series temporales.

## 8.1 Particiones temporales con `TimeSeriesSplit`

```{code-cell} python
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(42)

df = pd.read_csv("data/hour.csv")
features = ["temp", "atemp", "hum", "windspeed", "hr", "workingday"]

X = df[features].to_numpy()
y = df["cnt"].to_numpy()

tscv = TimeSeriesSplit(n_splits=5)
mse_scores = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    mse_scores.append(mse)
    print(f"Fold {fold}: MSE = {mse:.2f}")

print("MSE medio:", np.mean(mse_scores))
```

## 8.2 Comparación de especificaciones

A partir de este esquema se pueden comparar distintas especificaciones de variables y transformaciones, eligiendo aquella con mejor desempeño promedio en el horizonte de validación.
