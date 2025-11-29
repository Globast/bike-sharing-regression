---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---


# 6. Validación y Selección del Modelo
OLS, Reach, Lasso

## 6.1. Importacion de librerias

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
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
 
# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
```

## 6.2. Lectura de datos

```{code-cell} ipython3
df = pd.read_csv('day_clean.csv', sep =";")
```

## 6.3 Split data set (Training and test)
En este apartado separaremos los datos en un 70% para entremaniento de los modelos y un 30% para testing de los mismos.

```{code-cell} ipython3
# df ordenado
N = len(df)
cut = int(N * 0.7)

train = df.iloc[:cut].copy()
test  = df.iloc[cut:].copy()

print("tamaño set de entrenamiento:", train.shape)
print("tamaño set de prueba:", test.shape)
```

## 6.4 Aplicacion de Ridge, Lasso y Elastic Net
Aqui se aplicara Ridge, Lasso y Elastic Net para encontrar si el modelo requiere de sus correcciones. 

### 6.4.1 Ridge y Lasso

```{code-cell} ipython3
# --------------------------
# 0. Partición temporal 70/30
# --------------------------
N = len(df)
cut = int(N * 0.7)

train = df.iloc[:cut].copy()
test  = df.iloc[cut:].copy()

print("tamaño set de entrenamiento:", train.shape)
print("tamaño set de prueba:", test.shape)

# --------------------------
# 1. Subdivisión del conjunto de entrenamiento
# --------------------------
X_train = train[['yr', 'temp', 'hum', 'windspeed', 'season_2', 'season_3', 'season_4',
       'weathersit_2', 'weathersit_3', 'holiday_1']]

y_train = train['cnt']

X_test = test[['yr', 'temp', 'hum', 'windspeed', 'season_2', 'season_3', 'season_4',
       'weathersit_2', 'weathersit_3', 'holiday_1']]

y_test = test['cnt']

# ----------------------------------------------------------
# 1b. Partición interna también NO aleatoria (validación)
# ----------------------------------------------------------
N2 = len(X_train)
cut2 = int(N2 * 0.8)

X_train_2 = X_train.iloc[:cut2]
y_train_2 = y_train.iloc[:cut2]

X_test_2 = X_train.iloc[cut2:]
y_test_2 = y_train.iloc[cut2:]

# --------------------------
# 2. Búsqueda de alpha + guardar R2
# --------------------------
np.random.seed(42)
alphas = np.logspace(-4, 3, 100)

mejor_alpha_ridge = None
mejor_r2_ridge = -np.inf

mejor_alpha_lasso = None
mejor_r2_lasso = -np.inf

# listas para graficar
r2_ridge_list = []
r2_lasso_list = []

for a in alphas:

    # Ridge
    modelo_r = Ridge(alpha=a)
    modelo_r.fit(X_train_2, y_train_2)
    pred_r = modelo_r.predict(X_test_2)
    r2_r = r2_score(y_test_2, pred_r)
    r2_ridge_list.append(r2_r)

    if r2_r > mejor_r2_ridge:
        mejor_r2_ridge = r2_r
        mejor_alpha_ridge = a

    # Lasso
    modelo_l = Lasso(alpha=a, max_iter=10000)
    modelo_l.fit(X_train_2, y_train_2)
    pred_l = modelo_l.predict(X_test_2)
    r2_l = r2_score(y_test_2, pred_l)
    r2_lasso_list.append(r2_l)

    if r2_l > mejor_r2_lasso:
        mejor_r2_lasso = r2_l
        mejor_alpha_lasso = a

print(f"Ridge --> Mejor alpha = {mejor_alpha_ridge:.4f} | R² validación = {mejor_r2_ridge:.4f}")
print(f"Lasso --> Mejor alpha = {mejor_alpha_lasso:.4f} | R² validación = {mejor_r2_lasso:.4f}")

# --------------------------
# 2b. Gráfica alpha vs R2
# --------------------------
plt.figure(figsize=(10,6))
plt.plot(alphas, r2_ridge_list, label='Ridge', marker='o')
plt.plot(alphas, r2_lasso_list, label='Lasso', marker='s')

plt.xscale('log')
plt.xlabel("Alpha (escala log)")
plt.ylabel("R² en validación")
plt.title("Evolución de R² en función de alpha")
plt.grid(True)
plt.legend()
plt.show()

# --------------------------
# 3. Métricas
# --------------------------
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# --------------------------
# 4. Reentrenar con TODO el train
# --------------------------
ridge_final = Ridge(alpha=mejor_alpha_ridge)
ridge_final.fit(X_train, y_train)
ridge_pred_train = ridge_final.predict(X_train)
ridge_pred_test = ridge_final.predict(X_test)

resultados_ridge = {
    "R2 Train": r2_score(y_train, ridge_pred_train),
    "R2 Test": r2_score(y_test, ridge_pred_test),
    "RMSE Test": np.sqrt(mean_squared_error(y_test, ridge_pred_test)),
    "MAE Test": mean_absolute_error(y_test, ridge_pred_test),
    "MAPE Test (%)": mape(y_test, ridge_pred_test)
}

# LASSO final
lasso_final = Lasso(alpha=mejor_alpha_lasso, max_iter=10000)
lasso_final.fit(X_train, y_train)
lasso_pred_train = lasso_final.predict(X_train)
lasso_pred_test = lasso_final.predict(X_test)

resultados_lasso = {
    "R2 Train": r2_score(y_train, lasso_pred_train),
    "R2 Test": r2_score(y_test, lasso_pred_test),
    "RMSE Test": np.sqrt(mean_squared_error(y_test, lasso_pred_test)),
    "MAE Test": mean_absolute_error(y_test, lasso_pred_test),
    "MAPE Test (%)": mape(y_test, lasso_pred_test)
}

# --------------------------
# 5. Tabla final
# --------------------------
df_resultados = pd.DataFrame([resultados_ridge, resultados_lasso], index=["Ridge", "Lasso"])
display(df_resultados)
```

### 6.4.2. Elastic Net

```{code-cell} ipython3
# ---------------------------------
# 1. Partición temporal 70 / 30
# ---------------------------------
N = len(df)
cut = int(N * 0.7)

train = df.iloc[:cut].copy()
test  = df.iloc[cut:].copy()

# ---------------------------------
# 2. Definir X/Y
# ---------------------------------
X_train = train[['yr', 'temp', 'hum', 'windspeed', 'season_2', 'season_3', 'season_4',
       'weathersit_2', 'weathersit_3', 'holiday_1']]
y_train = train['cnt']

X_test = test[['yr', 'temp', 'hum', 'windspeed', 'season_2', 'season_3', 'season_4',
       'weathersit_2', 'weathersit_3', 'holiday_1']]
y_test = test['cnt']

# ---------------------------------
# 3. Partición interna también NO aleatoria (80/20)
# ---------------------------------
N2 = len(X_train)
cut2 = int(N2 * 0.8)

X_train_2 = X_train.iloc[:cut2]
y_train_2 = y_train.iloc[:cut2]

X_test_2 = X_train.iloc[cut2:]
y_test_2 = y_train.iloc[cut2:]

# ---------------------------------
# 4. Búsqueda de hiperparámetros
# ---------------------------------
alphas = np.logspace(-4, 3, 40)
l1_ratios = np.linspace(0.1, 0.9, 9)

mejor_alpha = None
mejor_l1 = None
mejor_r2 = -np.inf

for a in alphas:
    for l1 in l1_ratios:
        modelo = ElasticNet(alpha=a, l1_ratio=l1, max_iter=10000)
        modelo.fit(X_train_2, y_train_2)
        pred = modelo.predict(X_test_2)
        r2_val = r2_score(y_test_2, pred)

        if r2_val > mejor_r2:
            mejor_r2 = r2_val
            mejor_alpha = a
            mejor_l1 = l1

print(f"ElasticNet --> Mejor alpha = {mejor_alpha:.4f}, Mejor l1_ratio = {mejor_l1:.2f}")
print(f"R² Validación = {mejor_r2:.4f}")

# ---------------------------------
# 5. Entrenamiento final
# ---------------------------------
elastic_final = ElasticNet(alpha=mejor_alpha, l1_ratio=mejor_l1, max_iter=10000)
elastic_final.fit(X_train, y_train)

pred_train = elastic_final.predict(X_train)
pred_test = elastic_final.predict(X_test)

# ---------------------------------
# 6. Métricas finales
# ---------------------------------
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

resultados_elastic = {
    "R2 Train": r2_score(y_train, pred_train),
    "R2 Test": r2_score(y_test, pred_test),
    "RMSE Test": np.sqrt(mean_squared_error(y_test, pred_test)),
    "MAE Test": mean_absolute_error(y_test, pred_test),
    "MAPE Test (%)": mape(y_test, pred_test)
}

pd.DataFrame(resultados_elastic, index=["Elastic Net"])

```

### 6.4.3. Recopilacion resultados Ridge, Lasso y Elastic Net y contraste con OLS

```{code-cell} ipython3
# 1. Cargar el modelo OLS
with open("../models/ols_model.pkl", "rb") as f:
    ols_model = pickle.load(f)

# 2. Preparar X con constante
X_train_const = sm.add_constant(X_train, has_constant='add')
X_test_const = sm.add_constant(X_test, has_constant='add')

# 3. Predicciones
y_pred_train = ols_model.predict(X_train_const)
y_pred_test = ols_model.predict(X_test_const)

# 4. Métricas
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae_test = mean_absolute_error(y_test, y_pred_test)
mape_test = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100

# 5. Diccionario de resultados OLS
resultados_ols = {
    "R2 Train": r2_train,
    "R2 Test": r2_test,
    "RMSE Test": rmse_test,
    "MAE Test": mae_test,
    "MAPE Test (%)": mape_test
}

# 6. Añadir al DataFrame final
df_resultados_final = pd.DataFrame(
    [resultados_ridge, resultados_lasso, resultados_elastic, resultados_ols],
    index=["Ridge", "Lasso", "Elastic Net", "OLS"]
)

print(f"Ridge --> Mejor alpha = {mejor_alpha_ridge:.4f} | R² validación = {mejor_r2_ridge:.4f}")
print(f"Lasso --> Mejor alpha = {mejor_alpha_lasso:.4f} | R² validación = {mejor_r2_lasso:.4f}")
print(f"ElasticNet --> Mejor alpha = {mejor_alpha:.4f}, Mejor l1_ratio = {mejor_l1:.2f} | R² Validación = {mejor_r2:.4f}")
display(df_resultados_final)
```

```{code-cell} ipython3
plt.figure(figsize=(10, 6))

# Scatter de predicciones
plt.scatter(y_test, ridge_pred_test, alpha=0.6, label="Ridge", s=20)
plt.scatter(y_test, lasso_pred_test, alpha=0.6, label="Lasso", s=20)
plt.scatter(y_test, pred_test, alpha=0.6, label="Elastic Net", s=20)
plt.scatter(y_test, y_pred_test, alpha=0.6, label="OLS model", s=20)


# Línea ideal (predicción perfecta)
min_val = min(
    y_test.min(), 
    ridge_pred_test.min(), 
    lasso_pred_test.min(),
    pred_test.min()
)
max_val = max(
    y_test.max(), 
    ridge_pred_test.max(), 
    lasso_pred_test.max(),
    pred_test.max()
)

plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Línea ideal")

plt.xlabel("Valores reales (y_test)")
plt.ylabel("Predicciones")
plt.title("Comparación: Predicciones vs Valores Reales (Ridge, Lasso y Elastic Net)")
plt.legend()
plt.show()

```

### 6.5. Interpretación de resultados

Los tres modelos —**Ridge, Lasso y Elastic Net**— presentan desempeños prácticamente idénticos. Tanto los valores de ( $R^2$ ) en validación como las métricas en el conjunto de prueba muestran diferencias mínimas entre ellos. Esto permite concluir lo siguiente:

* **La multicolinealidad no es un problema relevante** en este conjunto de variables, ya que ni Lasso ni Elastic Net reducen coeficientes a cero ni muestran mejoras sustanciales frente a Ridge.
* **Todas las variables aportan información útil** al modelo; las penalizaciones no seleccionan ni descartan predictores, lo que confirma una buena preparación previa de las variables climáticas y temporales.
* **La estructura del modelo es estable**: las regularizaciones no alteran significativamente los coeficientes.
* **La relación entre clima, temporalidad y demanda es moderada**, suficiente para explicar parte del comportamiento, pero no lo bastante fuerte como para que las técnicas penalizadas marquen diferencias.

En términos de generalización, el rendimiento es consistente: el ( $R^2$ ) en entrenamiento y prueba se mantiene alrededor de 0.56, lo que indica una **ligera pérdida de capacidad predictiva** pero sin señales de sobreajuste importante.

Finalmente, existe espacio para mejorar el modelo explorando:

* **Interacciones entre variables** (por ejemplo, temperatura × humedad, hora × estación).
* **Interacciones No lineales** 
* **Modelos específicos para series de tiempo**, que capturen patrones más complejos como tendencia, estacionalidad y autocorrelación a largo plazo.


### 6.6. Seleccion del modelo optimo

Luego de la comparacion entre modelos, nos quedamos con OLS, luego de su correccion de errores, y haber notado que sus coeficientes eran robistos, podemos con el explicar buena parte de los datos, y evitamos el gasto computacional de correr nuevamnete los modelos.
