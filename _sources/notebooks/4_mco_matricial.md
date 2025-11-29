---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---


# 4. Implementación Matricial de MCO y diagnosticos de supuestos

## 4.1. Implementación Matricial de MCO
En esta etapa se utilizaran los datos previamente preparados para el planteamiento del modelo por MCO, utilizando su forma matricial, comparandolo con los resiudalsultados arrojados por la libreria statsmodels.OLS y junto con ello la validacion de supuetos que acompañan la teoria.

### 4.1.1. Importacion de librerias

```{code-cell} ipython3
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.api import OLS, add_constant
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
df = pd.read_csv("../data/day_clean.csv", sep =";")
```

### 4.1.2. Split data set (Training and test)
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

### 4.1.3. cosntruccion de matrices $X$ y $y$

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

### 4.1.4. Calculo de Coeficientes Regresion

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
## 4.2. Inferencia estadística en datos temporales
En este capítulo discutimos la inferencia sobre los coeficientes del modelo de regresión en presencia de dependencia temporal.

### Tabla de coeficientes

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


### Síntesis económica y urbana
 
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

## 4.3. Diagnóstico de supuestos del modelo
En esta seccion, a partir del modelo anterior, se revisara elñ cumplimiento de la teoria sobre los errores, para entender si el modelo es valido o no

### 4.3.1. Revision de residuos (Normalidad, Homocedasticidad e independencia)

Se parte de la inspeccion visual de residuos
- Disribucion de residuos
- Q-Q Plot de residuos
- Densiad de residuales

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

## 4.3.2. Resultados validacion supuestos

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
