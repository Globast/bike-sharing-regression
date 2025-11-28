---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---

# 7. Métodos robustos para datos temporales

Dado los desarrollos anteriores, se obtuvo que en el modelo de ajuste lineal los errores no satisfacen los supuestos de normalidad y homocedasticidad (varianza costante)  
Para lo anterior, en esta seccion estaremos explorando el uso de tecnicas como:  
1. Errores estandar HAC (Newey West) 
2. Estimadores robustos M (Huber, Tukey)
3. Boostrap temporal (Block boostrap)

## 7.1. Importacion de librerias

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

## 7.2. Lectura de datos

```{code-cell} ipython3
df = pd.read_csv('../data/hour_clean.csv', sep =";")
```

## 7.3 Split data set (Training and test)
En este apartado separaremos los datos en un 70% para entremaniento de los modelos y un 30% para testing de los mismos.

```{code-cell} ipython3
np.random.seed(42)
train, test = train_test_split(df, train_size=0.7, test_size=0.3, random_state=42) ## aqui separamos uno en 70% del data set y otro 30%, la separacion es aleatoria
print("tamaño set de entrenemaiento ", train.shape)
print("tamaño set de prueba ", test.shape)
```

## 7.4 Errores estándar HAC (Newey–West).  

### 7.4.1. Método HAC (Newey–West)  

El estimador de **Errores estándar HAC (Heteroskedasticity and Autocorrelation Consistent)** ajusta la matriz de varianzas y covarianzas de los coeficientes OLS para ser robusta frente a heterocedasticidad y autocorrelación.

#### 7.4.2. Modelo OLS
$$
y_t = X_t \beta + u_t
$$

#### 7.4.3. Estimador HAC
$$
\widehat{V}_{HAC} = (X'X)^{-1} \, \widehat{S} \, (X'X)^{-1}
$$

donde:
$$
\widehat{S} = \sum_{t=1}^{T} \hat{u}_t^2 X_t X_t' 
+ \sum_{l=1}^{q} w_l \sum_{t=l+1}^{T} \hat{u}_t \hat{u}_{t-l} 
\big( X_t X_{t-l}' + X_{t-l} X_t' \big)
$$

#### 7.4.4. Parámetros
- $\hat{u}_t$: residuos estimados del modelo OLS  
- $X_t$: vector de regresores en el tiempo $t$  
- $q$: número de rezagos (`maxlags`) especificado  
- $w_l = 1 - \frac{l}{q+1}$: pesos de Bartlett que decrecen linealmente con el rezago

#### 7.4.5. Interpretación
- El primer término corrige la **heterocedasticidad**.  
- El segundo término corrige la **autocorrelación** hasta el rezago $q$.  
- Los coeficientes $\beta$ no cambian, pero los errores estándar y las pruebas de significancia se ajustan.

```{code-cell} ipython3
import pickle
from statsmodels.stats.diagnostic import acorr_ljungbox

# 1) Cargar el modelo OLS guardado
with open("ols_model.pkl", "rb") as f:
    ols_model = pickle.load(f)

# 2) Heurística de maxlags según tamaño de muestra (Newey–West)
resid = ols_model.resid
T = len(resid)
heuristic_maxlags = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
heuristic_maxlags = max(1, heuristic_maxlags)

# 3) Opcional: ajustar por frecuencia (ejemplo: mensual, cap en 12)
# freq_cap = 12
# maxlags = min(heuristic_maxlags, freq_cap)

maxlags = heuristic_maxlags

# 4) Diagnóstico rápido con Ljung–Box en el lag elegido
lb = acorr_ljungbox(resid, lags=[maxlags], return_df=True)
print(f"Ljung–Box p-value at lag={maxlags}: {lb['lb_pvalue'].iloc[-1]:.4f}")

# 5) Aplicar HAC (Newey–West)
modelo_hac = ols_model.get_robustcov_results(cov_type="HAC", maxlags=maxlags)

# 6) Resumen con errores estándar corregidos
print(modelo_hac.summary())
```

## 7.5. Estimadores M Robust (Huber y Tukey)  
### 7.5.1. Aplicación sobre un modelo OLS

Cuando hacemos regresión OLS, los coeficientes se estiman minimizando:

$$
\min_\beta \sum_{i=1}^n (y_i - x_i^\top \beta)^2
$$

Esto hace que OLS sea **muy sensible a valores atípicos**, porque los residuos grandes se elevan al cuadrado.

Los **estimadores M** reducen la influencia de estos valores usando una función de pérdida $ \rho(\cdot) $ que penaliza menos a los outliers.

### 7.5.2. Estimador M general

Los estimadores M encuentran los coeficientes resolviendo:

$$
\hat{\beta} = \arg\min_\beta \sum_{i=1}^n \rho(r_i)
$$

donde:

- $ r_i = \frac{y_i - x_i^\top\beta}{\hat{\sigma}}$ es el residuo estandarizado  
- $ \rho(\cdot)$ es una función de pérdida robusta  

La ecuación de primer orden usa la función $\psi(r) = \rho'(r)$, y se resuelve:

$$
\sum_{i=1}^n \psi(r_i)x_i = 0
$$

### 7.5.3. Estimador de Huber

La función de pérdida es:

$$
\rho(r)=
\begin{cases}
\frac{1}{2}r^2 & |r| \le c \\
c|r| - \frac{1}{2}c^2 & |r| > c
\end{cases}
$$

La función derivada $\psi(r)$ es:

$$
\psi(r)=
\begin{cases}
r & |r|\le c \\
c \,\text{sign}(r) & |r| > c
\end{cases}
$$

Reduce la influencia de outliers moderados sin descartarlos.

### 7.5.4. Estimador Tukey Bisquare (Biweight)

Función de pérdida:

$$
\rho(r) =
\begin{cases}
\frac{c^2}{6}\left[1 - \left(1 - \left(\frac{r}{c}\right)^2\right)^3\right] & |r| \le c \\
\frac{c^2}{6} & |r| > c
\end{cases}
$$

Derivada $\psi(r)$:

$$
\psi(r)=
\begin{cases}
r \left(1 - \left(\frac{r}{c}\right)^2\right)^2 & |r| \le c \\
0 & |r| > c
\end{cases}
$$

Los outliers muy extremos reciben peso $0$ (se excluyen del ajuste).

### 7.5.5. Metodo Huber y Tukey

```{code-cell} ipython3
# Recuperar X y y con nombres reales desde el modelo
X = ols_model.model.exog
y = ols_model.model.endog
feature_names = ols_model.model.exog_names

X = pd.DataFrame(X, columns=feature_names)
y = pd.Series(y, name=ols_model.model.endog_names)

```

```{code-cell} ipython3
# Huber
huber_model = sm.RLM(y, X, M=sm.robust.norms.HuberT())
huber_results = huber_model.fit()

# Tukey
tukey_model = sm.RLM(y, X, M=sm.robust.norms.TukeyBiweight())
tukey_results = tukey_model.fit()

print("=== Huber ===")
print(huber_results.summary())

print("=== Tukey ===")
print(tukey_results.summary())

```

### 7.5.6. Bootstrap temporal (Moving Block Bootstrap) — 2000 réplicas

Cuando los datos son temporales (autocorrelación, dependencias en el tiempo) no se puede re-muestrear observaciones individuales de forma independiente sin romper la estructura temporal. El **block bootstrap** re-muestrea **bloques contiguos** de observaciones para preservar la dependencia local.

En el **Moving Block Bootstrap (MBB)**:
- Elegimos una longitud de bloque $ \ell $.
- Construimos bloques superpuestos de longitud $ \ell$: bloque 0 = observaciones [0:$\ell$-1], bloque 1 = [1:$\ell$], ..., hasta el final.
- Para generar una muestra bootstrap de tamaño $T$, extraemos $k = \lceil T / \ell \rceil$ bloques *con reemplazo* de la lista de bloques y los concatenamos; finalmente recortamos a longitud $T$.
- Repetimos esto $B$ veces (para el ejercicio $B=2000$), y para cada muestra bootstrap re-estimamos el modelo OLS y guardamos los coeficientes.
- Las inferencias se hacen sobre la distribución empírica de los coeficientes media, desviación estándar, intervalos percentiles, y demas.

### 7.5.7. Elección de la longitud de bloque  $\ell$
- Regla de pulgar común: $\ell \approx T^{1/3}$ 

```{code-cell} ipython3
# parametros para el bootstrap
T = len(y)
print(f"Tamaño de la muestra T = {T}, variables = {X.shape[1]}")

B = 2000  # número de replicas

# regla de pulgar para longitud de bloque: l = int(T**(1/3))
l = max(1, int(np.round(T ** (1/3))))
print(f"Block length l = {l} (regla ~ T^(1/3))")

k = int(np.ceil(T / l))  # número de bloques a concatenar por réplica

# listado de indice de bloques
# cada bloque comienza en i y cubre [i, i+l-1]
block_starts = np.arange(0, T - l + 1)
blocks = [np.arange(s, s + l) for s in block_starts]
n_blocks = len(blocks)
print(f"Número de bloques = {n_blocks}")

# funcion para el muestreo
def generate_mbb_indices(T, l, blocks, n_blocks):
    """
    Genera un vector de indices de longitud T usando Moving Block Bootstrap (MBB)
    """
    k = int(np.ceil(T / l))
    
    # elegir k bloques con reemplazo
    chosen_block_idxs = np.random.randint(0, n_blocks, size=k)
    
    chosen_indices = np.concatenate([blocks[i] for i in chosen_block_idxs])
    
    # recortar a longitud T
    chosen_indices = chosen_indices[:T]
    return chosen_indices


# Ejecutar bootstrap y estimar modelo OLS en cada replica
coefs = np.zeros((B, X.shape[1]))  # guardar coeficientes por réplica

# Para reproducibilidad, puedes fijar semilla:
np.random.seed(42)

for b in range(B):

    idx = generate_mbb_indices(T, l, blocks, n_blocks)

    Xb = X.iloc[idx].reset_index(drop=True)
    yb = y.iloc[idx].reset_index(drop=True)

    try:
        model_b = sm.OLS(yb, Xb).fit()
        coefs[b, :] = model_b.params.values
    except Exception as e:
        # En caso de error numérico, asignar Nan
        coefs[b, :] = np.nan
        print(f"Replica {b} falló: {e}")


# quitar replicas malas
valid = ~np.isnan(coefs).any(axis=1)
n_valid = valid.sum()
print(f"Réplicas válidas: {n_valid} / {B}")

coefs_valid = coefs[valid, :]

# metricas para el bootstrap 
coef_mean = np.mean(coefs_valid, axis=0)
coef_std = np.std(coefs_valid, axis=0, ddof=1)
ci_lower = np.percentile(coefs_valid, 2.5, axis=0)
ci_upper = np.percentile(coefs_valid, 97.5, axis=0)

results_df = pd.DataFrame({
    "coef_mean": coef_mean,
    "coef_std": coef_std,
    "ci_2.5%": ci_lower,
    "ci_97.5%": ci_upper
}, index=feature_names)

print("\nResultados bootstrap (MBB, 2000 réplicas):")
print(results_df)

```

## 7.5.8. Tabla comparativa, robustez de los coeficientes por metodo

```{code-cell} ipython3
# Extraccion de coeficientes
ols_coef = ols_model.params
hac_coef = modelo_hac.params
huber_coef = huber_results.params
tukey_coef = tukey_results.params

# 2) Bootstrap: ya tienes results_df con:
# coef_mean, coef_std, ci_2.5%, ci_97.5%
boot_mean = results_df["coef_mean"]

# 3) Combinar todo en una sola tabla

comparativa = pd.DataFrame({
    "OLS": ols_coef,
    "HAC (Newey-West)": hac_coef,
    "Huber": huber_coef,
    "Tukey": tukey_coef,
    "Bootstrap Mean": boot_mean,
    "Bootstrap Std": results_df["coef_std"],
    "Boot CI 2.5%": results_df["ci_2.5%"],
    "Boot CI 97.5%": results_df["ci_97.5%"]
})

print("\nTabla comparativa de robustez de coeficientes:\n")
comparativa
```

#### ¿Qué significa que un coeficiente es “robusto”?  

Un estimador de un coeficiente es robusto cuando al violar alguno de los supuestos del modelo lineal **NO  cambian mucho** su valor ni su significancia, es decir, se mantiene estable.  

Los supuestos típicos que se violan en datos reales son:  

- heteroscedasticidad
- autocorrelación
- colas pesadas / outliers
- dependencia temporal
- no normalidad de residuos

Es por ello que al revisar los diferentes metodos, estos abordan cada uno un aspecto distinto:  

| Metodo                          | Corrige o soluciona                     |
| ------------------------------- | -------------------------------------------- |
| **HAC (Newey-West)**            | Autocorrelacion + heteroscedasticidad        |
| **Huber (M-estimation)**        | Outliers moderados                           |
| **Tukey biweight**              | Outliers severos                             |
| **Bootstrap (block bootstrap)** | Distribucion no normal, dependencia temporal |

### 7.5.9. ¿Criterios de Robustez utilizados?  
En la evaluacion de robustez, se aplica entonces validaciones entre coeficientes de los diferentes metodos para evaluar:

1. **Estabilidad del coeficiente entre metodos:**   
    La magbitud del coeficiente debe ser "parecida" (toleremos una variacion del 20%, tomando por base OLS), por lo que si el valor cambia mucho, entonces el efecto no es tan fiable.

2. **Estabilidad del signo:**  
    El signo del coeficiente debe ser el mismo en todas los metodos. Si se mueve entre positivo y negativo, la relación no es robusta, ya que la relacion no puede indicar "a veces aumenta, aveces disminuye".

3. **Estabilidad de la significancia estadística:**  
    Si en para OLS, HACy Huber/Tukey para el coeficiente su $p-value < 0.05$, y el intervalo de bootstrap no contiene al 0, el coeficnicoeficiente es **robusto**.

4. **Ancho del intervalo bootstrap vs tamaño del coeficiente OLS:**
    Caundo el intervalo de confianza, en este caso 95%, el ancho o tamaño del inetervalo es igual a la magnitud del coeficiente es reflejo de una alta variabilidad, por lo que no es tan "robusto" entonces.  
    Es por esto que 1.5 veces (IC95%) la desviacion de bootstrap determinamos sea menor que la magnitud del coeficiente OLS 





```{code-cell} ipython3
df_rob = comparativa.copy()

# Criterios de robustez

## signo estable
def is_sign_stable(row):
    signs = np.sign([row['OLS'], row['Huber'], row['Tukey'], row['Bootstrap Mean']])
    return len(set(signs)) == 1  # todos iguales

## Consistencia entre metodos (tolerancia 20%)
def is_consistent_methods(row, tol=0.20):
    """
    Cambios menores al 20% entre:
    - OLS vs Huber
    - OLS vs Tukey
    - OLS vs Bootstrap Mean
    """
    base = row['OLS']
    if base == 0:
        return False
    
    ratios = [
        abs(row['Huber'] - base) / abs(base),
        abs(row['Tukey'] - base) / abs(base),
        abs(row['Bootstrap Mean'] - base) / abs(base)
    ]
    return all(r < tol for r in ratios)

## Se encuentran dentro del intervalo de bootstrap
def is_inside_bootstrap_CI(row):
    ci_low = row['Boot CI 2.5%']
    ci_high = row['Boot CI 97.5%']
    return (ci_low <= row['Huber'] <= ci_high) and (ci_low <= row['Tukey'] <= ci_high)

##Intervalo bootstrap es estable 
def is_bootstrap_stable(row):
    """
    Intervalo no excesivamente ancho.
    Definimos inestabilidad si: Std > 1.5 * |coef|
    """
    return row['Bootstrap Std'] <= 1.5 * abs(row['OLS'])

#evaluar criterios

df_rob['Signo estable'] = comparativa.apply(is_sign_stable, axis=1)
df_rob['Métodos consistentes'] = comparativa.apply(is_consistent_methods, axis=1)
df_rob['Dentro CI bootstrap'] = comparativa.apply(is_inside_bootstrap_CI, axis=1)
df_rob['Bootstrap estable'] = comparativa.apply(is_bootstrap_stable, axis=1)

# Criterio final
df_rob['ROBUSTO'] = np.where(
    df_rob[['Signo estable','Métodos consistentes','Dentro CI bootstrap','Bootstrap estable']].all(axis=1),
    "Sí", "No"
)

# Mostrar tabla final de robustez
print("\n============================")
print(" TABLA FINAL DE ROBUSTEZ")
print("============================\n")
df_rob
```

```{code-cell} ipython3
print ("Variables ROBUSTAS, luego de aplicar las validaciones definidas:")
for i in list(df_rob[df_rob['ROBUSTO']=="Sí"].index):
    print(i)
```
