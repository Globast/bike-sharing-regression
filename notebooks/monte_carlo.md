# 9. Simulación Monte Carlo

En este capítulo empleamos simulaciones Monte Carlo para estudiar la variabilidad de los estimadores bajo escenarios controlados.

## 9.1 Esquema básico de simulación

```{code-cell} python
import numpy as np

np.random.seed(42)

def simulate_ols(n=500, beta0=10.0, beta1=2.0, sigma=5.0):
    x = np.linspace(0, 10, n)
    eps = np.random.normal(0, sigma, size=n)
    y = beta0 + beta1 * x + eps

    X = np.column_stack([np.ones(n), x])
    beta_hat = np.linalg.inv(X.T @ X) @ (X.T @ y)
    return beta_hat

n_sim = 1000
betas = np.array([simulate_ols() for _ in range(n_sim)])

betas.mean(axis=0), betas.std(axis=0)
```

## 9.2 Relación con el caso de demanda de bicicletas

Aunque esta simulación usa un modelo muy simplificado, sirve para ilustrar la distribución muestral de los estimadores definidos por la ecuación {eq}`eq:ols_matrix` y compararla con los resultados observados en el conjunto de datos real.
