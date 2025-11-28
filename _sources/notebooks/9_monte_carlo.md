---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---

# 9. Simulación Monte Carlo

En este capítulo empleamos simulaciones Monte Carlo para estudiar la variabilidad de los estimadores bajo escenarios controlados.

# Monte Carlo: efectos de autocorrelación en estimadores OLS

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

np.random.seed(123)

# ---------- Parámetros del experimento ----------
B = 500                # réplicas Monte Carlo (recomendar 500-2000)
Ns = [200, 500, 1000, 2000]   # tamaños muestrales a probar (learning curves)
rhos = [0.0, 0.3, 0.6, 0.9]   # autocorrelación en errores
beta0 = 2.0
beta1 = 3.0
sigma_eps = 1.0        # desviación del ruido blanco
# -------------------------------------------------

# Función para simular una serie con errores AR(1)
def simulate_ar1_errors(n, rho, sigma):
    eps = np.random.normal(0, sigma, size=n)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = rho * u[t-1] + eps[t]
    return u

# Guardar resultados: multiindex (rho, N) -> arrays de betas
results = []

for rho in rhos:
    for N in Ns:
        betas_hat = np.zeros(B)
        for b in range(B):
            # Generar X (puede ser iid normal o también autocorrelado; usaremos iid normal)
            x = np.random.normal(0, 1, size=N)
            # Generar AR(1) errores
            u = simulate_ar1_errors(N, rho, sigma_eps)
            # Construir y
            y = beta0 + beta1 * x + u

            # Ajustar OLS (sklearn)
            model = LinearRegression(fit_intercept=True)
            model.fit(x.reshape(-1,1), y)
            betas_hat[b] = model.coef_[0]

        # Estadísticos Monte Carlo
        bias = betas_hat.mean() - beta1
        var = betas_hat.var(ddof=1)
        mse = np.mean((betas_hat - beta1)**2)
        se_emp = betas_hat.std(ddof=1)

        results.append({
            "rho": rho,
            "N": N,
            "bias": bias,
            "var": var,
            "se_emp": se_emp,
            "mse": mse,
            "mean_beta_hat": betas_hat.mean()
        })

# DataFrame con resultados resumen
df_res = pd.DataFrame(results)

# ---------- Gráficas: curvas de aprendizaje ----------
sns.set(style="whitegrid", palette="tab10")

fig, axes = plt.subplots(1, 3, figsize=(18,4), sharex=True)
for rho in rhos:
    df_plot = df_res[df_res["rho"] == rho].sort_values("N")
    axes[0].plot(df_plot["N"], df_plot["bias"], marker='o', label=f"rho={rho}")
    axes[1].plot(df_plot["N"], df_plot["var"], marker='o', label=f"rho={rho}")
    axes[2].plot(df_plot["N"], df_plot["mse"], marker='o', label=f"rho={rho}")

axes[0].axhline(0, color="k", linestyle="--", alpha=0.6)
axes[0].set_title("Bias (beta1_hat - beta1) vs N")
axes[0].set_xlabel("N")
axes[0].set_ylabel("Bias")
axes[1].set_title("Varianza (empírica) vs N")
axes[1].set_xlabel("N")
axes[1].set_ylabel("Var(beta1_hat)")
axes[2].set_title("MSE vs N")
axes[2].set_xlabel("N")
axes[2].set_ylabel("MSE")

axes[0].legend()
plt.tight_layout()
plt.show()

# ---------- Estabilidad: histogramas de betas para un N elegido ----------
N_show = Ns[-1]  # tamaño grande para ver estabilidad
fig, axs = plt.subplots(1, len(rhos), figsize=(16,3), sharey=True)
for i, rho in enumerate(rhos):
    # recalcular para mostrar distribuciones (puedes reducir B para velocidad)
    betas_hat = []
    for b in range(B):
        x = np.random.normal(0,1,size=N_show)
        u = simulate_ar1_errors(N_show, rho, sigma_eps)
        y = beta0 + beta1 * x + u
        model = LinearRegression().fit(x.reshape(-1,1), y)
        betas_hat.append(model.coef_[0])
    sns.histplot(betas_hat, bins=30, kde=True, ax=axs[i])
    axs[i].axvline(beta1, color="red", linestyle="--")
    axs[i].set_title(f"Distribución beta_hat (N={N_show}, rho={rho})")
    axs[i].set_xlabel("beta1_hat")
plt.tight_layout()
plt.show()

# ---------- Tablas resumen ----------
pd.set_option('display.float_format', '{:.6f}'.format)
display(df_res.pivot(index="N", columns="rho", values="bias"))
display(df_res.pivot(index="N", columns="rho", values="var"))
display(df_res.pivot(index="N", columns="rho", values="mse"))
```