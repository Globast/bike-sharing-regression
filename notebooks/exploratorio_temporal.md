# 3. Análisis exploratorio de series temporales

En esta sección realizamos un análisis exploratorio básico de la serie temporal de demanda horaria.

## 3.1 Evolución temporal del conteo total

```{code-cell} python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/hour.csv")
df["dteday"] = pd.to_datetime(df["dteday"])
df = df.sort_values(["dteday", "hr"]).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(df["dteday"] + pd.to_timedelta(df["hr"], unit="h"), df["cnt"], linewidth=0.5)
ax.set_xlabel("Fecha-hora")
ax.set_ylabel("Conteo total de bicicletas")
ax.set_title("Serie temporal de demanda horaria")

# Guardar la figura antes de mostrarla
plt.tight_layout()
plt.savefig("../data/ts_cnt_overview.png")
plt.show()
```

```{figure} ../data/ts_cnt_overview.png
:name: fig:ts_cnt_overview

Serie temporal completa de demanda horaria de bicicletas.
```

## 3.2 Distribución de la demanda por hora del día

```{code-cell} python
import seaborn as sns

fig, ax = plt.subplots(figsize=(8, 4))
sns.boxplot(x="hr", y="cnt", data=df, ax=ax)
ax.set_xlabel("Hora del día")
ax.set_ylabel("Conteo total de bicicletas")
ax.set_title("Distribución de la demanda por hora del día")

plt.tight_layout()
plt.savefig("../data/boxplot_cnt_hour.png")
plt.show()
```

```{figure} ../data/boxplot_cnt_hour.png
:name: fig:boxplot_cnt_hour

Distribución del conteo horario de bicicletas por hora del día.
```

Estas figuras, {numref}`fig:ts_cnt_overview` y {numref}`fig:boxplot_cnt_hour`, motivan la necesidad de capturar patrones diarios y efectos de hora en el modelo lineal.
