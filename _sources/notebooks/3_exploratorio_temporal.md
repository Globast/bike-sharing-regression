---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---

# 3. Análisis exploratorio de series temporales

## 3.1. Analisis exploratorio de Series Temporales
Se realiza una exploracion sobre el promedio de bicis rentadas respecto a diferentes temporalidades
1. Bicis rentadas diariamente
2. Bicis promedio rentadaspor hora segun dia de la semana
3. Bicis rentadas promedio por hora del dia
4. Bicis rentadas promedio por hora por mes

```{code-cell} ipython3
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np

DATA_PATH = Path("../data/hour.csv")
data = pd.read_csv(DATA_PATH)
data["dteday"] = pd.to_datetime(data["dteday"])
data.shape

DATA_PATH_CLEAN = Path("../data/hour_clean.csv")
df = pd.read_csv(DATA_PATH_CLEAN)
df.shape


fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# PALETA DE COLORES
weekday_colors = plt.cm.Set3(np.linspace(0, 1, data["weekday"].nunique()))
month_colors   = plt.cm.Paired(np.linspace(0, 1, data["mnth"].nunique()))
season_colors  = plt.cm.Set2(np.linspace(0, 1, data["season"].nunique()))

# 1. Línea suavizada (7 días)
data.set_index("dteday")["cnt"].rolling(7).mean().plot(
    ax=axs[0,0], linewidth=2
)
axs[0,0].set_title("Cnt suavizado (7 días)")
axs[0,0].set_ylabel("Cnt")

# 2. Boxplot weekday vs cnt
bp_weekday = axs[0,1].boxplot(
    [data[data["weekday"]==w]["cnt"] for w in sorted(data["weekday"].unique())],
    patch_artist=True,
    tick_labels=sorted(data["weekday"].unique()) 
)


for patch, color in zip(bp_weekday['boxes'], weekday_colors):
    patch.set_facecolor(color)

axs[0,1].set_title("Distribución cnt por weekday")
axs[0,1].set_xlabel("weekday")
axs[0,1].set_ylabel("cnt")

# 3. Boxplot month vs cnt
bp_month = axs[1,0].boxplot(
    [data[data["mnth"]==m]["cnt"] for m in sorted(data["mnth"].unique())],
    patch_artist=True,
    tick_labels=sorted(data["mnth"].unique())  
)


for patch, color in zip(bp_month['boxes'], month_colors):
    patch.set_facecolor(color)

axs[1,0].set_title("Distribución cnt por mes")
axs[1,0].set_xlabel("mes")
axs[1,0].set_ylabel("cnt")

# 4. Boxplot season vs cnt
bp_season = axs[1,1].boxplot(
    [data[data["season"]==s]["cnt"] for s in sorted(data["season"].unique())],
    patch_artist=True,
    tick_labels=sorted(data["season"].unique())  
)


for patch, color in zip(bp_season['boxes'], season_colors):
    patch.set_facecolor(color)

axs[1,1].set_title("Distribución cnt por season")
axs[1,1].set_xlabel("season")
axs[1,1].set_ylabel("cnt")

plt.tight_layout()
plt.show()

```

1. **grafico diario suavizado (7 dias)**
    Este gráfico muestra la evolución del uso de bicicletas por parte de usuarios entre enero de 2011 y diciembre de 2012.
    Se aplicó un suavizado de 7 días para eliminar el ruido diario y resaltar la tendencia general.
    Los usuarios tienen consistentemente mayor volumen en estaciones particulares, los meses cálidos, lo que sugiere una fuerte influencia del clima en la demanda.
   
2. **Total de bicicletas por según día de la semana**
    Este gráfico de barras compara el total de bicicletas por día de la semana.
    Los usuarios muestran mayor actividad en todos los días, con una distribución relativamente estable.

3. **Total de bicicletas según mes del año**
    Aquí se observa el uso en cada mes del año.
    los usuarios incrementan su actividad en los meses cálidos (mayo a octubre).
    
4. **Total de biciletas por temporada del año**
    Los usuarios muestran nuevamnete en funcion del clima y de las estaciones dle año una preferencia por el uso de la bicileta cuando las condiciones ambientales son agradables.

## 3.2. Influencia de factores ambientales
Para entender el efecto de las diferentes variables ambientales se traza un grafico de dispersion de la temperatura, humedad y velocidad del viento vs el promedio de bicis rentadas, esto con el fin de encontrar relaciones o influencias de las variables en la variable objetivo.

```{code-cell} ipython3
## Graficos de dispersion 

fig, axs = plt.subplots(1, 3, figsize=(18, 5)) ## ajustar a 1 fila 3 columnas de graficos

#--------------------------
sns.scatterplot(x=data["temp"], y=data["cnt"], ax=axs[0])
axs[0].set_title("Dispersión: Temp vs cnt")
#--------------------------
sns.scatterplot(x=data["hum"], y=data["cnt"], ax=axs[1])
axs[1].set_title("Dispersión: Humidity vs cnt")
#--------------------------
sns.scatterplot(x=data["windspeed"], y=data["cnt"], ax=axs[2])
axs[2].set_title("Dispersión: Windvelocity vs cnt")
#--------------------------
plt.tight_layout()
plt.show()
```

```{code-cell} ipython3
plt.figure(figsize=(8,6))
sns.heatmap(
    data[["temp","hum","windspeed","cnt"]].corr(),
    annot=True, cmap="coolwarm"
)
plt.title("Mapa de calor de correlaciones")
plt.show()
```

Notemos que la influencia de la temperatura es positiva sobre la cantidad de bicis rentadas, no es la razon principal de su demanada pero si explica en parte la tendencia a que cuando hace buen clima, las personas prefieran este medio de transporte.
A su vez, la humedad muestra disminuir las rentas de bicis a medida esta aumenta, indicador de que si el clima se torna con mucha humedad, existira un efecto a la baja en la renta de bicicletas. 
Por otro lado, la velocidad del viento, no denota una relacion lineal clara a la renta de bicicletas, esto puede denotar corrientes de aire moderadas dentro de la ciudad que no afectan a la utilizacion de este medio de transporte o variaciones a lo largo del dia.

