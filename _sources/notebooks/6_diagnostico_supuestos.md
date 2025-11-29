---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---
# 6. Diagnóstico de supuestos del modelo
En esta seccion, a partir del modelo anterior, se revisara elñ cumplimiento de la teoria sobre los errores, para entender si el modelo es valido o no

### 6.1. Revision de residuos (Normalidad, Homocedasticidad e independencia)

Se parte de la inspeccion visual de residuos
- Disribucion de residuos
- Q-Q Plot de residuos
- Densiad de residuales

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

N = len(df)
cut = int(N * 0.7)

train = df.iloc[:cut].copy()
test  = df.iloc[cut:].copy()
```
