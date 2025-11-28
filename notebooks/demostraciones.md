# 0.2 Demostraciones solicitadas

Este libro está diseñado para servir como soporte a demostraciones prácticas de:

- Preparación y limpieza de datos horarios de movilidad urbana.
- Construcción de variables temporales (tendencias, estacionalidad, lags).
- Ajuste de un modelo de regresión lineal múltiple mediante la formulación matricial de MCO.
- Evaluación de supuestos clásicos del modelo lineal en presencia de dependencia temporal.
- Uso de errores estándar robustos para series temporales.
- Validación cruzada temporal mediante particiones `TimeSeriesSplit`.
- Simulación Monte Carlo para estudiar la estabilidad y el sesgo de los estimadores.

Los capítulos posteriores contienen ejemplos de código reproducibles y referencias cruzadas a figuras, tablas y ecuaciones. Por ejemplo, la estimación matricial se basa en la ecuación {eq}`eq:ols_matrix`, mientras que la validación cruzada temporal se implementa en el capítulo 8 utilizando `TimeSeriesSplit` de `scikit-learn` [@scikit-learn].
