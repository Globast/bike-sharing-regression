# 11. Conclusiones y extensiones futuras

Este capítulo resume los hallazgos principales del análisis y sugiere líneas de trabajo adicionales.

## 11.1 Conclusiones principales

- La demanda horaria de bicicletas está fuertemente condicionada por la hora del día y el calendario laboral.
- Las variables climáticas aportan capacidad explicativa adicional y permiten mejorar las predicciones.
- La validación cruzada temporal (TimeSeriesSplit) ofrece una estimación más realista del desempeño predictivo que una partición aleatoria.

## 11.2 Extensiones futuras

Entre las posibles extensiones se encuentran:

- Modelos con estructuras ARIMA o modelos mixtos de efectos aleatorios [@boxjenkins2016].
- Integración de información espacial (estaciones) para construir modelos espaciotemporales.
- Uso de técnicas de aprendizaje automático más complejas (por ejemplo, *gradient boosting* o redes neuronales), siempre acompañadas de un protocolo de validación temporal riguroso.
