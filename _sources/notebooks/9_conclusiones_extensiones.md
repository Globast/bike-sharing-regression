# 9. Conclusiones y extensiones futuras

## 9.1. Respuestas a las tres preguntas de investigación
 ¿en qué medida las variables climáticas y temporales explican la demanda?
El modelo lineal estimado con las 10 variables seleccionadas (año, temperatura, humedad, velocidad del viento, estaciones, estado del clima y festivos) muestra que el clima y la temporalidad explican una fracción importante, pero no total, de la variabilidad en los préstamos de bicicletas. En el conjunto de entrenamiento se obtiene un R² ≈ 0,82, mientras que en el conjunto de prueba el R² cae a ≈ 0,56, es decir una capacidad explicativa moderada en datos no vistos. La señal de los coeficientes es coherente con la teoría: la temperatura tiene un efecto positivo fuerte; la humedad y la velocidad del viento reducen la demanda; las estaciones “más cálidas” aumentan los préstamos; y las categorías de clima desfavorable (lluvia, nubes densas) disminuyen significativamente el uso del sistema. 

¿qué especificación ofrece mejor equilibrio entre capacidad predictiva, interpretabilidad y estabilidad?
La comparación entre Ridge, Lasso, Elastic Net y OLS, muestra desempeños prácticamente idénticos. Ni Lasso ni Elastic Net llevan coeficientes a cero ni muestran mejoras sustanciales frente a Ridge u OLS, sugiriendo la multicolinealidad no es problemática y que todas las variables seleccionadas aportan información útil. 
¿cómo pueden usarse las predicciones y parámetros para orientar políticas de movilidad sostenible?
Los coeficientes estimados entregan una lectura cuantitativa de cómo responden los préstamos de bicicletas ante cambios en clima y calendario: por ejemplo, cuánto cae la demanda ante un empeoramiento del clima (pasar de despejado a lluvia ligera), cuánto aumenta cuando se pasa de una estación fría a una más cálida, o el efecto de días festivos frente a laborales. 
 Estas relaciones pueden trasladarse a decisiones concretas de política: dimensionar flota y estaciones en temporadas de alta demanda; definir estrategias de comunicación e incentivos en días con clima favorable; anticipar caídas de uso por eventos climáticos adversos y compensarlas con otras medidas; o estimar el potencial de sustitución de viajes motorizados por bicicleta en determinados escenarios. 
 ## 9.2.Limitaciones del modelo

El modelo se entrena con el “Bike Sharing Dataset” **agregado a nivel diario**, en un contexto urbano específico (2011–2012) que no corresponde exactamente a Bogotá. Esto implica que los coeficientes capturan patrones de una ciudad de referencia y solo se “trasladan conceptualmente” a Bogotá; para decisiones operativas finas en la ciudad sería imprescindible recalibrar el modelo con datos locales. Además, al trabajar con datos agregados por día, se pierde información horaria limitando el detalle con el que pueden formularse decisiones de gestión.

c) Supuestos del modelo lineal y comportamiento de los errores.
Aun cuando se realizan diagnósticos y se aplican métodos robustos (errores estándar HAC Newey–West, estimadores M de Huber/Tukey, etc.) para tratar heterocedasticidad y autocorrelación, los datos de demanda diaria siguen siendo una serie temporal potencialmente afectada por shocks, tendencias y patrones de dependencia más complejos. 
 La simulación Monte Carlo muestra que la presencia de autocorrelación en los errores (procesos AR(1) con distintos rhos) afecta el sesgo, la varianza y el MSE de los estimadores, aunque OLS resulte “razonablemente robusto” en muchos escenarios. Esto indica que, en la práctica, las inferencias deben tomarse con cautela: los intervalos de confianza y pruebas de significancia pueden no ser exactos si la estructura de dependencia real es más compleja de lo que el modelo y las correcciones capturan.

d) Desempeño predictivo y errores relativos.
Aunque el R² de prueba es aceptable para fenómenos de demanda con alta variabilidad, las métricas de error relativo (MAPE) en el conjunto de prueba son elevadas (del orden de 90 %), debido en parte a la presencia de días con conteos bajos en los que pequeños errores absolutos se traducen en grandes errores porcentuales. 
 Esto limita el uso del modelo para predicciones puntuales muy precisas en días específicos; su mayor valor está en capturar tendencias y efectos medios, más que en pronosticar con exactitud el número de viajes de un día concreto.

 ## 9.3. Extensiones Futuras
Si se dispone de datos por estación o por zona de Bogotá, se podrían usar modelos de panel espacio–temporales (efectos fijos/aleatorios por zona + estructura temporal ARMA/ARIMA) o modelos de regresión con efectos espaciales (SAR, CAR). Esto permitiría capturar dependencia entre estaciones cercanas y diferencias estructurales entre barrios.
Modelos aditivos generalizados (GAM): Mantienen interpretabilidad, pero permiten no linealidades suaves (splines de temperatura, humedad, hora del día). Así podrías capturar zonas de confort térmico, umbrales de precipitación a partir de los cuales la demanda cae abruptamente, etc.
Redes recurrentes (RNN, LSTM, GRU): Para series temporales de demanda por estación o por zona. Estas redes captan dependencias de largo plazo, estacionalidades complejas y efectos de eventos. Se pueden usar arquitecturas “many-to-one” (historial → demanda futura) u “encoder–decoder” para producir múltiples pasos de pronóstico.

