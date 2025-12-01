---
title: "Introducción y contexto de movilidad urbana"
layout: page
---

# 1. Introducción y contexto de movilidad urbana

## 1.1 Contexto del problema

En las últimas décadas, muchas ciudades han recurrido a la bicicleta como pieza clave de sus estrategias de movilidad sostenible, tanto por su bajo impacto ambiental como por sus beneficios en salud y eficiencia del espacio urbano [@eren2020_review_bikesharing](10_referencias.md). Los sistemas de bicicletas compartidas y el aumento del uso cotidiano de la bicicleta se han consolidado como alternativas reales al automóvil privado, especialmente para viajes cortos e intermodales.

### 1.1.1 Reducción de CO₂

La evidencia empírica en ciudades europeas muestra que sustituir desplazamientos en automóvil por viajes en bicicleta puede reducir de manera significativa las emisiones de CO₂ asociadas a la movilidad diaria. Estudios de ciclo de vida encuentran reducciones cercanas a dos tercios de las emisiones individuales cuando se cambia sistemáticamente del coche a modos activos como la bicicleta [@brand2021_active_travel](10_referencias.md). A escala global, una expansión masiva del uso de la bicicleta podría recortar alrededor de un 10 % las emisiones de transporte hacia 2050 [@mason2015_high_shift_cycling](10_referencias.md).

Bogotá se ha posicionado históricamente como “capital de la bicicleta” en América Latina gracias a políticas pioneras como la Ciclovía dominical, los días sin carro y sin moto, y la expansión continua de su red de ciclorrutas. La ciudad supera los 630 km de infraestructura ciclable segregada, una de las redes más extensas de la región [@alcaldia2023_ciclorrutas_bogota; @tmc2022_cycling_infrastructure_bogota](10_referencias.md).

### 1.1.2 Retos

Desde 2022, Bogotá cuenta con un sistema público de bicicletas compartidas en concesión con un operador privado. En pocos años el sistema superó los tres millones de viajes, mostrando que existe una demanda real por este tipo de servicio [@larepublica2024_tembici_bogota](10_referencias.md). Sin embargo, distintos análisis periodísticos y técnicos han señalado retos importantes: concentración espacial en áreas de renta media y alta, dificultades financieras del operador y baja participación relativa del sistema en el total de viajes urbanos [@eltiempo2022_retos_bicicletas_compartidas; @elpais2025_modelo_bicis_bogota](10_referencias.md).

Estos problemas muestran la necesidad de alinear mejor la oferta con la demanda efectiva y potencial. Un dimensionamiento inadecuado de la red (número y localización de estaciones, tamaños de flota, tarifas, integración tarifaria) puede traducirse en estaciones vacías o saturadas, baja rotación de bicicletas y, finalmente, en la percepción de un servicio poco confiable.

La literatura internacional sobre sistemas de bicicletas compartidas enfatiza que la demanda está fuertemente condicionada por factores meteorológicos, estacionales, de entorno construido, características socioeconómicas y condiciones de seguridad vial [@eren2020_review_bikesharing; @wang2021_spatial_regression_bikesharing](10_referencias.md). En consecuencia, disponer de modelos cuantitativos que capten la relación entre estas variables y el número de viajes en bicicleta resulta esencial para rediseñar y hacer sostenible el sistema bogotano.

### 1.1.3. Evaluación de políticas públicas de movilidad sostenible

La capacidad de predecir la demanda de bicicletas también es fundamental para evaluar ex ante y ex post distintas políticas públicas de movilidad sostenible, tales como:

- Nuevas jornadas de Día sin carro y sin moto, o ampliaciones de horarios de Ciclovía, analizando cómo podrían redistribuirse los flujos de viaje hacia la bicicleta [@elpais2025_dia_sin_carro](10_referencias.md).
- Cambios en normas de velocidad máxima, pacificación de tráfico o implementación de zonas de bajas emisiones, estimando el impacto sobre la elección modal.
- Programas de incentivos económicos (subsidios a bicicletas eléctricas, estacionamientos gratuitos, incentivos empresariales) o de educación vial.

Modelos de demanda calibrados con datos locales de Bogotá permiten simular estos escenarios y cuantificar beneficios ambientales (reducción de emisiones y ruido), de salud (mayor actividad física) y de tiempo de viaje. Los resultados pueden compararse con la evidencia internacional, que muestra que aumentar el uso de modos activos tiene un efecto significativo en la reducción de emisiones y costes asociados al cambio climático [@brand2021_active_travel; @mason2015_high_shift_cycling](10_referencias.md).

Contextualizar el estudio de predicción de demanda de bicicletas en Bogotá implica reconocer que, más allá de su valor académico, los modelos desarrollados pueden alimentar decisiones de planeación estratégica, diseño de infraestructura, operación de sistemas de micromovilidad y evaluación de políticas públicas. Dada la escala actual de la red ciclable y la ambición de la ciudad en materia de acción climática, contar con herramientas cuantitativas robustas para anticipar la demanda de bicicletas es un componente imprescindible de la gestión de la movilidad sostenible bogotana.

## 1.2. Valor como sensor Urbano
En el proyecto de regresión para bike sharing, el análisis exploratorio muestra  que la demanda de bicicletas responde de forma sistemática a patrones de clima, estacionalidad y calendario permitiendo entender en qué contextos las personas están dispuestas a sustituir el automóvil por la bicicleta 
La ingeniería de variables incoincorpora  transformaciones horarias y estacionales, así como dummies de clima y tipo de día, con el fin de capturar con precisión las condiciones bajo las cuales se generan más viajes en bicicleta. Sobre esta base, el modelo lineal estimado por MCO cuantifica el efecto marginal de temperatura, humedad, velocidad del viento, estaciones y festivos sobre el número de préstamos, proporcionando una lectura económica y urbana de los determinantes de la demanda ciclista. En conjunto, este andamiaje de datos y modelos ofrece valor como sensor urbano al dar a las autoridades herramientas concretas para formular políticas de movilidad que reduzcan emisiones (al favorecer el trasvase modal hacia la bicicleta), mejoren la eficiencia energética del sistema de transporte y permitan focalizar inversiones e intervenciones (infraestructura ciclista, integración con transporte público, gestión de la oferta de bicicletas) allí donde el potencial de mitigación climática y ahorro energético es mayor.

## 1.3. Objetivos
### 1.3.1. Objetivo general
Desarrollar y validar un modelo de regresión para la predicción de la demanda de bicicletas compartidas en Bogotá, a partir de variables climáticas, temporales y de calendario, que sea interpretable para los tomadores de decisión y sirva como insumo para diseñar políticas de movilidad sostenible orientadas a la reducción de emisiones y a la mejora de la eficiencia energética del sistema de transporte.

### 1.3.2. Objetivos específicos
1. Construir y depurar una base de datos integrada que combine información de uso del sistema de bicicletas compartidas con variables meteorológicas, temporales y de calendario relevantes para la ciudad de Bogotá, asegurando su calidad para el análisis estadístico.

2. Formular, estimar y seleccionar modelos de regresión para la predicción de la demanda de bicicletas, priorizando especificaciones que sean estadísticamente sólidas y al mismo tiempo interpretables para los tomadores de decisión, mediante el análisis detallado de coeficientes, efectos marginales y visualizaciones explicativas.

3. Validar rigurosamente el desempeño y la robustez del modelo seleccionado mediante técnicas de validación fuera de muestra, métodos robustos para datos temporales y ejercicios de simulación, cuantificando la incertidumbre de las predicciones y verificando la estabilidad de los resultados bajo diferentes configuraciones de datos.

## 1.4. Preguntas de investigación

¿En qué medida las variables climáticas (temperatura, precipitación, humedad, velocidad del viento) y las variables temporales y de calendario (hora del día, día de la semana, temporada, festivos, eventos especiales) explican la variación en la demanda de bicicletas compartidas en Bogotá?

¿Qué modelo de regresión ofrece el mejor equilibrio entre capacidad predictiva, interpretabilidad de los coeficientes y estabilidad de los resultados al predecir la demanda de bicicletas compartidas en Bogotá?

¿Cómo pueden utilizarse las predicciones del modelo y la interpretación de sus parámetros para orientar decisiones de política pública en movilidad sostenible, particularmente en la reducción de emisiones asociadas al transporte y en la mejora de la eficiencia energética del sistema de movilidad bogotano?

## 1.5. Justificación
Técnicamente, el estudio se justifica porque permite desarrollar y evaluar modelos cuantitativos de predicción de demanda en un contexto real de ciudad lainoamericana, con datos horarios y temporales que presentan autocorrelación, estacionalidad y posibles valores atípicos. El uso de diferentes especificaciones de regresión (clásicas, penalizadas y robustas) y de esquemas de validación fuera de muestra aporta evidencia sobre qué enfoques son más adecuados para datos de movilidad con estructura temporal. 
Socialmente, entender qué condiciones fomentan o inhiben el uso de la bicicleta permite diseñar políticas e infraestructuras que reduzcan barreras de acceso  y promuevan una movilidad más equitativa. Un modelo robusto de demanda ayuda a focalizar intervenciones donde más beneficien a grupos vulnerables, a mejorar la confiabilidad del sistema de bicicletas compartidas y, en consecuencia, a consolidar la bicicleta como opción real de movilidad cotidiana.

Desde el punto de vista ambiental, promover el uso de la bicicleta y de sistemas de micromovilidad implica la posibilidad de sustituir viajes en vehículo motorizado por un modo de muy baja huella de carbono. Para que esa sustitución sea efectiva, es necesario planear de forma estratégica la oferta de bicicletas, la infraestructura ciclable y la integración con el transporte público. El estudio aporta un instrumento para estimar cuánta demanda puede generarse bajo distintas condiciones y escenarios ayudando a cuantificar el potencial de reducción de emisiones y de mejora en la eficiencia energética del sistema de movilidad. De este modo, el modelo no solo describe el comportamiento actual, sino que se convierte en base técnica para diseñar políticas de movilidad sostenible coherentes con las metas climáticas y de calidad del aire de la ciudad.