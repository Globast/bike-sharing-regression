
# 1. Introducción y contexto de movilidad urbana

## 1.1 Contexto del problema

En las últimas décadas, muchas ciudades han recurrido a la bicicleta como pieza clave de sus estrategias de movilidad sostenible, tanto por su bajo impacto ambiental como por sus beneficios en salud y eficiencia del espacio urbano {cite}`eren2020_review_bikesharing`. Los sistemas de bicicletas compartidas y el aumento del uso cotidiano de la bicicleta se han consolidado como alternativas reales al automóvil privado, especialmente para viajes cortos e intermodales.

### 1.1.1 Reducción de CO₂

La evidencia empírica en ciudades europeas muestra que sustituir desplazamientos en automóvil por viajes en bicicleta puede reducir de manera significativa las emisiones de CO₂ asociadas a la movilidad diaria. Estudios de ciclo de vida encuentran reducciones cercanas a dos tercios de las emisiones individuales cuando se cambia sistemáticamente del coche a modos activos como la bicicleta [@brand2021_active_travel]. A escala global, una expansión masiva del uso de la bicicleta podría recortar alrededor de un 10 % las emisiones de transporte hacia 2050 [@mason2015_high_shift_cycling].

Bogotá se ha posicionado históricamente como “capital de la bicicleta” en América Latina gracias a políticas pioneras como la Ciclovía dominical, los días sin carro y sin moto, y la expansión continua de su red de ciclorrutas. La ciudad supera los 630 km de infraestructura ciclable segregada, una de las redes más extensas de la región [@alcaldia2023_ciclorrutas_bogota; @tmc2022_cycling_infrastructure_bogota].

## 1.2 Retos

Desde 2022, Bogotá cuenta con un sistema público de bicicletas compartidas en concesión con un operador privado. En pocos años el sistema superó los tres millones de viajes, mostrando que existe una demanda real por este tipo de servicio [@larepublica2024_tembici_bogota]. Sin embargo, distintos análisis periodísticos y técnicos han señalado retos importantes: concentración espacial en áreas de renta media y alta, dificultades financieras del operador y baja participación relativa del sistema en el total de viajes urbanos [@eltiempo2022_retos_bicicletas_compartidas; @elpais2025_modelo_bicis_bogota].

Estos problemas muestran la necesidad de alinear mejor la oferta con la demanda efectiva y potencial. Un dimensionamiento inadecuado de la red (número y localización de estaciones, tamaños de flota, tarifas, integración tarifaria) puede traducirse en estaciones vacías o saturadas, baja rotación de bicicletas y, finalmente, en la percepción de un servicio poco confiable.

La literatura internacional sobre sistemas de bicicletas compartidas enfatiza que la demanda está fuertemente condicionada por factores meteorológicos, estacionales, de entorno construido, características socioeconómicas y condiciones de seguridad vial [@eren2020_review_bikesharing; @wang2021_spatial_regression_bikesharing]. En consecuencia, disponer de modelos cuantitativos que capten la relación entre estas variables y el número de viajes en bicicleta resulta esencial para rediseñar y hacer sostenible el sistema bogotano.

## 1.3 Evaluación de políticas públicas de movilidad sostenible

La capacidad de predecir la demanda de bicicletas también es fundamental para evaluar ex ante y ex post distintas políticas públicas de movilidad sostenible, tales como:

- Nuevas jornadas de Día sin carro y sin moto, o ampliaciones de horarios de Ciclovía, analizando cómo podrían redistribuirse los flujos de viaje hacia la bicicleta [@elpais2025_dia_sin_carro].
- Cambios en normas de velocidad máxima, pacificación de tráfico o implementación de zonas de bajas emisiones, estimando el impacto sobre la elección modal.
- Programas de incentivos económicos (subsidios a bicicletas eléctricas, estacionamientos gratuitos, incentivos empresariales) o de educación vial.

Modelos de demanda calibrados con datos locales de Bogotá permiten simular estos escenarios y cuantificar beneficios ambientales (reducción de emisiones y ruido), de salud (mayor actividad física) y de tiempo de viaje. Los resultados pueden compararse con la evidencia internacional, que muestra que aumentar el uso de modos activos tiene un efecto significativo en la reducción de emisiones y costes asociados al cambio climático [@brand2021_active_travel; @mason2015_high_shift_cycling].

Contextualizar el estudio de predicción de demanda de bicicletas en Bogotá implica reconocer que, más allá de su valor académico, los modelos desarrollados pueden alimentar decisiones de planeación estratégica, diseño de infraestructura, operación de sistemas de micromovilidad y evaluación de políticas públicas. Dada la escala actual de la red ciclable y la ambición de la ciudad en materia de acción climática, contar con herramientas cuantitativas robustas para anticipar la demanda de bicicletas es un componente imprescindible de la gestión de la movilidad sostenible bogotana.




## 1.2 Descripción del conjunto de datos

El archivo `data/hour.csv` contiene observaciones horarias con información sobre:

- Fecha y hora (`dteday`, `hr`).
- Condiciones climáticas (`temp`, `atemp`, `hum`, `windspeed`, `weathersit`).
- Variables de calendario (`season`, `yr`, `mnth`, `holiday`, `weekday`, `workingday`).
- Conteos de bicicletas: usuarios registrados, casuales y totales (`registered`, `casual`, `cnt`).

```{code-cell} python
import numpy as np
import pandas as pd

# Semilla global para reproducibilidad en todo el libro
np.random.seed(42)

data_path = "data/hour.csv"
df = pd.read_csv(data_path)

df.head()
```

## 1.3 Objetivo general

Nuestro objetivo general es ajustar y analizar un modelo de regresión que explique el conteo total de bicicletas (`cnt`) a partir de variables temporales y climáticas, respetando la estructura de serie temporal de los datos y utilizando técnicas de validación cruzada temporal descritas en capítulos posteriores.
