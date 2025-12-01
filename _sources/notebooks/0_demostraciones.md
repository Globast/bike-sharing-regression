---
jupytext:
  formats: md:myst
  text_representation: {extension: .md, format_name: myst}
kernelspec: {name: python3, display_name: Python 3}
---


# 0 Demostraciones Solicitadas.

## Modelo de regresión lineal múltiple

Consideremos el modelo:

$$
y = X\beta + \epsilon
$$

donde:

- $y $ es un vector $n \times 1 $ de observaciones  
- $X $ es una matriz $n \times p $ de rango completo (por lo tanto, $X^\top X $ es invertible)  
- $\beta $ es un vector $p \times 1 $ de parámetros desconocidos  
- $\epsilon $ es un vector $n \times 1 $ de errores con $\mathbb{E}(\epsilon) = 0 $ y $\text{Var}(\epsilon) = \sigma^2 I_n $

El estimador de mínimos cuadrados de $\beta $ es:

$$
\hat{\beta} = (X^\top X)^{-1} X^\top y
$$

El vector de residuos se define como:

$$
e = y - X\hat{\beta}
$$

Definimos:

- **Suma de cuadrados de los residuos**:

$$
SS_{Res} = e^\top e = (y - X\hat{\beta})^\top (y - X\hat{\beta})
$$

- **Cuadrado medio de los residuos**:

$$
MS_{Res} = \frac{SS_{Res}}{n - p}
$$


En el contexto del modelo de regresión lineal múltiple $Y = X\beta + \epsilon $, donde $\epsilon \sim N(0, \sigma^2 I) $

## 0.1. Grados de libertad de $SS_{Res}$

### **Idea Principal**

- Las predicciones $X\hat{\beta}$ viven en el subespacio columna de $X$, de dimensión $p$.
- Los residuos $e = y - X\hat{\beta}$ son la proyección ortogonal de $y$ sobre el complemento de ese subespacio.
- Por tanto, $e$ vive en un espacio de dimensión $n - p$, lo que determina los grados de libertad de $SS_{Res} = e^\top e$.


### Desarrollemos

- Definimos la matriz de proyección:

$$
H = X(X^\top X)^{-1}X^\top
$$

- Entonces:

$$
\hat{y} = Hy, \quad e = y - \hat{y} = (I - H)y
$$

- Sea $M = I - H$, entonces:

$$
SS_{Res} = e^\top e = y^\top M^\top M y = y^\top M y
$$

- Propiedades de $M$:

$$
M^\top = M, \quad M^2 = M, \quad \text{rango}(M) = n - p
$$

### Bajo normalidad

Si $ \epsilon \sim N(0, \sigma^2 I_n) $, entonces:

$$
y = X\beta + \epsilon \quad \Rightarrow \quad e = My = M\epsilon
$$

Como $M$ es un proyector de rango $n - p$:

$$
\frac{SS_{Res}}{\sigma^2} = \frac{\epsilon^\top M \epsilon}{\sigma^2} \sim \chi^2_{n - p}
$$


### Concluimos 

- $SS_{Res}$ tiene $n - p$ grados de libertad porque los residuos viven en un subespacio de dimensión $n - p$.
- Bajo normalidad, esta cantidad sigue una distribución $\chi^2_{n - p}$, confirmando los grados de libertad.

## 0.2. $MS_{Res}$ como estimador insesgado de $\sigma^2$

- **Objetivo:** Mostrar que $MS_{Res} = \dfrac{SS_{Res}}{n - p}$ estima en promedio el verdadero $\sigma^2$.


### Desarrollemos

- **Matriz de proyección y residuos:**

$$
H = X(X^\top X)^{-1}X^\top, \quad M = I - H, \quad e = My.
$$

Dado $y = X\beta + \epsilon$ y $MX\beta = 0$,

$$
e = M\epsilon.
$$

- **Suma de cuadrados de residuos como forma cuadrática:**

$$
SS_{Res} = e^\top e = \epsilon^\top M \epsilon.
$$

- **Esperanza de una forma cuadrática con ruido homocedástico:**

Si $\mathbb{E}(\epsilon) = 0$ y $\operatorname{Var}(\epsilon) = \sigma^2 I_n$,

$$
\mathbb{E}(SS_{Res}) = \mathbb{E}(\epsilon^\top M \epsilon)
= \operatorname{tr}\!\left(M\,\mathbb{E}(\epsilon\epsilon^\top)\right)
= \operatorname{tr}(M\,\sigma^2 I_n)
= \sigma^2 \operatorname{tr}(M).
$$

Como $M$ es simétrica, idempotente y $\operatorname{rank}(M) = n - p$,

$$
\operatorname{tr}(M) = n - p \quad \Rightarrow \quad \mathbb{E}(SS_{Res}) = (n - p)\sigma^2.
$$

- **Insesgadez de $MS_{Res}$:**

$$
MS_{Res} = \frac{SS_{Res}}{n - p}
\quad \Rightarrow \quad
\mathbb{E}(MS_{Res}) = \frac{\mathbb{E}(SS_{Res})}{n - p} = \sigma^2.
$$

### Concluimos

- **Resultado:** $MS_{Res}$ es un estimador insesgado de $\sigma^2$.
- **Razón:** La esperanza de $SS_{Res}$ es $(n - p)\sigma^2$ porque proyecta $\epsilon$ sobre un subespacio de dimensión $n - p$; al dividir por $n - p$, se obtiene exactamente $\sigma^2$.

## 0.3. Valor esperado del cuadrado medio de regresión

### Idea principal

- El cuadrado medio de regresión $MSR$ mide cuánta variabilidad explica el modelo.
- Su valor esperado tiene dos componentes:
  - La **variabilidad aleatoria** del error: $\sigma^2$
  - La **variabilidad explicada** por el modelo: depende de $\beta^*$ y de la estructura de $X_c$

### Desarrollemos

- Sea $X_c$ la matriz de diseño centrada (sin la constante), de dimensión $n \times k$, y $\beta^*$ el vector de coeficientes asociado.
- El modelo es:

$$
y = X_c \beta^* + \epsilon, \quad \epsilon \sim N(0, \sigma^2 I)
$$

- La suma de cuadrados de regresión es:

$$
SS_{Reg} = \hat{y}^\top \hat{y} - n\bar{y}^2
$$

Pero en modelos centrados (sin intercepto), se usa:

$$
SS_{Reg} = \hat{y}^\top \hat{y} = y^\top H y
\quad \text{donde} \quad H = X_c (X_c^\top X_c)^{-1} X_c^\top
$$

- Entonces:

$$
MSR = \frac{SS_{Reg}}{k} = \frac{y^\top H y}{k}
$$

- Como $y = X_c \beta^* + \epsilon$, sustituimos:

$$
y^\top H y = (X_c \beta^* + \epsilon)^\top H (X_c \beta^* + \epsilon)
$$

Expandimos:

$$
= \beta^{*\top} X_c^\top H X_c \beta^* 
+ 2 \beta^{*\top} X_c^\top H \epsilon 
+ \epsilon^\top H \epsilon
$$

Usamos propiedades de $H$:

- $H X_c = X_c$, porque $H$ proyecta sobre $\mathcal{C}(X_c)$
- $H$ es simétrica e idempotente
- $\mathbb{E}(\epsilon) = 0$, $\mathbb{E}(\epsilon \epsilon^\top) = \sigma^2 I$

Entonces:

- $\mathbb{E}(2 \beta^{*\top} X_c^\top H \epsilon) = 0$
- $\mathbb{E}(\epsilon^\top H \epsilon) = \sigma^2 \operatorname{tr}(H) = \sigma^2 k$

Y:

$$
\mathbb{E}(y^\top H y) = \beta^{*\top} X_c^\top X_c \beta^* + \sigma^2 k
$$

Por tanto:

$$
\mathbb{E}(MSR) = \frac{1}{k} \mathbb{E}(y^\top H y) 
= \frac{1}{k} \left( \beta^{*\top} X_c^\top X_c \beta^* + \sigma^2 k \right)
= \sigma^2 + \frac{1}{k} \beta^{*\top} X_c^\top X_c \beta^*
$$


### Concluimos

El cuadrado medio de regresión $MSR$ tiene como valor esperado la suma de:
- La varianza del error $\sigma^2$
- Una componente que refleja la magnitud del efecto de los regresores: $\frac{1}{k} \beta^{*\top} X_c^\top X_c \beta^*$

## 0.4. Distribución no central de $F_0$

### Planteamiento y notación

- **Modelo con intercepto:** $y = \alpha \mathbf{1} + X_c \beta^* + \epsilon$, con $\epsilon \sim N(0, \sigma^2 I_n)$, y $X_c$ centrada de dimensión $n \times k$.
- **Proyectores:**
  - **Modelo completo:** $H = X(X^\top X)^{-1}X^\top$, donde $X = [\mathbf{1}, X_c]$ tiene rango $k+1$.
  - **Modelo nulo (solo intercepto):** $H_0 = \dfrac{1}{n}\mathbf{1}\mathbf{1}^\top$.
  - **Residuos:** $M = I - H$.
- **Sumas de cuadrados:**
  - **Regresión:** $SS_{Reg} = y^\top (H - H_0) y$, de rango $k$.
  - **Residuos:** $SS_{Res} = y^\top M y$, de rango $n - k - 1$.
- **Cuadrados medios:** $MSR = SS_{Reg}/k$, $MS_{Res} = SS_{Res}/(n - k - 1)$.

### Distribución de $SS_{Reg}$ y del parámetro de no centralidad

- **Descomposición de $y$:** $y = \alpha \mathbf{1} + X_c \beta^* + \epsilon$.
- **Anulación del intercepto en $SS_{Reg}$:** $(H - H_0)\mathbf{1} = 0$, de modo que
  $$
  SS_{Reg} = y^\top (H - H_0) y
  = (X_c \beta^* + \epsilon)^\top (H - H_0) (X_c \beta^* + \epsilon).
  $$
- **Propiedad clave:** $(H - H_0) X_c = X_c$ y $(H - H_0)$ es simétrica e idempotente con $\operatorname{rank}(H - H_0) = k$.
- **Forma cuadrática gaussiana no central:**
  $$
  \frac{SS_{Reg}}{\sigma^2}
  = \frac{(X_c \beta^* + \epsilon)^\top (H - H_0) (X_c \beta^* + \epsilon)}{\sigma^2}
  \sim \chi^2_k\!\left(\lambda\right),
  $$
  con
  $$
  \lambda = \frac{1}{\sigma^2}\beta^{*\top} X_c^\top X_c \beta^*.
  $$

### Distribución de $SS_{Res}$ e independencia

- **Residuos:** $SS_{Res} = y^\top M y = \epsilon^\top M \epsilon$, pues $MX = 0$.
- **Distribución central:**
  $$
  \frac{SS_{Res}}{\sigma^2} \sim \chi^2_{\,n - k - 1},
  $$
  ya que $M$ es simétrica, idempotente y $\operatorname{rank}(M) = n - k - 1$.
- **Independencia:** $(H - H_0)M = 0$ (proyectores ortogonales). Para normales, formas cuadráticas con proyectores ortogonales son independientes; por tanto, $SS_{Reg}$ y $SS_{Res}$ son independientes.

### Conclusión: distribución no central de $F_0$

- **Estadístico:**
  $$
  F_0 = \frac{MSR}{MS_{Res}}
  = \frac{SS_{Reg}/k}{SS_{Res}/(n - k - 1)}.
  $$
- **Resultado:**
  $$
  F_0 \sim F_{\,k,\;n - k - 1}\!\left(\lambda\right),
  \qquad
  \lambda = \frac{1}{\sigma^2}\beta^{*\top} X_c^\top X_c \beta^*.
  $$
- **Interpretación:**
  - Bajo $H_0$ ($\beta^* = 0$), $\lambda = 0$ y $F_0 \sim F_{k,\,n-k-1}$ central.
  - Bajo $H_1$, $\lambda > 0$ y $F_0$ sigue una $F$ no central con el parámetro indicado.

