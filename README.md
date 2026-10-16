# Predicción del precio de portátiles con Machine Learning

🏆 **1.er puesto de 11** en la competición interna de Kaggle del Bootcamp de Data Science de The Bridge (Valencia, noviembre de 2025), con un **MAE de 151 €** en la clasificación final y un único envío.
[Ver la competición y la clasificación en Kaggle](https://www.kaggle.com/competitions/adivina-su-precio-y-haz-tuyo-el-mejor-portatil/leaderboard)

## El reto
Una tienda quiere lanzar portátiles al mercado a un precio competitivo. A partir de datos de la competencia,
el objetivo era predecir el precio en euros de cada portátil según su marca y sus características.
La métrica de evaluación era el **MAE** (error absoluto medio).

## Datos
- 912 portátiles para entrenar y 391 para predecir.
- 11 variables en bruto: fabricante, modelo, tipo, pulgadas, resolución de pantalla, CPU, RAM, almacenamiento, GPU, sistema operativo y peso.
- Casi todas venían como texto ("8GB", "2.3kg", "256GB SSD + 1TB HDD", "Intel Core i7 7700HQ 2.8GHz"), así que la mayor parte del trabajo fue convertirlas en variables útiles.

## Qué hice

**1. Limpieza y creación de variables**
- RAM y peso convertidos a número.
- Pantalla: táctil (sí/no), panel IPS (sí/no) y densidad de píxeles (PPI), calculada a partir de la resolución y las pulgadas.
- CPU: fabricante, velocidad en GHz y gama (Core i3/i5/i7, Ryzen, Celeron…).
- Almacenamiento: separado en SSD, HDD, Flash e Híbrido, todo en GB.
- GPU: fabricante, serie (GTX, MX, Quadro, Intel HD…) y número de modelo.
- One-hot encoding de las variables categóricas, con un total de 117 variables.

**2. Modelos**
- Punto de partida: regresión lineal con las variables numéricas originales, con un **MAE de 322 €**.
- Probé Random Forest y XGBoost, con y sin transformación logarítmica del precio.
- Modelo final: **XGBoost** ajustado con GridSearchCV (72 combinaciones de hiperparámetros, validación cruzada de 3 particiones, optimizando directamente el MAE).

## Resultado
**Validación cruzada (en mis datos de entrenamiento)**

| Modelo | MAE |
|---|---|
| Regresión lineal (punto de partida) | 322 € |
| XGBoost ajustado a mano | 177 € |
| **XGBoost + GridSearchCV** | **166 €** |

Mejores parámetros: `max_depth=4`, `learning_rate=0.05`, `n_estimators=1000`, `subsample=0.8`, `colsample_bytree=1.0`.

**Clasificación final de Kaggle (datos de test ocultos)**

| | MAE |
|---|---|
| **Mi modelo (1.er puesto de 11)** | **151,26 €** |
| 2.º puesto | 157,68 € |
| Modelo de referencia de la competición | 317,15 € |

El error medio queda en menos de la mitad del modelo de referencia, y lo conseguí con un único envío: el resultado no sale de ir probando contra la clasificación.

## Herramientas
Python · Pandas · NumPy · Scikit-learn · XGBoost · expresiones regulares

## Archivos
- `prediccion_precio_portatiles.ipynb`: exploración, limpieza, creación de variables, modelo y predicción final.
- `submission.csv`: predicciones enviadas a Kaggle.
