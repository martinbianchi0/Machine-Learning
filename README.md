# Machine learning desde cero — cuatro trabajos sin librerías de ML

Cuatro trabajos prácticos **individuales** de la materia **Aprendizaje Automático y Aprendizaje Profundo (I302)** — Universidad de San Andrés, 1er semestre 2025. Nota de la cursada: 8/10.

**No hay ni un import de scikit-learn en este repositorio, y no es una elección de estilo: la consigna lo prohibía.** Textual: *"No está permitido el uso de librerías de Machine Learning como scikit-learn"*, y en el TP4 tampoco librerías de clustering ni de reducción de dimensionalidad. Los algoritmos, las métricas, el preprocesamiento y el particionado de datos son módulos propios en `src/`. La única excepción autorizada fue PyTorch, y se usó justamente para **contrastar** contra la implementación propia.

## Contenido

| Carpeta | Tema | Qué está implementado a mano |
|---|---|---|
| [`01-regression-from-scratch/`](01-regression-from-scratch/) | Regresión | Regresión lineal simple y multivariable, **Ridge** y **Lasso**, feature engineering y selección de modelos con MAE y RMSE |
| [`02-classification/`](02-classification/) | Clasificación | Regresión logística, **LDA** y **Random Forest**, sobre un problema binario desbalanceado (diagnóstico de cáncer de mama) y uno multiclase |
| [`03-neural-net-from-scratch/`](03-neural-net-from-scratch/) | Redes neuronales | **Un perceptrón multicapa completo**: forward, **backpropagation**, optimizador **ADAM**, regularización L2, dropout, early stopping y búsqueda de hiperparámetros — todo en NumPy |
| [`04-unsupervised-and-vae/`](04-unsupervised-and-vae/) | Aprendizaje no supervisado | **K-means**, **Gaussian Mixture Models**, **DBSCAN** y **PCA**, más un **Variational Autoencoder** |

## El resultado que vale la pena mirar (TP3)

El MLP escrito en NumPy se validó contra una implementación equivalente en PyTorch sobre clasificación de caracteres japoneses manuscritos (**49 clases**):

| Modelo | Implementación | Accuracy train | Accuracy validación |
|---|---|---|---|
| M0 — baseline, capas [100, 80] | propia (NumPy) | 1.00 | 0.62 |
| **M1 — mejor modelo tras búsqueda** (L2=0.01, capas [256, 128]) | **propia (NumPy)** | 0.96 | **0.64** |
| M3 — mejor generalización, capa [1024] | PyTorch | 0.86 | 0.61 |
| M4 — sobreajuste máximo, capas [512, 256, 128, 64] | PyTorch | 1.00 | 0.62 |

**Lo que muestra:** la implementación propia queda en el mismo orden que la de librería (0.64 contra 0.61), que era el objetivo — validar el backpropagation escrito a mano, no ganarle a PyTorch. Y el M4 está construido a propósito para sobreajustar: llega a accuracy 1.00 en entrenamiento con la peor loss de validación de las cuatro, que es la demostración pedida del fenómeno.

Con 49 clases balanceadas, el azar da ~0.02 de accuracy. El resto de las métricas está en los notebooks de cada carpeta.

## Alcance

Es **coursework**, y está bien dicho así: son trabajos de cursada, individuales, con datasets acotados. El valor no está en las métricas —una librería optimizada las superaría— sino en que los algoritmos están escritos y son defendibles línea por línea.
