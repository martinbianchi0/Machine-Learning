# Aprendizaje Automático y Profundo 

Este repositorio contiene cuatro trabajos realizados en base a Aprendizaje Automático y Aprendizaje Profundo. Cada trabajo aborda una temática distinta del aprendizaje automático, con implementaciones propias, análisis experimentales y documentación completa.

---

## 📁 Contenido

### 🔹 Trabajo 1 - Regresión
Implementación desde cero de modelos de regresión lineal para estimar precios de viviendas.  
Incluye:
- Regresión simple y multivariable
- Feature engineering
- Regularización (Ridge y Lasso)
- Selección de modelos con métricas MAE y RMSE  
📂 Carpeta: [`TP1/`](./TP1)

### 🔹 Trabajo 2 - Clasificación
Dos problemas: 
- Diagnóstico de cáncer de mama (clasificación binaria con datos desbalanceados)
- Rendimiento de jugadores de basketball (clasificación multiclase)  
Modelos implementados:
- Regresión logística
- LDA
- Random Forest  
📂 Carpeta: [`TP2/`](./TP2)

### 🔹 Trabajo 3 - Redes Neuronales
Clasificación multiclase de caracteres japoneses con redes neuronales.  
Incluye:
- Implementación propia de MLP
- Entrenamiento con backpropagation, ADAM, early stopping
- Versiones equivalentes en PyTorch
- Comparación de arquitecturas  
📂 Carpeta: [`TP3/`](./TP3)

### 🔹 Trabajo 4 - Clustering y Reducción de Dimensionalidad
Análisis sin supervisión sobre dos datasets usando:
- K-means, GMM, DBSCAN
- PCA implementado desde cero
- Autoencoder variacional (VAE) con PyTorch  
📂 Carpeta: [`TP4/`](./TP4)

---

## ▶️ Requisitos

Cada carpeta tiene su propio `requirements.txt` con las dependencias específicas. Se recomienda crear un entorno virtual por TP.

```bash
cd TP1/
python -m venv env
source env/bin/activate  # o env\Scripts\activate en Windows

pip install -r requirements.txt
