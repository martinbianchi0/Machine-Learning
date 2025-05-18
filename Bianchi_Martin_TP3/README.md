# Trabajo Práctico 3 - Redes Neuronales  
**Materia:** I302 - Aprendizaje Automático y Aprendizaje Profundo  
**Alumno:** Bianchi, Martin

## Descripción general

Este trabajo práctico consiste en el desarrollo, entrenamiento y evaluación de redes neuronales para un problema de clasificación multiclase utilizando imágenes de caracteres japoneses (49 clases). Se exploran diversas arquitecturas y técnicas de entrenamiento, tanto con implementaciones propias como utilizando PyTorch.

## Estructura del repositorio

Bianchi_Martin_TP3/
│── data/                             # Archivos de datos (.npy provistos)
│   │── X_COMP.npy
│   │── X_images.npy
│   │── y_images.npy
│
│── src/                              # Código fuente modularizado
│   │── preprocessing.py              # Funciones de preprocesamiento de datos
│   │── visualization.py              # Funciones de visualización y gráficos
│   │── models.py                     # Implementación de redes neuronales
│   │── metrics.py                    # Métricas de evaluación implementadas
│   │── evaluation.py                 # Scripts de entrenamiento y evaluación
│
│── Bianchi_Martin_Informe_TP3.pdf    # Informe teórico-metodológico del trabajo
│── Bianchi_Martin_Notebook_TP3.ipynb # Desarrollo técnico y pruebas
│── predicciones.csv                  # Predicciones sobre el dataset X_COMP.npy
│── requirements.txt                  # Dependencias utilizadas
│── README.md                         # Este archivo

## Modelos desarrollados

- **M0**: Red neuronal básica (2 capas ocultas: [100, 80]) implementada desde cero, entrenamiento con descenso de gradiente estándar y backpropagation.
- **M1**: Arquitectura optimizada con mejoras como mini-batch SGD, ADAM, regularización L2, early stopping y exploración de hiperparámetros.
- **M2**: Mismo modelo que M1, pero implementado en PyTorch.
- **M3**: Mejor arquitectura encontrada utilizando PyTorch.
- **M4**: Arquitectura en PyTorch que produce overfitting, utilizada para análisis comparativo.

## Cómo reproducir los experimentos

1. Clonar el repositorio y descomprimir el archivo `.zip`.
2. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt

## Ejecución

Ejecutar el notebook `Bianchi_Martin_Notebook_TP3.ipynb` en orden.

Ver el informe PDF para el análisis metodológico y teórico.

## Predicciones

El archivo `Bianchi_Martin_predicciones.csv` contiene las probabilidades a posteriori por clase del dataset `X_COMP.npy`, generadas utilizando el mejor modelo identificado en el trabajo (M3 o M1, según resultados).

## Créditos

Trabajo realizado por Bianchi, Martin para la materia I302 - Aprendizaje Automático y Aprendizaje Profundo, año 2025.