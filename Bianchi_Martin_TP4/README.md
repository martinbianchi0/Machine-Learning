# Trabajo Práctico 4 - Clustering y Reducción de Dimensionalidad

**Materia:** I302 - Aprendizaje Automático y Aprendizaje Profundo  
**Semestre:** 1er Semestre 2025  
**Fecha de entrega:** Viernes 30 de mayo, 23:59 hs  
**Formato de entrega:** Archivo comprimido `.zip` en el Campus Virtual  
**Lenguajes/Librerías permitidas:**  
- **NumPy** y **Pandas** para procesamiento numérico y manejo de datos  
- **Matplotlib** y **Seaborn** para visualización  
- **No está permitido el uso de librerías de clustering (como `sklearn.cluster`) o reducción de dimensionalidad prehechas** (salvo PyTorch para el VAE del punto 2.c)  

---

## 📌 Descripción

Este trabajo práctico tiene como objetivo implementar y comparar distintos algoritmos de clustering y reducción de dimensionalidad. Se trabajará con dos datasets:  
- `clustering.csv`: para aplicar K-means, GMM y DBSCAN  
- `MNIST_dataset.csv`: para aplicar PCA y opcionalmente entrenar un VAE usando PyTorch.  

El proyecto se compone de un informe principal (`Bianchi_Martin_Informe_TP4.pdf`) y un notebook de respaldo técnico (`Bianchi_Martin_notebook_TP4.ipynb`), junto con el código modularizado en scripts `.py`.

---

## 📂 Estructura del Proyecto

```
Bianchi_Martin_TP4/
│── data/                             # Archivos de datos (.npy provistos)
│   │── clustering.csv
│   │── MNIST_dataset.csv
│
│── src/                              # Código fuente modularizado
│   │── preprocessing.py              # Funciones de preprocesamiento de datos
│   │── visualization.py              # Funciones de visualización y gráficos
│   │── exploration.py                # Exploracion y evaluación de hiperparametros en modelos
│   │── metrics.py                    # Métricas de evaluación implementadas
│   │── clustering.py                 # Modelos de clustering no supervisado
│   │── dreduction.py                 # Métodos de reducción de dimensionalidad
│
│── Bianchi_Martin_Informe_TP4.pdf    # Informe teórico-metodológico del trabajo
│── Bianchi_Martin_Notebook_TP4.ipynb # Desarrollo técnico y pruebas
│── requirements.txt                  # Dependencias utilizadas
│── README.md                         # Este archivo
```

---

## 📊 Contenido del Trabajo

### 1️⃣ Clustering de Datos (`clustering.csv`)

- **K-means:**  
  Implementación desde cero, selección de K con el método de “ganancias decrecientes” y visualización de los clusters con sus centroides.

- **GMM (Gaussian Mixture Model):**  
  Implementación propia, inicialización con K-means, cálculo de responsabilidades y visualización de los clusters.

- **DBSCAN:**  
  Implementación del algoritmo con distintos valores de `ε` y `min_samples`, identificación de ruido y clusters densos. Visualización del resultado con colores por cluster.

### 2️⃣ Reducción de Dimensionalidad (`MNIST_dataset.csv`)

- **PCA (Principal Component Analysis):**  
  Cálculo de componentes principales, gráfico del error cuadrático medio vs. cantidad de componentes, y reconstrucción visual de las primeras 10 imágenes con una cantidad elegida de componentes.

- **VAE (Autoencoder Variacional) - OPCIONAL:**  
  Entrenamiento de un modelo VAE con PyTorch, comparación de reconstrucciones frente a PCA. División del dataset en entrenamiento y validación. Evaluación de la calidad de las imágenes generadas.

---

## 🛠 Instalación y Ejecución

1. Descomprimir el archivo `.zip`:
   ```bash
   unzip Bianchi_Martin_TP4.zip
   cd Bianchi_Martin_TP4

2. Crear un entorno virtual (opcional pero recomendado):

   ```sh
   python -m venv env
   source env/bin/activate  # En Windows: env\Scripts\activate
   ```

3. Instalar dependencias:

   ```sh
   pip install -r requirements.txt
   ```

4. Ejecutar el Jupyter Notebook:
   ```sh
   jupyter notebook Bianchi_Martin_notebook_TP4.ipynb
   ```