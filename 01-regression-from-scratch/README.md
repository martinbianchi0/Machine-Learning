# Trabajo Práctico 1 - Regresión

**Materia:** I302 - Aprendizaje Automático y Aprendizaje Profundo  
**Semestre:** 1er Semestre 2025  
**Fecha de entrega:** Lunes 24 de marzo de 2025, 23:59 hs  
**Formato de entrega:** Archivo comprimido `.zip` en el Campus Virtual  
**Lenguajes/Librerías permitidas:**  
- **NumPy** para cálculos matriciales y funciones numéricas.  
- **Pandas** para manipulación de datos.  
- **Matplotlib/Seaborn** para visualización de datos.  
- **No está permitido el uso de librerías de Machine Learning como scikit-learn.**  

---

## 📌 Descripción

Este trabajo práctico tiene como objetivo desarrollar y evaluar diversos modelos de regresión para estimar el precio de venta de una vivienda. Se utilizarán datos inmobiliarios que incluyen características como el área, número de habitaciones y año de construcción. Se trabajará con archivos `.csv` divididos en conjuntos de desarrollo (`casas_dev.csv`) y de prueba (`casas_test.csv`).

## 📂 Estructura del Proyecto

```
Apellido_Nombre_TP1.zip
│── data/                     # Datos del proyecto
│   │── raw/                   # Datos originales
│   │   │── casas_dev.csv
│   │   │── casas_test.csv
│   │── processed/              # Datos procesados
│
│── src/                      # Código fuente
│   │── preprocessing.py       # Procesamiento de datos sucios a limpios
│   │── data_splitting.py      # Métodos de división de datos (train_val, cross_val)
│   │── models.py              # Implementación de Linear Regression y métodos asociados
│   │── metrics.py             # Métricas de desempeño (MSE, MAE, RMSE)
│   │── utils.py               # Funciones auxiliares (gráficos, código reutilizable)
│
│── notebooks/                 # Notebooks de Jupyter
│   │── Entrega_TP1.ipynb      # Respuestas a los ejercicios
│
│── requirements.txt           # Dependencias del proyecto
│── README.md                  # Documentación del proyecto
```

---

## 📊 Contenido del Trabajo

### 1️⃣ Exploración de Datos  
- Análisis inicial del dataset, detección de valores erróneos o faltantes.  
- Estadísticas básicas y visualización de relaciones entre variables mediante histogramas y scatterplots.  
- División del dataset en 80% entrenamiento y 20% validación.  

### 2️⃣ Implementación de Regresión Lineal  
- Implementación de una clase `LinearRegression` con entrenamiento mediante pseudo-inversa y descenso por gradiente.  
- Cálculo de la función de pérdida usando **Error Cuadrático Medio (MSE)**.  
- Verificación de la implementación con distintas configuraciones de datos.  

### 3️⃣ Aplicación de Modelos de Regresión  
- Regresión lineal simple usando el **área** como variable predictora.  
- Regresión multivariable con selección de características relevantes.  
- Estimación del valor promedio por metro cuadrado y análisis del impacto de una pileta en el precio.  

### 4️⃣ Feature Engineering  
- Generación de nuevas características derivadas para mejorar la predicción.  
- Creación de modelos con hasta 300 características adicionales basadas en potencias de variables existentes.  

### 5️⃣ Regularización  
- Implementación de **Regresión Ridge (L2)** y **Regresión Lasso (L1)**.  
- Comparación de coeficientes y efectos de la regularización en el modelo.  
- Ajuste del hiperparámetro **λ** mediante validación cruzada.  

### 6️⃣ Selección de Modelo y Evaluación  
- Selección del mejor modelo basado en **MAE** y **RMSE** en el conjunto de prueba.  
- Justificación del modelo elegido para una posible implementación en producción.  

---

## 🛠 Instalación y Ejecución

1. Clonar el repositorio o descomprimir el archivo `.zip`:
   ```sh
   unzip Apellido_Nombre_TP1.zip
   cd Apellido_Nombre_TP1
   ```

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
   jupyter notebook notebooks/Entrega_TP1.ipynb
   ```

---

## 📌 Notas Importantes

- **Cumplir con la nomenclatura del archivo de entrega** (`Apellido_Nombre_TP1.zip`).  
- **Documentar claramente el código** y mantener una estructura modular en `src/`.  
- **Incluir gráficos y análisis** en el Jupyter Notebook `Entrega_TP1.ipynb`.  
- **El uso de librerías no permitidas conllevará a la no corrección del TP**.  

