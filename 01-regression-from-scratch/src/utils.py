from models import LinearRegression
from metrics import mse, mae, rmse
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def save_results():
    return

def load_model(x_train, y_train, method, L1=0, L2=0):
    """Entrena un modelo con el método especificado.  
    Params: x_train (DataFrame), y_train (Series), metodo (str), L1 (float), L2 (float).  
    Returns: Modelo entrenado.
    """
    modelo = LinearRegression(x_train, y_train, L1, L2)
    if method == "gd":
        modelo.gradiente_descendiente()
    elif method == "pinv":
        modelo.pseudoinversa()
    return modelo

def data_info(df_dev, df_test):
    """Muestra información general del dataset.  
    Params: df_dev (DataFrame), df_test (DataFrame).  
    """
    df = pd.concat([df_dev, df_test])
    print("Fragmento aleatorio\n", df.sample(7))
    print("\nRango de valores de cada columna\n", df.describe().loc[['min', 'max']])
    print("\nCategorías con valores faltantes\n", df.isna().sum()[df.isna().sum() > 0].to_string())
    print("\nFilas duplicadas:", df.duplicated().sum())

def data_analisis(df_dev, df_test):
    """Genera gráficos exploratorios del dataset.  
    Params: df_dev (DataFrame), df_test (DataFrame).  
    """
    df = pd.concat([df_dev, df_test], ignore_index=True)

    num_vars = ['area', 'price', 'rooms', 'age']
    bin_vars = ['is_house', 'has_pool']

    # Pairplot con color según la nueva categoría
    sns.pairplot(df, vars=num_vars, hue="zona", palette="coolwarm")
    plt.show()

    # Boxplots
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))

    # Numéricas contra binarias
    for ax, (num_var, bin_var) in zip(axes.flat, [(x, y) for x in num_vars for y in bin_vars]):
        sns.boxplot(x=df[bin_var], y=df[num_var], ax=ax)
        ax.set_title(f'{num_var} vs {bin_var}')

    # Ocultar gráficos vacíos si sobran espacios
    for ax in axes.flat[len(num_vars) * len(bin_vars):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    # Countplots de binarios contra binarios
    fig, axes = plt.subplots(1, 3, figsize=(12, 5))

    # Comparación de zona con is_house
    sns.countplot(x=df["zona"], hue=df["is_house"], palette="coolwarm", ax=axes[0])
    axes[0].set_title("Distribución de is_house según zona")

    # Comparación de zona con has_pool
    sns.countplot(x=df["zona"], hue=df["has_pool"], palette="coolwarm", ax=axes[1])
    axes[1].set_title("Distribución de has_pool según zona")

    # Comparación de is_house con has_pool
    sns.countplot(x=df["is_house"], hue=df["has_pool"], palette="coolwarm", ax=axes[2])
    axes[2].set_title("Distribución de has_pool según is_house")

    plt.tight_layout()
    plt.show()

def model_onefeature(x_train, y_train, X_val, y_val):
    """Entrena y evalúa modelos de regresión con una variable.  
    Params: x_train (DataFrame), y_train (Series), X_val (DataFrame), y_val (Series).  
    Returns: None (imprime métricas y muestra gráficos de regresión).
    """
    modelos = {
        "Gradiente Descendiente": load_model(x_train, y_train, "gd"),
        "Pseudoinversa": load_model(x_train, y_train, "pinv")
    }

    print("\n--- Comparación de métodos ---")
    
    for nombre, modelo in modelos.items():
        coef = modelo.obtener_coeficientes()
        print(f"\n{nombre}:")
        print("MSE Train:", mse(x_train, y_train, coef), "| MSE Validation:", mse(X_val, y_val, coef))

        # Graficar regresión
        X_plot = np.column_stack((np.ones(x_train.shape[0]), x_train))
        y_pred = X_plot @ coef

        plt.scatter(x_train.iloc[:, 0], y_train, color="blue", label="Datos")
        plt.plot(np.sort(x_train.iloc[:, 0]), np.sort(y_pred), color='red', lw=2, label="Recta de Regresión")
        plt.xlabel("Área")
        plt.ylabel("Precio")
        plt.legend()
        plt.title(f"Regresión Lineal: Precio vs Área ({nombre})")
        plt.show()


def feature_engineering(df):
    """Crea nuevas variables a partir de las existentes.  
    Params: df (DataFrame).  
    Returns: None (modifica el DataFrame in-place).
    """
    df["area_per_room"] = df["area"] / df["rooms"]
    df["luxury"] = df["is_house"] * df["has_pool"] * df["rooms"]
    df["log_area"] = np.log1p(df["area"])
    df["is_house_age"] = df["is_house"] * df["age"]
    df["zona_area"] = df["zona"] * df["area"]

def excessive_feature_engineering(dx, xn_train, xn_val, dy_train, dy_val):
    """Genera excesivas features elevando las columnas originales a diferentes potencias.  
    Luego entrena un modelo con pseudoinversa y calcula el error cuadrático medio (MSE).  
    Params: dx (DataFrame), xn_train (DataFrame), dy_train (Series), xn_val (DataFrame), dy_val (Series).
    """
    features = 0
    potencia = 0
    columnas_originales = list(dx.columns)  # Fija las columnas originales
    nuevas_columnas_train = {}
    nuevas_columnas_val = {}

    while features < 300:
        potencia += 1
        for column in columnas_originales:
            nuevo_nombre = f"{column}_{potencia}"
            nuevas_columnas_train[nuevo_nombre] = xn_train[column] ** potencia
            nuevas_columnas_val[nuevo_nombre] = xn_val[column] ** potencia
            features += 1
            if features >= 300:
                break

    nuevas_columnas_train = pd.DataFrame(nuevas_columnas_train)
    nuevas_columnas_val = pd.DataFrame(nuevas_columnas_val)
    model = load_model(nuevas_columnas_train, dy_train, "pinv")
    b = model.obtener_coeficientes()

    X_bias = np.hstack([np.ones((nuevas_columnas_train.shape[0], 1)), nuevas_columnas_train])
    print("MSE train:", mse(X_bias, dy_train, b))
    print("MSE val:", mse(np.hstack([np.ones((nuevas_columnas_val.shape[0], 1)), nuevas_columnas_val]), dy_val, b))

def metricas(x_train, y_train, X_val, y_val, test=None, L1=0, L2=0):
    """Muestra métricas de entrenamiento y validación para distintos métodos.  
    Params: x_train (DataFrame), y_train (Series), X_val (DataFrame), y_val (Series), test (DataFrame, opcional), L1 (float), L2 (float).  
    Returns: None (imprime resultados)."""
    
    modelos = {
        "Gradiente Descendiente": load_model(x_train, y_train, "gd", L1, L2),
        "Pseudoinversa": load_model(x_train, y_train, "pinv", L1, L2),
    }

    print("\n--- Comparación de métodos ---")
    for nombre, modelo in modelos.items():
        print(f"\n{nombre}:")
        metricas_train = mse(x_train, y_train, modelo.obtener_coeficientes())
        metricas_val = mse(X_val, y_val, modelo.obtener_coeficientes())
        print(f"MSE Train: {metricas_train:.2f} | MSE Validation: {metricas_val:.2f}")

        if isinstance(test, pd.DataFrame):
            print("Predicción de price:", modelo.predict(test))

def calcular_coefs_vs_l(x_train, y_train, reg, lambdas, metodo="gd"):
    """Calcula coeficientes para distintos valores de regularización.  
    Params:  
        x_train (DataFrame), y_train (Series), reg (str), lambdas (array),  
        metodo (str): "gd" para gradiente descendiente, "pinv" para pseudoinversa.  
    Returns: array con coeficientes para cada λ.
    """
    coefs = []
    for L in lambdas:
        if reg == "L2":
            modelo = LinearRegression(x_train, y_train, L2=L)
            if metodo == "gd":
                modelo.gradiente_descendiente(learning_rate=0.001)  # Permitir GD para L2
            else:
                modelo.pseudoinversa()  # Pseudoinversa por defecto
        else:  # L1
            modelo = LinearRegression(x_train, y_train, L1=L)
            modelo.gradiente_descendiente()
        coefs.append(modelo.obtener_coeficientes())

    return np.array(coefs)

def graficar_coeficientes(lambdas, coefs, reg, metodo, feature_names):
    """Grafica coeficientes en función de la regularización.  
    Params: lambdas (array), coefs (array), reg (str), metodo (str), feature_names (list).  
    Returns: None (muestra el gráfico)."""
    
    plt.figure(figsize=(10, 6))
    for i, name in enumerate(feature_names):
        plt.plot(lambdas, coefs[:, i], label=name)  # Usa el nombre real de la feature

    plt.xscale("log")
    plt.xlabel(f"Regularización {reg} (λ)")
    plt.ylabel("Valor de los coeficientes")
    plt.title(f"Coeficientes vs Regularización {reg} ({metodo})")
    plt.legend()
    plt.show()

def graficar_coef_vs_l(x_train, y_train, reg):
    """Calcula y grafica coeficientes en función de la regularización.  
    Params: x_train (DataFrame), y_train (Series), reg (str).  
    Returns: None (muestra gráficos)."""
    
    lambdas = np.logspace(-4, 2, 20)
    lambdas2 = np.logspace(-4, 3, 40)
    feature_names = ["Bias"] + list(x_train.columns)

    if reg == "L2":
        coefs_pinv = calcular_coefs_vs_l(x_train, y_train, "L2", lambdas2, "pinv")
        coefs_gd = calcular_coefs_vs_l(x_train, y_train, "L2", lambdas)

        graficar_coeficientes(lambdas2, coefs_pinv, "L2", "Pseudoinversa", feature_names)
        graficar_coeficientes(lambdas, coefs_gd, "L2", "Gradiente Descendiente", feature_names)
    else:
        coefs_gd = calcular_coefs_vs_l(x_train, y_train, "L1", lambdas2)
        graficar_coeficientes(lambdas2, coefs_gd, "L1", "Gradiente Descendiente", feature_names)