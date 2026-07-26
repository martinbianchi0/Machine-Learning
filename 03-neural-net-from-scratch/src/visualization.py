from src.evaluation import get_numpy_metrics, get_torch_metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_images(X_images, n):
    """
    Muestra las primeras 'n' imágenes en escala de grises.

    Args:
        X_images: Arreglo de imágenes en formato plano.
        n: Número de imágenes a mostrar.
    """
    for i in range(n):
        img = X_images[i].reshape(28, 28)
        plt.subplot(1, n, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        plt.title(f'Imagen {i}')

    plt.tight_layout()
    plt.show()


def show_metrics_and_matrices(m0, X_train, y_train, X_val, y_val, labels=None):
    """
    Calcula métricas y matrices de confusión para un modelo en NumPy.

    Args:
        m0: Modelo entrenado.
        X_train, X_val: Features de entrenamiento y validación.
        y_train, y_val: Etiquetas de entrenamiento y validación.
        labels: Lista opcional de etiquetas para las matrices.
    """
    res = get_numpy_metrics(m0, X_train, y_train, X_val, y_val)

    print("Entrenamiento:")
    print(f" - Accuracy: {res['acc_train']:.2f}")
    print(f" - Loss: {res['loss_train']:.2f}")

    print("\nValidación:")
    print(f" - Accuracy: {res['acc_val']:.2f}")
    print(f" - Loss: {res['loss_val']:.2f}")

    # Visualización
    num_classes = res['cm_train'].shape[0]
    labels = list(range(num_classes)) if labels is None else labels
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, cm, title in zip(axes, [res['cm_train'], res['cm_val']], ["Entrenamiento", "Validación"]):
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        sns.heatmap(df_cm, annot=False, fmt="d", cmap="Blues", cbar=True, ax=ax)
        ax.set_title(f"Matriz de Confusión - {title}")
        ax.set_xlabel("Etiqueta predicha")
        ax.set_ylabel("Etiqueta verdadera")

    plt.tight_layout()
    plt.show()

def show_torch_metrics_and_matrices(m0, X_train, y_train, X_val, y_val, labels=None):
    """
    Calcula métricas y matrices de confusión para un modelo en PyTorch.

    Args:
        m0: Modelo entrenado.
        X_train, X_val: Features de entrenamiento y validación.
        y_train, y_val: Etiquetas de entrenamiento y validación.
        labels: Lista opcional de etiquetas para las matrices.
    """
    res = get_torch_metrics(m0, X_train, y_train, X_val, y_val)

    print("Entrenamiento:")
    print(f" - Accuracy: {res['acc_train']:.2f}")
    print(f" - Loss: {res['loss_train']:.2f}")

    print("\nValidación:")
    print(f" - Accuracy: {res['acc_val']:.2f}")
    print(f" - Loss: {res['loss_val']:.2f}")

    num_classes = res['cm_train'].shape[0]
    labels = list(range(num_classes)) if labels is None else labels
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, cm, title in zip(axes, [res['cm_train'], res['cm_val']], ["Entrenamiento", "Validación"]):
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        sns.heatmap(df_cm, annot=False, fmt="d", cmap="Blues", cbar=True, ax=ax)
        ax.set_title(f"Matriz de Confusión - {title}")
        ax.set_xlabel("Etiqueta predicha")
        ax.set_ylabel("Etiqueta verdadera")

    plt.tight_layout()
    plt.show()


def display_results(results):
    """
    Muestra un resumen visual (heatmap) con métricas de varios modelos.

    Args:
        results: Diccionario con nombre del modelo y sus métricas.
    """
    data = {
        "Técnica": [],
        "Tiempo (s)": [],
        "Acc Train": [],
        "Acc Val": [],
        "Loss Train": [],
        "Loss Val": []
    }
    for name, metrics in results.items():
        data["Técnica"].append(name)
        data["Tiempo (s)"].append(round(metrics["time"], 2))
        data["Acc Train"].append(round(metrics["train_acc"], 4))
        data["Acc Val"].append(round(metrics["val_acc"], 4))
        data["Loss Train"].append(round(metrics["train_loss"], 4))
        data["Loss Val"].append(round(metrics["val_loss"], 4))

    df = pd.DataFrame(data)

    # Creamos un heatmap para resaltar valores
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        df.iloc[:, 1:].astype(float),
        annot=df.iloc[:, 1:].astype(float), 
        fmt=".2f",
        cmap="coolwarm", 
        linewidths=0.5,
        linecolor="black",
        cbar=False,
        ax=ax,
        xticklabels=df.columns[1:],  
        yticklabels=df["Técnica"]   
    )

    # Ajustamos el formato del heatmap
    ax.set_title("Resumen de Resultados", fontsize=16, pad=20)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=10, rotation=45)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    plt.tight_layout()
    plt.show()


def comparar_modelos(modelos_numpy, modelos_torch, datos_numpy, datos_torch):
    """
    Compara modelos de NumPy y PyTorch mostrando métricas y matrices.

    Args:
        modelos_numpy: Lista de modelos NumPy.
        modelos_torch: Lista de modelos PyTorch.
        datos_numpy: Datos de entrenamiento y validación (NumPy).
        datos_torch: Datos de entrenamiento y validación (PyTorch).
    """
    nombres = [f'Modelo {i}' for i in range(len(modelos_numpy) + len(modelos_torch))]

    resultados = []
    
    # Datos
    X_train_np, y_train_np, X_val_np, y_val_np = datos_numpy
    X_train_pt, y_train_pt, X_val_pt, y_val_pt = datos_torch

    for model in modelos_numpy:
        res = get_numpy_metrics(model, X_train_np, y_train_np, X_val_np, y_val_np)
        resultados.append(res)

    for model in modelos_torch:
        res = get_torch_metrics(model, X_train_pt, y_train_pt, X_val_pt, y_val_pt)
        resultados.append(res)

    # === Gráfico de accuracy y loss ===
    acc_train_list = [r['acc_train'] for r in resultados]
    acc_val_list = [r['acc_val'] for r in resultados]
    loss_train_list = [r['loss_train'] for r in resultados]
    loss_val_list = [r['loss_val'] for r in resultados]

    x = np.arange(len(resultados))

    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.bar(x - 0.2, acc_train_list, width=0.4, label='Train')
    plt.bar(x + 0.2, acc_val_list, width=0.4, label='Test')
    plt.xticks(x, nombres, rotation=45)
    plt.ylabel("Accuracy")
    plt.title("Accuracy por modelo")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(x - 0.2, loss_train_list, width=0.4, label='Train')
    plt.bar(x + 0.2, loss_val_list, width=0.4, label='Test')
    plt.xticks(x, nombres, rotation=45)
    plt.ylabel("Loss")
    plt.title("Loss por modelo")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # === Gráfico de matrices de confusión (solo validación para ahorrar espacio) ===
    fig, axes = plt.subplots(1, len(resultados), figsize=(5 * len(resultados), 5))
    if len(resultados) == 1:
        axes = [axes]

    for ax, res, name in zip(axes, resultados, nombres):
        cm = res['cm_val']
        labels = list(range(cm.shape[0]))
        df_cm = pd.DataFrame(cm, index=labels, columns=labels)
        sns.heatmap(df_cm, annot=False, fmt="d", cmap="Blues", cbar=True, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Predicha")
        ax.set_ylabel("Verdadera")

    plt.tight_layout()
    plt.show()
