import numpy as np

def accuracy(y_true, y_pred):
    """
    Calcula la exactitud del modelo.

    Args:
        y_true: Etiquetas reales.
        y_pred: Etiquetas predichas.

    Returns:
        Porcentaje de aciertos.
    """
    y_true = np.array(y_true) 
    return np.mean(y_true == y_pred)

def cross_entropy(y_true, y_pred):
    """
    Calcula la pérdida cross-entropy.

    Args:
        y_true: Etiquetas verdaderas (one-hot).
        y_pred: Probabilidades predichas.

    Returns:
        Pérdida promedio.
    """
    epsilon = 1e-8
    return -np.mean(np.sum(y_true * np.log(y_pred + epsilon), axis=0))

def matriz_de_confusion(y_true, y_pred):
    """
    Calcula la matriz de confusión.

    Args:
        y_true: Etiquetas reales.
        y_pred: Etiquetas predichas.

    Returns:
        Matriz de confusión (2D numpy array).
    """
    clases = np.unique(np.concatenate([y_true, y_pred]))
    n = len(clases)
    matriz = np.zeros((n, n), dtype=int)
    for i, actual in enumerate(clases):
        for j, predicho in enumerate(clases):
            matriz[i, j] = np.sum((y_true == actual) & (y_pred == predicho))
    return matriz