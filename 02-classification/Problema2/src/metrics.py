import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def precision(y_true, y_pred):
    """Calcula la precisión.

    Parámetros:
    - y_true (array): etiquetas verdaderas.
    - y_pred (array): etiquetas predichas.

    Retorna:
    - float: precisión promedio entre clases. Retorna -1 si no se puede calcular.
    """
    clases = np.unique(y_true)
    precisiones = []
    for c in clases:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        p = tp / (tp + fp) if (tp + fp) != 0 else -1
        precisiones.append(p)
    return np.mean([p for p in precisiones if p != -1])

def recall(y_true, y_pred):
    """Calcula el recall.

    Parámetros:
    - y_true (array): etiquetas verdaderas.
    - y_pred (array): etiquetas predichas.

    Retorna:
    - float: recall promedio entre clases. Retorna -1 si no se puede calcular.
    """
    clases = np.unique(y_true)
    recalls = []
    for c in clases:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        r = tp / (tp + fn) if (tp + fn) != 0 else -1
        recalls.append(r)
    return np.mean([r for r in recalls if r != -1])

def f_score(y_true, y_pred):
    """Calcula el F-score a partir de precisión y recall.

    Parámetros:
    - y_true (array): etiquetas verdaderas.
    - y_pred (array): etiquetas predichas.

    Retorna:
    - float: F1-score macro. Retorna -1 si no se puede calcular.
    """
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p == -1 or r == -1 or (p + r) == 0:
        return -1
    return 2 * (p * r) / (p + r)

def accuracy(y_true, y_pred):
    """Calcula la exactitud (accuracy).

    Parámetros:
    - y_true (array): etiquetas verdaderas.
    - y_pred (array): etiquetas predichas.

    Retorna:
    - float: proporción de predicciones correctas.
    """
    return np.mean(y_true == y_pred)

def matriz_de_confusion(y_true, y_pred):
    """Genera la matriz de confusión entre las clases.

    Parámetros:
    - y_true (array): etiquetas verdaderas.
    - y_pred (array): etiquetas predichas.

    Retorna:
    - ndarray: matriz de confusión (clases reales vs. predichas).
    """
    clases = np.unique(np.concatenate([y_true, y_pred]))
    n = len(clases)
    matriz = np.zeros((n, n), dtype=int)
    for i, actual in enumerate(clases):
        for j, predicho in enumerate(clases):
            matriz[i, j] = np.sum((y_true == actual) & (y_pred == predicho))
    return matriz

def pr_auc(y_true, y_scores):
    """
    Calcula la curva Precisión-Recall y el área bajo la curva (AUC-PR) combinando
    todos los verdaderos y falsos positivos de todas las clases (micro-promedio).

    Parámetros:
    - y_true: array de etiquetas verdaderas (n_samples,)
    - y_scores: array (n_samples, n_clases) de probabilidades predichas

    Retorna:
    - recalls: lista de recall para cada umbral
    - precisions: lista de precision para cada umbral
    - auc_pr: área bajo la curva precision-recall
    """
    clases = np.unique(y_true)
    n_samples, n_classes = y_scores.shape

    # Convertir y_true a one-hot
    y_true_onehot = np.zeros_like(y_scores)
    for i, c in enumerate(clases):
        y_true_onehot[:, i] = (y_true == c).astype(int)

    # Aplanar ambos arrays para combinar todos los valores
    y_true_flat = y_true_onehot.ravel()
    y_scores_flat = y_scores.ravel()

    # Ordenar por score descendente
    orden = np.argsort(-y_scores_flat)
    y_true_sorted = y_true_flat[orden]
    y_scores_sorted = y_scores_flat[orden]

    tp = 0
    fp = 0
    precisions = []
    recalls = []

    total_positivos = np.sum(y_true_flat)

    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
        else:
            fp += 1

        precision = tp / (tp + fp)
        recall = tp / total_positivos

        precisions.append(precision)
        recalls.append(recall)

    # Agregamos punto inicial
    recalls = [0.0] + recalls
    precisions = [1.0] + precisions

    auc_pr = np.trapz(precisions, recalls)
    return recalls, precisions, auc_pr

def roc_auc(y_true, y_scores):
    """
    Calcula la curva ROC y el área bajo la curva (AUC-ROC) combinando
    todos los verdaderos y falsos positivos de todas las clases (micro-promedio).

    Parámetros:
    - y_true: array de etiquetas verdaderas (n_samples,)
    - y_scores: array (n_samples, n_clases) de probabilidades predichas

    Retorna:
    - fpr: lista de tasa de falsos positivos
    - tpr: lista de tasa de verdaderos positivos
    - auc_roc: área bajo la curva ROC
    """
    clases = np.unique(y_true)
    n_samples, n_classes = y_scores.shape

    # Convertir y_true a one-hot
    y_true_onehot = np.zeros_like(y_scores)
    for i, c in enumerate(clases):
        y_true_onehot[:, i] = (y_true == c).astype(int)

    # Aplanar
    y_true_flat = y_true_onehot.ravel()
    y_scores_flat = y_scores.ravel()

    # Ordenar por score descendente
    orden = np.argsort(-y_scores_flat)
    y_true_sorted = y_true_flat[orden]

    tp = 0
    fp = 0
    fn = np.sum(y_true_flat)
    tn = len(y_true_flat) - fn

    tpr = []
    fpr = []

    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
            tn -= 1

        tpr_val = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0

        tpr.append(tpr_val)
        fpr.append(fpr_val)

    auc_roc = np.trapz(tpr, fpr)
    return fpr, tpr, auc_roc