import numpy as np

def precision(y_true, y_pred):
    """Calcula la precisión de las predicciones (TP / (TP + FP)).

    Parámetros:
    - y_true (Series): etiquetas verdaderas.
    - y_pred (Series): etiquetas predichas.

    Retorna:
    - float: valor de precisión, o -1 si no hay positivos predichos."""
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    predicted_positives = np.sum(y_pred == 1)
    return true_positives / predicted_positives if predicted_positives != 0 else -1

def recall(y_true, y_pred):
    """Calcula el recall de las predicciones (TP / (TP + FN)).

    Parámetros:
    - y_true (Series): etiquetas verdaderas.
    - y_pred (Series): etiquetas predichas.

    Retorna:
    - float: valor de recall, o -1 si no hay positivos reales."""
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    actual_positives = np.sum(y_true == 1)
    return true_positives / actual_positives if actual_positives != 0 else -1

def f_score(y_true, y_pred):
    """Calcula el F1-score como la media armónica entre precisión y recall.

    Parámetros:
    - y_true (Series): etiquetas verdaderas.
    - y_pred (Series): etiquetas predichas.

    Retorna:
    - float: valor del F1-score, o -1 si precisión o recall no están definidos."""
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    if p == -1 or r == -1 or p + r == 0:
        return -1
    return 2 * (p * r) / (p + r)

def accuracy(y_true, y_pred):
    """Calcula la exactitud del modelo (proporción de aciertos).

    Parámetros:
    - y_true (Series): etiquetas verdaderas.
    - y_pred (Series): etiquetas predichas.

    Retorna:
    - float: valor de accuracy."""
    return np.mean(y_true == y_pred)

def matriz_de_confusion(y_true, y_pred):
    """Calcula y devuelve la matriz de confusión.

    Parámetros:
    - y_true (Series): etiquetas verdaderas.
    - y_pred (Series): etiquetas predichas.

    Retorna:
    - cm (ndarray): matriz de confusión 2x2.
    """
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return np.array([[tn, fp], [fn, tp]])

def pr_auc(y_true, y_pred):
    """Calcula la curva Precisión-Recall y su área bajo la curva (AUC-PR).

    Parámetros:
    - y_true (Series): etiquetas verdaderas.
    - y_pred (Series): probabilidades predichas.

    Retorna:
    - tuple: (recalls, precisions, auc_pr), donde cada uno es una lista o float."""
    thresholds = np.sort(np.unique(y_pred))[::-1]
    precisions = []
    recalls = []
    for t in thresholds:
        y_pred_t = (y_pred >= t).astype(int)
        precisions.append(precision(y_true, y_pred_t))
        recalls.append(recall(y_true, y_pred_t))

    # Agregar punto extremo
    recalls = [0.0] + recalls
    precisions = [1.0] + precisions

    auc_pr = np.trapz(precisions, recalls)
    return recalls, precisions, auc_pr


def roc_auc(y_true, y_scores):
    """Calcula la curva ROC y su área bajo la curva (AUC-ROC).

    Parámetros:
    - y_true (Series): etiquetas verdaderas.
    - y_scores (Series): probabilidades predichas.

    Retorna:
    - tuple: (fpr, tpr, auc_roc), donde fpr y tpr son listas, auc_roc es un float."""
    thresholds = np.linspace(0, 1, 100)
    tpr = []
    fpr = []

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        tpr.append(tp / (tp + fn))
        fpr.append(fp / (fp + tn))


    sorted_pairs = sorted(zip(fpr, tpr))
    fpr_sorted, tpr_sorted = zip(*sorted_pairs)

    auc_roc = np.trapz(tpr_sorted, fpr_sorted)
    return fpr_sorted, tpr_sorted, auc_roc