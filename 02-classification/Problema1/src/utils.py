import numpy as np
from .metrics import f_score, accuracy, precision, recall, roc_auc, pr_auc
from Problema1.src.models import LogisticRegression 

def train_best_logistic_model(X_train, y_train, X_val, y_val, weights=None):
    """Entrena un modelo de regresión logística con los mejores hiperparámetros 
    según el F-score en el set de validación.

    Parámetros:
    - X_train, y_train: datos y etiquetas de entrenamiento.
    - X_val, y_val: datos y etiquetas de validación.
    - weights (array, opcional): pesos por muestra.

    Retorna:
    - model (LogisticRegression): modelo entrenado con los mejores hiperparámetros."""
    best_fscore = 0
    best_params = {'lr': None, 'lambda': None, 'threshold': None}
    lambdas = [0, 1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100]
    learning_rates = [0.1, 0.01, 0.001]
    thresholds = np.linspace(0, 1, 100)

    for lr in learning_rates:
        for l in lambdas:
            model = LogisticRegression(X_train, y_train, lr=lr, L2=l, weights=weights)
            y_scores = model.predict_proba(X_val)

            for t in thresholds:
                y_pred = (y_scores >= t).astype(int)
                fs = f_score(y_val, y_pred)

                if fs > best_fscore:
                    best_fscore = fs
                    best_params['lr'] = lr
                    best_params['lambda'] = l
                    best_params['threshold'] = t
    model = LogisticRegression(X_train, y_train, lr=best_params['lr'], threshold=best_params['threshold'], L2=best_params['lambda'], weights=weights)
    return model


def evaluar_modelo(nombre, model, X_val, y_val):
    """Evalúa un modelo de clasificación binaria sobre un conjunto de validación.

    Parámetros:
    - nombre (str): nombre del modelo.
    - model (objeto): modelo entrenado que implemente `predict` y `predict_proba`.
    - X_val (array): features del conjunto de validación.
    - y_val (Series): etiquetas verdaderas del conjunto de validación.

    Retorna:
    - dict: métricas del modelo (accuracy, precisión, recall, F1, AUC-ROC, AUC-PR, y scores)."""
    y_scores = model.predict_proba(X_val)
    y_pred = model.predict(X_val)
    _, _, auc_r = roc_auc(y_val, y_scores)
    _, _, auc_pr = pr_auc(y_val, y_scores)

    return {
        "Modelo": nombre,
        "Accuracy": accuracy(y_val, y_pred),
        "Precision": precision(y_val, y_pred),
        "Recall": recall(y_val, y_pred),
        "F-Score": f_score(y_val, y_pred),
        "AUC-ROC": auc_r,
        "AUC-PR": auc_pr,
        "y_scores": y_scores
    }