import numpy as np
import pandas as pd

def mse(X, y, b):
    """Calcula el error cuadrático medio (MSE)."""
    X, y = preparar_datos(X, y)
    y_pred = X @ b
    return np.mean((y_pred - y) ** 2)

def rmse(X, y, b):
    """Calcula la raíz del error cuadrático medio (RMSE)."""
    return np.sqrt(mse(X, y, b))

def mae(X, y, b):
    """Calcula el error absoluto medio (MAE)."""
    X, y = preparar_datos(X, y)
    y_pred = X @ b
    return np.mean(np.abs(y_pred - y))

def preparar_datos(X, y):
    """Convierte X y y a numpy y agrega la columna de bias si falta."""
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    if isinstance(y, pd.Series):
        y = y.to_numpy()
    if X.ndim == 1 or X.shape[1] == 1:
        X = X.reshape(-1, 1)
    if not np.all(X[:, 0] == 1):
        X = np.column_stack((np.ones(X.shape[0]), X))
    return X, y
