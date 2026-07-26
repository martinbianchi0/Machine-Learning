import numpy as np

def standard_fit(X):
    """
    Calcula media y desvío estándar.
    Parámetros: - X: Serie o array unidimensional.
    Retorna: - media y desvío estándar.
    """
    return X.mean(), X.std()

def standard_transform(X, media, std):
    """
    Aplica estandarización con media y desvío dado.
    Parámetros:
        - X: Serie o array a escalar.
        - media: Media a usar.
        - std: Desvío estándar a usar.
    Retorna:
        - Datos escalados.
    """
    return (X - media) / std

def train_test_split(X, y, test_size=0.2):
    """
    Divide el dataset en entrenamiento y prueba.

    Parámetros:
        - X: Features.
        - y: Etiquetas.
        - test_size: Proporción del conjunto de prueba.

    Retorna:
        - X_train, y_train, X_test, y_test.
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split_index = int(n_samples * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]