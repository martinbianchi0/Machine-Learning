import numpy as np

def train_test_split(X, y, test_size=0.2):
    """
    Divide el dataset en conjuntos de entrenamiento y prueba.

    Args:
        X: Features.
        y: Etiquetas.
        test_size: Proporción del conjunto de prueba.

    Returns:
        X_train, y_train, X_test, y_test
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    split_index = int(n_samples * (1 - test_size))
    train_indices = indices[:split_index]
    test_indices = indices[split_index:]
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]


def one_hot(Y, num_classes):
    """
    Codifica las etiquetas como vectores one-hot.

    Args:
        Y: Etiquetas (valores enteros).
        num_classes: Cantidad total de clases.

    Returns:
        Matriz one-hot de shape (num_classes, m)
    """
    m = Y.shape[0]
    one_hot_Y = np.zeros((num_classes, m))
    one_hot_Y[Y, np.arange(m)] = 1
    return one_hot_Y