import numpy as np

def silhouette_manual(X, labels):
    """
    Calcula el `silhouette score` excluyendo puntos con label -1.

    Parámetros:
      - X: Datos (2D array).
      - labels: Etiquetas de clustering (1D array).

    Returns:
      - Promedio del silhouette score (float), ignorando ruido.
    """
    X = np.array(X)
    labels = np.array(labels)
    mask = labels != -1
    X = X[mask]
    labels = labels[mask]

    if len(np.unique(labels)) < 2:
        return np.nan  # No tiene sentido con <2 clusters

    s = []
    for i in range(len(X)):
        xi = X[i]
        li = labels[i]

        # Distancia intra-cluster
        same_cluster = (labels == li)
        if np.sum(same_cluster) <= 1:
            a = 0
        else:
            a = np.mean(np.linalg.norm(X[same_cluster] - xi, axis=1)[1:])

        # Distancia al siguiente cluster más cercano
        b = np.inf
        for l in np.unique(labels):
            if l == li:
                continue
            cluster_l = (labels == l)
            dist = np.mean(np.linalg.norm(X[cluster_l] - xi, axis=1))
            b = min(b, dist)

        s_i = (b - a) / max(a, b) if max(a, b) > 0 else 0
        s.append(s_i)

    return np.mean(s)

def mse(X, X_rec):
    """
    Error cuadrático medio entre datos originales y reconstruidos.

    Parámetros:
      - X: Datos originales.
      - X_rec: Datos reconstruidos.

    Returns:
      - MSE como float.
    """
    return np.mean((X - X_rec) ** 2)

def final_inertia(model):
    """
    Devuelve la última `inercia` de un modelo KMeans.

    Parámetros:
      - model: Objeto KMeans entrenado, con historial de inercia.

    Returns:
      - Valor final de inercia (float).
    """
    return model.inertia[np.nonzero(model.inertia)][-1]

def final_loglikelihood(model):
    """
    Devuelve la última `log-verosimilitud` de un modelo GMM.

    Parámetros:
      - model: Objeto GMM entrenado, con historial de log-likelihoods.

    Returns:
      - Último valor de log-likelihood (float).
    """
    return model.log_likelihoods[-1]