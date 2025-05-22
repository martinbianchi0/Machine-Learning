import numpy as np

class kmeans():
    """
    Algoritmo K-Means con inicialización KMeans++ y selección entre 15 inicializaciones.

    Parámetros:
        - X (ndarray): Datos de entrada.
        - n_clusters (int): Número de clusters.
        - max_iter (int): Iteraciones máximas.
        - tol (float): Tolerancia para convergencia.
        - fit (bool): Si entrena el modelo al instanciar.

    Atributos:
        - centroids (ndarray): Coordenadas de centroides.
        - labels (ndarray): Etiquetas asignadas.
        - inertia (ndarray): Inercia por iteración (de la mejor corrida).
    """
    def __init__(self, X, n_clusters=3, max_iter=300, tol=1e-4, fit=True):
        X = np.array(X)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        if fit:
            self.fit(X)

    def init_kmeans_pp(self):
        """
        Inicializa centroides usando KMeans++.

        Returns:
            - ndarray: Centroides iniciales.
        """
        n_samples = self.X.shape[0]
        centroids = [self.X[np.random.randint(n_samples)]]
        for _ in range(1, self.n_clusters):
            distances = np.min(np.linalg.norm(self.X[:, np.newaxis] - np.array(centroids), axis=2)**2, axis=1)
            probs = distances / distances.sum()
            next_centroid = self.X[np.random.choice(n_samples, p=probs)]
            centroids.append(next_centroid)
        return np.array(centroids)

    def fit(self, X):
        """
        Ajusta el modelo K-Means a los datos, probando 15 inicializaciones.

        Parámetros:
            - X (ndarray): Datos 2D de entrada.
        """
        self.X = X
        self.n_samples, self.n_features = X.shape

        best_inertia = np.inf
        best_centroids = None
        best_labels = None
        best_inertia_list = None

        for _ in range(15):  # 15 inicializaciones
            centroids = self.init_kmeans_pp()
            labels = np.zeros(self.n_samples)
            inertia_list = []

            for _ in range(self.max_iter):
                distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
                labels = np.argmin(distances, axis=1)
                inertia = np.sum(np.min(distances, axis=1) ** 2)
                inertia_list.append(inertia)

                new_centroids = np.array([
                    X[labels == j].mean(axis=0) if np.any(labels == j) else centroids[j]
                    for j in range(self.n_clusters)
                ])
                if np.linalg.norm(new_centroids - centroids) < self.tol:
                    break
                centroids = new_centroids

            if inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels
                best_inertia_list = inertia_list

        self.centroids = best_centroids
        self.labels = best_labels
        self.inertia = np.array(best_inertia_list)

class GMM:
    """
    Modelo de mezcla gaussiana entrenado con EM.

    Parámetros:
        - X (ndarray): Datos de entrada.
        - n_components (int): Número de gaussianas.
        - max_iter (int): Iteraciones máximas.
        - tol (float): Tolerancia de convergencia.
        - means, labels (ndarray): Inicialización opcional.
        - fit (bool): Si ajusta al instanciar.

    Atributos:
        - means, covariances, weights (ndarray): Parámetros del modelo.
        - responsibilities (ndarray): Matriz de pertenencia.
        - labels (ndarray): Etiquetas asignadas.
        - log_likelihoods (list): Evolución de la log-verosimilitud.
    """
    def __init__(self, X, n_components=3, max_iter=100, tol=1e-4,
        means=None, labels=None, fit=True):
        X = np.array(X)
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.means = means
        self.labels = labels
        if fit:
            self.fit(X)


    def _gaussian(self, X, mean, cov):
        """
        Evalúa la gaussiana multivariada.

        Parámetros:
            - X (ndarray): Datos de entrada.
            - mean, cov (ndarray): Parámetros de la gaussiana.

        Returns:
            - ndarray: Probabilidades por muestra.
        """
        n = X.shape[1]
        diff = X - mean
        inv_cov = np.linalg.inv(cov)
        det_cov = np.linalg.det(cov)
        norm_const = 1.0 / (np.power(2 * np.pi, n / 2) * np.sqrt(det_cov))
        exponent = -0.5 * np.sum(diff @ inv_cov * diff, axis=1)
        return norm_const * np.exp(exponent)

    def initialize_parameters(self, X):
        """
        Inicializa medias, covarianzas y pesos.

        Parámetros:
            - X (ndarray): Datos de entrada.
        """
        n_samples, n_features = X.shape
        if self.means is not None and self.labels is not None:
            self.means = self.means
            self.covariances = np.zeros((self.n_components, n_features, n_features))
            self.weights = np.zeros(self.n_components)
            for k in range(self.n_components):
                cluster_data = X[self.labels == k]
                self.weights[k] = len(cluster_data) / n_samples
                self.covariances[k] = np.cov(cluster_data.T) + 1e-6 * np.eye(n_features)
        else:
            rng = np.random.default_rng()
            indices = rng.choice(n_samples, self.n_components, replace=False)
            self.means = X[indices]
            self.covariances = np.array([np.cov(X.T) + 1e-6 * np.eye(n_features) for _ in range(self.n_components)])
            self.weights = np.ones(self.n_components) / self.n_components
        self.responsibilities = np.zeros((n_samples, self.n_components))

    def e_step(self, X):
        """
        Calcula responsabilidades (E-step).

        Parámetros:
            - X (ndarray): Datos de entrada.
        """
        for k in range(self.n_components):
            self.responsibilities[:, k] = self.weights[k] * self._gaussian(X, self.means[k], self.covariances[k])
        self.responsibilities /= self.responsibilities.sum(axis=1, keepdims=True)

    def m_step(self, X):
        """
        Actualiza parámetros (M-step).

        Parámetros:
            - X (ndarray): Datos de entrada.
        """
        n_samples = X.shape[0]
        Nk = self.responsibilities.sum(axis=0)

        self.means = np.dot(self.responsibilities.T, X) / Nk[:, np.newaxis]

        for k in range(self.n_components):
            diff = X - self.means[k]
            weighted_cov = np.dot((self.responsibilities[:, k][:, np.newaxis] * diff).T, diff) / Nk[k]
            self.covariances[k] = weighted_cov + 1e-6 * np.eye(X.shape[1])  # evitar singularidad

        self.weights = Nk / n_samples

    def compute_log_likelihood(self, X):
        """
        Calcula la log-verosimilitud total.

        Parámetros:
            - X (ndarray): Datos de entrada.

        Returns:
            - float: Log-verosimilitud.
        """
        total_likelihood = np.zeros(X.shape[0])
        for k in range(self.n_components):
            total_likelihood += self.weights[k] * self._gaussian(X, self.means[k], self.covariances[k])
        return np.sum(np.log(total_likelihood))

    def fit(self, X):
        """
        Ejecuta EM para ajustar el modelo.

        Parámetros:
            - X (ndarray): Datos 2D de entrada.
        """
        X = np.array(X)
        self.initialize_parameters(X)
        self.log_likelihoods = []

        for i in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
            log_likelihood = self.compute_log_likelihood(X)
            self.log_likelihoods.append(log_likelihood)

            if i > 0 and np.abs(log_likelihood - self.log_likelihoods[-2]) < self.tol:
                break

        self.labels = np.argmax(self.responsibilities, axis=1)

class DBSCAN:
    """
    Algoritmo DBSCAN para clustering por densidad.

    Parámetros:
        - X (ndarray): Datos de entrada.
        - eps (float): Radio de vecindad.
        - min_samples (int): Vecinos mínimos por punto.
        - fit (bool): Si ajusta el modelo al instanciar.

    Atributos:
        - labels (ndarray): Etiquetas de cluster (-1 = ruido).
    """
    def __init__(self, X, eps=0.5, min_samples=5, fit=True):
        X = np.array(X)
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None
        if fit:
            self.fit(X)

    def region_query(self, X, point_idx):
        """
        Busca vecinos dentro de 'eps'.

        Parámetros:
            - X (ndarray): Datos de entrada.
            - point_idx (int): Índice del punto.

        Returns:
            - ndarray: Índices de vecinos.
        """
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]

    def expand_cluster(self, X, labels, point_idx, cluster_id):
        """
        Expande un cluster desde un punto semilla.

        Parámetros:
            - X (ndarray): Datos de entrada.
            - labels (ndarray): Etiquetas actuales.
            - point_idx (int): Punto de inicio.
            - cluster_id (int): ID del cluster.

        Returns:
            - bool: True si se formó un cluster.
        """
        neighbors = self.region_query(X, point_idx)
        if len(neighbors) < self.min_samples:
            labels[point_idx] = -1  # ruido
            return False

        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            neighbor_idx = neighbors[i]
            if labels[neighbor_idx] == -1:
                labels[neighbor_idx] = cluster_id
            elif labels[neighbor_idx] == -2:
                labels[neighbor_idx] = cluster_id
                new_neighbors = self.region_query(X, neighbor_idx)
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate((neighbors, new_neighbors))
            i += 1
        return True

    def fit(self, X):
        """
        Ejecuta DBSCAN sobre los datos.

        Parámetros:
            - X (ndarray): Datos 2D de entrada.
        """
        self.X = X
        n_samples = X.shape[0]
        self.labels = np.full(n_samples, -2)  # -2: no visitado
        cluster_id = 0

        for point_idx in range(n_samples):
            if self.labels[point_idx] != -2:
                continue
            if self.expand_cluster(X, self.labels, point_idx, cluster_id):
                cluster_id += 1

        self.labels = np.array(self.labels)