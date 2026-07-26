import numpy as np
import pandas as pd

class LinearRegression():
    def __init__(self, X, y, L1=0, L2=0):
        """Inicializa el modelo con los datos 'X' (features), 'y' (target), y los coeficientes de regularización."""
        self.features = ['bias'] + list(X.columns)
        self.X = np.column_stack((np.ones(X.shape[0]), X if isinstance(X, np.ndarray) else X.to_numpy()))
        self.y = y if isinstance(y, np.ndarray) else y.to_numpy()
        self.coef = np.zeros(self.X.shape[1])
        self.L1 = L1
        self.L2 = L2

    def gradiente(self, b0):
        """Calcula el gradiente de la función de error cuadrático medio (MSE)."""
        n = len(self.y)
        return 2/n * self.X.T @ ((self.X @ b0) - self.y) + self.L1 * np.sign(b0) + 2 * self.L2 * b0

    def gradiente_descendiente(self, learning_rate=0.01, tolerancia=1e-6, max_iter=1000):
        """Optimiza los coeficientes usando el método de Gradiente Descendiente."""
        k = 0
        while np.linalg.norm(self.gradiente(self.coef)) > tolerancia and k < max_iter:
            self.coef -= learning_rate * self.gradiente(self.coef)
            k += 1

    def pseudoinversa(self):
        """Calcula los coeficientes usando el metodo de la Pseudoinversa."""
        m, n = self.X.shape
        I = np.eye(n)
        I[0, 0] = 0
        self.coef = np.linalg.pinv(self.X.T @ self.X + self.L2 * I) @ self.X.T @ self.y

    def mostrar_coeficientes(self):
        """Imprime los coeficientes del modelo con los nombres de features."""
        print("Coeficientes del modelo:")
        for nombre, valor in zip(self.features, self.coef):
            print(f"{nombre}: {valor:.4f}")

    def obtener_coeficientes(self):
        """Devuelve los coeficientes del modelo."""
        return self.coef
    
    def predict(self, X):
        """Realiza predicciones con el modelo entrenado."""
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if X.ndim == 1 or X.shape[1] == 1:
            X = X.reshape(-1, 1)
        if not np.all(X[:, 0] == 1):
            X = np.column_stack((np.ones(X.shape[0]), X))
            
        return X @ self.coef