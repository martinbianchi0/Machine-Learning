import numpy as np
import pandas as pd
from IPython.display import display
    
class LogisticRegression():
    def __init__(self, X, y, lr=0.01, threshold=0.5, L2=0, fit=True, weights=None):
        """Inicializa el modelo con los datos 'X' (features), 'y' (target), y los coeficientes de regularización."""
        self.features = ['bias'] + list(X.columns)
        self.X = np.column_stack((np.ones(X.shape[0]), X if isinstance(X, np.ndarray) else X.to_numpy()))
        self.y = y if isinstance(y, np.ndarray) else y.to_numpy()
        self.coef = np.zeros(self.X.shape[1])
        self.lr = lr
        self.threshold = threshold
        self.L2 = L2
        self.weights = weights
        if fit:
            self.fit()
    
    def sigmoid(self, z):
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


    def gradiente(self, b0):
        """Calcula el gradiente de la función de error logístico."""
        n = len(self.y)
        h = self.sigmoid(self.X @ b0)

        if self.weights is not None:
            # Cost re-weighting activo
            sample_weights = np.array([self.weights[y_i] for y_i in self.y])
            grad = -1/n * self.X.T @ ((self.y - h) * sample_weights)
        else:
            grad = -1/n * self.X.T @ (self.y - h)

        grad[1:] += 2 * self.L2 * b0[1:]  # solo regularizamos del índice 1 en adelante
        return grad

    
    def fit(self, tolerancia=1e-6, max_iter=1000):
        """Entrena el modelo usando el método de Gradiente Descendiente."""
        k = 0
        while np.linalg.norm(self.gradiente(self.coef)) > tolerancia and k < max_iter:
            self.coef -= self.lr * self.gradiente(self.coef)
            k += 1
    
    def predict_proba(self, X):
        """Realiza predicciones de probabilidad con el modelo entrenado."""
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if X.ndim == 1 or X.shape[1] == 1:
            X = X.reshape(-1, 1)
        if not np.all(X[:, 0] == 1):
            X = np.column_stack((np.ones(X.shape[0]), X))
            
        return self.sigmoid(X @ self.coef)

    def predict(self, X):
        """Realiza predicciones de clase con el modelo entrenado."""
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def show_hyperparameters(self, name="Hyperparameters"):
        """Muestra los hiperparámetros del modelo."""
        hyperparams = {
            'learning_rate': self.lr,
            'threshold': self.threshold,
            'L2': self.L2,
            'weights': str(self.weights)  
        }
        df = pd.DataFrame(hyperparams, index=[name])
        display(df)
    
