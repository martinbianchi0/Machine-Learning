import numpy as np
import pandas as pd

class LinearDiscriminantAnalysis:
    def __init__(self, X, y, fit=True):
        self.Xtrain = X if isinstance(X, np.ndarray) else X.to_numpy()
        self.ytrain = y if isinstance(y, np.ndarray) else y.to_numpy()
        self.classes = None
        self.coef = None
        self.intercepts = None
        if fit:
            self.fit()

    def fit(self):
        self.classes = np.unique(self.ytrain)
        self.n_classes = len(self.classes)
        n_features = self.Xtrain.shape[1]
        
        means = np.array([np.mean(self.Xtrain[self.ytrain == cls], axis=0) for cls in self.classes])

        cov = np.zeros((n_features, n_features))
        for cls, mean in zip(self.classes, means):
            X_cls = self.Xtrain[self.ytrain == cls]
            cov += (X_cls - mean).T @ (X_cls - mean)
        cov /= len(self.Xtrain) - self.n_classes 

        cov_inv = np.linalg.pinv(cov)  

        self.coef = cov_inv @ means.T 

        # Priors por clase
        priors = np.array([np.mean(self.ytrain == cls) for cls in self.classes])

        self.intercepts = []
        for i in range(self.n_classes):
            mean_i = means[i]
            term = -0.5 * mean_i @ cov_inv @ mean_i + np.log(priors[i])
            self.intercepts.append(term)
        self.intercepts = np.array(self.intercepts)  


    def predict(self, X):
        scores = X @ self.coef + self.intercepts 
        preds = self.classes[np.argmax(scores, axis=1)]
        return preds
    
    def predict_proba(self, X):
        X = X if isinstance(X, np.ndarray) else X.to_numpy()
        scores = X @ self.coef + self.intercepts
        e_z = np.exp(scores - np.max(scores, axis=1, keepdims=True)) 
        probs = e_z / np.sum(e_z, axis=1, keepdims=True)
        return probs




def entropy(y):
    counts = np.bincount(y)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities + 1e-9)) 

def split_dataset(X, y, feature_index, threshold):
    left_mask = X[:, feature_index] <= threshold
    right_mask = X[:, feature_index] > threshold
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]


class TreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value 

    def is_leaf(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.root = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))
        parent_entropy = entropy(y)  # Optimización 1

        if (self.max_depth is not None and depth >= self.max_depth) or \
        num_samples < self.min_samples_split or \
        num_classes == 1:
            leaf_value = self._most_common_label(y)
            return TreeNode(value=leaf_value)

        best_gain = -1
        best_split = None

        for feature_index in range(num_features):
            col = X[:, feature_index]
            sorted_idx = np.argsort(col)
            col_sorted = col[sorted_idx]
            y_sorted = y[sorted_idx]

            thresholds = []
            for i in range(1, len(y_sorted)):
                if y_sorted[i] != y_sorted[i - 1]:  # Cambio de clase
                    t = (col_sorted[i] + col_sorted[i - 1]) / 2
                    thresholds.append(t)

            for threshold in thresholds:
                X_left, y_left, X_right, y_right = split_dataset(X, y, feature_index, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                gain = self._information_gain(parent_entropy, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    best_split = (feature_index, threshold, X_left, y_left, X_right, y_right)

        if best_gain == -1:
            return TreeNode(value=self._most_common_label(y))

        feature_index, threshold, X_left, y_left, X_right, y_right = best_split
        left = self._build_tree(X_left, y_left, depth + 1)
        right = self._build_tree(X_right, y_right, depth + 1)
        return TreeNode(feature_index, threshold, left, right)

        
    def _information_gain(self, parent_entropy, left, right):
        weight_left = len(left) / (len(left) + len(right))
        weight_right = len(right) / (len(left) + len(right))
        return parent_entropy - (weight_left * entropy(left) + weight_right * entropy(right))


    def _most_common_label(self, y):
        return np.bincount(y).argmax()

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _predict_sample(self, x, node):
        if node.is_leaf():
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)

class RandomForest:
    def __init__(self, n_trees=100, max_depth=None, min_samples_split=2):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float64)
        if isinstance(y, pd.Series):
            y = y.to_numpy(dtype=int)

        n_samples = X.shape[0]
        self.trees = []

        for _ in range(self.n_trees):
            # Bootstrap sampling
            indices = np.random.randint(0, n_samples, n_samples)
            X_sample = X[indices]
            y_sample = y[indices]

            # Entrenar árbol
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float64)

        # Cada árbol predice
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # Voto por mayoría
        return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=predictions)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy(dtype=np.float64)

        n_samples = X.shape[0]
        predictions = np.array([tree.predict(X) for tree in self.trees]) 

        # Obtener todas las clases posibles de todos los árboles
        all_classes = np.unique(np.concatenate([tree.classes_ for tree in self.trees]))
        class_to_index = {cls: i for i, cls in enumerate(all_classes)}
        n_classes = len(all_classes)

        proba = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            sample_votes = predictions[:, i]
            for vote in sample_votes:
                proba[i, class_to_index[vote]] += 1
            proba[i] /= self.n_trees

        return proba




class MulticlassLogisticRegression:
    def __init__(self, X, y, lr=0.01, L2=0.0, fit=True):
        self.X = np.column_stack((np.ones(X.shape[0]), X.to_numpy() if isinstance(X, pd.DataFrame) else X))
        self.y = y if isinstance(y, np.ndarray) else y.to_numpy()
        self.classes = np.unique(self.y)
        self.n_classes = len(self.classes)
        self.lr = lr
        self.L2 = L2
        self.coef = np.zeros((self.X.shape[1], self.n_classes)) 
        if fit:
            self.fit()

    def softmax(self, z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True)) 
        return e_z / np.sum(e_z, axis=1, keepdims=True)

    def one_hot(self, y):
        onehot = np.zeros((len(y), self.n_classes))
        for i, val in enumerate(y):
            class_idx = np.where(self.classes == val)[0][0]
            onehot[i, class_idx] = 1
        return onehot

    def compute_gradient(self, y_onehot, probs):
        n = self.X.shape[0]
        grad = -1/n * self.X.T @ (y_onehot - probs)
        grad[1:] += 2 * self.L2 * self.coef[1:] 
        return grad

    def fit(self, tolerancia=1e-6, max_iter=1000):
        y_onehot = self.one_hot(self.y)
        for _ in range(max_iter):
            logits = self.X @ self.coef
            probs = self.softmax(logits)
            grad = self.compute_gradient(y_onehot, probs)
            if np.linalg.norm(grad) < tolerancia:
                break
            self.coef -= self.lr * grad

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = np.column_stack((np.ones(X.shape[0]), X))
        logits = X @ self.coef
        return self.softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        preds_idx = np.argmax(probs, axis=1)
        return self.classes[preds_idx]
