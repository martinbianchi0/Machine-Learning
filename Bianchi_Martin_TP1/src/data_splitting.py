import numpy as np
from utils import load_model
from metrics import mse

def train_val_split(dx, dy, split=0.8):
    """Divide dx y dy en conjuntos de entrenamiento y validación.  
    Params: dx (DataFrame), dy (Series), split (float). Returns: (dx_train, dx_val, dy_train, dy_val).
    """
    n = len(dx)
    n = int(n * split)
    train_dx = dx[:n]
    val_dx = dx[n:]
    train_dy = dy[:n]
    val_dy = dy[n:]
    return train_dx, val_dx, train_dy, val_dy

def cross_val(x_train, y_train, l2, k=5):
    """Divide x_train en k conjuntos de entrenamiento y validación.  
    Params: x_train (DataFrame), y_train (Series), l2 (float), k (int).  
    Return: Promedio de MSE (float)."
    """
    n = len(x_train)
    indices = np.arange(n)
    np.random.shuffle(indices)  # Mezclar los datos para asegurar variedad

    folds = []
    for i in range(k):
        val_indices = indices[i::k]  # Selecciona cada k-ésimo elemento
        train_indices = np.setdiff1d(indices, val_indices)  # Resto es entrenamiento
        
        xval_fold = x_train.iloc[val_indices]
        xtrain_fold = x_train.iloc[train_indices]
        yval_fold = y_train.iloc[val_indices]
        ytrain_fold = y_train.iloc[train_indices]

        
        folds.append((xtrain_fold, ytrain_fold, xval_fold, yval_fold))

    mses = []
    for xtrain, ytrain, xval, yval in folds:
        model = load_model(xtrain, ytrain, "gd", L2=l2)
        mses.append(mse(xval, yval, model.obtener_coeficientes()))
    
    return np.mean(mses)
