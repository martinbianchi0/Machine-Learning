from .metrics import accuracy, cross_entropy, matriz_de_confusion
from .models import TorchNeuralNet
import torch.nn as nn
from .models import RedNeuronal
import numpy as np
import torch
import time

def get_metrics(model, X_train, y_train, X_val, y_val, duration):
    """
    Calcula métricas del modelo, incluyendo exactitud y pérdida 
    para entrenamiento y validación.

    Args:
        model: Modelo entrenado.
        X_train, X_val: Features de entrenamiento y validación.
        y_train, y_val: Etiquetas de entrenamiento y validación.
        duration: Tiempo de entrenamiento.

    Returns:
        dict con tiempo, exactitud y pérdida.
    """
    y_train_true = np.argmax(y_train, axis=0)
    y_val_true = np.argmax(y_val, axis=0)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    Y_train_hat = model.forward(X_train, training=False)
    Y_val_hat = model.forward(X_val, training=False)

    acc_train = accuracy(y_train_true, y_train_pred)
    acc_val = accuracy(y_val_true, y_val_pred)

    loss_train = cross_entropy(y_train, Y_train_hat)
    loss_val = cross_entropy(y_val, Y_val_hat)

    return {
        'time': duration,
        'train_acc': acc_train,
        'val_acc': acc_val,
        'train_loss': loss_train,
        'val_loss': loss_val
    }

def evaluate_training_techniques(X_train, y_train, X_val, y_val):
    """
    Compara técnicas de entrenamiento sobre un dataset.

    Args:
        X_train, X_val: Features de entrenamiento y validación.
        y_train, y_val: Etiquetas de entrenamiento y validación.

    Returns:
        dict con métricas de cada técnica evaluada.
    """
    results = {}
    input_size = X_train.shape[0]
    hidden = [100, 80]
    output_size = y_train.shape[0]
    patience = 50

    # m0
    start = time.time()
    m0 = RedNeuronal(input_size, hidden, output_size, lr=1)
    m0.train(X_train, y_train, epochs=500)
    duration = time.time() - start
    results["m0"] = get_metrics(m0, X_train, y_train, X_val, y_val, duration)

    # ADAM
    start = time.time()
    adam = RedNeuronal(input_size, hidden, output_size, lr=0.001, solver='adam')
    adam.train(X_train, y_train, epochs=500, batch_size=64, X_val=X_val, Y_val=y_val, early_stopping=True, patience=patience)
    duration = time.time() - start
    results["adam"] = get_metrics(adam, X_train, y_train, X_val, y_val, duration)

    # Scheduler Exponencial
    start = time.time()
    exp = RedNeuronal(input_size, hidden, output_size, lr=2)
    exp.train(X_train, y_train, epochs=500, lr_schedule='exp', X_val=X_val, Y_val=y_val, early_stopping=True, patience=patience)
    duration = time.time() - start
    results["schedule_exp"] = get_metrics(exp, X_train, y_train, X_val, y_val, duration)

    # Scheduler Lineal
    start = time.time()
    lin = RedNeuronal(input_size, hidden, output_size, lr=1)
    lin.train(X_train, y_train, epochs=500, lr_schedule='lin', X_val=X_val, Y_val=y_val, early_stopping=True, patience=patience)
    duration = time.time() - start
    results["schedule_lin"] = get_metrics(lin, X_train, y_train, X_val, y_val, duration)

    # SGD puro
    start = time.time()
    sgd = RedNeuronal(input_size, hidden, output_size, lr=0.01, solver='sgd')
    sgd.train(X_train, y_train, epochs=500, batch_size=64, X_val=X_val, Y_val=y_val, early_stopping=True, patience=patience)
    duration = time.time() - start
    results["sgd"] = get_metrics(sgd, X_train, y_train, X_val, y_val, duration)

    # L2 Regularization
    start = time.time()
    l2 = RedNeuronal(input_size, hidden, output_size, lr=1, L2=0.01)
    l2.train(X_train, y_train, epochs=500, X_val=X_val, Y_val=y_val, early_stopping=True, patience=patience)
    duration = time.time() - start
    results["L2"] = get_metrics(l2, X_train, y_train, X_val, y_val, duration)

    # Dropout
    start = time.time()
    drop = RedNeuronal(input_size, hidden, output_size, lr=1, dropout_rate=0.2)
    drop.train(X_train, y_train, epochs=500, X_val=X_val, Y_val=y_val, early_stopping=True, patience=patience)
    duration = time.time() - start
    results["dropout"] = get_metrics(drop, X_train, y_train, X_val, y_val, duration)

    return results

def find_best_model(X_train, y_train, X_val, y_val):
    """
    Busca el modelo con los mejores hiperparámetros.

    Args:
        X_train, X_val: Features de entrenamiento y validación.
        y_train, y_val: Etiquetas de entrenamiento y validación.

    Returns:
        Mejor modelo y sus hiperparámetros.
    """
    lr_dict = {'adam': [0.001], 'sgd': [0.01], 'gd': [1]}
    lr_bs = {'adam': [64, 128], 'sgd': [64, 128], 'gd':[3000]}
    L2 = 0.01
    solvers = ['sgd', 'adam', 'gd']
    dropouts = [0, 0.2]
    hidden_cfgs = [[256], [100, 80], [256, 128], [128, 64, 32]]
    best_loss, best_model, best_hp = 10, None, None

    for h in hidden_cfgs:
        for s in solvers:
            for lr in lr_dict[s]:
                for bs in lr_bs[s]:
                    for d in dropouts:
                        model = RedNeuronal(X_train.shape[0], h, 49, lr=lr, L2=L2, solver=s, dropout_rate=d)
                        model.train(X_train, y_train, epochs=500, batch_size=bs,
                                    X_val=X_val, Y_val=y_val, early_stopping=True, patience=30)
                        Y_val_hat = model.forward(X_val)
                        loss = cross_entropy(y_val, Y_val_hat)
                        if loss < best_loss:
                            best_loss, best_model = loss, model
                            best_hp = {
                                'lr': lr, 'L2': L2, 'solver': s, 'dropout': d,
                                'hidden': h, 'batch_size': bs
                            }

    print(f"Mejor modelo (acc val={best_loss:.2f}):")
    print(best_hp)
    return best_model

def find_M3_model(X_train, y_train, X_val, y_val, hidden_layer_options,
                  input_dim=784, output_dim=49, epochs=500, batch_size=3000,
                  lr=0.01, L2=0.01, patience=10, show_progress=False):
    """
    Encuentra el modelo con mejor generalización.

    Args:
        X_train, X_val: Features de entrenamiento y validación.
        y_train, y_val: Etiquetas de entrenamiento y validación.
        hidden_layer_options: Configuraciones de capas ocultas.
        Otros: Parámetros de entrenamiento.

    Returns:
        Mejor modelo, configuración y pérdida de validación.
    """
    best_model = None
    best_config = None
    lowest_val_loss = float('inf')

    for hidden_layers in hidden_layer_options:
        
        model = TorchNeuralNet(input_dim, hidden_layers, output_dim, lr=lr, L2=L2)

        _, val_loss = model.train_model(
            X_train, y_train, epochs=epochs, batch_size=batch_size,
            X_val=X_val, Y_val=y_val, early_stopping=True,
            patience=patience, show_progress=show_progress
        )

        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = nn.CrossEntropyLoss()(val_logits, y_val).item()

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_model = model
            best_config = hidden_layers

    print(f"\n[M3] Mejor generalización: {best_config}, Val Loss = {lowest_val_loss:.4f}")
    return best_model, best_config, lowest_val_loss

def find_M4_model(X_train, y_train, hidden_layer_options,
                  input_dim=784, output_dim=49, epochs=500, batch_size=3000,
                  lr=0.01, show_progress=False):
    """
    Encuentra el modelo con menor pérdida de entrenamiento.

    Args:
        X_train: Features de entrenamiento.
        y_train: Etiquetas de entrenamiento.
        hidden_layer_options: Configuraciones de capas ocultas.
        Otros: Parámetros de entrenamiento.

    Returns:
        Mejor modelo, configuración y pérdida de entrenamiento.
    """
    best_model = None
    best_config = None
    lowest_train_loss = float('inf')

    for hidden_layers in hidden_layer_options:
        
        model = TorchNeuralNet(input_dim, hidden_layers, output_dim, lr=lr, L2=0.0)

        train_loss, _ = model.train_model(
            X_train, y_train, epochs=epochs, batch_size=batch_size,
            X_val=None, Y_val=None, early_stopping=False,
            patience=0, show_progress=show_progress
        )


        if train_loss < lowest_train_loss:
            lowest_train_loss = train_loss
            best_model = model
            best_config = hidden_layers

    print(f"\n[M4] Más overfitting: {best_config}, Train Loss = {lowest_train_loss:.4f}")
    return best_model, best_config, lowest_train_loss

def get_numpy_metrics(model, X_train, y_train, X_val, y_val):
    """
    Calcula métricas y matrices de confusión para un modelo en numpy.

    Args:
        model: Modelo entrenado.
        X_train, X_val: Features de entrenamiento y validación.
        y_train, y_val: Etiquetas de entrenamiento y validación.

    Returns:
        dict con métricas de exactitud, pérdida y matrices de confusión.
    """
    Y_train_hat = model.forward(X_train)
    Y_val_hat = model.forward(X_val)

    y_train_labels = np.argmax(y_train, axis=0)
    y_val_labels = np.argmax(y_val, axis=0)

    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    acc_train = accuracy(y_train_labels, y_train_pred)
    acc_val = accuracy(y_val_labels, y_val_pred)

    loss_train = cross_entropy(y_train, Y_train_hat)
    loss_val = cross_entropy(y_val, Y_val_hat)

    cm_train = matriz_de_confusion(y_train_labels, y_train_pred)
    cm_val = matriz_de_confusion(y_val_labels, y_val_pred)

    return {
        "acc_train": acc_train,
        "acc_val": acc_val,
        "loss_train": loss_train,
        "loss_val": loss_val,
        "cm_train": cm_train,
        "cm_val": cm_val
    }

def get_torch_metrics(model, X_train, y_train, X_val, y_val):
    """
    Calcula métricas y matrices de confusión para un modelo en PyTorch.

    Args:
        model: Modelo entrenado.
        X_train, X_val: Features de entrenamiento y validación.
        y_train, y_val: Etiquetas de entrenamiento y validación.

    Returns:
        dict con métricas de exactitud, pérdida y matrices de confusión.
    """
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = X_val.detach().clone().float()
    y_val_tensor = y_val.detach().clone().long()

    y_train_pred = model.predict(X_train_tensor)
    y_val_pred = model.predict(X_val_tensor)

    y_train_labels = y_train_tensor.cpu().numpy()
    y_val_labels = y_val_tensor.cpu().numpy()

    acc_train = accuracy(y_train_labels, y_train_pred)
    acc_val = accuracy(y_val_labels, y_val_pred)

    with torch.no_grad():
        Y_train_logits = model(X_train_tensor)
        Y_val_logits = model(X_val_tensor)

    criterion = nn.CrossEntropyLoss()
    loss_train = criterion(Y_train_logits, y_train_tensor.long()).item()
    loss_val = criterion(Y_val_logits, y_val_tensor.long()).item()

    cm_train = matriz_de_confusion(y_train_labels, y_train_pred)
    cm_val = matriz_de_confusion(y_val_labels, y_val_pred)

    return {
        'loss_train': loss_train,
        'loss_val': loss_val,
        'acc_train': acc_train,
        'acc_val': acc_val,
        'cm_train': cm_train,
        'cm_val': cm_val
    }