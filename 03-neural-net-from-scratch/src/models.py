import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch
import copy

class RedNeuronal:
    """
    Red neuronal multicapa con entrenamiento por gradiente descendente o Adam.

    Args:
        input_size (int): Tamaño de la capa de entrada.
        hidden_layers (list): Lista con el número de neuronas por capa oculta.
        output_size (int): Tamaño de la capa de salida.
        lr (float): Tasa de aprendizaje.
        L2 (float): Coeficiente de regularización L2.
        solver (str): Optimizador ('gd' o 'adam').
        beta1 (float): Parámetro beta1 para Adam.
        beta2 (float): Parámetro beta2 para Adam.
        epsilon (float): Término de estabilidad numérica para Adam.
        dropout_rate (float): Tasa de dropout.
    """
    def __init__(self, input_size, hidden_layers, output_size, lr=0.01, L2=0, solver='gd',
                 beta1=0.9, beta2=0.999, epsilon=1e-8, dropout_rate=0.0):
        self.L = len(hidden_layers)
        self.pesos = []
        self.sesgos = []
        self.lr = lr
        self.L2 = L2
        self.solver = solver
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.dropout_rate = dropout_rate
        self.m_W = []
        self.v_W = []
        self.m_b = []
        self.v_b = []
        capas = [input_size] + hidden_layers + [output_size]

        for i in range(len(capas) - 1):
            W = np.random.randn(capas[i+1], capas[i]) * np.sqrt(2 / capas[i])
            b = np.zeros((capas[i+1], 1))
            self.pesos.append(W)
            self.sesgos.append(b)
            self.m_W.append(np.zeros_like(W))
            self.v_W.append(np.zeros_like(W))
            self.m_b.append(np.zeros_like(b))
            self.v_b.append(np.zeros_like(b))

    def relu(self, z):
        """
        Aplica la función ReLU.
        Args:
            z (ndarray): Entrada.
        Returns:
            ndarray: Salida de ReLU.
        """
        return np.maximum(0, z)

    def relu_deriv(self, z):
        """
        Derivada de la función ReLU.
        Args:
            z (ndarray): Entrada.
        Returns:
            ndarray: Derivada de ReLU.
        """
        return (z > 0).astype(float)

    def softmax(self, z):
        """
        Aplica softmax por columnas.
        Args:
            z (ndarray): Logits.
        Returns:
            ndarray: Distribución de probabilidad.
        """
        e_z = np.exp(z - np.max(z, axis=0, keepdims=True))
        return e_z / np.sum(e_z, axis=0, keepdims=True)

    def forward(self, X, training=True):
        """
        Propagación hacia adelante.

        Args:
            X (ndarray): Entradas (shape: features x muestras).
            training (bool): Si es True, aplica dropout.

        Returns:
            ndarray: Salidas predichas (probabilidades).
        """
        A = X
        self.As = [A]
        self.Zs = []
        self.dropout_masks = []

        for i in range(self.L):
            Z = np.dot(self.pesos[i], A) + self.sesgos[i]
            A = self.relu(Z)
            self.Zs.append(Z)

            if training and self.dropout_rate > 0:
                mask = (np.random.rand(*A.shape) > self.dropout_rate).astype(float)
                A *= mask
                A /= (1.0 - self.dropout_rate)
                self.dropout_masks.append(mask)
            else:
                self.dropout_masks.append(np.ones_like(A))

            self.As.append(A)

        Z = np.dot(self.pesos[-1], A) + self.sesgos[-1]
        A = self.softmax(Z)
        self.Zs.append(Z)
        self.As.append(A)
        return A

    def predict(self, X):
        """
        Predice clases para nuevas entradas.
        Args:
            X (ndarray): Entradas.
        Returns:
            ndarray: Índices de clases predichas.
        """
        Y_hat = self.forward(X, training=False)
        return np.argmax(Y_hat, axis=0)

    def compute_loss(self, Y_hat, Y):
        """
        Calcula la pérdida cross-entropy con regularización L2.

        Args:
            Y_hat (ndarray): Probabilidades predichas.
            Y (ndarray): One-hot labels reales.

        Returns:
            float: Pérdida total promedio.
        """
        m = Y.shape[1]
        loss = -np.sum(Y * np.log(Y_hat + 1e-8)) / m
        loss += self.L2 * np.sum([np.sum(W**2) for W in self.pesos]) / (2 * m)
        return loss

    def backward(self, X, Y):
        """
        Propagación hacia atrás y actualización de pesos.

        Args:
            X (ndarray): Entradas.
            Y (ndarray): Etiquetas verdaderas (one-hot).
        """
        m = X.shape[1]
        grads_W = [None] * len(self.pesos)
        grads_b = [None] * len(self.sesgos)

        dZ = self.As[-1] - Y
        grads_W[-1] = np.dot(dZ, self.As[-2].T) / m
        grads_b[-1] = np.sum(dZ, axis=1, keepdims=True) / m

        for l in reversed(range(self.L)):
            dA = np.dot(self.pesos[l+1].T, dZ)
            dA *= self.dropout_masks[l]  # aplica máscara
            dZ = dA * self.relu_deriv(self.Zs[l])
            grads_W[l] = np.dot(dZ, self.As[l].T) / m
            grads_b[l] = np.sum(dZ, axis=1, keepdims=True) / m

        if self.solver == 'adam':
            self.t += 1
            for i in range(len(self.pesos)):
                self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * grads_W[i]
                self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (grads_W[i] ** 2)
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * grads_b[i]
                self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (grads_b[i] ** 2)

                m_W_corr = self.m_W[i] / (1 - self.beta1 ** self.t)
                v_W_corr = self.v_W[i] / (1 - self.beta2 ** self.t)
                m_b_corr = self.m_b[i] / (1 - self.beta1 ** self.t)
                v_b_corr = self.v_b[i] / (1 - self.beta2 ** self.t)

                self.pesos[i] -= self.lr * m_W_corr / (np.sqrt(v_W_corr) + self.epsilon)
                self.sesgos[i] -= self.lr * m_b_corr / (np.sqrt(v_b_corr) + self.epsilon)
        else:
            for i in range(len(self.pesos)):
                self.pesos[i] -= self.lr * grads_W[i]
                self.sesgos[i] -= self.lr * grads_b[i]

    def train(self, X, Y, epochs=500, batch_size=64, X_val=None, Y_val=None,
             early_stopping=False, patience=10, lr_schedule=None, show_progress=False):
        """
        Entrena la red neuronal.

        Args:
            X (ndarray): Datos de entrenamiento.
            Y (ndarray): Etiquetas (one-hot).
            epochs (int): Número de épocas.
            batch_size (int): Tamaño del mini-batch.
            X_val (ndarray): Conjunto de validación (opcional).
            Y_val (ndarray): Etiquetas de validación.
            early_stopping (bool): Si usar parada temprana.
            patience (int): Épocas de espera para early stopping.
            lr_schedule (str): 'exp' o 'lin' para ajuste de tasa de aprendizaje.
            show_progress (bool): Si mostrar gráficos o imprimir pérdidas.
        """
        m = X.shape[1]
        if self.solver == 'gd':
            batch_size = m

        best_loss = float('inf')
        patience_counter = 0

        initial_lr = self.lr
        decay_rate = 0.95  # para decaimiento exponencial
        linear_decay = self.lr / epochs  # para decaimiento lineal
        min_lr = 1e-5  # mínimo para evitar que lr llegue a cero

        best_pesos = copy.deepcopy(self.pesos)
        best_sesgos = copy.deepcopy(self.sesgos)

        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Actualización de tasa de aprendizaje
            if lr_schedule == 'exp':
                self.lr = initial_lr * (decay_rate ** epoch)
            elif lr_schedule == 'lin':
                self.lr = max(initial_lr - linear_decay * epoch, min_lr)

            # Barajar los datos
            indices = np.random.permutation(m)
            X_shuffled = X[:, indices]
            Y_shuffled = Y[:, indices]

            # Mini-batches
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[:, i:i+batch_size]
                Y_batch = Y_shuffled[:, i:i+batch_size]

                Y_hat = self.forward(X_batch)
                self.backward(X_batch, Y_batch)

            # Pérdida en todo el conjunto de entrenamiento
            Y_hat_full = self.forward(X)
            loss = self.compute_loss(Y_hat_full, Y)
            train_losses.append(loss)

            # Validación
            if X_val is not None and Y_val is not None:
                Y_val_hat = self.forward(X_val)
                val_loss = self.compute_loss(Y_val_hat, Y_val)
                val_losses.append(val_loss)

                if early_stopping:
                    if val_loss < best_loss - 1e-4:
                        best_loss = val_loss
                        patience_counter = 0
                        best_pesos = copy.deepcopy(self.pesos)
                        best_sesgos = copy.deepcopy(self.sesgos)
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if show_progress:
                                print(f"Early stopping at epoch {epoch}, val_loss: {val_loss:.4f}")
                                plt.plot(train_losses, label="Train Loss")
                                plt.plot(val_losses, label="Val Loss")
                                plt.xlabel("Epoch")
                                plt.ylabel("Loss")
                                plt.title("Train vs Val Loss")
                                plt.legend()
                                plt.show()
                            self.pesos = best_pesos
                            self.sesgos = best_sesgos
                            return

                if epoch % 50 == 0 and show_progress:
                    print(f"Epoch {epoch}, Train Loss: {loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if epoch % 50 == 0 and show_progress:
                    print(f"Epoch {epoch}, Train Loss: {loss:.4f}")

        # Mostrar gráfico final si corresponde
        if show_progress and (X_val is not None and Y_val is not None):
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Train vs Val Loss")
            plt.legend()
            plt.show()

class TorchNeuralNet(nn.Module):
    """
    Red neuronal construida con PyTorch.

    Args:
        input_dim (int): Dimensión de entrada.
        hidden_layers (list): Neuronas por capa oculta.
        output_dim (int): Dimensión de salida.
        lr (float): Tasa de aprendizaje.
        L2 (float): Peso de regularización L2.
    """
    def __init__(self, input_dim, hidden_layers, output_dim, lr=1.0, L2=0.0):
        super().__init__()
        self.lr = lr
        self.L2 = L2

        layers = []
        in_dim = input_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # Inicialización personalizada (como la de tu red numpy)
        self.network.apply(self.init_weights_he_normal)

    def init_weights_he_normal(self, layer):
        """
        Inicializa pesos con He normal para capas lineales.

        Args:
            layer (nn.Module): Capa a inicializar.
        """
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')  # Igual a tu inicialización
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Propagación hacia adelante.
        Args:
            x (Tensor): Entrada.
        Returns:
            Tensor: Salida.
        """
        return self.network(x)
    
    def predict(self, X):
        """
        Predice clases desde un array o tensor.

        Args:
            X (array o Tensor): Datos de entrada.

        Returns:
            ndarray: Predicciones (clase más probable).
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        self.eval()
        with torch.no_grad():
            logits = self(X)
            return torch.argmax(logits, dim=1).numpy()

    def train_model(self, X_train, y_train, epochs, batch_size=64,
                X_val=None, Y_val=None, early_stopping=False,
                patience=20, show_progress=False):
        """
        Entrena la red con validación y early stopping opcional.

        Args:
            X_train (array): Datos de entrenamiento.
            y_train (array): Etiquetas de entrenamiento.
            epochs (int): Número de épocas.
            batch_size (int): Tamaño del batch.
            X_val (Tensor): Datos de validación (opcional).
            Y_val (Tensor): Etiquetas de validación.
            early_stopping (bool): Si se usa early stopping.
            patience (int): Épocas de paciencia.
            show_progress (bool): Si imprimir progreso.

        Returns:
            tuple: (pérdida final de entrenamiento, pérdida de validación)
        """
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)

        if X_val is not None:
            X_val = X_val.float()
            Y_val = Y_val.long()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.L2)

        best_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            perm = torch.randperm(X_train.size(0))
            for i in range(0, X_train.size(0), batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = X_train[idx], y_train[idx]

                optimizer.zero_grad()
                out = self(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

            if X_val is not None:
                self.eval()
                with torch.no_grad():
                    val_out = self(X_val)
                    val_loss = criterion(val_out, Y_val).item()

                if show_progress and epoch % 50 == 0:
                    print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}")

                if early_stopping:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_state = {k: v.clone() for k, v in self.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if show_progress:
                                print("Early stopping.")
                            break

        if early_stopping and best_state is not None:
            self.load_state_dict(best_state)

        # Calcular pérdida final con mejor modelo (ya cargado)
        self.eval()
        with torch.no_grad():
            train_out = self(X_train)
            final_train_loss = criterion(train_out, y_train).item()

            final_val_loss = None
            if X_val is not None:
                val_out = self(X_val)
                final_val_loss = criterion(val_out, Y_val).item()

        return final_train_loss, final_val_loss