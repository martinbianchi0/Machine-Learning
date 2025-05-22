from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch


def pca_transform_reconstruct(X, k):
    """
    Aplica PCA y reconstruye los datos con k componentes.

    Parámetros:
      - X: Datos originales (n x d).
      - k: Número de componentes principales.

    Returns:
      - Z: Representación reducida (n x k).
      - X_reconstructed: Datos reconstruidos (n x d).
    """
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    W = Vt[:k, :]         # (k x d)
    Z = X_centered @ W.T  # (n x k)
    X_reconstructed = Z @ W + X_mean  # (n x d)

    return Z, X_reconstructed

class VAE(nn.Module):
    """
    Autoencoder Variacional totalmente conectado.
    
    Inicializa un VAE con encoder y decoder simétricos.
    """
    def __init__(self, input_dim=784, hidden_dims=[400], latent_dim=20, lr=1e-3, L2=0.0):
        super().__init__()
        self.lr = lr
        self.L2 = L2

        # Encoder dinámico
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        self.mu_layer = nn.Linear(hidden_dims[-1], latent_dim)
        self.logvar_layer = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder simétrico
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*decoder_layers)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.L2)

    def encode(self, x):
        """
        Codifica la entrada y devuelve media y log-varianza.

        Parámetros:
          - x: Tensor de entrada.

        Returns:
          - mu: Media del espacio latente.
          - logvar: Log-varianza del espacio latente.
        """
        h = self.encoder(x)
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Aplica el truco de reparametrización.

        Parámetros:
          - mu: Media.
          - logvar: Log-varianza.

        Returns:
          - z: Vector muestrado del espacio latente.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decodifica un vector latente a la reconstrucción.
        Parámetros: - z: Vector latente.
        Returns: - Reconstrucción del dato original.
        """
        return self.decoder(z)

    def forward(self, x):
        """
        Pasa el dato por encoder, sampling y decoder.

        Parámetros:
        - x: Tensor de entrada.

        Returns:
        - x_recon: Reconstrucción.
        - mu: Media latente.
        - logvar: Log-varianza latente.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss_function(self, x_recon, x, mu, logvar):
        """
        Calcula la pérdida total (reconstrucción + KL).

        Parámetros:
        - x_recon: Salida del decoder.
        - x: Entrada original.
        - mu, logvar: Parámetros del espacio latente.

        Returns:
        - Pérdida total como escalar.
        """
        recon_loss = nn.functional.binary_cross_entropy(x_recon, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div

    def fit(self, X_train, X_val=None, epochs=50, batch_size=128, early_stopping=False, patience=10, verbose=True):
        """
        Entrena el VAE con (opcional) early stopping.

        Parámetros:
        - X_train: Datos de entrenamiento.
        - X_val: Datos de validación.
        - epochs: Cantidad de épocas.
        - batch_size: Tamaño del batch.
        - early_stopping: Si se usa early stopping.
        - patience: Épocas de espera sin mejora.
        - verbose: Si se imprime el progreso.
        """
        X_train = torch.tensor(X_train, dtype=torch.float32)
        train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)

        if X_val is not None:
            X_val = torch.tensor(X_val, dtype=torch.float32)
            val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size)

        best_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(epochs):
            self.train()
            total_loss = 0

            for batch in train_loader:
                x = batch[0]
                self.optimizer.zero_grad()
                x_recon, mu, logvar = self.forward(x)
                loss = self.loss_function(x_recon, x, mu, logvar)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {total_loss:.2f}", end="")

            if X_val is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = 0
                    for batch in val_loader:
                        x = batch[0]
                        x_recon, mu, logvar = self.forward(x)
                        loss = self.loss_function(x_recon, x, mu, logvar)
                        val_loss += loss.item()
                if verbose:
                    print(f" - Val Loss: {val_loss:.2f}")
                if early_stopping:
                    if val_loss < best_loss:
                        best_loss = val_loss
                        best_state = {k: v.clone() for k, v in self.state_dict().items()}
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose:
                                print("Early stopping.")
                            break
            else:
                if verbose:
                    print("")

        if early_stopping and best_state is not None:
            self.load_state_dict(best_state)

    def reconstruct(self, x):
        """
        Reconstruye los datos pasándolos por el VAE entrenado.

        Parámetros:
        - x: Datos a reconstruir.

        Returns:
        - Reconstrucciones como arreglo numpy.
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float32)
            x_recon, _, _ = self.forward(x)
        return x_recon.numpy()