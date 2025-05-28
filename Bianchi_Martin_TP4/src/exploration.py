from .metrics import mse, silhouette_manual, final_inertia, final_loglikelihood
from .dreduction import VAE, pca_transform_reconstruct
from .clustering import kmeans, GMM, DBSCAN
from .visualization import plot_mse_vs_latent
import pandas as pd
import numpy as np

def explore_kmeans(data, max_k=20):
    """
    Evalúa KMeans para distintos valores de K, registrando la inercia final.

    Parámetros:
        - data: Datos de entrada (ndarray o DataFrame), sin etiquetas.
        - max_k: Máximo número de clusters a evaluar (int).

    Retorna:
        - Lista con la inercia final del modelo para cada valor de K (del 1 a max_k).
    """
    Ls = []
    for k in range(1, max_k + 1):
        modelo = kmeans(data, n_clusters=k)
        Ls.append(final_inertia(modelo))
    return Ls

def explore_gmm(data, max_k=20):
    """
    Evalúa GMM para distintos valores de K, registrando la log-verosimilitud final.
    Para inicializar los parámetros, se usa una corrida previa de KMeans.

    Parámetros:
        - data: Datos de entrada (ndarray o DataFrame), sin etiquetas.
        - max_k: Máximo número de componentes a evaluar (int).

    Retorna:
        - Lista con la log-verosimilitud final para cada valor de K (del 1 a max_k).
    """
    Ls = []
    for k in range(1, max_k + 1):
        model_kmeans = kmeans(data, n_clusters=k)
        modelo = GMM(data, n_components=k, means=model_kmeans.centroids, labels=model_kmeans.labels)
        Ls.append(final_loglikelihood(modelo))
    return Ls

def explore_dbscan(X, lista_eps, lista_min_samples, ruido_max=0.2):
    """
    Evalúa combinaciones de DBSCAN filtrando por ruido y cantidad de clusters.

    Parámetros:
      - X: Datos a clusterizar.
      - lista_eps: Lista de valores `eps` a probar.
      - lista_min_samples: Lista de valores `min_samples` a probar.
      - ruido_max: Máximo porcentaje de puntos ruido permitido (default 0.4).

    Returns:
      - DataFrame pivot con `silhouette score` para cada configuración.
    """
    resultados = []

    for eps in lista_eps:
        for min_samples in lista_min_samples:
            modelo = DBSCAN(X, eps=eps, min_samples=min_samples)
            etiquetas = np.array(modelo.labels)
            n_clusters = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
            ruido = np.mean(etiquetas == -1)

            if n_clusters >= 2 and ruido <= ruido_max:
                score = silhouette_manual(X, etiquetas)
            else:
                score = np.nan

            resultados.append((eps, min_samples, score))

    df = pd.DataFrame(resultados, columns=["eps", "min_samples", "silhouette"])
    tabla = df.pivot(index="eps", columns="min_samples", values="silhouette")
    return tabla

def explore_pca(X, max_k, step=1):
    """
    Evalúa PCA para distintos números de componentes principales.

    Parámetros:
        - X: Datos de entrada (ndarray).
        - max_k: Máximo número de componentes a evaluar.
        - step: Incremento entre valores de k.

    Retorna:
        - ks: Lista de componentes evaluadas.
        - errors: Lista de errores de reconstrucción (MSE) por cada k.
    """
    ks = []
    errors = []

    for k in range(1, max_k + 1, step):
        _, X_reconstructed = pca_transform_reconstruct(X, k)
        error = mse(X, X_reconstructed)
        ks.append(k)
        errors.append(error)

    return ks, errors

def explore_vaes(X_train, X_val, configs, epochs=30, early_stopping=True, plot_latent_mse=False):
    """
    Entrena varios modelos VAE y selecciona el que tiene menor MSE en validación.

    Parámetros:
        - X_train: Datos de entrenamiento (ndarray).
        - X_val: Datos de validación (ndarray).
        - configs: Lista de configuraciones VAE a evaluar.
        - epochs: Número de épocas de entrenamiento.
        - early_stopping: Si usar detención temprana.
        - plot_latent_mse: Si graficar MSE vs. dimensión latente al final.

    Retorna:
        - El mejor modelo VAE entrenado.
    """
    resultados = []
    latents = []
    mses = []

    for config in configs:
        vae = VAE(
            input_dim=784,
            hidden_dims=config["hidden_dims"],
            latent_dim=config["latent_dim"],
            lr=config["lr"],
            L2=config["L2"]
        )
        vae.fit(
            X_train, X_val,
            epochs=epochs,
            batch_size=config["batch_size"],
            early_stopping=early_stopping,
            patience=3,
            verbose=False
        )
        recon = vae.reconstruct(X_val)
        error = mse(X_val, recon)
        resultados.append((vae, config, error))
        latents.append(config["latent_dim"])
        mses.append(error)
        print(f"Capas: {config['hidden_dims']} | MSE validación: {error:.4f}")
    
    mejor = min(resultados, key=lambda x: x[2])
    print("\n=== Mejor modelo ===")
    print(f"Config: {mejor[1]}")
    print(f"Capas: {mejor[1]['hidden_dims']} | MSE validación: {mejor[2]:.4f}")

    if plot_latent_mse:
        plot_mse_vs_latent(latents, mses)

    return mejor[0]