from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import pandas as pd
import numpy as np

def view_data(df):
    """Imprime un resumen del dataset combinado: muestra aleatoria, rango de valores,
    columnas con nulos y cantidad de duplicados.

    Parámetros:
        - df (DataFrame): conjunto de datos a explorar."""
    print("Fragmento aleatorio de muestras")
    display(df.sample(7))
    print("\nRango de valores de cada columna")
    display(df.describe().loc[['min', 'max']])
    print("\nCategorías con valores faltantes\n", df.isna().sum()[df.isna().sum() > 0].to_string())
    print("\nFilas duplicadas:", df.duplicated().sum())

def plot_histograms(data):
    """
    Muestra histogramas de las columnas 'A' y 'B'.

    Parámetros:
        - data: DataFrame con las columnas 'A' y 'B'.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))  # 1 fila, 2 columnas

    axes[0].hist(data['A'], bins=30, edgecolor='black')
    axes[0].set_title("Histograma de A")
    axes[0].set_xlabel("Valor")
    axes[0].set_ylabel("Frecuencia")

    axes[1].hist(data['B'], bins=30, edgecolor='black')
    axes[1].set_title("Histograma de B")
    axes[1].set_xlabel("Valor")
    axes[1].set_ylabel("Frecuencia")

    plt.tight_layout()
    plt.show()

def plot_data(data):
    """
    Grafica los datos sin clasificar en 2D.

    Parámetros:
        - data: DataFrame o array con dos columnas.
    """
    plt.figure(figsize=(8, 6))
    data_np = data.to_numpy()
    plt.scatter(data_np[:, 0], data_np[:, 1], c='gray', alpha=0.7, label="Datos originales")

    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_elbow_method(Ls, model_type='kmeans', max_k=20):
    """
    Aplica el método del codo para elegir K en clustering.

    Parámetros:
        - Ls: Lista de métricas (inercia o log-verosimilitud).
        - model_type: 'kmeans' o 'gmm'.
        - max_k: Máximo valor de K evaluado.
    """
    if model_type == 'kmeans':
        ylabel = "Suma de distancias (L)"
    elif model_type == 'gmm':
        ylabel = "Log-verosimilitud"
    else:
        raise ValueError("Tipo de modelo no reconocido: usa 'kmeans' o 'gmm'")

    Ks = list(range(1, max_k + 1))
    plt.plot(Ks, Ls, marker='o')
    plt.xlabel("Cantidad de clusters (K)")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

    # Mostrar tabla
    tabla = pd.DataFrame({'K': Ks, 'L': Ls})
    tabla['K'] = tabla['K'].astype(int)
    tabla['L'] = tabla['L'].round(2)
    tabla_horizontal = tabla.T
    tabla_horizontal.columns = [f'Cluster {i+1}' for i in range(len(tabla_horizontal.columns))]
    display(tabla_horizontal)


def plot_metric_dbscan(tabla):
    """
    Grafica heatmap del `silhouette score` para combinaciones de DBSCAN.

    Parámetros:
      - tabla: DataFrame con scores, indexado por `eps` y `min_samples`.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(tabla, annot=True, fmt=".2f", cmap="YlOrRd", cbar_kws={"label": "Silhouette Score"})
    plt.xlabel("min_samples (k)")
    plt.ylabel("eps (ε)")
    plt.title("Silhouette Score por configuración DBSCAN")
    plt.tight_layout()
    plt.show()

def plot_clusters(data, model, model_type='kmeans'):
    """
    Grafica los clusters detectados por un modelo.

    Parámetros:
        - data: Array 2D con los datos.
        - model: Modelo entrenado (KMeans, GMM o DBSCAN).
        - model_type: Tipo de modelo ('kmeans', 'gmm' o 'dbscan').
    """
    if model_type == 'kmeans':
        labels = model.labels
        centroids = model.centroids
        plot_centroids = True
    elif model_type == 'gmm':
        labels = model.labels
        centroids = model.means
        plot_centroids = True
    elif model_type == 'dbscan':
        labels = model.labels
        centroids = None
        plot_centroids = False
    else:
        raise ValueError("Tipo de modelo no reconocido: usa 'kmeans', 'gmm' o 'dbscan'")

    k = len(set(labels)) - (1 if -1 in labels else 0)  # excluye ruido (-1)
    plt.figure(figsize=(8, 6))

    unique_labels = np.unique(labels)
    colors = sns.color_palette("tab20", len(unique_labels))

    for i, cluster in enumerate(unique_labels):
        mask = labels == cluster
        label_name = "Ruido" if cluster == -1 else f"{cluster + 1}"
        color = 'k' if cluster == -1 else colors[i]
        plt.scatter(data[mask, 0], data[mask, 1], label=label_name, alpha=0.7, c=[color])

    if plot_centroids and centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='black', marker='X', label='Centroides')

    plt.xlabel("Eje X")
    plt.ylabel("Eje Y")
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.show()

def plot_pca_errors(ks, errors):
    """
    Muestra el error de reconstrucción según la cantidad de componentes.

    Parámetros:
        - ks: Lista de componentes principales.
        - errors: Errores de reconstrucción (MSE) para cada k.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(ks, errors, marker='o')
    plt.xlabel('Cantidad de componentes principales (k)')
    plt.ylabel('Error cuadrático medio (MSE)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_reconstructions(X, X_reconstructed, titles=["Original", "Reconstruida"]):
    """
    Muestra comparaciones entre imágenes originales y reconstruidas.
    Incluye una grilla principal con 10 imágenes y una comparación destacada al costado.

    Parámetros:
        - X: Imágenes originales.
        - X_reconstructed: Imágenes reconstruidas.
        - titles: Títulos para cada fila (original y reconstruida).
    """
    n = 10
    fig = plt.figure(figsize=(14, 4))
    gs = gridspec.GridSpec(2, n + 3, width_ratios=[1]*n + [0.3, 0.3, 1.5], wspace=0.1, hspace=0.05)

    for i in range(n):
        ax_orig = plt.subplot(gs[0, i])
        ax_recon = plt.subplot(gs[1, i])
        
        ax_orig.imshow(X[i].reshape(28, 28), cmap='gray')
        ax_recon.imshow(X_reconstructed[i].reshape(28, 28), cmap='gray')
        
        ax_orig.axis('off')
        ax_recon.axis('off')
        
        if i == 0:
            ax_orig.set_title(titles[0])
            ax_recon.set_title(titles[1])

    # Comparación destacada al costado (índice 2)
    ax_big_orig = plt.subplot(gs[0, -1])
    ax_big_recon = plt.subplot(gs[1, -1])
    
    ax_big_orig.imshow(X[2].reshape(28, 28), cmap='gray')
    ax_big_recon.imshow(X_reconstructed[2].reshape(28, 28), cmap='gray')
    
    ax_big_orig.set_title(titles[0], fontsize=10)
    ax_big_recon.set_title(titles[1], fontsize=10)

    ax_big_orig.axis('off')
    ax_big_recon.axis('off')

    plt.tight_layout()
    plt.show()

def plot_mse_vs_latent(latent_dims, mses):
    """
    Grafica un histograma del MSE de validación en función del tamaño del espacio latente.

    Parámetros:
        - latent_dims: Lista de dimensiones latentes evaluadas.
        - mses: Lista de errores MSE correspondientes a cada configuración.
    """
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 4))
    plt.bar(latent_dims, mses, width=1.5, color='skyblue', edgecolor='black')
    plt.xlabel("Dimensión del espacio latente")
    plt.ylabel("MSE en validación")
    plt.title("Error de reconstrucción vs. dimensión latente")
    plt.xticks(sorted(set(latent_dims)))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
