import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display
from .metrics import pr_auc, roc_auc, matriz_de_confusion, accuracy, precision, recall, f_score

def explore_data(df):
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

def plot_correlation_matrix(df):
    """Muestra un mapa de calor con la correlación entre variables numéricas.

    Parámetros:
    - df (DataFrame): dataset a analizar."""
    corr = df.select_dtypes(include='number').corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
    plt.title("Matriz de correlación")
    plt.tight_layout()
    plt.show()

def plot_boxplots(df, bin_cols=None):
    """Genera boxplots para columnas numéricas no binarias.

    Parámetros:
    - df (DataFrame): dataset a analizar.
    - bin_cols (list of str): columnas a excluir por ser binarias."""
    if bin_cols is None:
        bin_cols = []

    num_cols = [col for col in df.select_dtypes(include='number').columns if col not in bin_cols]
    n_cols = 4  # Ahora hay 4 columnas
    n_rows = (len(num_cols) + n_cols - 1) // n_cols  # Ajusta la cantidad de filas

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))  # Tamaño ajustado para 4 columnas
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.boxplot(y=df[col].reset_index(drop=True), ax=axes[i])
        axes[i].set_title(col)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_metrics_summary(y_true, y_scores, y_pred):
    """Grafica en una figura 2x2: curva PR, curva ROC, matriz de confusión y métricas.

    Parámetros:
    - y_true (array): etiquetas verdaderas.
    - y_scores (array): probabilidades predichas.
    - y_pred (array): etiquetas predichas.
    """

    recalls, precisions, auc_pr = pr_auc(y_true, y_scores)
    fpr, tpr, auc_roc = roc_auc(y_true, y_scores)
    cm = matriz_de_confusion(y_true, y_pred)

    # Calcular métricas
    metricas = {
        "Accuracy": round(accuracy(y_true, y_pred), 3),
        "Precision": round(precision(y_true, y_pred), 3),
        "Recall": round(recall(y_true, y_pred), 3),
        "F-Score": round(f_score(y_true, y_pred), 3)
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Curva PR
    axes[0, 0].plot(recalls, precisions, label=f'Curva PR (AUC={auc_pr:.4f})')
    axes[0, 0].fill_between(recalls, precisions, alpha=0.3)
    axes[0, 0].set_xlabel('Recall')
    axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Curva de Precisión-Recall')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Curva ROC
    axes[0, 1].plot(fpr, tpr, label=f'Curva ROC (AUC={auc_roc:.4f})')
    axes[0, 1].fill_between(fpr, tpr, alpha=0.3)
    axes[0, 1].set_xlabel('Tasa de Falsos Positivos')
    axes[0, 1].set_ylabel('Tasa de Verdaderos Positivos')
    axes[0, 1].set_title('Curva ROC')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Matriz de confusión
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['0', '1'], yticklabels=['0', '1'], ax=axes[1, 0])
    axes[1, 0].set_xlabel('Predicción')
    axes[1, 0].set_ylabel('Realidad')
    axes[1, 0].set_title('Matriz de Confusión')

    # Métricas en tabla
    axes[1, 1].axis('off')
    table_data = [[k, v] for k, v in metricas.items()]
    # Tabla con estilo
    table = axes[1, 1].table(
        cellText=table_data,
        colLabels=["Métrica", "Valor"],
        cellLoc='center',
        loc='center'
    )

    # Negrita para headers
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # fila del header
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')  # fondo header azul oscuro
        else:
            cell.set_facecolor('#f1f1f2')  # fondo clarito para el resto

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)  # ajustar tamaño de celdas

    axes[1, 1].set_title('Resumen de Métricas')

    plt.tight_layout()
    plt.show()


def plot_all_metrics(resultados, y_val):
    """Grafica en una figura 1x3: curva PR, curva ROC y una tabla resumen de métricas.

    Parámetros:
    - resultados (list of dict): lista de resultados por modelo. Cada dict debe incluir:
        - "Modelo" (str): nombre del modelo.
        - "y_scores" (array): probabilidades predichas.
        - métricas como "Accuracy", "Precision", "Recall", "F1", etc.
    - y_val (array): etiquetas verdaderas del conjunto de validación.
    """

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(26, 6))

    # === 1. Curvas PR ===
    for result in resultados:
        nombre = result["Modelo"]
        y_scores = result["y_scores"]
        recalls, precisions, auc_pr = pr_auc(y_val, y_scores)
        axes[0].plot(recalls, precisions, label=f'{nombre} (AUC-PR={auc_pr:.4f})')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Curvas Precisión-Recall')
    axes[0].legend(loc='lower left')
    axes[0].grid(True)

    # === 2. Curvas ROC ===
    for result in resultados:
        nombre = result["Modelo"]
        y_scores = result["y_scores"]
        fpr, tpr, auc_roc = roc_auc(y_val, y_scores)
        axes[1].plot(fpr, tpr, label=f'{nombre} (AUC-ROC={auc_roc:.4f})')
    axes[1].plot([0, 1], [0, 1], linestyle='--', color='grey', label='Línea aleatoria')
    axes[1].set_xlabel('Tasa de Falsos Positivos (FPR)')
    axes[1].set_ylabel('Tasa de Verdaderos Positivos (TPR)')
    axes[1].set_title('Curvas ROC')
    axes[1].legend(loc='lower right')
    axes[1].grid(True)

    # === 3. Tabla de métricas ===
    tabla = pd.DataFrame(resultados).drop(columns=["y_scores"])

    # Preparar contenido
    table_data = tabla.round(3).values.tolist()
    col_labels = tabla.columns.tolist()

    # Insertar tabla en el subplot
    table = axes[2].table(cellText=table_data, colLabels=col_labels, 
                          cellLoc='center', loc='center')

    # Estilo
    for (row, col), cell in table.get_celld().items():
        if row == 0:  # Header
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        else:
            cell.set_facecolor('#f9f9f9')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.8, 2.5)
    axes[2].axis('off')
    axes[2].set_title('Resumen de Métricas')

    plt.tight_layout()
    plt.show()
