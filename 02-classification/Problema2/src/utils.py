from .models import RandomForest, MulticlassLogisticRegression
from .metrics import f_score, accuracy, precision, recall, pr_auc, roc_auc, matriz_de_confusion
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def train_best_random_forest(X_train, y_train, X_val, y_val):
    """
    Entrena un Random Forest con los mejores hiperparámetros según F-score (rápido).

    Parámetros:
    - X_train (array o DataFrame): datos de entrenamiento.
    - y_train (array o Series): etiquetas de entrenamiento.
    - X_val (array o DataFrame): datos de validación.
    - y_val (array o Series): etiquetas de validación.

    Retorna:
    - model (RandomForest): modelo entrenado con mejores hiperparámetros.
    """
    best_fscore = 0
    best_params = {}

    # Reducción fuerte del grid
    n_trees_list = [5, 10, 20]
    max_depth_list = [5, 10, 15]
    min_samples_split_list = [2]

    # Usamos un subconjunto de datos para evaluar hiperparámetros
    idx = np.random.choice(len(X_train), size=min(300, len(X_train)), replace=False)
    X_sample = X_train.iloc[idx]
    y_sample = y_train.iloc[idx]

    for n_trees in n_trees_list:
        for max_depth in max_depth_list:
            for min_samples_split in min_samples_split_list:
                model = RandomForest(
                    n_trees=n_trees,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split
                )
                model.fit(X_sample, y_sample)
                y_pred = model.predict(X_val)
                fs = f_score(y_val, y_pred)

                if fs > best_fscore:
                    best_fscore = fs
                    best_params = {
                        'n_trees': n_trees,
                        'max_depth': max_depth,
                        'min_samples_split': min_samples_split
                    }

    # Entrenar modelo final con todos los datos
    model = RandomForest(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    fs = f_score(y_val, y_pred)
    print(f"Mejor F-score: {fs} con n_trees={best_params['n_trees']}, max_depth={best_params['max_depth']}, min_samples_split={best_params['min_samples_split']}")
    return model

def train_best_multiclass_logistic_model(X_train, y_train, X_val, y_val):
    """Entrena un modelo de regresión logística multiclase (softmax) 
    con los mejores hiperparámetros según el F-score en validación.

    Retorna:
    - model (MulticlassLogisticRegression): modelo entrenado con mejores hiperparámetros.
    """

    best_fscore = 0
    best_params = {'lr': None, 'lambda': None}
    lambdas = [0, 1e-4, 1e-3, 1e-2, 0.1, 1, 10]
    learning_rates = [0.1, 0.01, 0.001]

    for lr in learning_rates:
        for l in lambdas:
            model = MulticlassLogisticRegression(X_train, y_train, lr=lr, L2=l)
            y_pred = model.predict(X_val)
            fs = f_score(y_val, y_pred)

            if fs > best_fscore:
                best_fscore = fs
                best_params['lr'] = lr
                best_params['lambda'] = l

    final_model = MulticlassLogisticRegression(X_train, y_train,
                                               lr=best_params['lr'],
                                               L2=best_params['lambda'])
    print(f"Mejor F-score: {best_fscore} con lr={best_params['lr']}, lambda={best_params['lambda']}")
    return final_model

def compare_all_metrics(y_vals, y_probs, y_preds, nombres=["LDA", "Logistic", "Random Forest"]):
    """
    Compara múltiples modelos graficando:
    - PR Curve
    - ROC Curve
    - Tabla resumen de métricas
    - Matrices de confusión (todo en una grilla 2x3)

    Params: y_vals (list), y_probs (list), y_preds (list), nombres (list).
    """
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))

    # === 1. Curva PR ===
    for i in range(len(y_vals)):
        recalls, precisions, auc_pr = pr_auc(y_vals[i], y_probs[i])
        axes[0, 0].plot(recalls, precisions, label=f'{nombres[i]} (AUC={auc_pr:.3f})')
    axes[0, 0].set_title("Curva Precisión-Recall")
    axes[0, 0].set_xlabel("Recall")
    axes[0, 0].set_ylabel("Precision")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # === 2. Curva ROC ===
    for i in range(len(y_vals)):
        fpr, tpr, auc_roc = roc_auc(y_vals[i], y_probs[i])
        axes[0, 1].plot(fpr, tpr, label=f'{nombres[i]} (AUC={auc_roc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axes[0, 1].set_title("Curva ROC")
    axes[0, 1].set_xlabel("FPR")
    axes[0, 1].set_ylabel("TPR")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # === 3. Tabla de métricas ===
    resultados = []
    for i in range(len(y_vals)):
        y_true = y_vals[i]
        y_pred = y_preds[i]
        resultados.append({
            "Modelo": nombres[i],
            "Accuracy": round(accuracy(y_true, y_pred), 3),
            "Precision": round(precision(y_true, y_pred), 3),
            "Recall": round(recall(y_true, y_pred), 3),
            "F1": round(f_score(y_true, y_pred), 3),
        })
    tabla = pd.DataFrame(resultados)
    axes[0, 2].axis('off')
    table_data = tabla.values.tolist()
    col_labels = tabla.columns.tolist()
    table = axes[0, 2].table(cellText=table_data, colLabels=col_labels,
                             cellLoc='center', loc='center')

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        else:
            cell.set_facecolor('#f9f9f9')

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.8, 2.5)
    axes[0, 2].set_title('Resumen de Métricas')

    # === 4-6. Matrices de confusión ===
    for i in range(len(y_vals)):
        cm = matriz_de_confusion(y_vals[i], y_preds[i])
        clases = np.unique(y_vals[i])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=clases, yticklabels=clases, ax=axes[1, i])
        axes[1, i].set_title(f'Matriz de Confusión - {nombres[i]}')
        axes[1, i].set_xlabel('Predicción')
        axes[1, i].set_ylabel('Realidad')

    plt.tight_layout()
    plt.show()
