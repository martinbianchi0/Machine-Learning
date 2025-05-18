import numpy as np
import pandas as pd

def label_encode(data, column, mapping=None):
    """Codifica una columna categórica a valores numéricos.

    Parámetros:
    - data (pd.DataFrame): DataFrame de entrada.
    - column (str): Columna a codificar.
    - mapping (dict, opcional): Diccionario de mapeo personalizado.

    Retorna:
    - pd.DataFrame: DataFrame con la columna codificada."""
    if mapping is not None:
        data[column] = data[column].map(mapping)
    else:
        data[column] = data[column].astype('category').cat.codes
    return data

def one_hot_encode(data, column, categories=None, drop_first=False):
    """
    Codifica una columna categórica usando one-hot encoding, conservando los NaNs
    y solo creando columnas para categorías específicas si se proveen.

    Parámetros:
    - data (pd.DataFrame): DataFrame de entrada.
    - column (str): Columna a codificar.
    - categories (list, opcional): Lista de categorías a codificar. Las que no estén se marcan como NaN.
    - drop_first (bool): Si True, elimina la primera categoría para evitar multicolinealidad.

    Retorna:
    - pd.DataFrame: DataFrame con la columna codificada y original eliminada.
    """
    col_values = data[column].astype(str).copy()

    if categories is not None:
        col_values = col_values.where(col_values.isin(categories))  # los no deseados se vuelven NaN

    # One-hot encode (solo las que quedaron en categories o NaN)
    dummies = pd.get_dummies(col_values, prefix=column)

    # Si querés dropear la primera categoría para evitar multicolinealidad
    if drop_first:
        dummies = dummies.iloc[:, 1:]

    # Asegurar que sea float (para aceptar NaNs)
    dummies = dummies.astype(float)

    # Restaurar NaNs en las filas que eran originalmente NaN o que fueron filtradas
    is_nan = col_values.isna()
    dummies[is_nan.values] = np.nan

    # Reemplazar columna original
    data = data.drop(column, axis=1)
    data = pd.concat([data, dummies], axis=1)

    return data


def clean_outliers_iqr(df):
    """Devuelve una copia del DataFrame con outliers removidos usando IQR (calculado por columna)."""
    df_clean = df.copy()
    columnas = df.select_dtypes(include=np.number).columns

    for col in columnas:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        df_clean[col] = df[col].where(df[col].between(lower, upper) | df[col].isna())

    return df_clean

def clean_outliers(df, limits):
    """Elimina los outliers de las columnas numéricas según límites predefinidos.

    Parámetros:
    - df (pd.DataFrame): DataFrame de entrada."""

    for col, (min_val, max_val) in limits.items():
        df[col] = df[col].where(df[col].between(min_val, max_val) | df[col].isna())

def train_val_split(dx, dy=None, split=0.8, shuffle=False, random_state=None):
    """Divide dx y dy en conjuntos de entrenamiento y validación.

    Parámetros:
    - dx (pd.DataFrame): features.
    - dy (pd.Series, opcional): etiquetas.
    - split (float): proporción para el set de entrenamiento.
    - shuffle (bool): si True, desordena los datos antes de dividir.
    - random_state (int, opcional): semilla para reproducibilidad del shuffle.

    Retorna:
    - tuple: subconjuntos divididos (x_train, x_val, y_train, y_val) si dy está dado, sino solo (x_train, x_val)."""
    if shuffle:
        if dy is not None:
            combined = pd.concat([dx, dy], axis=1)
            combined = combined.sample(frac=1, random_state=random_state).reset_index(drop=True)
            dx = combined.iloc[:, :-1]
            dy = combined.iloc[:, -1]
        else:
            dx = dx.sample(frac=1, random_state=random_state).reset_index(drop=True)

    n = int(len(dx) * split)
    x_train = dx[:n].copy()
    x_val = dx[n:].copy()
    if dy is None:
        return x_train, x_val
    y_train = dy[:n].copy()
    y_val = dy[n:].copy()
    return x_train, x_val, y_train, y_val

def robust_fit(X):
    """Calcula mediana e IQR.  
    Params: X (Series). Returns: (float, float)."""
    return X.median(), X.quantile(0.75) - X.quantile(0.25)

def robust_transform(X, mediana, iqr):
    """Escala usando mediana e IQR.  
    Params: X (Series), mediana (float), iqr (float). Returns: Series."""
    return (X - mediana) / iqr

def minmax_fit(X):
    """Calcula mínimo y máximo.  
    Params: X (Series). Returns: (float, float)."""
    return X.min(), X.max()

def minmax_transform(X, minimos, maximos):
    """Escala entre 0 y 1.  
    Params: X (Series), minimos (float), maximos (float). Returns: Series."""
    return (X - minimos) / (maximos - minimos)

def get_knn(fila, df_ref, col, k, features=None):
    """Devuelve los k vecinos más cercanos a 'fila'.

    Parámetros:
    - fila (pd.Series): fila a imputar.
    - df_ref (pd.DataFrame): dataset de referencia.
    - col (str): columna objetivo.
    - k (int): cantidad de vecinos.
    - features (list[str], opcional): columnas a usar como features.

    Retorna:
    - pd.DataFrame: k vecinos más cercanos ordenados por distancia."""
    if features is None:
        features = df_ref.drop(columns=[col]).columns.tolist()

    df_tmp = df_ref[features]
    fila_tmp = fila[features]

    distancias = np.linalg.norm(df_tmp.values - fila_tmp.values, axis=1)
    vecinos = df_ref.copy()
    vecinos["dist"] = distancias
    vecinos = vecinos.sort_values("dist").drop(columns="dist")
    return vecinos.head(k)


def knn_regression(df_target, df_ref, col, k=5, features=None, binary=False):
    """Imputa valores faltantes en df_target usando KNN.

    Parámetros:
    - df_target (pd.DataFrame): dataset con valores nulos.
    - df_ref (pd.DataFrame): dataset de referencia sin nulos.
    - col (str): columna a imputar.
    - k (int): cantidad de vecinos.
    - features (list[str], opcional): columnas a usar como features.
    - binary (bool): si True, redondea la media como 0 o 1.

    Retorna:
    - pd.DataFrame: dataset imputado."""
    df_out = df_target.copy()
    filas_nulas = df_out[df_out[col].isna()]

    for idx, fila in filas_nulas.iterrows():
        vecinos = get_knn(fila, df_ref[df_ref[col].notna()], col, k, features)
        valores_vecinos = vecinos[col]
        valor = int(valores_vecinos.mean() >= 0.5) if binary else valores_vecinos.mean()
        df_out.at[idx, col] = valor

    return df_out

def under_sample(df, col):
    """Submuestrea el DataFrame para balancear la clase minoritaria.

    Parámetros:
    - df (pd.DataFrame): dataset de entrada.
    - col (str): nombre de la columna objetivo.

    Retorna:
    - pd.DataFrame: dataset balanceado por submuestreo."""
    n_1 = len(df[df[col] == 1])
    n_0 = len(df[df[col] == 0])
    n_samples = min(n_1, n_0)
    df_1 = df[df[col] == 1].sample(n=n_samples, random_state=42)
    df_0 = df[df[col] == 0].sample(n=n_samples, random_state=42)
    return pd.concat([df_1, df_0]).sample(frac=1, random_state=42).reset_index(drop=True)

def oversample_duplication(df, col):
    """Sobremuestrea el DataFrame duplicando la clase minoritaria.

    Parámetros:
    - df (pd.DataFrame): dataset de entrada.
    - col (str): nombre de la columna objetivo.

    Retorna:
    - pd.DataFrame: dataset balanceado por sobremuestreo."""
    n_1 = len(df[df[col] == 1])
    n_0 = len(df[df[col] == 0])
    
    if n_1 > n_0:
        df_min = df[df[col] == 0].sample(n=n_1, replace=True, random_state=42)
        df_maj = df[df[col] == 1]
    else:
        df_min = df[df[col] == 1].sample(n=n_0, replace=True, random_state=42)
        df_maj = df[df[col] == 0]

    return pd.concat([df_min, df_maj]).sample(frac=1, random_state=42).reset_index(drop=True)

def smote(df, col, k=5, features=None):
    """
    Aplica SMOTE para balancear clases (binaria o multiclase).

    Parámetros:
    - df (pd.DataFrame): dataset de entrada.
    - col (str): columna objetivo.
    - k (int): número de vecinos para SMOTE.
    - features (list[str], opcional): columnas a usar como features.

    Retorna:
    - pd.DataFrame: dataset con ejemplos sintéticos añadidos.
    """
    import pandas as pd
    from random import randint, uniform

    if features is None:
        features = df.columns.drop(col)

    counts = df[col].value_counts()
    max_class = counts.idxmax()
    max_count = counts.max()

    sinteticos = []

    for cls, count in counts.items():
        if cls == max_class:
            continue
        x = df[df[col] == cls].reset_index(drop=True)
        n_sinteticos = max_count - count

        for _ in range(n_sinteticos):
            i = randint(0, len(x) - 1)
            punto = x.loc[i]
            vecinos = get_knn(punto, x, col, k, features)
            if vecinos is None or len(vecinos) == 0:
                continue
            vecino = vecinos.sample(1).iloc[0]
            nuevo = {}
            for f in features:
                nuevo[f] = punto[f] + uniform(0, 1) * (vecino[f] - punto[f])
            nuevo[col] = cls
            sinteticos.append(nuevo)

    df_sint = pd.DataFrame(sinteticos)
    return pd.concat([df, df_sint], ignore_index=True)
