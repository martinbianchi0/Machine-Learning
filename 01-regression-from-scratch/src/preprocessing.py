def one_hot_encoder(df):
    """Codifica las columnas 'lat' y 'lon' en una nueva variable categórica 'zona'.  
    Params: df (DataFrame).  
    Return: None (modifica el DataFrame en el lugar).
    """
    df["zona"] = df["lat"].apply(lambda x: 0 if -35.4 <= x <= -33.5 else 1)
    df.drop(columns="lat", inplace=True)
    df.drop(columns="lon", inplace=True)

def convert_areaunits(df):
    """Convierte las columnas de área de pies cuadrados a metros cuadrados.  
    Params: df (DataFrame). 
    """
    for fila in df.itertuples(index=True):
        if fila.area_units == 'sqft':
            df.at[fila.Index, 'area'] *= 0.092903
    df.drop(columns='area_units', inplace=True)

def normalize(train, val):
    """Normaliza los datos con media y desviación estándar de train.  
    Params: train (DataFrame), val (DataFrame).  
    Returns: train_norm (DataFrame), val_norm (DataFrame), medias (Series), desv (Series).
    """
    medias = train.mean()
    desv = train.std()
    
    train_norm = (train - medias) / desv
    val_norm = (val - medias) / desv

    return train_norm, val_norm, medias, desv


def handle_missing_values(df):
    """Imputa valores faltantes en 'rooms' y 'age' según reglas específicas.  
    Params: df (DataFrame).  
    Return: df (DataFrame) con valores imputados.
    """
    rooms_1 = df.loc[df["rooms"] == 1, "area"].max()
    rooms_2 = df.loc[df["rooms"] == 2, "area"].max()
    rooms_3 = df.loc[df["rooms"] == 3, "area"].max()
    rooms_4 = df.loc[df["rooms"] == 4, "area"].max()
    rooms_5 = df.loc[df["rooms"] == 5, "area"].max()
    rooms = [rooms_1, rooms_2, rooms_3, rooms_4, rooms_5]
    for index, row in df.loc[df["rooms"].isna()].iterrows():
        area_value = row["area"]
        for i, max_area in enumerate(rooms):
            if area_value <= max_area:
                df.at[index, "rooms"] = i+1
                break

    age_house_1 = df.loc[df["is_house"] == 1, "age"]
    age_house_0 = df.loc[df["is_house"] == 0, "age"]
    age_house_1_mean = round(age_house_1.mean() if not age_house_1.isna().all() else df["age"].mean())
    age_house_0_mean = round(age_house_0.mean() if not age_house_0.isna().all() else df["age"].mean())

    for index, row in df.loc[df["age"].isna()].iterrows():
        if row["is_house"] == 1:
            df.at[index, "age"] = age_house_1_mean
        else:
            df.at[index, "age"] = age_house_0_mean

    return df