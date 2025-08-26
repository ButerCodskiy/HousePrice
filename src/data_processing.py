import pandas as pd

def load_data(file_path):
    """Загружает данные из CSV файла."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Выполняет предобработку данных: заполнение пропусков, преобразование типов, удаление столбцов."""
    # Заполнение пропусков в last_review и преобразование типа
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
    df["last_review"] = df["last_review"].fillna(pd.NaT)

    # Заполнение пропусков в reviews_per_month медианой
    df["reviews_per_month"] = df["reviews_per_month"].fillna(df["reviews_per_month"].median())

    # Удаление неинформативных колонок
    df.drop(columns=["id", "name", "host_name", "host_id"], inplace=True)

    return df
