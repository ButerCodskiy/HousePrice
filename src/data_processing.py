import pandas as pd

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
    df["last_review"] = df["last_review"].fillna(pd.NaT)
    df["reviews_per_month"] = df["reviews_per_month"].fillna(df["reviews_per_month"].median())
    df.drop(columns=["id", "name", "host_name", "host_id"], inplace=True)

    return df

