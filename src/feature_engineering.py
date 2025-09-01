import pandas as pd
from geopy.distance import geodesic

def create_price_per_night(df):
    df["price_per_night"] = df["price"] / df["minimum_nights"]
    return df

def calculate_center_distance(df):
    centr = (40.7128, -74.0060)

    def dist(row):
        n = (row["latitude"], row["longitude"])
        return geodesic(n, centr).kilometers

    df["center_distance"] = df.apply(dist, axis=1)
    df.drop(columns=["latitude", "longitude"], inplace=True)
    return df

def extract_date_features(df):
    df["year"] = df["last_review"].dt.year
    df["month"] = df["last_review"].dt.month
    df["year"] = df["year"].fillna(-1)
    df["month"] = df["month"].fillna(-1)
    df.drop(columns=["last_review"], inplace=True)
    return df

