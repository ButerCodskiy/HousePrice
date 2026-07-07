import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.data_processing import load_data, preprocess_data, remove_outliers
from src.feature_engineering import calculate_center_distance, extract_date_features
from src.models import get_preprocessor, train_stacked_model, evaluate_model

def main():
    print("Загрузка данных...")
    df = load_data("data/AirBnb.csv")
    
    print("Предобработка данных...")
    df = preprocess_data(df)
    
    print("Фильтрация выбросов...")
    df = remove_outliers(df)
    
    print("Генерация признаков (Feature Engineering)...")
    # Признак price_per_night был удален, чтобы избежать Data Leakage
    df = calculate_center_distance(df)
    df = extract_date_features(df)
    
    # Отделяем целевую переменную
    if "price" not in df.columns:
        print("Ошибка: колонка price не найдена!")
        return
        
    print("Логарифмирование целевой переменной...")
    y = np.log1p(df["price"])
    X = df.drop(columns=["price"])
    
    print(f"Размер датасета: {X.shape}")
    print("Разбиение данных (Train/Test split)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Инициализация Preprocessor (с исправленным OneHotEncoder)...")
    preprocessor = get_preprocessor(X_train)
    
    print("Обучение стекинговой модели (StackingRegressor)...")
    print("Это может занять пару минут...")
    model = train_stacked_model(X_train, y_train, preprocessor)
    
    print("Оценка модели на тестовой выборке...")
    mse, r2 = evaluate_model(model, X_test, y_test)
    print("================")
    print(f"Test MSE: {mse:.2f}")
    print(f"Test R2:  {r2:.4f}")
    print("================")
    print("Готово!")

if __name__ == "__main__":
    main()
