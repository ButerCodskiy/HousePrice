import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from src.data_processing import load_data, preprocess_data, remove_outliers
from src.feature_engineering import calculate_center_distance, extract_date_features

def run_econometric_analysis():
    print("Загрузка и подготовка данных для эконометрики...")
    df = load_data("data/AirBnb.csv")
    df = preprocess_data(df)
    df = remove_outliers(df)
    df = calculate_center_distance(df)
    df = extract_date_features(df)
    
    y = np.log1p(df["price"])
    X = df.drop(columns=["price"])
    
    # 1. Избегаем Dummy Variable Trap
    print("Кодирование категориальных признаков (drop_first=True)...")
    if "name" in X.columns:
        X = X.drop(columns=["name"])
        
    # Преобразуем строковые категории в dummy-переменные (drop_first=True уберет строгую мультиколлинеарность)
    X_encoded = pd.get_dummies(X, drop_first=True, dtype=float)
    
    # Чтобы коэффициенты числовых признаков были интерпретируемы и сопоставимы, мы их стандартизируем (Z-score)
    num_cols = X.select_dtypes(include=np.number).columns
    for col in num_cols:
        X_encoded[col] = (X_encoded[col] - X_encoded[col].mean()) / X_encoded[col].std()

    # Добавляем константу
    X_encoded = sm.add_constant(X_encoded)
    
    # 2. Анализ Мультиколлинеарности (VIF)
    print("Расчет Variance Inflation Factor (VIF) для числовых признаков...")
    vif_data = pd.DataFrame()
    # Считаем VIF только для числовых фичей, так как для 200+ дамми-переменных это избыточно
    num_cols_with_const = ["const"] + list(num_cols)
    X_num = X_encoded[num_cols_with_const]
    vif_data["Feature"] = X_num.columns
    vif_data["VIF"] = [variance_inflation_factor(X_num.values, i) for i in range(X_num.shape[1])]
    
    print("Обучение OLS модели с робастными ошибками (HC3)...")
    # 3. Обучение OLS с HC3 поправкой Уайта на гетероскедастичность
    model = sm.OLS(y.values, X_encoded)
    # Используем cov_type='HC3' для робастных ошибок
    results = model.fit(cov_type='HC3')
    
    summary = results.summary()
    
    # 4. Извлечение ТОП-20 самых значимых коэффициентов
    print("Извлечение Топ-20 предикторов...")
    # Исключаем константу из рейтинга
    pvalues = results.pvalues.drop("const")
    tvalues = results.tvalues.drop("const")
    coefs = results.params.drop("const")
    std_errs = results.bse.drop("const")
    
    # Собираем DataFrame
    coef_df = pd.DataFrame({
        "Коэффициент": coefs,
        "Std Err (HC3)": std_errs,
        "t-статистика": tvalues,
        "p-value": pvalues,
        "Abs_t": tvalues.abs()
    })
    
    # Сортируем по абсолютной t-статистике (самые надежные и сильные предикторы)
    top_20 = coef_df.sort_values(by="Abs_t", ascending=False).head(20)
    top_20 = top_20.drop(columns=["Abs_t"])
    
    print("\nСохранение результатов в statistical_analysis.md...")
    with open("statistical_analysis.md", "w", encoding="utf-8") as f:
        f.write("# Профессиональный Эконометрический Анализ\n\n")
        f.write("Анализ проведен на логарифме цены `np.log1p(price)`. Использованы **робастные стандартные ошибки Уайта (HC3)** для корректировки гетероскедастичности. Категориальные признаки закодированы с удалением первой категории (`drop_first=True`) во избежание строгой мультиколлинеарности.\n\n")
        
        f.write("## 1. Топ-20 самых значимых предикторов\n")
        f.write("Отсортированы по t-статистике (надежности влияния на цену). Коэффициенты стандартизированных числовых признаков показывают изменение лог-цены при изменении признака на 1 стандартное отклонение.\n\n")
        f.write(top_20.to_markdown())
        f.write("\n\n")
        
        f.write("## 2. Анализ Мультиколлинеарности (VIF)\n")
        f.write("Значение VIF > 10 указывает на сильную мультиколлинеарность. В нашей числовой выборке:\n\n")
        f.write(vif_data[vif_data["Feature"] != "const"].to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 3. Полный лог регрессии (с поправками HC3)\n```text\n")
        f.write(summary.as_text())
        f.write("\n```\n")

if __name__ == "__main__":
    run_econometric_analysis()
