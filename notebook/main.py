#!/usr/bin/env python
# coding: utf-8

# 

# # Описание датасета
# С 2008 года пользователи Airbnb могут находить уникальные и персонализированные варианты проживания для путешествий по всему миру. Этот набор данных содержит информацию о листингах и их активности в Нью-Йорке в 2019 году. В нем представлены данные о хостах, географическом расположении объектов и ключевых показателях, которые могут быть использованы для анализа и прогнозирования. В данном проекте будет выполнена задача прогнозирования цены.
# 

# # Описание признаков
# id — уникальный идентификатор объявления.
# 
# name — название объявления.
# 
# host_id — идентификатор владельца жилья.
# 
# neighbourhood_group — район города (например, Манхэттен, Бруклин).
# 
# neighbourhood — подрайон (конкретное местоположение внутри района).
# 
# latitude, longitude — географические координаты объекта(longitude -долгота, latitude - широта).
# 
# room_type — тип жилья (например, "Весь дом/квартира", "Частная комната" и т.д.).
# 
# price — стоимость ночи проживания (целевая переменная).
# 
# minimum_nights — минимальное количество ночей для бронирования.
# 
# number_of_reviews — общее количество отзывов.
# 
# last_review — дата последнего отзыва.
# 
# reviews_per_month — среднее количество отзывов в месяц.
# 
# calculated_host_listings_count — количество всех объявлений от одного хозяина.
# 
# availability_365 — количество дней, когда объект доступен для бронирования в течение года.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('dark_background')


# ##1. Загрузка датасета

# In[ ]:


get_ipython().system('gdown --id 1jguoHyAi6QXgDlTIkIpYXZGy6vBsIckv')
import pandas as pd
df = pd.read_csv('AirBnb.csv')
df.head()


# ##2.Просмотр и предобработка данных##

# Необходимо посмотреть на размер датафрейма, оценить количество пропусков и типы данных для признаков.

# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.isna().sum()


# Посмотрим на типы данных в колонках с пропусками, чтобы правильно понимать, как их следует заполнять.
# 

# In[ ]:


df[['name', 'last_review', 'reviews_per_month', "host_name"]].dtypes


# 1) name и host_name - не будем заполнять, эти признаки стоит удалить, они не несут полезной информации для анализа зависимостей и будут только мешать в построении модели.
# 
# 2) Пропуски в last_review заполним NaT, посколько их довольно большое количество, помимо этого необходмо конвертировать тип данных, поскольку в дейстивительности это информация о последней дате просмотра объявления. Также создадим новый признак - есть ли дата.
# 
# 3) reviews_per_month заполним медианой, поскольку это числовой столбец и медиана устойчива к выбросам.
# 
# Кроме этого, удалим колонки host_name и id, поскольку они также неинформативны и могут мешать модели.

# In[ ]:


df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
df['last_review'] = df['last_review'].fillna(pd.NaT)


# In[ ]:


df['reviews_per_month']=df['reviews_per_month'].fillna(df['reviews_per_month'].median())


# In[ ]:


df.drop(columns=['id', 'name', "host_name", 'host_id'], inplace=True)


# In[ ]:


df.describe()


# Как можно видеть из описания датасета, вероятно ряд колонок содержит выбросы, это можно заметить по тому, как сильно отличаются 75% квартиль и максимум. Также мы можем оценить диапазон значений, например, для целевой пременной - цена, он составляет от 0 до 10000.

# Теперь посмотрим на наличие дубликатов в данных.

# In[ ]:


df.duplicated().sum()


# ## 3. Создание новых признаков

# Создадим новый признаки, во-первых, цену за ночь, это позволит лучше понимать цену хоста. Во-вторых, используем наши данные о широте и долготе для расчеты актуального расстояния до центра Нью-Йорка, сами по себе эти признаки не очень информативны, а в таком виде они становятся очень полезными. Поэтому после создания нового признака, широту и долготу следует удалить.

# In[ ]:


df['price_per_night'] = df['price'] / df['minimum_nights']


# In[ ]:


from geopy.distance import geodesic

centr = (40.7128, -74.0060)
def dist(row):
    n = (row['latitude'], row['longitude'])
    return geodesic(n, centr).kilometers

df['center_distance'] = df.apply(dist, axis=1)



# In[ ]:


df.drop(columns=['latitude', 'longitude'], inplace=True)


# In[ ]:


df.head(10)


# Оценим количество уникальных значений в колонках и затем посмотрим на уникальные значения в колонках, где их меньше 10.

# In[ ]:


df.nunique()


# In[ ]:


df['room_type'].value_counts()


# In[ ]:


df['neighbourhood_group'].value_counts()


# Теперь у нас есть представление о структуре данных, о том как работать с каждым из признаков.

# ## 4. Разведывательный анализ данных и визуализация

# ##4.1. Распредление категориальных данных

# In[ ]:


import math
df_cat =  df.select_dtypes(include=['object', 'category']).columns
df_cat = df_cat[df_cat != 'neighbourhood']
num_col = len(df_cat)
num_rows = math.ceil(num_col / 3)
num_cols = min(3, num_col)


# Находим все категориальные столбцы из датасета, выбрав столбцы с типами данных object или category. Затем из полученного списка удаляем столбец neighbourhood, так как он содержит слишком много уникальных значений, что делает его нецелесообразным для дальнейшего анализа в графическом виде. После этого вычисляется количество строк и столбцов для размещения подграфиков, с учётом того, что в каждом ряду будет не более трех графиков. Таким образом, мы динамически адаптируем количество подграфиков, чтобы все они помещались в удобном формате.

# In[ ]:


import matplotlib.pyplot as plt
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))

for i, col in enumerate(df_cat):
    df[col].value_counts().plot.bar(ax=axes[i], color='grey', edgecolor='black')
    axes[i].set_title(f'Распределение {col}')

plt.tight_layout()



# Из результатов можно увидеть, что основной рынок аренды сосредоточен в Манхэттене и Бруклине, а большая часть предложений – это либо отдельные квартиры, либо частные комнаты. Bronx и Staten Island почти не представлены, а совместное проживание (shared room) крайне непопулярно.

# 

# Посмотрим на группы разных комнат и районов по средней цене.

# In[ ]:


df.groupby('room_type')['price'].mean().reset_index()


# Как видно, одна из групп, а именно целый дом и квартира,сильно выделяется высокой ценой. Это в свою очередь объясняется тем, что такие варианты предоставляют полный доступ к жилью, что является более привлекательным для гостей, которые ищут приватность и удобство, что создает высокий спрос.

# In[ ]:


df.groupby('neighbourhood_group')['price'].mean().reset_index()


# В данном случае, можно увидеть, что в Манхеттене средняя цена намного превосходит средние цены других районов, самый дешевый Бронкс - но он не сильно отвличется от остальной массы. В целом, цены Манхеттена объяснимы, ведь он является сердцем города Нью-Йорк, где сосредоточены финансовые, культурные и деловые центры.

# ##4.2 Распределение числовых данных

# In[ ]:


df_num = df.select_dtypes(include=['number']).columns
num_col = len(df_num)
num_rows = math.ceil(num_col / 3)
num_cols = min(3, num_col)


# In[ ]:


import seaborn as sns
fig, axes = plt.subplots(num_rows, 3, figsize=(12, num_rows * 5))

for i, col in enumerate(df_num, 1):
    plt.subplot(num_rows, 3, i)
    sns.histplot(df[col], kde=True, color='pink', bins=30)
    plt.title(f'Распределение {col}')

plt.tight_layout()

Данные содержат сильную асимметрию в распределении цен, количества ночей, отзывов и доступности. Есть признаки выбросов, особенно в цене и минимальном количестве ночей.
# ## 4.3. Анализ выбросов.

# In[ ]:


df_num = df.select_dtypes(include=['number']).columns
for col in df_num:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
    print(f"{col}: количество выбросов = {outliers_count}")

    df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)


# In[ ]:


num_cols = len(df_num)
num_rows = (num_cols + 2) // 3


# In[ ]:


plt.figure(figsize=(10, num_rows * 5))
for i, col in enumerate(df_num):
    plt.subplot(num_rows, 3, i + 1)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot для {col}')
    plt.ylabel(col)

plt.tight_layout()

Выбрасов нет, по каждому графику можно опрделить вариативности, размах, медианы и пр.
# ## 4.4 Корреляция между целевой переменной и числовыми признаками, визуализация.

# In[ ]:


df_nums = df.select_dtypes(include=['number'])
corr = df_nums.corrwith(df['price']).sort_values()

sns.barplot(y=corr.index, x=corr, palette='hls')
plt.xlabel('Корреляция')
plt.ylabel('Признак')

Есть отрицательная корреляция с признаком center_distance, она составляет около -0.3. Cреди положительной самая сильная с price_per_night, корреляция целевой пременной с остальными признаками не очень значительна, но третья по значимости с availability_365, однако только 10%.
# In[ ]:


df.info()


# In[ ]:


df['distance_group'] = pd.cut(df['center_distance'], bins=10)
plt.figure(figsize=(12, 6))
sns.boxplot(x=df['distance_group'], y=df['price'])
plt.xticks(rotation=45)
plt.title('Распределение цены в зависимости от расстояния до центра')
plt.xlabel('Группы расстояний')
plt.ylabel('Цена')

Как видно из графика, с ростом расстояния цена в среднем падает.
# In[ ]:


import numpy as np
plt.figure(figsize=(10, 6))
sns.lineplot(x='availability_365', y='price', data=df, label='Price')

z = np.polyfit(df['availability_365'], df['price'], 1)
p = np.poly1d(z)

plt.plot(df['availability_365'], p(df['availability_365']), color='red', label='Trend Line')

plt.title('Line Plot with Trend Line: Availability_365 vs Price')
plt.xlabel('Availability_365')
plt.ylabel('Price')
plt.legend()


# Из графика можно увидеть незначительную связь количеством доступных дней в году и ценой, в целом с доступностью цена в среднем растет.

# In[ ]:


plt.figure(figsize=(10, 6))
sns.histplot(df['price_per_night'], kde=True, bins=30)
plt.title('Histogram: Price_per_night Distribution')
plt.xlabel('Price_per_night')
plt.ylabel('Frequency')



# Это гистограмма распределению цены за ночь, в целом, ц графика нисходящий тренд - с ростом цены частота снижается, отднако при значении 175 заметен скачок.

# # 4.5 Корреляционная матрица для числовых признаков и визуализация##

# Кроме оценки связи признаков с целевой пременной, интересно посмотреть на некоторые зависимости признаков между собой.

# In[ ]:


plt.figure(figsize=(10, 6))
corr = df.select_dtypes(include=np.number).corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Корреляционная матрица')


# Можно увидеть из тепловой карты корреляции какие признаки между собой сильнее всего коррелируют. Но в целом сильной корреляции не наблюдается, что в целом может быть хорошо в контексте применения линейной модели, так как это убирает риск линейной зависимости между признаками.

# In[ ]:


plt.figure(figsize=(10, 6))
corr_spearman = df.select_dtypes(include=np.number).corr(method='spearman')
sns.heatmap(corr_spearman, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляционная матрица (Метод Спирмена)')


# Однако мы использовали метод Пирсона и могли упустить ряд связей из-за его ограниченности, полноценно посмотреть на связи можно используя также матод Спирмена. Можно заметить, что в такой корелляцоинной матрице, например, прослеживается более сильная корреляция между количество просмотров объявления за месяц и его доступностью в году. Хотя часть корреляций усилилась, некоторые напротив, стали слабее.

# In[ ]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='minimum_nights', y='price_per_night', data=df)
plt.title('Boxplot: минимум ночей и цена за ночь')
plt.xlabel('Минимум ночей')
plt.ylabel('Цена за ночь')


# Как видно из гарфика, цена за ночь в среднем падает с ростом количества минимальных ночей, это говорит о том, что хозяева жилья часто предлагают скидки для гостей, которые бронируют проживание на более длительный срок. Это стимулирует гостей выбирать более долгие периоды проживания, что может снизить среднюю цену за ночь. Это похоже на оптовую продажу, где низкая цена компенсируется большим количеством.

# In[ ]:


grouped = df.groupby("calculated_host_listings_count")["availability_365"].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.lineplot(data=grouped, x="calculated_host_listings_count", y="availability_365", marker="o")
plt.title("Средняя доступность (availability_365) по количеству объявлений у хоста")
plt.xlabel("Количество объявлений у хоста")
plt.ylabel("Средняя доступность (дней в году)")


# In[ ]:


plt.figure(figsize=(12, 6))
ax = sns.countplot(x='room_type', hue='neighbourhood_group', data=df)

plt.title('Распределение типов жилья по районам')
plt.xlabel('Тип жилья')
plt.ylabel('Количество объявлений')

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, title='Районы', bbox_to_anchor=(1.05, 1), loc='upper left')


# Из данного графика можно увидеть, как распрелено количество объявлений по типам жилья в разных районах. В целом распредление везде похожее, а вот количество объявлений в Манхетене самое большое, хотя Бруклин не сильно отстает. В Манхетене в основном размещены объявления для отедльного дома и квартиры, а в бруклине для отдельной комнаты.

# In[ ]:


def format_range(x):
    return f"{x.left:.3f} – {x.right:.3f}"

dbins = pd.cut(df['center_distance'], bins=10)
df['distance_range'] = dbins.map(format_range)

pivot_table = df.groupby(['distance_range', 'neighbourhood_group'])['price'].mean().unstack()

plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, annot=True, fmt=".0f")
plt.title('Средняя цена по расстоянию до центра и районам')
plt.xlabel('Район')
plt.ylabel('Расстояние до центра')
plt.xticks(rotation=45)


# Видна разница в районнах по ценам, Манхэттенне средние цены выше, как уже выяснилось до этого, но самая высокая цена на удивление не находится в области ближашей к центру, она располагается в 5-7 км. от центра и составляет 199 долларов. Хотя примечательно, что в Бруклине в ближайшему к центру области средняя цена превосходит  соответствующую у Манхеттена. В Квинс средняя цена тоже относительно высокая ( 184 долларов для ближайшей к центру области)
# 

# # 5. Машинное обучение.
**Причины выбора линейной регрессии:**
1. Легкость интерпретации: коэффициенты модели показывают влияние каждого признака на результат.
2. Линейные зависимости: в данных есть линейные связи, что в целом может делать выбор оправданным, но важно уточнить, что далеко не все зависимости линейные.
3. Низкие вычислительные затраты: линейная регрессия требует меньше ресурсов, особенно для больших датасетов.
4. Быстрая тренировка: обучение модели происходит быстрее, чем у более сложных алгоритмов.
5. Простота настройки: линейная регрессия не требует сложной настройки гиперпараметров.
# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder


# In[ ]:


df["year"] = df["last_review"].dt.year
df["month"] = df["last_review"].dt.month

df["year"] = df["year"].fillna(-1)
df["month"] = df["month"].fillna(-1)

df.drop(columns=["last_review"], inplace=True)

y = df["price"]
X = df.drop(columns=["price"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=161)


# Используем one-hot-encoding, встроенный в pandas(его использование проще, чем из sklearn), а также делим наши данные на тестовые и тренировочные. В качестве модели берем Ridge, в ней есть встроенная регуляризация, которая помогает бороться с переобучением.

# In[ ]:


param_grid = {
    'ridge__alpha': [0.01, 0.1, 1, 10, 100],
    'ridge__solver': ['auto', 'lsqr', 'sparse_cg', 'sag'],
    'ridge__max_iter': [1000, 5000]

}
num_cols = X_train.select_dtypes(include=['number']).columns
cat_cols = X_train.select_dtypes(include=['object']).columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('ridge', Ridge())
])

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
y_pred = grid_search.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")


# Находим лучшие гиперпараметры для нашей модели используя GridSearchCV, для этого нужно предварительно отмасшатбировали данные, поскольку для линейных моделей это критично, хорошим выбором будет StandardScaler, поскольку он относительно устойчив к выбросам. Сделали словарь параметров и их значений и передали его в GridSearchCV с 5-ти разовой кросс-валидацией и отрицательной средне-квадратичной функцией потерь(поскольку GridSearchCV по умолчанию находит гиперпараметры для максимизации функции).

# Модель имеет относительно хорошее качество (r2 около 0.7 - для линейной модели довольно неплохое описание дисперсии) и не переобучена, что ожидаемо для линейной модели с регуляризацией.

# In[ ]:


best = grid_search.best_estimator_
coeff = best.named_steps['ridge'].coef_
feature_names = best.named_steps['preprocessor'].get_feature_names_out()
coeff_df = pd.DataFrame({'Feature': feature_names,'Coefficient': coeff})
coeff_df['Abs_Coefficient'] = coeff_df['Coefficient'].abs()
coeff_df = coeff_df.sort_values(by='Abs_Coefficient', ascending=False).reset_index(drop=True)
coeff_df


# Наиболее важные признаки - цена за ночь и весь дом или аппартаменты, причем оба признака с положительным коэффициентом, а самый большой отрицательный коэффициент имеет общая комната. Результаты довольно ожидаемые и очевидные.

# In[ ]:


plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred, alpha=0.7, color='blue', edgecolor='white', s=40)

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='y = x')

plt.grid(True, linestyle='-', alpha=0.5)
plt.xlabel("Фактическая цена")
plt.ylabel("Предсказанная цена")
plt.title(f"Сравнение фактической и предсказанной цены\nR² = {r2:.2f}")

plt.legend()


# ##5.1 Стекинг

# In[ ]:


get_ipython().system('pip install optuna')


# In[ ]:


import optuna
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


# In[ ]:


def objective_rf(trial):
    param_grid = {"n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 30),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"])}

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_errors = np.empty(cv.get_n_splits())

    for idx, (train_idx, test_idx) in enumerate(cv.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[test_idx]
        X_tr_processed = preprocessor.fit_transform(X_tr)
        X_val_processed = preprocessor.transform(X_val)

        model = RandomForestRegressor(**param_grid, random_state=42, n_jobs=-1)
        model.fit(X_tr_processed, y_tr)
        preds = model.predict(X_val_processed)
        cv_errors[idx] = mean_squared_error(y_val, preds)

    return np.mean(cv_errors)


# In[ ]:


study_rf = optuna.create_study(direction='minimize')
study_rf.optimize(objective_rf, n_trials=20)


# In[ ]:


rf_params = study_rf.best_params
print(rf_params)


# In[ ]:


def objective_knn(trial):
    param_grid = {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 30),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "metric": trial.suggest_categorical("metric", ["euclidean", "manhattan"])
    }

    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    cv_errors = np.empty(cv.get_n_splits())

    for idx, (train_idx, test_idx) in enumerate(cv.split(X_train, y_train)):
        X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_tr = X_train_fold.sample(frac=0.2, random_state=42)
        y_tr = y_train_fold.loc[X_tr.index]

        X_val = X_train.iloc[test_idx]
        y_val = y_train.iloc[test_idx]
        X_tr_processed = preprocessor.fit_transform(X_tr)
        X_val_processed = preprocessor.transform(X_val)


        model = Pipeline([
            ("scaler", StandardScaler(with_mean=False)),
            ("knn", KNeighborsRegressor(**param_grid))
        ])

        model.fit(X_tr_processed, y_tr)
        preds = model.predict(X_val_processed)
        cv_errors[idx] = mean_squared_error(y_val, preds)

    return np.mean(cv_errors)


# In[ ]:


study_knn = optuna.create_study(direction='minimize')
study_knn.optimize(objective_knn, n_trials=20)


# In[ ]:


knn_params = study_knn.best_params
print(knn_params)


# In[ ]:


get_ipython().system('pip install optuna-integration[lightgbm]')


# In[ ]:


import lightgbm as lgb
from optuna.integration import LightGBMPruningCallback


# In[ ]:


X_train.columns = X_train.columns.str.replace(r"[^\w]", "_", regex=True)
X_test.columns = X_test.columns.str.replace(r"[^\w]", "_", regex=True)


# In[ ]:


from optuna.pruners import MedianPruner

def objective(trial):
    params = {
        "objective": "regression",
        "metric": "l2",
        "n_estimators": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 127),
        "max_depth": trial.suggest_int("max_depth", 3, 6),
        "min_child_samples": trial.suggest_int("min_child_samples", 50, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.7, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 1.0),
        "verbosity": -1,
        "n_jobs": -1
    }

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    val_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        X_tr_processed = preprocessor.fit_transform(X_tr)
        X_val_processed = preprocessor.transform(X_val)
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr_processed, y_tr)
        preds = model.predict(X_val_processed)

        mse = mean_squared_error(y_val, preds)
        val_scores.append(mse)
        trial.report(mse, fold_idx)
        if trial.should_prune():
            raise optuna.TrialPruned()


    return np.mean(val_scores)

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=MedianPruner(
        n_startup_trials=3,
        n_warmup_steps=2,
        interval_steps=1
    )
)

study.optimize(objective, n_trials=20, timeout=3600)


# In[ ]:


best_params = study.best_params
print(best_params)


# In[ ]:


if isinstance(X_train, np.ndarray):
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)


# In[ ]:


from sklearn.ensemble import StackingRegressor

base_models = [
    ('ridge', Ridge(alpha=1, max_iter=1000, solver='auto', random_state=42)),
    ('rf', RandomForestRegressor(
        n_estimators=237,
        max_depth=27,
        min_samples_split=17,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )),
    ("lgb", lgb.LGBMRegressor(
    learning_rate=0.068,
    num_leaves=31,
    max_depth=5,
    min_data_in_leaf=20,
    feature_fraction=0.79,
    bagging_fraction=0.86,
    bagging_freq=5,
    lambda_l1=0.1,
    n_estimators=100,
    random_state=42,
    verbose=-1
    ))
]

stacked_model = StackingRegressor(
    estimators=base_models,
    final_estimator=KNeighborsRegressor(
        n_neighbors=9,
        weights='distance',
        metric='euclidean',
        n_jobs=-1
    ),
    cv=3,
    n_jobs=-1,
    passthrough=True
)


# In[ ]:


num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ],
    verbose_feature_names_out=False
)

stacked_model.set_params(verbose=0)


# In[ ]:


final_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('stacked_model', stacked_model)
])

final_pipeline.fit(X_train, y_train)


# In[ ]:


y_fpred = final_pipeline.predict(X_test)


# In[ ]:


mean_squared_error(y_test, y_fpred)


# In[ ]:


r2_score(y_test, y_fpred)


# ##6. Кластеризация
# 

# In[ ]:


from sklearn.cluster import KMeans
features = ['price', 'price_per_night', 'center_distance', 'minimum_nights', 'availability_365', 'reviews_per_month']
X_clust = df[features]


# Выбираем те столбцы, которые будут использоваться для кластеризации. Это позволяет сфокусироваться на признаках, влияющих на распределение объектов (цена, цена за ночь, расстояние до центра, минимальное количество ночей, доступность за год и отзывы).

# In[ ]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clust)
eps_values = np.arange(0.05, 2.0, 0.15)
min_samples = 5


# In[ ]:


from sklearn.metrics import silhouette_score


# Масштабирование признаков необходимо для метода К ближайших соседей.

# In[ ]:


wcss = []
silhouette_scores = []
K_range=range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_avg = silhouette_score(X_scaled, labels)
    silhouette_scores.append(silhouette_avg)


# In[ ]:


fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(K_range, wcss, marker='o', color='tab:blue')
ax1.set_xlabel('Количество кластеров')
ax1.set_ylabel('WCSS', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')
ax1.set_title('Метод локтя и силуэт для выбора числа кластеров')

ax2 = ax1.twinx()
ax2.plot(K_range, silhouette_scores, marker='s', color='tab:red')
ax2.set_ylabel('сsilhouette', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')


# Как можно наблюдать из синего графика, оптимальным количеством кластеров является число 10, где wcss минимален. По видимому, график и далее будет показавыть нисходящее движение, но темп снижения замедляется, поскольку у него есть горизонтальная асимптота. Так что дальнейшие значения не сильно изменят ошибку.
# Оптимальный силуэт достигается при 4 кластерах, поэтому столько и следует взять.

# In[ ]:


k = 4
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)


# In[ ]:


df['cluster_k'] = clusters


# In[ ]:


plt.scatter(df['center_distance'], df['price_per_night'],
            c=df['cluster_k'], cmap='viridis', alpha=0.6, edgecolors='w', s=50)
plt.xlabel('Расстояние до центра (км)')
plt.ylabel('Цена за ночь')
plt.title('Кластеры по расстоянию и цене')
plt.colorbar(label='Кластер')


# На графике воспрнять 4 кластера довольно тяжело, хотя видны некоторые границы разделения.

# In[ ]:


centers_scaled = kmeans.cluster_centers_
centers = scaler.inverse_transform(centers_scaled)
centers_df = pd.DataFrame(centers, columns=features)
centers_df


# Первый кластер характеризуется относительно низкой ценой
#  около 136 долларов, но при этом цена за ночь всего 8, что говорит о долгосрочной аренде и значительных скидках. Минимальное количество ночей выше среднего, удаленность примерно 6,5 км от центра. Доступность высокая, почти весь год, а количество отзывов в месяц низкое.
# 
# Второй кластер отличается умеренной стоимостью около 100 долларов, с ценой за ночь 44. Это больше похоже на краткосрочную аренду. Минимальное количество ночей небольшое, около 3, доступность низкая. Объекты находятся дальше — более 7 км от центра.
# 
# Третий кластер это дорогие варианты с ценой около 253 долларов и ценой за ночь более 135. Эти предложения ближе к центру , минимальное количество ночей примерно 2, доступность умеренная, а отзывов больше, чем в других кластерах.
# 
# Четвёртый кластер снова относится к более доступному сегменту цена около 92 долларов, цена за ночь 54. Удаленность максимальная почти 10 км, минимальные ночи около 2, доступность высокая, а количество отзывов в месяц самое большое, что указывает на популярность среди гостей.

# In[ ]:


plt.figure(figsize=(12,6))
sns.boxplot(x=df["cluster_k"], y=df["price"])
plt.title("Распределение цен по кластерам")
plt.show()


# Здесь как раз можно видеть что средняя цена во 2 кластре значительно выделяется.

# In[ ]:


plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="cluster_k", y="reviews_per_month", palette="coolwarm")
plt.title("Популярность кластеров по числу отзывов в месяц")
plt.xlabel("Кластер")
plt.ylabel("Среднее число отзывов в месяц")
plt.show()



# 3 и 4 кластеры самые популярные.

# In[ ]:


plt.figure(figsize=(12, 6))
sns.boxplot(data=df, x="cluster_k", y="availability_365", palette="magma")
plt.title("Доступность жилья в кластерах")
plt.xlabel("Кластер")
plt.ylabel("Число доступных дней в году")
plt.show()



# 0 и 3 кластеры характеризуются высокой доступностью жилья.
Возможные дальнейшие действия:

**Расширенная гиперпараметрическая оптимизация**:

Применение более детального подбора параметров, например, через,  RandomizedSearchCV, ведб  он может выявить еще более оптимальные настройки.
Также расширение диапазона параметров и использование более сложных регуляризаций.

**Использование альтернативных моделей**:

Возможность применить ансамблевые методы (например, бэггинг или градиентный бустинг), которые способны уловить нелинейные зависимости.

**Дополнительное создание и отбор признаков**:

Генерация новых признаков, например, взаимодействий между существующими или полиномиальных признаков, может помочь модели лучше уловить сложные зависимости.
Применение методов отбора признаков (например, Lasso, деревья решений для определения важности признаков) поможет устранить лишний шум и повысить качество модели.

**Углубленный анализ данных и обработка выбросов**:

Анализ остатков модели позволит выявить систематические ошибки и дополнительные закономерности, которые модель не смогла уловить.

**Более тщательная валидация**:

Использование k-fold кросс-валидации для оценки стабильности модели.
Построение графиков остатков (residual plots) для проверки предположений линейной регрессии

**Использование альтернативных методов кластеризации**

Например, DBSCAN или иерархической кластеризации для более точного сегментирования.Применение результатов кластеризации для сегментации клиентов и создания персонализированных предложений
Интеграция дополнительных источников данных:

**Объединение с данными с новыми данными**
Например, о событиях, туристической активности или экономических индикаторах для повышения точности прогнозов
Анализ отзывов и социальных медиа для определения восприятия предложений и улучшения качества рекомендаций


**Разработка рекомендательной системы:**

Сегментация объектов в сочетании с анализом предпочтений пользователей может использоваться для создания системы рекомендаций, предлагающей наиболее подходящие варианты жилья исходя из бюджета и предпочтений клиента