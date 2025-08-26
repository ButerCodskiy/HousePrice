import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.linear_model import Ridge
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
import lightgbm as lgb
import optuna

def get_preprocessor(X_train):
    """Возвращает ColumnTransformer для предобработки числовых и категориальных признаков."""
    num_cols = X_train.select_dtypes(include=np.number).columns
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        verbose_feature_names_out=False
    )
    return preprocessor

def train_ridge_model(X_train, y_train, preprocessor):
    """Обучает модель Ridge с использованием GridSearchCV."""
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("ridge", Ridge())
    ])

    param_grid = {
        "ridge__alpha": [0.01, 0.1, 1, 10, 100],
        "ridge__solver": ["auto", "lsqr", "sparse_cg", "sag"],
        "ridge__max_iter": [1000, 5000]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="neg_mean_squared_error", n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search

def optimize_lgbm(X_train, y_train, preprocessor):
    """Оптимизирует гиперпараметры LightGBM с использованием Optuna."""
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
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3,
            n_warmup_steps=2,
            interval_steps=1
        )
    )
    study.optimize(objective, n_trials=20, timeout=3600)
    return study.best_params

def train_stacked_model(X_train, y_train, preprocessor):
    """Обучает стековую модель (StackingRegressor)."""
    base_models = [
        ("ridge", Ridge(alpha=1, max_iter=1000, solver="auto", random_state=42)),
        ("rf", RandomForestRegressor(
            n_estimators=237,
            max_depth=27,
            min_samples_split=17,
            min_samples_leaf=2,
            max_features="sqrt",
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
            weights="distance",
            metric="euclidean",
            n_jobs=-1
        ),
        cv=3,
        n_jobs=-1,
        passthrough=True
    )

    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("stacked_model", stacked_model)
    ])

    final_pipeline.fit(X_train, y_train)
    return final_pipeline

def evaluate_model(model, X_test, y_test):
    """Оценивает модель и возвращает MSE и R2."""
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2
