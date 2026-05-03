"""
Train regression models to predict energy usage.

Two modeling tracks:
1. pre-run model:
   uses only features known before inference
2. post-run model:
   uses features known after inference (more explanatory / usually more accurate)
"""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import CLEAN_RUNS_CSV, MODELS_DIR, PLOTS_DIR

# Attempt XGBoost

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

TARGET = "energy_joules"


def build_preprocessor(num_features, cat_features):
    """
    Build a preprocessing pipeline:
    - numeric: median impute + scale
    - categorical: most-frequent impute + one-hot encode
    """
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_features),
        ("cat", cat_pipe, cat_features),
    ])


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """
    Fit one model and return metrics.
    """
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5

    metrics = {
        "model": name,
        "r2": r2_score(y_test, preds),
        "mae": mean_absolute_error(y_test, preds),
        "rmse": rmse,
    }
    return model, metrics


def run_experiment(df: pd.DataFrame, feature_set_name: str, features: list[str]):
    """
    Train multiple regression models for a given feature set.
    Save results, best model, and feature importance plots.
    """
    X = df[features].copy()
    y = df[TARGET].copy()

    cat_features = [c for c in features if X[c].dtype == "object"]
    num_features = [c for c in features if c not in cat_features]

    preprocessor = build_preprocessor(num_features, cat_features)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
        )

    # Standard 80/20 split
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    results = []
    fitted = {}

    for name, estimator in models.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", estimator),
        ])

        fitted_model, metrics = evaluate_model(
            name,
            pipe,
            X_train,
            X_test,
            y_train,
            y_test,
        )
        fitted[name] = fitted_model
        results.append(metrics)

    # Rank models by R² score
    
    results_df = pd.DataFrame(results).sort_values("r2", ascending=False)

    # Save model comparison table
    
    results_path = MODELS_DIR / f"{feature_set_name}_results.csv"
    results_df.to_csv(results_path, index=False)

    # Save best model
    
    best_name = results_df.iloc[0]["model"]
    best_model = fitted[best_name]
    joblib.dump(best_model, MODELS_DIR / f"{feature_set_name}_best_model.joblib")

    # Try feature importance plot for a tree-based model
    
    tree_pipe = None
    chosen_tree_name = None
    for candidate in ["ExtraTrees", "RandomForest", "XGBoost"]:
        if candidate in fitted:
            tree_pipe = fitted[candidate]
            chosen_tree_name = candidate
            break

    if tree_pipe is not None:
        model = tree_pipe.named_steps["model"]
        try:
            importances = model.feature_importances_

            plt.figure(figsize=(8, 4))
            plt.bar(range(len(importances)), importances)
            plt.title(f"{feature_set_name} - {chosen_tree_name} Feature Importances")
            plt.xlabel("Encoded Feature Index")
            plt.ylabel("Importance")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"{feature_set_name}_{chosen_tree_name}_feature_importance.png")
            plt.close()
        except Exception:
            pass

    return results_df


def main():
    """
    Train both pre-run and post-run models.
    """
    df = pd.read_csv(CLEAN_RUNS_CSV)

    # Features available before inference
    
    pre_run_features = [
        "prompt_style",
        "task_type",
        "dataset_name",
        "input_tokens",
    ]

    # Features available after inference
    
    post_run_features = [
        "prompt_style",
        "task_type",
        "dataset_name",
        "input_tokens",
        "output_tokens",
        "latency_sec",
    ]

    # Train pre-run predictive models
    
    pre_results = run_experiment(df, "pre_run", pre_run_features)

    # Train post-run predictive models
    
    post_results = run_experiment(df, "post_run", post_run_features)

    print("=== PRE-RUN RESULTS ===")
    print(pre_results)

    print("\n=== POST-RUN RESULTS ===")
    print(post_results)


if __name__ == "__main__":
    main()
