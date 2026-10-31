"""
train.py
Trains a class-weighted LightGBM model on the preprocessed Home Credit
data, evaluates it with stratified cross-validation, reports AUC (not
accuracy -- the target is ~8% positive, so accuracy is misleading),
and saves the trained model for later use with SHAP / FastAPI / MLflow.

Usage:
    python train.py
Expects:
    data/processed/x_train.csv, y_train.csv, x_test.csv, y_test.csv
    (produced by preprocessing.py)
Produces:
    models/lgbm_model.pkl
"""

import os
import joblib
import numpy as np
import pandas as pd
import mlflow
import mlflow.lightgbm
import json
from mlflow.tracking import MlflowClient
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, classification_report


BEST_PARAMS_PATH = "models/best_params.json"
DATA_DIR = "data/processed"
MODEL_DIR = "models"


def load_processed(data_dir: str = DATA_DIR):
    x_train = pd.read_csv(f"{data_dir}/x_train.csv")
    x_test = pd.read_csv(f"{data_dir}/x_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze("columns")
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").squeeze("columns")

    # CSV round-tripping loses the 'category' dtype LightGBM needs for
    # native categorical handling -- restore it for every object column.
    cat_cols = x_train.select_dtypes(include="object").columns.tolist()
    for col in cat_cols:
        x_train[col] = x_train[col].astype("category")
        x_test[col] = x_test[col].astype("category")

    return x_train, x_test, y_train, y_test


def cross_validate(x: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> list:
    """Stratified K-fold CV so each fold keeps the same ~8% default rate
    as the full dataset, giving a realistic estimate of AUC."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y), start=1):
        x_tr, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = build_model()
        model.fit(
            x_tr, y_tr,
            eval_set=[(x_val, y_val)],
            eval_metric="auc",
            callbacks=[],
        )

        val_probs = model.predict_proba(x_val)[:, 1]
        auc = roc_auc_score(y_val, val_probs)
        fold_scores.append(auc)
        print(f"Fold {fold}: AUC = {auc:.4f}")

    print(f"\nMean CV AUC: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")
    return fold_scores


def build_model() -> LGBMClassifier:
    """
    LightGBM with class weighting instead of SMOTE: is_unbalance=True
    tells LightGBM to automatically weight the minority (default) class
    more heavily in the loss function, which is usually a stronger and
    simpler first move than generating synthetic rows.

    If models/best_params.json exists (produced by optimize.py), those
    Optuna-tuned hyperparameters are used instead of the hand-picked
    defaults below.
    """
    default_params = dict(
        n_estimators=500,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
    )

    if os.path.exists(BEST_PARAMS_PATH):
        with open(BEST_PARAMS_PATH) as f:
            tuned_params = json.load(f)
        default_params.update(tuned_params)
        print(f"Using Optuna-tuned hyperparameters from {BEST_PARAMS_PATH}")

    return LGBMClassifier(
        **default_params,
        is_unbalance=True,
        random_state=42,
        n_jobs=-1,
    )


def train_final_model(x_train, y_train, x_test, y_test):
    model = build_model()
    model.fit(
        x_train, y_train,
        eval_set=[(x_test, y_test)],
        eval_metric="auc",
    )

    test_probs = model.predict_proba(x_test)[:, 1]
    test_preds = (test_probs >= 0.5).astype(int)

    print("\nHeld-out test set performance:")
    print(f"AUC: {roc_auc_score(y_test, test_probs):.4f}")
    print(f"F1 (default class): {f1_score(y_test, test_preds):.4f}")
    print(classification_report(y_test, test_preds, target_names=["Repaid", "Default"]))

    return model

def train_on_pool(x: pd.DataFrame, y: pd.Series, run_name: str = "training_run"):
    model = build_model()
    params = model.get_params()

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            "n_estimators": params["n_estimators"],
            "learning_rate": params["learning_rate"],
            "num_leaves": params["num_leaves"],
            "min_child_samples": params["min_child_samples"],
        })
        model.fit(x, y)
        mlflow.log_metric("train_rows", len(x))
        mlflow.lightgbm.log_model(model, name="model")
        run_id = run.info.run_id

    return model, run_id

REGISTERED_MODEL_NAME = "smlops_credit_risk_champion"


def register_as_production(run_id: str, model_name: str = REGISTERED_MODEL_NAME):
    """Registers the model logged under run_id as a new version of
    model_name, promotes it to the 'Production' stage, and archives any
    previously-production version -- so the registry always reflects
    exactly one current production model."""
    client = MlflowClient()
    model_uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(model_uri, model_name)
    version = result.version

    for mv in client.get_latest_versions(model_name, stages=["Production"]):
        client.transition_model_version_stage(
            name=model_name, version=mv.version, stage="Archived"
        )

    client.transition_model_version_stage(
        name=model_name, version=version, stage="Production"
    )
    print(f"Registered {model_name} version {version} and promoted to Production")
    return version

def save_model(model, path: str = f"{MODEL_DIR}/lgbm_model.pkl"):
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def promote_or_reject(champion_probs, challenger_probs, y_eval, threshold: float = 0.005):
    champion_auc = roc_auc_score(y_eval, champion_probs)
    challenger_auc = roc_auc_score(y_eval, challenger_probs)
    print(f"Champion AUC: {champion_auc:.4f} | Challenger AUC: {challenger_auc:.4f}")

    if challenger_auc > champion_auc + threshold:
        print(f"Challenger beats champion by > {threshold}: PROMOTED")
        return True
    print("Challenger did not beat champion by threshold: REJECTED")
    return False

if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_processed()

    print("Running 5-fold cross-validation on training data...")
    cross_validate(x_train, y_train)

    print("\nTraining final model on full training set...")
    final_model = train_final_model(x_train, y_train, x_test, y_test)

    save_model(final_model)