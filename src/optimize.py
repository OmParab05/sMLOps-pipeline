"""
optimize.py
Uses Optuna to search for better LightGBM hyperparameters than the
hand-picked defaults currently in train.py's build_model(), optimizing
directly for AUC via stratified cross-validation on the initial
training pool.

Usage:
    python optimize.py
Expects:
    data/simulated/initial_train.csv
Produces:
    models/best_params.json
    (automatically picked up by train.py's build_model() on every future
    run, including inside retrain_pipeline.py -- no other wiring needed)
"""

import json
import os

import numpy as np
import optuna
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from preprocessing import process_file

DATA_PATH = "data/simulated/initial_train.csv"
OUT_PATH = "models/best_params.json"
N_TRIALS = 30
N_SPLITS = 5


def objective(trial, x, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    fold_scores = []

    for train_idx, val_idx in skf.split(x, y):
        x_tr, x_val = x.iloc[train_idx], x.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(
            **params,
            is_unbalance=True,
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
        model.fit(x_tr, y_tr)
        probs = model.predict_proba(x_val)[:, 1]
        fold_scores.append(roc_auc_score(y_val, probs))

    return float(np.mean(fold_scores))


def run():
    x, y = process_file(DATA_PATH, encode_for="xgboost")
    print(f"Tuning on {x.shape[0]} rows, {x.shape[1]} features")

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, x, y), n_trials=N_TRIALS)

    print(f"\nBest AUC found: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(study.best_params, f, indent=2)
    print(f"\nSaved best params to {OUT_PATH}")


if __name__ == "__main__":
    run()