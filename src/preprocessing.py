"""
preprocessing.py
Cleans application_train.csv, fixes known data issues, engineers the
ratio features that matter most for this dataset, and prepares train/test
splits ready for a LightGBM or XGBoost model.

"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/application_train.csv"
OUT_DIR = "data/processed"

# Loads the data from the data directory
def load_data(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    print("Target balance:\n", df["TARGET"].value_counts(normalize=True))
    return df


def fix_known_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Fix data problems specific to this dataset."""
    df = df.copy()

    # DAYS_EMPLOYED has a placeholder value of 365243 for people who are
    # not employed (pensioners, unemployed). This is not a real number of
    # days -- treat it as missing instead of letting the model learn from
    # a nonsense value.
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace({365243: np.nan})

    # DAYS_BIRTH and DAYS_EMPLOYED are stored as negative day counts.
    # Convert to a more readable positive "years" scale.
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365.25
    df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"] / 365.25

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the ratio and aggregate features that are consistently the
    strongest predictors in this dataset."""
    df = df.copy()

    # Loan-to-income and payment-burden ratios.
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_ANNUITY_RATIO"] = df["AMT_CREDIT"] / df["AMT_ANNUITY"]

    # How much of the client's life has been spent employed -- a proxy
    # for employment stability.
    df["EMPLOYED_AGE_RATIO"] = df["YEARS_EMPLOYED"] / df["AGE_YEARS"]

    # The three EXT_SOURCE columns are strong external risk scores but are
    # often partially missing. Rather than dropping rows, capture both
    # how many are present and their average.
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCE_COUNT"] = df[ext_cols].notnull().sum(axis=1)
    df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)

    return df


def prepare_features(df: pd.DataFrame, encode_for: str = "lightgbm"):
    """
    Split into x / y and encode categorical.

    encode_for="lightgbm" -> cast object columns to pandas 'category' dtype
        and let LightGBM handle them natively (no manual encoding needed).
    encode_for="xgboost"  -> one-hot encode object columns, since XGBoost's
        broad-compatibility path expects numeric input.
    """
    df = df.copy()
    y = df["TARGET"]
    x = df.drop(columns=["TARGET", "SK_ID_CURR"])

    cat_cols = x.select_dtypes(include="object").columns.tolist()
    print(f"{len(cat_cols)} categorical columns found")

    if encode_for == "lightgbm":
        for col in cat_cols:
            x[col] = x[col].astype("category")
    elif encode_for == "xgboost":
        x = pd.get_dummies(x, columns=cat_cols, drop_first=True)
        x = sanitize_feature_names(x)
    else:
        raise ValueError("encode_for must be 'lightgbm' or 'xgboost'")

    return x, y

def sanitize_feature_names(x: pd.DataFrame) -> pd.DataFrame:
    """LightGBM rejects feature names containing JSON special characters
    (commas, colons, brackets, quotes). One-hot encoding can generate
    exactly these from category values like "Spouse, partner" -- replace
    anything that isn't a letter, digit, or underscore with an underscore."""
    x = x.copy()
    x.columns = [re.sub(r"[^A-Za-z0-9_]+", "_", col) for col in x.columns]
    return x


def split_and_save(x: pd.DataFrame, y: pd.Series, out_dir: str = OUT_DIR):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    os.makedirs(out_dir, exist_ok=True)
    x_train.to_csv(f"{out_dir}/x_train.csv", index=False)
    x_test.to_csv(f"{out_dir}/x_test.csv", index=False)
    y_train.to_csv(f"{out_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{out_dir}/y_test.csv", index=False)

    print(f"Saved processed splits to {out_dir}/")
    print(f"x_train: {x_train.shape}, x_test: {x_test.shape}")
    return x_train, x_test, y_train, y_test

def process_file(path: str, encode_for: str = "lightgbm"):
    """Clean, fix, and feature-engineer a given CSV, returning x, y with
    no train/test split. Used for the initial pool and each new batch
    in the retraining simulation."""
    df = load_data(path)
    df = fix_known_issues(df)
    df = engineer_features(df)
    x, y = prepare_features(df, encode_for=encode_for)
    return x, y

def run_pipeline(encode_for: str = "lightgbm"):
    df = load_data()
    df = fix_known_issues(df)
    df = engineer_features(df)
    x, y = prepare_features(df, encode_for=encode_for)
    return split_and_save(x, y)


if __name__ == "__main__":
    run_pipeline(encode_for="lightgbm")