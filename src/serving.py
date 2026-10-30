"""
serving.py
Everything the API needs to turn one raw applicant into a model score,
using exactly the same cleaning / feature engineering as training.

Why this file exists: preprocessing.prepare_features() cannot be used on
a single row. It needs a TARGET column, and pd.get_dummies(drop_first=True)
on one row would drop the only category of every column (so every
applicant would look like the "baseline" category). Instead we build the
one-hot columns by hand, against the champion's saved training columns
(models/champion_columns.json, written by retrain_pipeline.py).

Run everything from the PROJECT ROOT so MLflow resolves the same tracking
store the pipeline used.
"""

import json
import re
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from preprocessing import fix_known_issues, engineer_features

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "smlops_credit_risk_champion"
MODEL_PKL_PATH = ROOT / "models" / "champion_model.pkl"
COLUMNS_PATH = ROOT / "models" / "champion_columns.json"

# Business rules BR-4 / BR-5 / BR-6 -- the ONLY place thresholds live.
DECISION_BANDS = {"reject_above": 0.7, "review_from": 0.4}

# The 16 text columns of application_train.csv (matches "16 categorical
# columns found" in the pipeline log). The pipeline one-hot encodes these.
CATEGORICAL_COLUMNS = [
    "NAME_CONTRACT_TYPE", "CODE_GENDER", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
    "NAME_TYPE_SUITE", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "OCCUPATION_TYPE",
    "WEEKDAY_APPR_PROCESS_START", "ORGANIZATION_TYPE", "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE",
]

# API field name -> raw dataset column. age_years and years_employed are
# handled separately (they become DAYS_BIRTH / DAYS_EMPLOYED).
CORE_RAW = {
    "amt_income_total": "AMT_INCOME_TOTAL",
    "amt_credit": "AMT_CREDIT",
    "amt_annuity": "AMT_ANNUITY",
    "amt_goods_price": "AMT_GOODS_PRICE",
    "cnt_children": "CNT_CHILDREN",
    "ext_source_1": "EXT_SOURCE_1",
    "ext_source_2": "EXT_SOURCE_2",
    "ext_source_3": "EXT_SOURCE_3",
    "name_contract_type": "NAME_CONTRACT_TYPE",
    "code_gender": "CODE_GENDER",
    "flag_own_car": "FLAG_OWN_CAR",
    "flag_own_realty": "FLAG_OWN_REALTY",
    "name_income_type": "NAME_INCOME_TYPE",
    "name_education_type": "NAME_EDUCATION_TYPE",
    "name_family_status": "NAME_FAMILY_STATUS",
    "name_housing_type": "NAME_HOUSING_TYPE",
}

# Columns a caller may NOT pass through extra_features.
RESERVED = set(CORE_RAW.values()) | {"TARGET", "SK_ID_CURR", "DAYS_BIRTH", "DAYS_EMPLOYED"}


def sanitize_name(name: str) -> str:
    """Must match preprocessing.sanitize_feature_names."""
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def dummy_columns(columns) -> list:
    """The one-hot columns among the champion's training columns."""
    prefixes = tuple(sanitize_name(c) + "_" for c in CATEGORICAL_COLUMNS)
    return [c for c in columns if c.startswith(prefixes)]


def allowed_extra_keys(columns) -> set:
    numeric = set(columns) - set(dummy_columns(columns))
    return (numeric | set(CATEGORICAL_COLUMNS)) - RESERVED


def build_raw_row(fields: dict, extras: dict) -> pd.DataFrame:
    """One-row DataFrame in the raw application_train.csv layout."""
    row = dict(extras)
    for field, raw_col in CORE_RAW.items():
        value = fields.get(field)
        row[raw_col] = np.nan if value is None else value

    row["DAYS_BIRTH"] = -float(fields["age_years"]) * 365.25
    years = fields.get("years_employed")
    # None = not employed. The dataset encodes that as the placeholder
    # 365243, which fix_known_issues() turns into NaN + an "anomaly" flag.
    row["DAYS_EMPLOYED"] = 365243.0 if years is None else -float(years) * 365.25
    return pd.DataFrame([row])


def prepare_for_serving(df: pd.DataFrame, columns) -> pd.DataFrame:
    """Raw (already cleaned + engineered) rows -> the exact column layout
    the champion was trained on.

    - numeric columns we were not given -> NaN (LightGBM handles missing,
      and that is what training saw for missing values)
    - one-hot columns -> 0, then 1 where the category matches
    """
    columns = list(columns)
    dummies = dummy_columns(columns)
    x = pd.DataFrame(np.nan, index=df.index, columns=columns, dtype="float64")

    for col in columns:
        if col in df.columns and col not in dummies:
            x[col] = pd.to_numeric(df[col])

    if dummies:
        x[dummies] = 0.0
    for col in CATEGORICAL_COLUMNS:
        if col not in df.columns:
            continue
        for idx, value in df[col].items():
            if pd.isna(value):
                continue
            name = sanitize_name(f"{col}_{value}")
            if name in x.columns:
                x.at[idx, name] = 1.0
    return x


def features_from_applicant(fields: dict, extras: dict, columns) -> pd.DataFrame:
    df = build_raw_row(fields, extras)
    df = fix_known_issues(df)
    df = engineer_features(df)
    return prepare_for_serving(df, columns)


def predict_proba(model, x: pd.DataFrame) -> np.ndarray:
    """Works whether MLflow hands back an LGBMClassifier or a raw Booster."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    return np.asarray(model.predict(x))


def decide(p: float) -> str:
    if p > DECISION_BANDS["reject_above"]:
        return "reject"
    if p >= DECISION_BANDS["review_from"]:
        return "manual_review"
    return "approve"


def load_champion() -> dict:
    """Load the Production model from the MLflow registry (falling back to
    models/champion_model.pkl) plus the columns it was trained on."""
    if not COLUMNS_PATH.exists():
        raise FileNotFoundError(
            f"{COLUMNS_PATH} not found. Run `python src/retrain_pipeline.py` "
            "once after adding the champion_columns.json change."
        )
    with open(COLUMNS_PATH) as f:
        columns = json.load(f)

    try:
        model = mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}/Production")
        version = MlflowClient().get_latest_versions(MODEL_NAME, stages=["Production"])[0].version
        source = "mlflow_registry"
    except Exception as e:  # registry unreachable / empty
        print(f"Registry load failed ({e}); falling back to {MODEL_PKL_PATH}")
        model = joblib.load(MODEL_PKL_PATH)
        version, source = "unknown", "champion_model.pkl"

    return {"model": model, "columns": columns, "version": str(version), "source": source}