"""
detect_drift.py
Compares a new batch's raw feature distributions against a reference
distribution (the initial training pool) using the Population Stability
Index (PSI), and flags whether retraining should be triggered.

This checks DATA drift (did the input distributions shift), which is a
separate question from MODEL PERFORMANCE drift (did AUC on a fixed eval
set get worse). retrain_pipeline.py currently only measures the latter --
this module adds the former, and is what should gate whether a round
retrains at all.

PSI interpretation (standard thresholds used across the industry):
    PSI < 0.1   -> no significant shift
    0.1 - 0.2   -> moderate shift, worth monitoring
    PSI > 0.2   -> significant shift, retraining is recommended

Usage (standalone, to inspect all simulated batches):
    python detect_drift.py
Expects:
    data/simulated/initial_train.csv, batch_1.csv, batch_2.csv, batch_3.csv
"""

import numpy as np
import pandas as pd

SIM_DIR = "data/simulated"

# Numeric columns checked for drift. Chosen because they're present in
# every row, rarely missing, and are the features most likely to reflect
# a genuine change in the incoming applicant population.
DRIFT_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

PSI_THRESHOLD = 0.2


def calculate_psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """
    Standard PSI calculation: bin the reference distribution into
    `bins` quantile buckets, then compare what fraction of the current
    distribution falls into each of those same buckets.
    """
    reference = reference.dropna()
    current = current.dropna()

    breakpoints = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(breakpoints) < 3:
        # Not enough distinct values to form meaningful bins (e.g. a
        # near-constant column) -- treat as no drift rather than error.
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    # Avoid divide-by-zero / log(0) for empty bins.
    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / ref_counts.sum())
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / cur_counts.sum())

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def check_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    features: list = None,
    threshold: float = PSI_THRESHOLD,
) -> dict:
    """
    Computes PSI for each feature, returning a dict with per-feature
    scores and an overall drift_detected flag (True if ANY feature
    crosses the threshold).
    """
    features = features or DRIFT_FEATURES
    scores = {}

    for feature in features:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        scores[feature] = calculate_psi(reference_df[feature], current_df[feature])

    drift_detected = any(score > threshold for score in scores.values())

    return {
        "scores": scores,
        "drift_detected": drift_detected,
        "threshold": threshold,
    }


def print_drift_report(batch_name: str, result: dict):
    print(f"\nDrift check: {batch_name}")
    for feature, score in result["scores"].items():
        flag = "DRIFTED" if score > result["threshold"] else "stable"
        print(f"  {feature:<20} PSI = {score:.4f}  [{flag}]")

    if result["drift_detected"]:
        print(f"  -> Overall: DRIFT DETECTED (threshold = {result['threshold']})")
    else:
        print(f"  -> Overall: no significant drift (threshold = {result['threshold']})")


if __name__ == "__main__":
    reference_df = pd.read_csv(f"{SIM_DIR}/initial_train.csv")

    for i in range(1, 4):
        batch_df = pd.read_csv(f"{SIM_DIR}/batch_{i}.csv")
        result = check_drift(reference_df, batch_df)
        print_drift_report(f"batch_{i}", result)