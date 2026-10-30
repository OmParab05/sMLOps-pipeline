"""
check_serving_skew.py
Proves the API scores an applicant exactly like the training pipeline does
(no training/serving skew), and shows how the 0.4 / 0.7 decision bands
split the evaluation set.

Start the API first, then from the PROJECT ROOT:
    python src/check_serving_skew.py
"""

import sys

import numpy as np
import pandas as pd
import requests

from preprocessing import process_file
from serving import CORE_RAW, DECISION_BANDS, decide, load_champion, predict_proba

API_URL = "http://127.0.0.1:8000/predict"
EVAL_PATH = "data/simulated/eval_fixed.csv"
TOLERANCE = 1e-6


def native(v):
    return v.item() if hasattr(v, "item") else v


def row_to_payload(r: pd.Series) -> dict:
    """Turn one raw dataset row into an API request (core fields + extras)."""
    payload = {}
    for field, raw_col in CORE_RAW.items():
        v = r[raw_col]
        payload[field] = None if pd.isna(v) else native(v)
    payload["age_years"] = float(-r["DAYS_BIRTH"] / 365.25)
    payload["years_employed"] = (
        None if r["DAYS_EMPLOYED"] == 365243 else float(-r["DAYS_EMPLOYED"] / 365.25)
    )
    skip = set(CORE_RAW.values()) | {"TARGET", "SK_ID_CURR", "DAYS_BIRTH", "DAYS_EMPLOYED"}
    payload["extra_features"] = {
        col: native(v) for col, v in r.items() if col not in skip and not pd.isna(v)
    }
    return payload


def main():
    champion = load_champion()
    raw = pd.read_csv(EVAL_PATH)

    # Offline path = exactly what retrain_pipeline.py does for evaluation.
    x_eval, _ = process_file(EVAL_PATH, encode_for="xgboost")
    x_eval = x_eval.reindex(columns=champion["columns"], fill_value=0)
    offline = predict_proba(champion["model"], x_eval)

    order = np.argsort(offline)
    picks = [order[0], order[len(order) // 4], order[len(order) // 2],
             order[3 * len(order) // 4], order[-1]]

    print(f"\n{'row':>6} {'offline':>9} {'api':>9} {'diff':>10}  decision")
    worst = 0.0
    for i in picks:
        try:
            resp = requests.post(API_URL, json=row_to_payload(raw.iloc[i]), timeout=30)
        except requests.ConnectionError:
            sys.exit("Could not reach the API. Start it first (see file header).")
        if resp.status_code != 200:
            sys.exit(f"Row {i}: API returned {resp.status_code}: {resp.text}")
        api_p = resp.json()["default_probability"]
        diff = abs(api_p - offline[i])
        worst = max(worst, diff)
        print(f"{i:>6} {offline[i]:>9.4f} {api_p:>9.4f} {diff:>10.6f}  {resp.json()['decision']}")

    # API rounds to 4 decimals, so allow that much slack.
    print("\nSKEW CHECK:", "PASS" if worst < 1e-4 else "FAIL", f"(max diff {worst:.6f})")

    print("\nProbability distribution on the evaluation set:")
    for q in (5, 25, 50, 75, 95):
        print(f"  {q:>2}th percentile: {np.percentile(offline, q):.3f}")
    decisions = pd.Series([decide(p) for p in offline]).value_counts(normalize=True)
    print(f"\nShare of applicants per decision band {DECISION_BANDS}:")
    print((decisions * 100).round(1).to_string())


if __name__ == "__main__":
    main()