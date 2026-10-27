"""
simulate_batches.py
Splits the raw application_train.csv into pools that simulate a
continuously-arriving-data MLOps scenario from a static dataset:

  1. A fixed evaluation set -- carved out once, never touched again.
     Every model (champion and every challenger) is scored against
     this same set, so comparisons stay fair across retraining rounds.
  2. An initial training pool -- what the first ("champion") model
     is trained on.
  3. Several "batches" -- held back entirely at first, released one
     at a time to simulate new data arriving. One batch is perturbed
     to guarantee a visible, detectable drift signal for the demo.

This script only splits and (optionally) perturbs the RAW data. It does
not clean, engineer features, or encode anything -- that still happens
in preprocessing.py, applied separately to whichever pool you're about
to train or evaluate on.

Usage:
    python simulate_batches.py
Expects:
    data/application_train.csv
Produces:
    data/simulated/eval_fixed.csv
    data/simulated/initial_train.csv
    data/simulated/batch_1.csv
    data/simulated/batch_2.csv   (perturbed -- simulates drift)
    data/simulated/batch_3.csv
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/application_train.csv"
OUT_DIR = "data/simulated"

# Fraction of the full dataset reserved as the permanent evaluation set.
EVAL_FRACTION = 0.20
# Of what's left after the eval set, how much goes to the initial pool
# vs. being held back for the simulated batches.
INITIAL_FRACTION_OF_REMAINDER = 0.5
N_BATCHES = 3
RANDOM_STATE = 42


def load_raw(path: str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def carve_eval_set(df: pd.DataFrame):
    """Split off the fixed evaluation set first, stratified on TARGET so
    the ~8% default rate is preserved. Everything downstream works only
    with the remainder -- the eval set is never touched again."""
    remainder, eval_fixed = train_test_split(
        df,
        test_size=EVAL_FRACTION,
        random_state=RANDOM_STATE,
        stratify=df["TARGET"],
    )
    print(f"Fixed evaluation set: {eval_fixed.shape[0]} rows (held out permanently)")
    print(f"Remainder for initial pool + batches: {remainder.shape[0]} rows")
    return remainder, eval_fixed


def split_initial_and_batches(remainder: pd.DataFrame, n_batches: int = N_BATCHES):
    """Split the remainder into the initial training pool and n_batches
    equally-sized simulated incoming batches."""
    initial, held_back = train_test_split(
        remainder,
        test_size=1 - INITIAL_FRACTION_OF_REMAINDER,
        random_state=RANDOM_STATE,
        stratify=remainder["TARGET"],
    )
    print(f"Initial training pool: {initial.shape[0]} rows")

    # Shuffle, then slice into n roughly-equal batches.
    held_back = held_back.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    batch_indices = np.array_split(held_back.index, n_batches)
    batches = [held_back.loc[idx].reset_index(drop=True) for idx in batch_indices]

    for i, batch in enumerate(batches, start=1):
        print(f"Batch {i}: {batch.shape[0]} rows")

    return initial, batches


def inject_synthetic_drift(batch: pd.DataFrame) -> pd.DataFrame:
    """
    Perturb one batch to deliberately create a detectable distribution
    shift, since the raw dataset has no real time ordering to exploit.
    This is what Evidently (or a PSI/KS-test check) should be able to
    flag when this batch is "released" into the pipeline.

    Document this clearly in the report: the drift here is simulated,
    not naturally occurring in the data.
    """
    batch = batch.copy()

    # Shift income down -- simulates a cohort of applicants with
    # generally lower reported income than the training distribution.
    batch["AMT_INCOME_TOTAL"] = batch["AMT_INCOME_TOTAL"] * 0.85

    # Shift external credit scores down slightly -- simulates a riskier
    # incoming applicant pool.
    for col in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]:
        batch[col] = (batch[col] * 0.90).clip(lower=0, upper=1)

    print("Synthetic drift injected: AMT_INCOME_TOTAL x0.85, EXT_SOURCE_* x0.90")
    return batch


def save_all(eval_fixed, initial, batches, out_dir: str = OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    eval_fixed.to_csv(f"{out_dir}/eval_fixed.csv", index=False)
    initial.to_csv(f"{out_dir}/initial_train.csv", index=False)
    for i, batch in enumerate(batches, start=1):
        batch.to_csv(f"{out_dir}/batch_{i}.csv", index=False)

    print(f"\nSaved all pools to {out_dir}/")


def run():
    df = load_raw()
    remainder, eval_fixed = carve_eval_set(df)
    initial, batches = split_initial_and_batches(remainder)

    # Perturb the second batch only -- the first batch stays "clean" so
    # you can demo both a false-alarm check (batch 1: no drift expected)
    # and a true detection (batch 2: drift expected).
    batches[1] = inject_synthetic_drift(batches[1])

    save_all(eval_fixed, initial, batches)


if __name__ == "__main__":
    run()