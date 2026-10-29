"""
retrain_pipeline.py
Runs the full simulated continuous-retraining loop:

  Round 0: train the champion on the initial pool, evaluate it on the
           fixed evaluation set.
  Round N: combine everything accumulated so far with the next batch,
           train a challenger on it, evaluate both the current champion
           and the new challenger on the SAME fixed evaluation set, and
           promote the challenger only if it beats the champion by the
           configured threshold.

This is the script that ties preprocessing.py and train.py together into
an actual pipeline -- it's also the logic that becomes your Prefect flow
later, once you add orchestration scheduling on top of it.

Usage:
    python retrain_pipeline.py
Expects:
    data/simulated/initial_train.csv, batch_1.csv, batch_2.csv, batch_3.csv,
    eval_fixed.csv (produced by simulate_batches.py)
Produces:
    models/champion_model.pkl (overwritten each time a challenger is promoted)
"""

import pandas as pd
import mlflow
from sklearn.metrics import roc_auc_score
from train import train_on_pool, promote_or_reject, save_model, register_as_production
from preprocessing import process_file
from train import train_on_pool, promote_or_reject, save_model
from detect_drift import check_drift

SIM_DIR = "data/simulated"
MODEL_PATH = "models/champion_model.pkl"
PROMOTION_THRESHOLD = 0.005
N_BATCHES = 3

# One-hot encoding ("xgboost" mode in process_file) is used here instead
# of LightGBM's native categorical dtype, specifically because different
# data pools (initial vs. combined-with-batches) can end up with slightly
# different category values present. One-hot columns can be reliably
# aligned across pools with align_columns() below; native categorical
# dtype codes can't be aligned this simply.
ENCODING = "xgboost"


def combine_raw_files(paths, out_path):
    """Concatenate the given raw CSVs into one file on disk, so
    process_file() can clean + engineer the combined pool as a whole."""
    frames = [pd.read_csv(p) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_path, index=False)
    return out_path


def align_columns(x: pd.DataFrame, reference_columns) -> pd.DataFrame:
    """Reindex x to match a reference column set, filling any column
    missing in x with 0. Needed because one-hot encoding can produce
    slightly different columns depending on which categories are present
    in a given pool."""
    return x.reindex(columns=reference_columns, fill_value=0)


def run():
    mlflow.set_experiment("smlops_credit_risk")

    print("=== Round 0: training initial champion ===")
    x_init, y_init = process_file(f"{SIM_DIR}/initial_train.csv", encode_for=ENCODING)
    champion, champion_run_id = train_on_pool(x_init, y_init, run_name="round_0_champion")
    champion_columns = x_init.columns

    x_eval_raw, y_eval = process_file(f"{SIM_DIR}/eval_fixed.csv", encode_for=ENCODING)
    x_eval_for_champion = align_columns(x_eval_raw, champion_columns)

    champion_probs = champion.predict_proba(x_eval_for_champion)[:, 1]
    champion_auc = roc_auc_score(y_eval, champion_probs)
    print(f"Initial champion AUC on fixed evaluation set: {champion_auc:.4f}")

    with mlflow.start_run(run_name="round_0_summary"):
        mlflow.log_metric("champion_auc", champion_auc)

    save_model(champion, MODEL_PATH)
    register_as_production(champion_run_id)

    # Reference distribution every batch is checked against -- the
    # initial pool, kept fixed throughout, same as detect_drift.py's
    # own standalone usage.
    reference_df = pd.read_csv(f"{SIM_DIR}/initial_train.csv")

    accumulated_paths = [f"{SIM_DIR}/initial_train.csv"]

    for i in range(1, N_BATCHES + 1):
        print(f"\n=== Round {i}: batch_{i} arrives ===")
        batch_path = f"{SIM_DIR}/batch_{i}.csv"

        # The batch is still considered "arrived" and gets folded into
        # future training pools regardless of this round's decision --
        # only THIS round's retraining is skipped if nothing has drifted.
        accumulated_paths.append(batch_path)

        batch_df = pd.read_csv(batch_path)
        drift_result = check_drift(reference_df, batch_df)

        if not drift_result["drift_detected"]:
            print(f"No significant drift in batch_{i} -- skipping retraining this round.")
            continue

        print(f"Drift detected in batch_{i} -- proceeding with retraining.")

        combined_path = f"{SIM_DIR}/_combined_round_{i}.csv"
        combine_raw_files(accumulated_paths, combined_path)

        x_round, y_round = process_file(combined_path, encode_for=ENCODING)
        challenger, challenger_run_id = train_on_pool(
            x_round, y_round, run_name=f"round_{i}_challenger"
        )
        challenger_columns = x_round.columns

        x_eval_for_champion = align_columns(x_eval_raw, champion_columns)
        x_eval_for_challenger = align_columns(x_eval_raw, challenger_columns)

        champion_probs = champion.predict_proba(x_eval_for_champion)[:, 1]
        challenger_probs = challenger.predict_proba(x_eval_for_challenger)[:, 1]

        promoted = promote_or_reject(
            champion_probs, challenger_probs, y_eval, threshold=PROMOTION_THRESHOLD
        )

        with mlflow.start_run(run_name=f"round_{i}_summary"):
            mlflow.log_metric("champion_auc", roc_auc_score(y_eval, champion_probs))
            mlflow.log_metric("challenger_auc", roc_auc_score(y_eval, challenger_probs))
            mlflow.log_metric("promoted", int(promoted))

        if promoted:
            champion = challenger
            champion_columns = challenger_columns
            save_model(champion, MODEL_PATH)
            register_as_production(challenger_run_id)

    print("\n=== Retraining simulation complete ===")

if __name__ == "__main__":
    run()