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
from sklearn.metrics import roc_auc_score

from preprocessing import process_file
from train import train_on_pool, promote_or_reject, save_model

SIM_DIR = "data/simulated"
MODEL_PATH = "models/champion_model.pkl"
PROMOTION_THRESHOLD = 0.02
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
    print("=== Round 0: training initial champion ===")
    x_init, y_init = process_file(f"{SIM_DIR}/initial_train.csv", encode_for=ENCODING)
    champion = train_on_pool(x_init, y_init)
    champion_columns = x_init.columns

    x_eval_raw, y_eval = process_file(f"{SIM_DIR}/eval_fixed.csv", encode_for=ENCODING)
    x_eval_for_champion = align_columns(x_eval_raw, champion_columns)

    champion_probs = champion.predict_proba(x_eval_for_champion)[:, 1]
    champion_auc = roc_auc_score(y_eval, champion_probs)
    print(f"Initial champion AUC on fixed evaluation set: {champion_auc:.4f}")

    save_model(champion, MODEL_PATH)

    accumulated_paths = [f"{SIM_DIR}/initial_train.csv"]

    for i in range(1, N_BATCHES + 1):
        print(f"\n=== Round {i}: batch_{i} arrives ===")
        accumulated_paths.append(f"{SIM_DIR}/batch_{i}.csv")

        combined_path = f"{SIM_DIR}/_combined_round_{i}.csv"
        combine_raw_files(accumulated_paths, combined_path)

        x_round, y_round = process_file(combined_path, encode_for=ENCODING)
        challenger = train_on_pool(x_round, y_round)
        challenger_columns = x_round.columns

        # Re-align the fixed eval set separately for each model, since
        # champion and challenger were trained on pools with potentially
        # different one-hot columns.
        x_eval_for_champion = align_columns(x_eval_raw, champion_columns)
        x_eval_for_challenger = align_columns(x_eval_raw, challenger_columns)

        champion_probs = champion.predict_proba(x_eval_for_champion)[:, 1]
        challenger_probs = challenger.predict_proba(x_eval_for_challenger)[:, 1]

        promoted = promote_or_reject(
            champion_probs, challenger_probs, y_eval, threshold=PROMOTION_THRESHOLD
        )

        if promoted:
            champion = challenger
            champion_columns = challenger_columns
            save_model(champion, MODEL_PATH)

    print("\n=== Retraining simulation complete ===")


if __name__ == "__main__":
    run()