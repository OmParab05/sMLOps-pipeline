"""
explain.py
Explains the current champion model's predictions using SHAP.

Produces two things:
  1. A global importance plot -- which features matter most across all
     applicants, overall.
  2. A single-prediction waterfall plot -- for one specific applicant,
     exactly which features pushed their risk score up or down, and by
     how much.

Usage:
    python explain.py
Expects:
    models/champion_model.pkl   (produced by retrain_pipeline.py)
    data/simulated/eval_fixed.csv
Produces:
    reports/shap/global_importance.png
    reports/shap/single_prediction.png
"""

import os
import joblib
import matplotlib.pyplot as plt
import shap
import pandas as pd

from preprocessing import process_file

MODEL_PATH = "models/champion_model.pkl"
DATA_PATH = "data/simulated/eval_fixed.csv"
OUT_DIR = "reports/shap"

# Which applicant (by row position in the evaluation set) to generate
# the single-prediction explanation for.
EXAMPLE_ROW = 0


def align_to_model(x, model):
    """Reindex x to exactly the feature columns the model was trained
    on, in the same order, filling anything missing with 0. Needed
    because one-hot encoding can produce slightly different columns
    depending on which categories are present in a given file."""
    expected = model.booster_.feature_name()
    return x.reindex(columns=expected, fill_value=0)


def load_model_and_data():
    model = joblib.load(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}")

    x, y = process_file(DATA_PATH, encode_for="xgboost")
    x = align_to_model(x, model)
    print(f"Loaded {x.shape[0]} rows, {x.shape[1]} features for explanation")

    return model, x, y


def get_shap_values(explainer, x):
    """Handles both SHAP API shapes: a single array (most current
    versions) or a [class_0, class_1] list (older versions, for binary
    classifiers). We always want the class 1 (default) explanation."""
    values = explainer.shap_values(x)
    if isinstance(values, list):
        return values[1]
    return values


def plot_global_importance(shap_values, x, out_path, max_display=15):
    plt.figure()
    shap.summary_plot(shap_values, x, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved global importance plot to {out_path}")


def plot_single_explanation(explainer, shap_values, x, row_index, out_path, max_display=15):
    base_value = explainer.expected_value
    if isinstance(base_value, list):
        base_value = base_value[1]

    explanation = shap.Explanation(
        values=shap_values[row_index],
        base_values=base_value,
        data=x.iloc[row_index],
        feature_names=x.columns.tolist(),
    )

    plt.figure()
    shap.plots.waterfall(explanation, show=False, max_display=max_display)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved single-prediction explanation to {out_path}")


def plot_bar_importance(shap_values, x, out_path, max_display=15):
    plt.figure()
    shap.summary_plot(shap_values, x, show=False, max_display=max_display, plot_type="bar")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved bar importance plot to {out_path}")


def plot_dependence(shap_values, x, feature, out_path):
    plt.figure()
    shap.dependence_plot(feature, shap_values, x, show=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved dependence plot for {feature} to {out_path}")


def export_importance_csv(shap_values, x, out_path, top_n=20):
    importance = pd.DataFrame({
        "feature": x.columns,
        "mean_abs_shap": abs(shap_values).mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).head(top_n)

    importance.to_csv(out_path, index=False)
    print(f"Saved top {top_n} feature importances to {out_path}")
    return importance


def find_example_rows(model, x):
    """Returns row indices for a high-risk, low-risk, and borderline
    applicant, so the single-prediction explanation can show contrasting
    cases instead of just one."""
    probs = model.predict_proba(x)[:, 1]
    probs_series = pd.Series(probs, index=x.index)

    high_risk_idx = probs_series.idxmax()
    low_risk_idx = probs_series.idxmin()
    borderline_idx = (probs_series - 0.5).abs().idxmin()

    return {
        "high_risk": x.index.get_loc(high_risk_idx),
        "low_risk": x.index.get_loc(low_risk_idx),
        "borderline": x.index.get_loc(borderline_idx),
    }


def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    model, x, y = load_model_and_data()

    explainer = shap.TreeExplainer(model)
    shap_values = get_shap_values(explainer, x)

    # Global views
    plot_global_importance(shap_values, x, f"{OUT_DIR}/global_importance.png")
    plot_bar_importance(shap_values, x, f"{OUT_DIR}/importance_bar.png")

    importance = export_importance_csv(shap_values, x, f"{OUT_DIR}/feature_importance.csv")
    top_feature = importance.iloc[0]["feature"]
    plot_dependence(shap_values, x, top_feature, f"{OUT_DIR}/dependence_{top_feature}.png")

    # Single-prediction explanations for three contrasting cases
    example_rows = find_example_rows(model, x)
    for label, row_index in example_rows.items():
        plot_single_explanation(
            explainer, shap_values, x, row_index, f"{OUT_DIR}/single_prediction_{label}.png"
        )
        predicted_prob = model.predict_proba(x.iloc[[row_index]])[:, 1][0]
        actual_label = y.iloc[row_index]
        print(f"\n{label.replace('_', ' ').title()} applicant (row {row_index}):")
        print(f"  Predicted default probability: {predicted_prob:.4f}")
        print(f"  Actual outcome: {'Defaulted' if actual_label == 1 else 'Repaid'}")


if __name__ == "__main__":
    run()