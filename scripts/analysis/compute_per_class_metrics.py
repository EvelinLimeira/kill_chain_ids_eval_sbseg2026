"""Compute per-class Precision/Recall/F1/Support from confusion matrices for the 19-class scenario.

Outputs a LaTeX-formatted table body for the 4 main models (RF, 1D-CNN, LightGBM Tuned, LR).
"""
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results" / "tables"

MODELS = {
    "RF": "cm_random_forest_19-class.csv",
    "CNN": "cm_1d-cnn_19-class.csv",
    "LGBM-T": "cm_lightgbm_tuned_19-class.csv",
    "LR": "cm_logistic_regression_19-class.csv",
}


def per_class_metrics(cm: pd.DataFrame) -> pd.DataFrame:
    """Compute precision, recall, F1, support from a CM where rows=true, cols=pred."""
    labels = list(cm.index)
    support = cm.sum(axis=1).values
    tp = np.diag(cm.values)
    fp = cm.values.sum(axis=0) - tp
    fn = cm.values.sum(axis=1) - tp
    precision = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    recall = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    return pd.DataFrame({
        "class": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    })


def load_cm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    return df


def main() -> None:
    results = {}
    for key, fname in MODELS.items():
        cm = load_cm(TABLES / fname)
        results[key] = per_class_metrics(cm).set_index("class")

    # Use RF class list as reference
    classes = list(results["RF"].index)

    # Build combined dataframe
    rows = []
    for cls in classes:
        row = {"class": cls, "support": int(results["RF"].loc[cls, "support"])}
        for key in MODELS.keys():
            row[f"{key}_P"] = round(float(results[key].loc[cls, "precision"]), 3)
            row[f"{key}_R"] = round(float(results[key].loc[cls, "recall"]), 3)
            row[f"{key}_F1"] = round(float(results[key].loc[cls, "f1"]), 3)
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = TABLES / "per_class_19-class.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    # Print LaTeX-formatted table body (class name, support, F1 for 4 models)
    print("\nCompact LaTeX rows (class, support, RF F1, CNN F1, LGBM-T F1, LR F1):\n")
    for _, r in out.iterrows():
        name = r["class"].replace("_", "\\_")
        print(
            f"{name} & {int(r['support']):,} & "
            f"{r['RF_F1']:.3f} & {r['CNN_F1']:.3f} & {r['LGBM-T_F1']:.3f} & {r['LR_F1']:.3f} \\\\"
        )

    # Print full per-class with P/R/F1 for RF and CNN (compact)
    print("\nFull LaTeX table body (class, support, RF P/R/F1, CNN P/R/F1):\n")
    for _, r in out.iterrows():
        name = r["class"].replace("_", "\\_")
        print(
            f"{name} & {int(r['support']):,} & "
            f"{r['RF_P']:.2f} & {r['RF_R']:.2f} & {r['RF_F1']:.2f} & "
            f"{r['CNN_P']:.2f} & {r['CNN_R']:.2f} & {r['CNN_F1']:.2f} \\\\"
        )


if __name__ == "__main__":
    main()
