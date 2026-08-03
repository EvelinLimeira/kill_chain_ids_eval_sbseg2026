#!/usr/bin/env python3
"""
compare_ae_cpu_gpu.py
=====================
Compara duas execuções do autoencoder canônico produzidas com o parâmetro
--tag (ex.: `cpu` no Windows/venv_iomt e `gpu` no WSL/venv_wsl).

Objetivo científico
--------------------
Quantificar de forma reprodutível o impacto do ambiente de treino (CPU oneDNN
vs GPU cuDNN) sobre:
    * as métricas binárias no threshold primário (P85);
    * a curva de sensibilidade P85-P99;
    * o vetor canônico de reconstruction errors (correlação de Spearman/Pearson,
      diferença absoluta máxima/média);
    * o threshold calibrado tau e a época selecionada por early stopping.

Isso permite reportar no artigo, de forma honesta, que o autoencoder é
sensível à ordem de operações de ponto flutuante do backend, e justificar qual
run foi congelada como canônica.

Uso
---
    python scripts/evaluation/compare_ae_cpu_gpu.py --tags cpu gpu

Requer que ambas as runs já tenham sido geradas:
    python scripts/evaluation/autoencoder_threshold_sensitivity.py --tag cpu   (Windows)
    ... --tag gpu   (WSL)

Saídas em results/tables/:
    ae_cpu_gpu_main_comparison.csv        <- métricas P85 lado a lado + delta
    ae_cpu_gpu_sensitivity_comparison.csv <- sensibilidade P85-P99 lado a lado
    ae_cpu_gpu_error_divergence.csv       <- estatísticas dos erros de teste
    ae_cpu_gpu_phase_comparison.csv       <- recall por fase lado a lado
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import TABLES_DIR

SCENARIO = "19-class"


def _p(name: str) -> str:
    return os.path.join(TABLES_DIR, name)


def _require(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Artefato ausente: {path}\n"
            f"Gere a run correspondente com --tag antes de comparar."
        )
    return path


def load_run(tag: str) -> dict:
    """Carrega os artefatos de uma run tagueada."""
    main_csv = _require(_p(f"ae_canonical_main_table_{tag}.csv"))
    sens_csv = _require(_p(f"ae_threshold_sensitivity_{SCENARIO}_{tag}.csv"))
    phase_csv = _require(_p(f"ae_phase_recall_summary_{SCENARIO}_{tag}.csv"))
    meta_json = _require(_p(f"ae_canonical_metadata_{SCENARIO}_{tag}.json"))
    errors_npy = _require(_p(f"ae_canonical_errors_{SCENARIO}_{tag}.npy"))
    val_errors_npy = _require(_p(f"ae_canonical_val_errors_{SCENARIO}_{tag}.npy"))

    with open(meta_json, encoding="utf-8") as fh:
        meta = json.load(fh)

    return {
        "tag": tag,
        "main": pd.read_csv(main_csv),
        "sensitivity": pd.read_csv(sens_csv),
        "phase": pd.read_csv(phase_csv),
        "metadata": meta,
        "test_errors": np.load(errors_npy),
        "val_errors": np.load(val_errors_npy),
    }


def compare_main(run_a: dict, run_b: dict) -> pd.DataFrame:
    """Métricas P85 lado a lado + delta (b - a)."""
    a, b = run_a["main"].iloc[0], run_b["main"].iloc[0]
    metrics = [
        "tau", "Recall (Attack)", "Precision (Attack)", "FPR", "n_FP",
        "Specificity", "Balanced Accuracy", "F1 (Attack)", "F1-macro",
        "AUC-ROC", "AP (Attack)", "AP (Benign)", "Accuracy", "Kappa",
        "n_alerts", "n_TP", "n_FN", "n_TN",
    ]
    rows = []
    for m in metrics:
        if m not in a.index or m not in b.index:
            continue
        va, vb = a[m], b[m]
        try:
            delta = float(vb) - float(va)
        except (TypeError, ValueError):
            delta = np.nan
        rows.append({
            "Metric": m,
            run_a["tag"]: va,
            run_b["tag"]: vb,
            f"delta ({run_b['tag']}-{run_a['tag']})": round(delta, 6)
            if not np.isnan(delta) else np.nan,
        })
    return pd.DataFrame(rows)


def compare_sensitivity(run_a: dict, run_b: dict) -> pd.DataFrame:
    """Sensibilidade P85-P99 lado a lado para métricas-chave."""
    a = run_a["sensitivity"].set_index("Percentile")
    b = run_b["sensitivity"].set_index("Percentile")
    cols = ["tau", "Recall (Attack)", "Precision (Attack)", "FPR", "n_FP",
            "Balanced Accuracy", "F1-macro"]
    rows = []
    for pct in sorted(set(a.index) | set(b.index)):
        row = {"Percentile": pct}
        for c in cols:
            if c in a.columns:
                row[f"{c} [{run_a['tag']}]"] = a.loc[pct, c]
            if c in b.columns:
                row[f"{c} [{run_b['tag']}]"] = b.loc[pct, c]
        rows.append(row)
    return pd.DataFrame(rows)


def compare_phase(run_a: dict, run_b: dict) -> pd.DataFrame:
    """Recall por fase (mean e weighted) lado a lado."""
    a = run_a["phase"].set_index("Phase")
    b = run_b["phase"].set_index("Phase")
    rows = []
    for phase in sorted(set(a.index) | set(b.index)):
        row = {"Phase": phase}
        for col in ["Recall (mean)", "Recall (weighted)", "Support"]:
            if phase in a.index and col in a.columns:
                row[f"{col} [{run_a['tag']}]"] = a.loc[phase, col]
            if phase in b.index and col in b.columns:
                row[f"{col} [{run_b['tag']}]"] = b.loc[phase, col]
        if "Recall (mean)" in a.columns and "Recall (mean)" in b.columns \
                and phase in a.index and phase in b.index:
            row["delta Recall (mean)"] = round(
                float(b.loc[phase, "Recall (mean)"])
                - float(a.loc[phase, "Recall (mean)"]), 6)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_errors(run_a: dict, run_b: dict) -> pd.DataFrame:
    """Divergência entre os vetores de reconstruction errors."""
    from scipy.stats import spearmanr

    ea, eb = run_a["test_errors"], run_b["test_errors"]
    if ea.shape != eb.shape:
        raise ValueError(
            f"Error vectors have different shapes: {ea.shape} vs {eb.shape}. "
            f"As runs devem usar a mesma partição de teste."
        )
    diff = np.abs(ea - eb)
    pearson = float(np.corrcoef(ea, eb)[0, 1])
    spearman = float(spearmanr(ea, eb).correlation)

    va, vb = run_a["val_errors"], run_b["val_errors"]
    val_pearson = (float(np.corrcoef(va, vb)[0, 1])
                   if va.shape == vb.shape else np.nan)

    rows = [
        {"Statistic": "n_test", run_a["tag"]: len(ea), run_b["tag"]: len(eb)},
        {"Statistic": "mean error", run_a["tag"]: round(float(ea.mean()), 6),
         run_b["tag"]: round(float(eb.mean()), 6)},
        {"Statistic": "std error", run_a["tag"]: round(float(ea.std()), 6),
         run_b["tag"]: round(float(eb.std()), 6)},
        {"Statistic": "P85 tau (val)",
         run_a["tag"]: round(float(np.percentile(va, 85)), 6),
         run_b["tag"]: round(float(np.percentile(vb, 85)), 6)},
        {"Statistic": "Pearson r (test errors)", run_a["tag"]: round(pearson, 6),
         run_b["tag"]: ""},
        {"Statistic": "Spearman rho (test errors)",
         run_a["tag"]: round(spearman, 6), run_b["tag"]: ""},
        {"Statistic": "Pearson r (val errors)",
         run_a["tag"]: round(val_pearson, 6) if not np.isnan(val_pearson) else "n/a",
         run_b["tag"]: ""},
        {"Statistic": "max |diff| (test)", run_a["tag"]: round(float(diff.max()), 6),
         run_b["tag"]: ""},
        {"Statistic": "mean |diff| (test)", run_a["tag"]: round(float(diff.mean()), 6),
         run_b["tag"]: ""},
    ]
    return pd.DataFrame(rows)


def compare_metadata(run_a: dict, run_b: dict) -> pd.DataFrame:
    keys = ["platform", "python_version", "tensorflow_version",
            "deterministic_ops", "selected_epoch", "best_val_loss",
            "train_time_seconds", "test_inference_time_seconds", "tau_p85"]
    ma, mb = run_a["metadata"], run_b["metadata"]
    rows = [{"Field": k, run_a["tag"]: ma.get(k), run_b["tag"]: mb.get(k)}
            for k in keys]
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compara duas runs tagueadas do autoencoder (ex.: cpu vs gpu)."
    )
    parser.add_argument(
        "--tags", nargs=2, default=["cpu", "gpu"], metavar=("TAG_A", "TAG_B"),
        help="Os dois tags a comparar (default: cpu gpu).",
    )
    args = parser.parse_args()
    tag_a, tag_b = args.tags

    print("=" * 70)
    print(f"  Comparação de runs do autoencoder: '{tag_a}' vs '{tag_b}'")
    print("=" * 70)

    run_a = load_run(tag_a)
    run_b = load_run(tag_b)

    df_meta = compare_metadata(run_a, run_b)
    df_main = compare_main(run_a, run_b)
    df_sens = compare_sensitivity(run_a, run_b)
    df_phase = compare_phase(run_a, run_b)
    df_err = compare_errors(run_a, run_b)

    df_main.to_csv(_p("ae_cpu_gpu_main_comparison.csv"), index=False)
    df_sens.to_csv(_p("ae_cpu_gpu_sensitivity_comparison.csv"), index=False)
    df_phase.to_csv(_p("ae_cpu_gpu_phase_comparison.csv"), index=False)
    df_err.to_csv(_p("ae_cpu_gpu_error_divergence.csv"), index=False)

    print("\n--- Ambiente / telemetria ---")
    print(df_meta.to_string(index=False))
    print("\n--- Métricas P85 (lado a lado) ---")
    print(df_main.to_string(index=False))
    print("\n--- Sensibilidade P85-P99 ---")
    print(df_sens.to_string(index=False))
    print("\n--- Recall por fase ---")
    print(df_phase.to_string(index=False))
    print("\n--- Divergência dos reconstruction errors ---")
    print(df_err.to_string(index=False))

    print("\nArquivos salvos em results/tables/:")
    print("  ae_cpu_gpu_main_comparison.csv")
    print("  ae_cpu_gpu_sensitivity_comparison.csv")
    print("  ae_cpu_gpu_phase_comparison.csv")
    print("  ae_cpu_gpu_error_divergence.csv")


if __name__ == "__main__":
    main()
