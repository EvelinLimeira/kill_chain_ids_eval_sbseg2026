"""
Classification metrics computation and reporting.
"""
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, classification_report, confusion_matrix,
)

from config import TABLES_DIR, REPORTS_DIR

log = logging.getLogger(__name__)


def compute_all_metrics(y_true, y_pred, average='macro') -> dict:
    """Compute comprehensive classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'kappa': cohen_kappa_score(y_true, y_pred),
    }


def save_classification_report(y_true, y_pred, class_names, model_name, scenario):
    """Save per-class classification report as CSV."""
    report = classification_report(y_true, y_pred, target_names=class_names,
                                    output_dict=True, zero_division=0)
    df = pd.DataFrame(report).transpose()
    safe = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    path = f"{REPORTS_DIR}/report_{safe}_{scenario}.csv"
    df.to_csv(path)
    return df


def save_confusion_matrix(y_true, y_pred, class_names, model_name, scenario):
    """Save confusion matrix as CSV."""
    cm = confusion_matrix(y_true, y_pred)
    df = pd.DataFrame(cm, index=class_names, columns=class_names)
    safe = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    path = f"{TABLES_DIR}/cm_{safe}_{scenario}.csv"
    df.to_csv(path)
    return cm


def generate_comparison_table(all_results: dict, scenario: str):
    """
    Generate comprehensive comparison table for a scenario.
    Includes classification metrics, resource usage, and cost estimates.
    Saves CSV and returns DataFrame.
    """
    rows = []
    for model_name, data in all_results.items():
        pred_time = data.get('pred_time', 0)
        n_samples = len(data.get('y_pred', []))
        per_sample_ms = (pred_time / n_samples * 1000) if n_samples > 0 else 0

        row = {
            'Model': model_name,
            # Classification metrics
            'Accuracy': data.get('accuracy', 0),
            'Precision (macro)': data.get('precision_macro', 0),
            'Recall (macro)': data.get('recall_macro', 0),
            'F1 (macro)': data.get('f1_macro', 0),
            'F1 (weighted)': data.get('f1_weighted', 0),
            'Kappa': data.get('kappa', 0),
            # Timing
            'Train Time (s)': data.get('train_time', 0),
            'Pred Time (s)': pred_time,
            'Latency per Sample (ms)': per_sample_ms,
            'Throughput (samples/s)': (n_samples / pred_time) if pred_time > 0 else 0,
        }

        # Resource profiling (if available)
        train_profile = data.get('train_profile')
        if train_profile:
            tp = train_profile if isinstance(train_profile, dict) else train_profile.to_dict()
            row['GPU Memory Peak (MB)'] = tp.get('gpu_memory_peak_mb', 0)
            row['CPU Memory Peak (MB)'] = tp.get('cpu_memory_peak_mb', 0)
            row['GPU Power Avg (W)'] = tp.get('gpu_power_avg_w', 0)
            row['GPU Power Max (W)'] = tp.get('gpu_power_max_w', 0)
            row['GPU Energy (Wh)'] = tp.get('gpu_energy_wh', 0)

        # Detailed latency (if available)
        latency_profile = data.get('latency_profile')
        if latency_profile:
            row['Latency Mean (ms)'] = latency_profile.get('latency_mean_ms', 0)
            row['Latency Std (ms)'] = latency_profile.get('latency_std_ms', 0)
            row['Latency P50 (ms)'] = latency_profile.get('latency_p50_ms', 0)
            row['Latency P95 (ms)'] = latency_profile.get('latency_p95_ms', 0)
            row['Latency P99 (ms)'] = latency_profile.get('latency_p99_ms', 0)
            row['Batch Throughput (samples/s)'] = latency_profile.get('throughput_samples_s', 0)

        # Model size (if available)
        model_size = data.get('model_size')
        if model_size:
            row['Model Disk Size (MB)'] = model_size.get('model_disk_size_mb', 0)
            row['Model RAM Size (MB)'] = model_size.get('model_ram_size_mb', 0)

        # GCP cost (if available)
        gcp_cost = data.get('gcp_cost')
        if gcp_cost:
            row['GCP Instance'] = gcp_cost.get('gcp_instance', '')
            row['GCP $/hr'] = gcp_cost.get('gcp_price_per_hour', 0)
            row['Cost Single Run ($)'] = gcp_cost.get('cost_single_run_usd', 0)
            row['Cost Annual Retrain ($)'] = gcp_cost.get('cost_annual_retrain_usd', 0)
            row['Cost Annual Total ($)'] = gcp_cost.get('cost_annual_total_usd', 0)

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"{TABLES_DIR}/comparison_{scenario}.csv", index=False)
    log.info(f"  Comparison table saved: comparison_{scenario}.csv")
    return df


def generate_latex_table(all_scenario_results: dict):
    """Generate publication-ready LaTeX table spanning all scenarios."""
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Comprehensive Model Comparison Across Classification Scenarios}",
        r"\label{tab:comprehensive_comparison}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{cl ccccc r rr}",
        r"\toprule",
        r"\textbf{Classes} & \textbf{Model} & \textbf{Accuracy} & "
        r"\textbf{Prec.} & \textbf{Recall} & \textbf{F1 (macro)} & "
        r"\textbf{$\kappa$} & \textbf{Time (s)} & "
        r"\textbf{Lat. (ms)} & \textbf{Energy (Wh)} \\",
        r"\midrule",
    ]

    for scenario, models in all_scenario_results.items():
        first = True
        # Find best F1 for bolding
        best_f1 = max(m.get('f1_macro', 0) for m in models.values())
        for name, m in models.items():
            sc = scenario if first else ""
            first = False
            f1_val = m.get('f1_macro', 0)
            f1_str = f"\\textbf{{{f1_val:.4f}}}" if f1_val == best_f1 else f"{f1_val:.4f}"

            # Latency and energy (from profiling if available)
            lp = m.get('latency_profile', {})
            tp = m.get('train_profile', {})
            lat_str = f"{lp.get('latency_mean_ms', 0):.3f}" if lp else "--"
            energy_str = f"{tp.get('gpu_energy_wh', 0):.4f}" if tp else "--"

            lines.append(
                f"  {sc} & {name} & {m.get('accuracy', 0):.4f} & "
                f"{m.get('precision_macro', 0):.4f} & {m.get('recall_macro', 0):.4f} & "
                f"{f1_str} & {m.get('kappa', 0):.4f} & {m.get('train_time', 0):.1f} & "
                f"{lat_str} & {energy_str} \\\\"
            )
        lines.append(r"\midrule")

    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table*}"])

    path = f"{TABLES_DIR}/table_comprehensive.tex"
    with open(path, 'w') as f:
        f.write("\n".join(lines))
    log.info(f"  LaTeX table saved: {path}")

def generate_full_comparison_table(all_results: dict, scenario: str):
    """
    Generate a comprehensive comparison table with ALL metrics for each model.
    Includes: classification, timing, latency, resource, energy, cost, model size.
    One row per model, all columns.
    """
    rows = []
    for model_name, data in all_results.items():
        pred_time = data.get('pred_time', 0)
        n_samples = len(data.get('y_pred', []))
        tp = data.get('train_profile', {})
        lp = data.get('latency_profile', {})
        ms = data.get('model_size', {})
        gc = data.get('gcp_cost', {})

        row = {
            'Model': model_name,
            # --- Classification ---
            'Accuracy': data.get('accuracy', 0),
            'Precision (macro)': data.get('precision_macro', 0),
            'Recall (macro)': data.get('recall_macro', 0),
            'F1 (macro)': data.get('f1_macro', 0),
            'Precision (weighted)': data.get('precision_weighted', 0),
            'Recall (weighted)': data.get('recall_weighted', 0),
            'F1 (weighted)': data.get('f1_weighted', 0),
            'Kappa': data.get('kappa', 0),
            # --- Timing ---
            'Train Time (s)': data.get('train_time', 0),
            'Pred Time (s)': pred_time,
            'Latency/Sample (ms)': (pred_time / n_samples * 1000) if n_samples > 0 else 0,
            'Throughput (samples/s)': (n_samples / pred_time) if pred_time > 0 else 0,
            # --- Detailed Latency (warmup) ---
            'Latency Mean (ms)': lp.get('latency_mean_ms', 0),
            'Latency Std (ms)': lp.get('latency_std_ms', 0),
            'Latency P50 (ms)': lp.get('latency_p50_ms', 0),
            'Latency P95 (ms)': lp.get('latency_p95_ms', 0),
            'Latency P99 (ms)': lp.get('latency_p99_ms', 0),
            'Batch Throughput (samples/s)': lp.get('throughput_samples_s', 0),
            # --- Resource Usage ---
            'GPU Memory Peak (MB)': tp.get('gpu_memory_peak_mb', 0),
            'CPU Memory Peak (MB)': tp.get('cpu_memory_peak_mb', 0),
            'GPU Power Avg (W)': tp.get('gpu_power_avg_w', 0),
            'GPU Power Max (W)': tp.get('gpu_power_max_w', 0),
            'GPU Energy (Wh)': tp.get('gpu_energy_wh', 0),
            # --- Model Size ---
            'Model Disk (MB)': ms.get('model_disk_size_mb', 0),
            'Model RAM (MB)': ms.get('model_ram_size_mb', 0),
            # --- GCP Cost ---
            'GCP Instance': gc.get('gcp_instance', ''),
            'GCP $/hr': gc.get('gcp_price_per_hour', 0),
            'Cost/Run ($)': gc.get('cost_single_run_usd', 0),
            'Cost Annual Retrain ($)': gc.get('cost_annual_retrain_usd', 0),
            'Cost Annual Total ($)': gc.get('cost_annual_total_usd', 0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"{TABLES_DIR}/full_comparison_{scenario}.csv", index=False)
    log.info(f"  Full comparison table saved: full_comparison_{scenario}.csv")
    return df


def generate_lgbm_variants_table(lgbm_results: dict, scenario: str):
    """
    Generate dedicated LightGBM variants comparison table.
    Shows all variants (original, balanced, aggressive, dart) with full metrics.
    """
    rows = []
    for config_name, data in lgbm_results.items():
        pred_time = data.get('pred_time', 0)
        n_samples = len(data.get('y_pred', []))

        row = {
            'Variant': config_name,
            'Accuracy': data.get('accuracy', 0),
            'Precision (macro)': data.get('precision_macro', 0),
            'Recall (macro)': data.get('recall_macro', 0),
            'F1 (macro)': data.get('f1_macro', 0),
            'F1 (weighted)': data.get('f1_weighted', 0),
            'Kappa': data.get('kappa', 0),
            'Best Iteration': data.get('best_iteration', 0),
            'Train Time (s)': data.get('train_time', 0),
            'Pred Time (s)': pred_time,
            'Latency/Sample (ms)': (pred_time / n_samples * 1000) if n_samples > 0 else 0,
            'Throughput (samples/s)': (n_samples / pred_time) if pred_time > 0 else 0,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Add delta columns relative to original
    if 'original' in lgbm_results:
        orig_acc = lgbm_results['original'].get('accuracy', 0)
        orig_f1 = lgbm_results['original'].get('f1_macro', 0)
        df['Δ Accuracy'] = df['Accuracy'] - orig_acc
        df['Δ F1 (macro)'] = df['F1 (macro)'] - orig_f1

    df.to_csv(f"{TABLES_DIR}/lgbm_variants_{scenario}.csv", index=False)
    log.info(f"  LightGBM variants table saved: lgbm_variants_{scenario}.csv")
    return df


