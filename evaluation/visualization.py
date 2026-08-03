"""
Publication-quality visualization for IoMT IDS paper.
All figures: 300 DPI, PDF + PNG, colorblind-friendly palette.
"""
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config import (
    FIGURES_DIR, FIG_DPI, FIG_FORMAT, PLOT_STYLE, MODEL_COLORS,
)

log = logging.getLogger(__name__)

# Apply publication style globally
plt.rcParams.update(PLOT_STYLE)


def _save_fig(fig, name):
    """Save figure in all configured formats (with timestamp to avoid overwrites)."""
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    stamped = f"{name}_{ts}"
    for fmt in FIG_FORMAT:
        path = f"{FIGURES_DIR}/{stamped}.{fmt}"
        fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    log.info(f"    Saved: {stamped}")


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, scenario):
    """
    Publication-quality confusion matrix heatmap.
    Uses percentage (row-normalized) with sample counts annotated.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    n = len(class_names)
    fig_size = max(6, n * 0.55)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

    # Heatmap
    sns.heatmap(
        cm_pct, annot=True, fmt='.1f', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, vmin=0, vmax=100, linewidths=0.5, linecolor='white',
        annot_kws={'size': max(7, 11 - n // 4)},
        cbar_kws={'label': 'Percentage (%)', 'shrink': 0.8},
    )

    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title(f'{model_name} ({scenario})', fontweight='bold')

    # Rotate labels for readability
    if n > 6:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    safe = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    _save_fig(fig, f"cm_{safe}_{scenario}")
    return cm


def plot_accuracy_heatmap(all_results: dict):
    """
    Accuracy heatmap: models x scenarios.
    Publication-quality replacement for bar charts.
    """
    scenarios = list(all_results.keys())
    models = list(all_results[scenarios[0]].keys())

    data = np.array([
        [all_results[s][m].get('accuracy', 0) for s in scenarios]
        for m in models
    ])

    fig, ax = plt.subplots(figsize=(8, max(4, len(models) * 0.6)))
    sns.heatmap(
        data * 100, annot=True, fmt='.2f', cmap='RdYlGn',
        xticklabels=scenarios, yticklabels=models,
        ax=ax, vmin=50, vmax=100, linewidths=0.8, linecolor='white',
        annot_kws={'size': 10, 'fontweight': 'bold'},
        cbar_kws={'label': 'Accuracy (%)'},
    )
    ax.set_title('Model Accuracy Across Classification Scenarios', fontweight='bold')
    ax.set_xlabel('Scenario')
    ax.set_ylabel('')
    _save_fig(fig, 'accuracy_heatmap')


def plot_f1_comparison_grouped(all_results: dict):
    """Grouped bar chart: F1-macro by model and scenario."""
    scenarios = list(all_results.keys())
    models = list(all_results[scenarios[0]].keys())

    x = np.arange(len(scenarios))
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        vals = [all_results[s][model].get('f1_macro', 0) for s in scenarios]
        color = MODEL_COLORS.get(model, f'C{i}')
        bars = ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                       label=model, color=color, edgecolor='white', linewidth=0.5)
        # Value labels
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_xlabel('Classification Scenario')
    ax.set_ylabel('F1-Score (Macro)')
    ax.set_title('F1-Macro Comparison Across Scenarios', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0, 1.15)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    _save_fig(fig, 'f1_macro_comparison')



def plot_training_time_comparison(all_results: dict):
    """Training time comparison — single grouped horizontal bar chart for all scenarios."""
    scenarios = list(all_results.keys())
    models = list(all_results[scenarios[0]].keys())
    n_models = len(models)
    n_scenarios = len(scenarios)

    bar_height = 0.8 / n_scenarios
    y_base = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(10, max(4, n_models * 0.9)))

    scenario_colors = ['#2196F3', '#FF9800', '#4CAF50']

    for i, scenario in enumerate(scenarios):
        times = [all_results[scenario][m].get('train_time', 0) for m in models]
        y_pos = y_base + i * bar_height - (n_scenarios - 1) * bar_height / 2

        bars = ax.barh(y_pos, times, height=bar_height * 0.9,
                        color=scenario_colors[i % len(scenario_colors)],
                        edgecolor='white', linewidth=0.5,
                        label=scenario, alpha=0.85)

        for bar, t in zip(bars, times):
            if t > 0:
                ax.text(bar.get_width() + ax.get_xlim()[1] * 0.005,
                        bar.get_y() + bar.get_height() / 2,
                        f'{t:.0f}s', va='center', fontsize=7)

    ax.set_yticks(y_base)
    ax.set_yticklabels(models, fontsize=9)
    ax.set_xlabel('Training Time (seconds)')
    ax.set_title('Training Time Comparison', fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.invert_yaxis()

    _save_fig(fig, 'training_time_comparison')



def plot_kappa_vs_accuracy(kappa_dfs: dict):
    """Scatter plot: Kappa vs Accuracy to show imbalance effect."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for scenario, df in kappa_dfs.items():
        for _, row in df.iterrows():
            color = MODEL_COLORS.get(row['Model'], 'gray')
            ax.scatter(row['Accuracy'], row['Kappa'], c=color, s=100,
                       edgecolors='black', linewidth=0.5, zorder=5)
            ax.annotate(f"{row['Model']}\n({scenario})",
                        (row['Accuracy'], row['Kappa']),
                        fontsize=6, ha='center', va='bottom',
                        xytext=(0, 8), textcoords='offset points')

    # Diagonal (perfect agreement line)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect agreement')
    ax.set_xlabel('Accuracy')
    ax.set_ylabel("Cohen's Kappa ($\\kappa$)")
    ax.set_title('Accuracy vs Kappa: Detecting Class Imbalance Effects', fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_xlim(0.5, 1.02)
    ax.set_ylim(0.0, 1.02)
    _save_fig(fig, 'kappa_vs_accuracy')


def plot_cd_diagram(avg_ranks, model_names, cd_value):
    """Critical Difference diagram (Demsar, 2006)."""
    k = len(model_names)
    fig, ax = plt.subplots(figsize=(12, 3.5))

    sorted_idx = np.argsort(avg_ranks)
    sorted_ranks = np.array(avg_ranks)[sorted_idx]
    sorted_names = [model_names[i] for i in sorted_idx]

    ax.set_xlim(0.5, k + 0.5)
    ax.set_ylim(-0.1, 1.15)

    # Rank axis
    ax.hlines(0.5, 1, k, colors='black', linewidth=1.5)
    for r in range(1, k + 1):
        ax.vlines(r, 0.45, 0.55, colors='black', linewidth=1.5)
        ax.text(r, 0.60, str(r), ha='center', va='bottom', fontsize=11, fontweight='bold')

    for i, (rank, name) in enumerate(zip(sorted_ranks, sorted_names)):
        y_text = 0.20 if i % 2 == 0 else 0.80
        y_line = 0.35 if i % 2 == 0 else 0.65

        ax.plot(rank, 0.5, 'ko', markersize=7, zorder=5)
        ax.vlines(rank, 0.5, y_line, colors='gray', linewidth=0.8)
        color = MODEL_COLORS.get(name, 'lightyellow')
        ax.text(rank, y_text, f"{name}\n({rank:.2f})",
                ha='center', va='center', fontsize=8,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                          edgecolor='gray', alpha=0.9))

    # CD bar
    ax.hlines(1.0, 1, 1 + cd_value, colors='red', linewidth=2.5)
    ax.text(1 + cd_value / 2, 1.07, f'CD = {cd_value:.2f}',
            ha='center', fontsize=10, color='red', fontweight='bold')

    ax.set_title('Critical Difference Diagram', fontweight='bold', fontsize=13)
    ax.axis('off')
    _save_fig(fig, 'cd_diagram')


def plot_autoencoder_error_distribution(ae_results: dict, scenario: str):
    """Plot reconstruction error distribution for autoencoder."""
    errors = ae_results['reconstruction_errors']
    y_test = ae_results['y_test_binary']
    threshold = ae_results['threshold']
    optimal_pct = ae_results.get('optimal_percentile', 95)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Separate errors by class
    benign_errors = errors[y_test == 0]
    attack_errors = errors[y_test == 1]

    ax.hist(benign_errors, bins=100, alpha=0.6, color='#2196F3',
            label=f'Benign (n={len(benign_errors):,})', density=True)
    ax.hist(attack_errors, bins=100, alpha=0.6, color='#F44336',
            label=f'Attack (n={len(attack_errors):,})', density=True)

    ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
               label=f'Threshold (p{optimal_pct}) = {threshold:.4f}')

    ax.set_xlabel('Reconstruction Error (MSE)')
    ax.set_ylabel('Density')
    ax.set_title(f'Autoencoder Reconstruction Error Distribution ({scenario})',
                 fontweight='bold')
    ax.legend()
    ax.set_xlim(0, np.percentile(errors, 99.5))
    _save_fig(fig, f'ae_error_dist_{scenario}')



def plot_autoencoder_threshold_curve(ae_results: dict, scenario: str):
    """
    Enhanced threshold analysis with accuracy on secondary axis.
    Left: Precision/Recall/F1/Accuracy vs Percentile (dual y-axis)
    Right: Precision-Recall trade-off curve
    Shows why accuracy is misleading for anomaly detection.
    """
    thr_df = ae_results.get('threshold_analysis')
    if thr_df is None or thr_df.empty:
        return

    optimal_pct = ae_results.get('optimal_percentile', 95)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # --- Left: Precision / Recall / F1 vs Percentile (primary axis) ---
    ax1.plot(thr_df['percentile'], thr_df['precision'], 'o-',
             color='#2196F3', linewidth=2, markersize=5, label='Precision')
    ax1.plot(thr_df['percentile'], thr_df['recall'], 's-',
             color='#F44336', linewidth=2, markersize=5, label='Recall')
    ax1.plot(thr_df['percentile'], thr_df['f1_macro'], 'D-',
             color='#4CAF50', linewidth=2.5, markersize=6, label='F1 (macro)')

    # Accuracy on secondary y-axis (to show why it's misleading)
    ax1_acc = ax1.twinx()
    ax1_acc.plot(thr_df['percentile'], thr_df['accuracy'], '^--',
                 color='#FF9800', linewidth=1.5, markersize=5, alpha=0.7,
                 label='Accuracy')
    ax1_acc.set_ylabel('Accuracy', color='#FF9800', fontsize=10)
    ax1_acc.tick_params(axis='y', labelcolor='#FF9800')
    ax1_acc.set_ylim(0, 1.05)

    # Shade the region where accuracy is misleading (drops sharply)
    if 'accuracy' in thr_df.columns:
        drop_mask = thr_df['accuracy'] < 0.5
        if drop_mask.any():
            drop_start = thr_df.loc[drop_mask, 'percentile'].iloc[0]
            ax1.axvspan(drop_start, thr_df['percentile'].max(),
                        alpha=0.08, color='#FF9800',
                        label='Accuracy misleading zone')

    # Highlight optimal
    opt_row = thr_df[thr_df['percentile'] == optimal_pct].iloc[0]
    ax1.axvline(optimal_pct, color='gray', linestyle=':', alpha=0.7)
    ax1.plot(optimal_pct, opt_row['f1_macro'], '*', color='#4CAF50',
             markersize=18, zorder=10, markeredgecolor='black', markeredgewidth=0.8)
    ax1.annotate(f'Optimal (p{optimal_pct})\nF1={opt_row["f1_macro"]:.4f}',
                 xy=(optimal_pct, opt_row['f1_macro']),
                 xytext=(optimal_pct - 4, opt_row['f1_macro'] - 0.08),
                 fontsize=9, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='gray'))

    ax1.set_xlabel('Threshold Percentile')
    ax1.set_ylabel('Score')
    ax1.set_title('Threshold Sensitivity Analysis', fontweight='bold')

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_acc.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # --- Right: Precision vs Recall (trade-off curve) ---
    # Color-coded by percentile for richer information
    scatter = ax2.scatter(thr_df['recall'], thr_df['precision'],
                          c=thr_df['percentile'], cmap='RdYlGn_r',
                          s=80, zorder=5, edgecolors='white', linewidth=1)
    ax2.plot(thr_df['recall'], thr_df['precision'], '-',
             color='#9C27B0', linewidth=1.5, alpha=0.4)

    cbar = plt.colorbar(scatter, ax=ax2, shrink=0.8)
    cbar.set_label('Percentile', fontsize=9)

    # Annotate key points
    for _, row in thr_df.iterrows():
        if row['percentile'] in [85, 90, optimal_pct, 99]:
            ax2.annotate(f'p{int(row["percentile"])}',
                         xy=(row['recall'], row['precision']),
                         fontsize=8, ha='left', fontweight='bold',
                         xytext=(5, 5), textcoords='offset points')

    ax2.plot(opt_row['recall'], opt_row['precision'], '*', color='#4CAF50',
             markersize=18, zorder=10, markeredgecolor='black', markeredgewidth=0.8)

    ax2.set_xlabel('Recall (Attack Detection Rate)')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Trade-off', fontweight='bold')
    ax2.set_xlim(-0.05, 1.05)
    ax2.set_ylim(0.7, 1.02)
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Autoencoder Anomaly Detection ({scenario})',
                 fontweight='bold', y=1.02, fontsize=13)
    _save_fig(fig, f'ae_threshold_curve_{scenario}')




def plot_lgbm_tuning_comparison(lgbm_results: dict, scenario: str):
    """
    Modern dot-plot with zoomed axis for LightGBM hyperparameter comparison.
    Shows multiple metrics side-by-side with delta annotations.
    Much better than bar charts when differences are small.
    """
    configs = list(lgbm_results.keys())
    metrics_to_plot = ['accuracy', 'f1_macro', 'kappa']
    metric_labels = ['Accuracy', 'F1-Score (Macro)', "Cohen's κ"]
    metric_colors = ['#2196F3', '#4CAF50', '#9C27B0']
    marker_styles = ['o', 'D', 's']

    # Collect data
    data = {}
    for m in metrics_to_plot:
        data[m] = [lgbm_results[c].get(m, 0) for c in configs]

    # Clean config names for display
    display_names = []
    for c in configs:
        name = c.replace('tuned_', '').replace('_', ' ').title()
        if c == 'original':
            name = 'Original'
        display_names.append(name)

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(14, 4.5))

    for idx, (m, label, color, marker) in enumerate(
            zip(metrics_to_plot, metric_labels, metric_colors, marker_styles)):
        ax = axes[idx]
        values = data[m]
        y_pos = np.arange(len(configs))

        # Dot plot (horizontal lollipop)
        ax.hlines(y_pos, xmin=min(values) - 0.001, xmax=values,
                  color=color, alpha=0.3, linewidth=2)
        ax.scatter(values, y_pos, color=color, s=120, zorder=5,
                   marker=marker, edgecolors='white', linewidth=1.5)

        # Annotate values with delta from original
        orig_val = values[0]
        for i, v in enumerate(values):
            delta = v - orig_val
            delta_str = f' ({delta:+.4f})' if i > 0 else ' (baseline)'
            delta_color = '#2E7D32' if delta > 0 else ('#C62828' if delta < 0 else 'gray')
            ax.annotate(f'{v:.4f}{delta_str}',
                        xy=(v, y_pos[i]),
                        xytext=(8, 0), textcoords='offset points',
                        fontsize=8, va='center', color=delta_color,
                        fontweight='bold' if i > 0 and delta > 0 else 'normal')

        # Highlight best
        best_idx = np.argmax(values)
        ax.scatter([values[best_idx]], [y_pos[best_idx]], color=color,
                   s=250, zorder=4, marker=marker, alpha=0.2)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(display_names, fontsize=9)
        ax.set_xlabel(label, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')

        # Zoom axis to show differences clearly
        val_range = max(values) - min(values)
        padding = max(val_range * 2, 0.002)
        ax.set_xlim(min(values) - padding, max(values) + padding * 4)

        ax.invert_yaxis()

    plt.suptitle(f'LightGBM Hyperparameter Tuning Comparison ({scenario})',
                 fontweight='bold', y=1.02, fontsize=13)
    _save_fig(fig, f'lgbm_tuning_{scenario}')


def plot_inference_latency(all_results: dict):
    """
    Inference latency comparison: per-sample time (ms) and throughput (samples/s).
    Dual-axis chart for publication.
    """
    scenarios = list(all_results.keys())
    models = list(all_results[scenarios[0]].keys())

    fig, axes = plt.subplots(1, len(scenarios), figsize=(6 * len(scenarios), 5),
                              sharey=False)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, scenario in zip(axes, scenarios):
        latencies = []
        throughputs = []
        for m in models:
            pred_time = all_results[scenario][m].get('pred_time', 0)
            n_samples = len(all_results[scenario][m].get('y_pred', []))
            per_sample_ms = (pred_time / n_samples * 1000) if n_samples > 0 else 0
            throughput = (n_samples / pred_time) if pred_time > 0 else 0
            latencies.append(per_sample_ms)
            throughputs.append(throughput)

        colors = [MODEL_COLORS.get(m, f'C{i}') for i, m in enumerate(models)]
        y_pos = np.arange(len(models))

        # Latency bars
        bars = ax.barh(y_pos, latencies, color=colors,
                        edgecolor='white', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(models, fontsize=9)
        ax.set_xlabel('Latency per Sample (ms)')
        ax.set_title(f'{scenario}', fontweight='bold')

        for bar, lat, thr in zip(bars, latencies, throughputs):
            ax.text(bar.get_width() + max(latencies) * 0.03,
                    bar.get_y() + bar.get_height() / 2,
                    f'{lat:.4f} ms\n({thr:,.0f} s/s)',
                    va='center', fontsize=7)

    plt.suptitle('Inference Latency & Throughput', fontweight='bold', y=1.02)
    _save_fig(fig, 'inference_latency')



def plot_resource_usage(all_results: dict):
    """
    Resource usage comparison: GPU memory, CPU memory, GPU energy.
    Triple subplot for publication.
    """
    scenarios = list(all_results.keys())
    # Use first scenario for resource comparison
    scenario = scenarios[0]
    models_data = all_results[scenario]
    models = list(models_data.keys())

    # Check if resource data exists
    has_resources = any(models_data[m].get('train_profile') for m in models)
    if not has_resources:
        log.info("    Skipping resource plot (no profiling data)")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    gpu_mem = []
    cpu_mem = []
    gpu_energy = []
    for m in models:
        tp = models_data[m].get('train_profile')
        if tp:
            tp_d = tp if isinstance(tp, dict) else tp.to_dict()
            gpu_mem.append(tp_d.get('gpu_memory_peak_mb', 0))
            cpu_mem.append(tp_d.get('cpu_memory_peak_mb', 0))
            gpu_energy.append(tp_d.get('gpu_energy_wh', 0))
        else:
            gpu_mem.append(0)
            cpu_mem.append(0)
            gpu_energy.append(0)

    colors = [MODEL_COLORS.get(m, f'C{i}') for i, m in enumerate(models)]
    y_pos = np.arange(len(models))

    # GPU Memory
    bars = axes[0].barh(y_pos, gpu_mem, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(models, fontsize=8)
    axes[0].set_xlabel('GPU Memory Peak (MB)')
    axes[0].set_title('GPU VRAM Usage', fontweight='bold')
    for bar, v in zip(bars, gpu_mem):
        if v > 0:
            axes[0].text(bar.get_width() + max(gpu_mem) * 0.02,
                         bar.get_y() + bar.get_height() / 2,
                         f'{v:.0f}', va='center', fontsize=8)

    # CPU Memory
    bars = axes[1].barh(y_pos, cpu_mem, color=colors, edgecolor='white', linewidth=0.5)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(models, fontsize=8)
    axes[1].set_xlabel('CPU Memory Peak (MB)')
    axes[1].set_title('CPU RAM Usage', fontweight='bold')
    for bar, v in zip(bars, cpu_mem):
        if v > 0:
            axes[1].text(bar.get_width() + max(cpu_mem) * 0.02,
                         bar.get_y() + bar.get_height() / 2,
                         f'{v:.0f}', va='center', fontsize=8)

    # GPU Energy
    bars = axes[2].barh(y_pos, gpu_energy, color=colors, edgecolor='white', linewidth=0.5)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(models, fontsize=8)
    axes[2].set_xlabel('GPU Energy (Wh)')
    axes[2].set_title('Energy Consumption', fontweight='bold')
    for bar, v in zip(bars, gpu_energy):
        if v > 0:
            axes[2].text(bar.get_width() + max(gpu_energy) * 0.02,
                         bar.get_y() + bar.get_height() / 2,
                         f'{v:.4f}', va='center', fontsize=8)

    plt.suptitle(f'Resource Usage Comparison ({scenario})', fontweight='bold', y=1.02)
    _save_fig(fig, 'resource_usage')


def plot_gcp_cost_comparison(all_results: dict):
    """
    GCP deployment cost comparison: single run vs annual.
    """
    scenarios = list(all_results.keys())
    scenario = scenarios[0]
    models_data = all_results[scenario]
    models = list(models_data.keys())

    has_cost = any(models_data[m].get('gcp_cost') for m in models)
    if not has_cost:
        log.info("    Skipping GCP cost plot (no cost data)")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    single_costs = []
    annual_costs = []
    labels = []
    for m in models:
        cost = models_data[m].get('gcp_cost')
        if cost:
            single_costs.append(cost.get('cost_single_run_usd', 0))
            annual_costs.append(cost.get('cost_annual_total_usd', 0))
            labels.append(m)

    colors = [MODEL_COLORS.get(m, f'C{i}') for i, m in enumerate(labels)]
    y_pos = np.arange(len(labels))

    # Single run cost
    bars = ax1.barh(y_pos, single_costs, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel('Cost (USD)')
    ax1.set_title('Single Run Cost', fontweight='bold')
    for bar, v in zip(bars, single_costs):
        ax1.text(bar.get_width() + max(single_costs) * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 f'${v:.4f}', va='center', fontsize=8)

    # Annual cost
    bars = ax2.barh(y_pos, annual_costs, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=8)
    ax2.set_xlabel('Cost (USD)')
    ax2.set_title('Annual Cost (weekly retrain + daily inference)', fontweight='bold')
    for bar, v in zip(bars, annual_costs):
        ax2.text(bar.get_width() + max(annual_costs) * 0.02,
                 bar.get_y() + bar.get_height() / 2,
                 f'${v:.2f}', va='center', fontsize=8)

    plt.suptitle(f'GCP Deployment Cost Estimation ({scenario})', fontweight='bold', y=1.02)
    _save_fig(fig, 'gcp_cost_comparison')


def plot_latency_detailed(all_results: dict):
    """
    Detailed inference latency: mean, P50, P95, P99 with error bars.
    """
    scenarios = list(all_results.keys())
    scenario = scenarios[0]
    models_data = all_results[scenario]
    models = list(models_data.keys())

    has_latency = any(models_data[m].get('latency_profile') for m in models)
    if not has_latency:
        log.info("    Skipping detailed latency plot (no profiling data)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    labels = []
    means = []
    stds = []
    p95s = []
    p99s = []
    for m in models:
        lp = models_data[m].get('latency_profile')
        if lp:
            labels.append(m)
            means.append(lp.get('latency_mean_ms', 0))
            stds.append(lp.get('latency_std_ms', 0))
            p95s.append(lp.get('latency_p95_ms', 0))
            p99s.append(lp.get('latency_p99_ms', 0))

    colors = [MODEL_COLORS.get(m, f'C{i}') for i, m in enumerate(labels)]
    x = np.arange(len(labels))
    width = 0.25

    ax.bar(x - width, means, width, label='Mean', color=colors, alpha=0.9,
           edgecolor='white', linewidth=0.5, yerr=stds, capsize=3)
    ax.bar(x, p95s, width, label='P95', color=colors, alpha=0.6,
           edgecolor='white', linewidth=0.5)
    ax.bar(x + width, p99s, width, label='P99', color=colors, alpha=0.35,
           edgecolor='white', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Latency (ms)')
    ax.set_title(f'Inference Latency Distribution ({scenario})', fontweight='bold')
    ax.legend()
    _save_fig(fig, 'latency_detailed')


def plot_power_timeline(all_results: dict):
    """
    GPU power draw comparison across models.
    Simple bar chart of average and max power.
    """
    scenarios = list(all_results.keys())
    scenario = scenarios[0]
    models_data = all_results[scenario]
    models = list(models_data.keys())

    has_power = any(
        models_data[m].get('train_profile') and
        (models_data[m]['train_profile'] if isinstance(models_data[m]['train_profile'], dict)
         else models_data[m]['train_profile'].to_dict()).get('gpu_power_avg_w', 0) > 0
        for m in models
    )
    if not has_power:
        log.info("    Skipping power plot (no power data)")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    labels = []
    avg_power = []
    max_power = []
    for m in models:
        tp = models_data[m].get('train_profile')
        if tp:
            tp_d = tp if isinstance(tp, dict) else tp.to_dict()
            if tp_d.get('gpu_power_avg_w', 0) > 0:
                labels.append(m)
                avg_power.append(tp_d['gpu_power_avg_w'])
                max_power.append(tp_d.get('gpu_power_max_w', 0))

    if not labels:
        plt.close(fig)
        return

    colors = [MODEL_COLORS.get(m, f'C{i}') for i, m in enumerate(labels)]
    x = np.arange(len(labels))
    width = 0.35

    ax.bar(x - width / 2, avg_power, width, label='Avg Power (W)',
           color=colors, alpha=0.9, edgecolor='white', linewidth=0.5)
    ax.bar(x + width / 2, max_power, width, label='Max Power (W)',
           color=colors, alpha=0.5, edgecolor='white', linewidth=0.5)

    for i, (a, m) in enumerate(zip(avg_power, max_power)):
        ax.text(i - width / 2, a + 0.5, f'{a:.1f}W', ha='center', fontsize=8)
        ax.text(i + width / 2, m + 0.5, f'{m:.1f}W', ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Power (Watts)')
    ax.set_title(f'GPU Power Draw During Training ({scenario})', fontweight='bold')
    ax.legend()
    _save_fig(fig, 'gpu_power_comparison')


def generate_all_figures(all_results, kappa_dfs=None, friedman=None,
                          lgbm_all=None, ae_results=None):
    """Generate all publication figures."""
    log.info("\n  Generating publication figures...")

    if all_results:
        plot_accuracy_heatmap(all_results)
        plot_f1_comparison_grouped(all_results)
        plot_training_time_comparison(all_results)
        plot_inference_latency(all_results)
        # New resource/energy/cost figures
        plot_resource_usage(all_results)
        plot_gcp_cost_comparison(all_results)
        plot_latency_detailed(all_results)
        plot_power_timeline(all_results)

    if kappa_dfs:
        plot_kappa_vs_accuracy(kappa_dfs)

    if friedman and friedman.get('avg_ranks'):
        plot_cd_diagram(
            np.array(friedman['avg_ranks']),
            friedman['models'],
            friedman['CD'],
        )

    log.info("  All figures generated.")

def plot_f1_per_class_boxplot(y_test, predictions, class_names, scenario):
    """
    Boxplot of per-class F1-scores across models.
    Reveals which attack classes each model struggles with.
    Essential for showing that high macro-F1 doesn't hide per-class weaknesses.
    """
    from sklearn.metrics import f1_score as _f1

    model_names = list(predictions.keys())
    # Compute per-class F1 for each model
    records = []
    for name in model_names:
        y_pred = predictions[name]['y_pred']
        per_class_f1 = _f1(y_test, y_pred, average=None, zero_division=0)
        for cls_idx, f1_val in enumerate(per_class_f1):
            cls_name = class_names[cls_idx] if cls_idx < len(class_names) else f'Class {cls_idx}'
            records.append({'Model': name, 'Class': cls_name, 'F1-Score': f1_val})

    df = pd.DataFrame(records)

    # Determine figure width based on number of models
    n_models = len(model_names)
    fig_width = max(10, n_models * 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))

    colors = [MODEL_COLORS.get(m, '#666666') for m in model_names]
    palette = dict(zip(model_names, colors))

    bp = sns.boxplot(data=df, x='Model', y='F1-Score', hue='Model',
                     palette=palette, legend=False,
                     width=0.6, fliersize=4, linewidth=1.2, ax=ax)

    # Overlay individual class points (strip plot)
    sns.stripplot(data=df, x='Model', y='F1-Score', color='black',
                  size=4, alpha=0.4, jitter=0.15, ax=ax)

    # Annotate median values
    medians = df.groupby('Model', sort=False)['F1-Score'].median()
    for i, name in enumerate(model_names):
        med = medians[name]
        ax.annotate(f'{med:.3f}', xy=(i, med), xytext=(0, -15),
                    textcoords='offset points', ha='center', fontsize=8,
                    fontweight='bold', color=palette.get(name, '#333'))

    ax.set_ylabel('F1-Score (per class)')
    ax.set_xlabel('')
    ax.set_title(f'Per-Class F1-Score Distribution ({scenario})', fontweight='bold')
    ax.set_ylim(-0.05, 1.08)
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=20, ha='right')

    _save_fig(fig, f'f1_per_class_boxplot_{scenario}')


def plot_bootstrap_boxplot(y_test, predictions, scenario, n_boot=500):
    """
    Boxplot of bootstrap F1-macro distributions per model.
    Shows stability/variance of each classifier's performance estimate.
    Uses fewer iterations than full CI analysis for speed (visualization only).
    """
    from sklearn.metrics import f1_score as _f1
    from config import SEED

    rng = np.random.RandomState(SEED)
    n = len(y_test)
    model_names = list(predictions.keys())

    records = []
    for name in model_names:
        y_pred = predictions[name]['y_pred']
        for _ in range(n_boot):
            idx = rng.randint(0, n, size=n)
            yt, yp = y_test[idx], y_pred[idx]
            if len(np.unique(yt)) < 2:
                continue
            score = _f1(yt, yp, average='macro', zero_division=0)
            records.append({'Model': name, 'F1-Macro': score})

    df = pd.DataFrame(records)

    n_models = len(model_names)
    fig_width = max(10, n_models * 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, 5))

    colors = [MODEL_COLORS.get(m, '#666666') for m in model_names]
    palette = dict(zip(model_names, colors))

    sns.boxplot(data=df, x='Model', y='F1-Macro', hue='Model',
                palette=palette, legend=False,
                width=0.5, fliersize=3, linewidth=1.2, ax=ax,
                showmeans=True,
                meanprops=dict(marker='D', markerfacecolor='white',
                               markeredgecolor='black', markersize=6))

    # Annotate median and IQR
    for i, name in enumerate(model_names):
        vals = df[df['Model'] == name]['F1-Macro']
        med = vals.median()
        q1, q3 = vals.quantile(0.025), vals.quantile(0.975)
        ax.annotate(f'{med:.4f}\n[{q1:.4f}, {q3:.4f}]',
                    xy=(i, q1), xytext=(0, -22),
                    textcoords='offset points', ha='center', fontsize=7,
                    color='#333')

    ax.set_ylabel('F1-Score (Macro)')
    ax.set_xlabel('')
    ax.set_title(f'Bootstrap F1-Macro Distribution ({scenario}) — {n_boot} samples',
                 fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    plt.xticks(rotation=20, ha='right')

    _save_fig(fig, f'bootstrap_f1_boxplot_{scenario}')


