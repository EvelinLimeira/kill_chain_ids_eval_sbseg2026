#!/usr/bin/env python3
"""
Main execution pipeline for IoMT IDS experiments.
SBSeg 2026 Submission.

Usage:
  python main.py --data_dir ./data --scenarios 2 6 19
  python main.py --data_dir ./data --scenarios 19 --skip_xai
  python main.py --data_dir ./data --scenarios 2 6 19 --only_stats
"""
import argparse
import gc
import logging
import os
import sys
import time
import warnings
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf

# ---------------------------------------------------------------------------
# Suppress noisy warnings that clutter logs
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", message=".*binary classifier with TreeExplainer.*")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF INFO/WARNING C++ logs

# ---------------------------------------------------------------------------
# GPU / CUDA configuration (must run before any TF operation)
# ---------------------------------------------------------------------------
def _configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                # Memory growth avoids TF grabbing all VRAM at once
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU config warning: {e}")
    return gpus

_GPUS = _configure_gpu()

# Project modules
from config import SEED, RESULTS_DIR, TABLES_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, XAI_DIR
from data_loader import load_data
from models.classical import train_random_forest, train_logistic_regression, train_lightgbm_all
from models.cnn_standard import train_standard_cnn
from models.cnn_light import train_light_cnn
from models.autoencoder import train_autoencoder
from evaluation.metrics import (
    compute_all_metrics, save_classification_report,
    save_confusion_matrix, generate_comparison_table, generate_latex_table,
    generate_full_comparison_table, generate_lgbm_variants_table,
)
from evaluation.statistical_tests import (
    run_bootstrap_analysis, run_mcnemar_analysis,
    run_kappa_analysis, run_friedman_nemenyi,
)
from evaluation.resource_profiler import (
    ResourceMonitor, profile_inference_latency,
    get_model_size, estimate_gcp_cost,
)
from evaluation.visualization import (
    plot_confusion_matrix, plot_accuracy_heatmap,
    plot_f1_comparison_grouped, plot_training_time_comparison,
    plot_kappa_vs_accuracy, plot_cd_diagram,
    plot_autoencoder_error_distribution, plot_autoencoder_threshold_curve,
    plot_lgbm_tuning_comparison,
    plot_f1_per_class_boxplot, plot_bootstrap_boxplot,
)
from explainability.shap_analysis import run_shap_analysis
from explainability.lime_analysis import run_lime_analysis


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging():
    fmt = '[%(asctime)s] %(levelname)s %(name)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt='%H:%M:%S',
                        handlers=[
                            logging.StreamHandler(sys.stdout),
                            logging.FileHandler(f'{RESULTS_DIR}/experiment.log', mode='w'),
                        ])

log = logging.getLogger('main')


def clear_memory():
    gc.collect()
    if tf.config.list_physical_devices('GPU'):
        tf.keras.backend.clear_session()
        tf.random.set_seed(SEED)


def _save_resource_table(predictions: dict, scenario: str):
    """Save dedicated resource profiling CSV with all models."""
    import pandas as pd
    rows = []
    for name, res in predictions.items():
        tp = res.get('train_profile', {})
        lp = res.get('latency_profile', {})
        ms = res.get('model_size', {})
        gc = res.get('gcp_cost', {})
        rows.append({
            'Model': name,
            'Train Time (s)': round(res.get('train_time', 0), 2),
            'Pred Time (s)': round(res.get('pred_time', 0), 4),
            'GPU Memory Peak (MB)': tp.get('gpu_memory_peak_mb', 0),
            'CPU Memory Peak (MB)': tp.get('cpu_memory_peak_mb', 0),
            'GPU Power Avg (W)': tp.get('gpu_power_avg_w', 0),
            'GPU Power Max (W)': tp.get('gpu_power_max_w', 0),
            'GPU Energy (Wh)': tp.get('gpu_energy_wh', 0),
            'Latency Mean (ms)': lp.get('latency_mean_ms', 0),
            'Latency Std (ms)': lp.get('latency_std_ms', 0),
            'Latency P50 (ms)': lp.get('latency_p50_ms', 0),
            'Latency P95 (ms)': lp.get('latency_p95_ms', 0),
            'Latency P99 (ms)': lp.get('latency_p99_ms', 0),
            'Batch Throughput (samples/s)': lp.get('throughput_samples_s', 0),
            'Model Disk (MB)': ms.get('model_disk_size_mb', 0),
            'Model RAM (MB)': ms.get('model_ram_size_mb', 0),
            'GCP Instance': gc.get('gcp_instance', ''),
            'GCP $/hr': gc.get('gcp_price_per_hour', 0),
            'Cost/Run ($)': gc.get('cost_single_run_usd', 0),
            'Cost Annual Retrain ($)': gc.get('cost_annual_retrain_usd', 0),
            'Cost Annual Total ($)': gc.get('cost_annual_total_usd', 0),
        })
    df = pd.DataFrame(rows)
    df.to_csv(f"{TABLES_DIR}/resource_profiling_{scenario}.csv", index=False)
    log.info(f"  Resource profiling table saved: resource_profiling_{scenario}.csv")


# =============================================================================
# SINGLE SCENARIO PIPELINE
# =============================================================================

def run_scenario(data_dir: str, class_config: int,
                  skip_xai: bool = False,
                  include_light_cnn: bool = False,
                  include_cnn_focal: bool = False,
                  include_light_cnn_v2: bool = False,
                  light_cnn_v2_epochs: int = None,
                  include_mlp_baseline: bool = False,
                  only_autoencoder: bool = False) -> dict:
    """
    Run complete pipeline for one classification scenario.

    Returns dict with all predictions and results.
    """
    scenario = f"{class_config}-class"
    log.info(f"\n{'#' * 70}")
    log.info(f"  SCENARIO: {scenario}")
    log.info(f"{'#' * 70}")

    # --- Load data ---
    data = load_data(data_dir, class_config)
    predictions = OrderedDict()

    # =====================================================================
    # AUTOENCODER-ONLY MODE (skip all other models)
    # =====================================================================
    if only_autoencoder:
        log.info(f"\n--- Autoencoder Only Mode ---")
        monitor = ResourceMonitor()
        monitor.start()
        ae_result = train_autoencoder(
            data['X_train_flat'], data['y_train'],
            data['X_val_flat'], data['y_val'],
            data['X_test_flat'], data['y_test'],
            data['label_encoder'], scenario)
        ae_result['train_profile'] = monitor.stop().to_dict()
        predictions['Autoencoder'] = ae_result

        plot_autoencoder_error_distribution(ae_result, scenario)
        plot_autoencoder_threshold_curve(ae_result, scenario)
        clear_memory()

        return {
            'predictions': predictions,
            'supervised_preds': {},
            'data': data,
            'lgbm_results': {},
            'ae_result': ae_result,
            'kappa_df': None,
            'boot_df': None,
        }

    # =====================================================================
    # CLASSICAL ML MODELS (use flat 2D arrays, combined train+val)
    # =====================================================================
    log.info(f"\n--- Classical ML Models ---")

    # Random Forest
    monitor = ResourceMonitor()
    monitor.start()
    rf_result = train_random_forest(
        data['X_train_combined'], data['y_train_combined'],
        data['X_test_flat'], data['y_test'], scenario)
    rf_result['train_profile'] = monitor.stop().to_dict()
    rf_result.update(compute_all_metrics(data['y_test'], rf_result['y_pred']))
    predictions['Random Forest'] = rf_result

    # Logistic Regression
    monitor = ResourceMonitor()
    monitor.start()
    lr_result = train_logistic_regression(
        data['X_train_combined'], data['y_train_combined'],
        data['X_test_flat'], data['y_test'], scenario)
    lr_result['train_profile'] = monitor.stop().to_dict()
    lr_result.update(compute_all_metrics(data['y_test'], lr_result['y_pred']))
    predictions['Logistic Regression'] = lr_result

    # LightGBM (original + all tuned variants)
    # Note: LightGBM uses X_train (without val) for training,
    # and X_val for early stopping — prevents data leakage
    monitor = ResourceMonitor()
    monitor.start()
    lgbm_results = train_lightgbm_all(
        data['X_train_flat'], data['y_train'],
        data['X_val_flat'], data['y_val'],
        data['X_test_flat'], data['y_test'],
        data['n_classes'], scenario)
    lgbm_profile = monitor.stop().to_dict()

    # Add original LightGBM
    lgbm_orig = lgbm_results['original']
    lgbm_orig['train_profile'] = lgbm_profile
    lgbm_orig.update(compute_all_metrics(data['y_test'], lgbm_orig['y_pred']))
    predictions['LightGBM (Original)'] = lgbm_orig

    # Add best tuned LightGBM
    tuned_keys = [k for k in lgbm_results if 'tuned' in k]
    if tuned_keys:
        best_tuned_key = max(tuned_keys, key=lambda k: lgbm_results[k]['f1_macro'])
        lgbm_tuned = lgbm_results[best_tuned_key]
        lgbm_tuned['train_profile'] = lgbm_profile
        lgbm_tuned.update(compute_all_metrics(data['y_test'], lgbm_tuned['y_pred']))
        predictions['LightGBM (Tuned)'] = lgbm_tuned

    # =====================================================================
    # DEEP LEARNING MODELS (use 3D arrays for CNN, flat for autoencoder)
    # =====================================================================
    log.info(f"\n--- Deep Learning Models ---")

    # Standard 1D-CNN
    monitor = ResourceMonitor()
    monitor.start()
    cnn_result = train_standard_cnn(
        data['X_train'], data['y_train'], data['X_val'], data['y_val'],
        data['X_test'], data['y_test'], data['n_classes'], scenario)
    cnn_result['train_profile'] = monitor.stop().to_dict()
    cnn_result.update(compute_all_metrics(data['y_test'], cnn_result['y_pred']))
    predictions['1D-CNN'] = cnn_result
    clear_memory()

    # Light-CNN (Depthwise Separable) - optional, for future work
    if include_light_cnn:
        monitor = ResourceMonitor()
        monitor.start()
        lcnn_result = train_light_cnn(
            data['X_train'], data['y_train'], data['X_val'], data['y_val'],
            data['X_test'], data['y_test'], data['n_classes'], scenario)
        lcnn_result['train_profile'] = monitor.stop().to_dict()
        lcnn_result.update(compute_all_metrics(data['y_test'], lcnn_result['y_pred']))
        predictions['Light-CNN'] = lcnn_result
        clear_memory()

    # CNN Focal Loss (NOVO — opcional)
    if include_cnn_focal:
        from models.cnn_focal import train_focal_cnn
        monitor = ResourceMonitor()
        monitor.start()
        focal_result = train_focal_cnn(
            data['X_train'], data['y_train'], data['X_val'], data['y_val'],
            data['X_test'], data['y_test'], data['n_classes'], scenario)
        focal_result['train_profile'] = monitor.stop().to_dict()
        focal_result.update(compute_all_metrics(data['y_test'], focal_result['y_pred']))
        predictions['CNN-FocalLoss'] = focal_result
        clear_memory()

    # Light-CNN V2 (NOVO — opcional)
    if include_light_cnn_v2:
        from models.cnn_light_v2 import train_light_cnn_v2
        monitor = ResourceMonitor()
        monitor.start()
        lv2_result = train_light_cnn_v2(
            data['X_train'], data['y_train'], data['X_val'], data['y_val'],
            data['X_test'], data['y_test'], data['n_classes'], scenario,
            epochs_override=light_cnn_v2_epochs)
        lv2_result['train_profile'] = monitor.stop().to_dict()
        lv2_result.update(compute_all_metrics(data['y_test'], lv2_result['y_pred']))
        predictions['Light-CNN V2'] = lv2_result
        clear_memory()

    # MLP Baseline (NOVO — opcional, usa dados flat 2D)
    if include_mlp_baseline:
        from models.mlp_baseline import train_mlp_baseline
        monitor = ResourceMonitor()
        monitor.start()
        mlp_result = train_mlp_baseline(
            data['X_train_flat'], data['y_train'],
            data['X_val_flat'], data['y_val'],
            data['X_test_flat'], data['y_test'],
            data['n_classes'], scenario)
        mlp_result['train_profile'] = monitor.stop().to_dict()
        mlp_result.update(compute_all_metrics(data['y_test'], mlp_result['y_pred']))
        predictions['MLP Baseline'] = mlp_result
        clear_memory()

    # Autoencoder (binary anomaly detection)
    monitor = ResourceMonitor()
    monitor.start()
    ae_result = train_autoencoder(
        data['X_train_flat'], data['y_train'],
        data['X_val_flat'], data['y_val'],
        data['X_test_flat'], data['y_test'],
        data['label_encoder'], scenario)
    ae_result['train_profile'] = monitor.stop().to_dict()
    predictions['Autoencoder'] = ae_result
    clear_memory()

    # =====================================================================
    # RESOURCE PROFILING: Latency, Model Size, GCP Cost
    # =====================================================================
    log.info(f"\n--- Resource Profiling ---")

    # GPU models (need GPU instance on GCP)
    gpu_models = {'1D-CNN', 'Light-CNN', 'CNN-FocalLoss', 'Light-CNN V2', 'MLP Baseline', 'Autoencoder'}

    for name, res in predictions.items():
        model = res.get('model')
        if model is None:
            continue

        is_keras = hasattr(model, 'predict') and hasattr(model, 'fit') and hasattr(model, 'layers')
        is_gpu = name in gpu_models

        # Inference latency profiling
        log.info(f"  Profiling inference: {name}...")
        if name == 'Autoencoder':
            X_sample = data['X_test_flat'][:1000]
        elif name in ('1D-CNN', 'Light-CNN', 'CNN-FocalLoss', 'Light-CNN V2'):
            X_sample = data['X_test'][:1000]
        else:
            X_sample = data['X_test_flat'][:1000]

        try:
            # Wrap LightGBM predict to suppress feature-name warnings
            if type(model).__name__ == 'LGBMClassifier':
                _orig_model = model
                class _LGBMWrapper:
                    def predict(self, X):
                        return _orig_model.predict(X, validate_features=False)
                profile_model = _LGBMWrapper()
            else:
                profile_model = model

            latency = profile_inference_latency(
                profile_model, X_sample,
                n_warmup=100, n_iterations=1000,
                batch_size=100, is_keras=is_keras,
            )
            res['latency_profile'] = latency
            log.info(f"    {name}: {latency['latency_mean_ms']:.4f} ms/sample, "
                     f"{latency['throughput_samples_s']:,.0f} samples/s")
        except Exception as e:
            log.warning(f"    Latency profiling failed for {name}: {e}")

        # Model size
        model_path = None
        if name == 'Random Forest':
            model_path = f"{MODELS_DIR}/rf_{scenario}.pkl"
        elif name == 'Logistic Regression':
            model_path = f"{MODELS_DIR}/lr_{scenario}.pkl"
        elif name == 'LightGBM (Original)':
            model_path = f"{MODELS_DIR}/lgbm_original_{scenario}.pkl"
        elif name == 'LightGBM (Tuned)':
            model_path = f"{MODELS_DIR}/lgbm_tuned_{scenario}.pkl"
        elif name == '1D-CNN':
            model_path = f"{MODELS_DIR}/cnn_standard_{scenario}.keras"
        elif name == 'Autoencoder':
            model_path = f"{MODELS_DIR}/autoencoder_{scenario}.keras"
        elif name == 'CNN-FocalLoss':
            model_path = f"{MODELS_DIR}/cnn_focal_{scenario}.keras"
        elif name == 'Light-CNN V2':
            model_path = f"{MODELS_DIR}/cnn_light_v2_{scenario}.keras"
        elif name == 'Light-CNN':
            model_path = f"{MODELS_DIR}/cnn_light_{scenario}.keras"
        elif name == 'MLP Baseline':
            model_path = f"{MODELS_DIR}/mlp_baseline_{scenario}.keras"

        try:
            res['model_size'] = get_model_size(model, model_path)
            log.info(f"    {name}: disk={res['model_size']['model_disk_size_mb']:.2f} MB, "
                     f"RAM={res['model_size']['model_ram_size_mb']:.2f} MB")
        except Exception as e:
            log.warning(f"    Model size failed for {name}: {e}")

        # GCP cost estimation
        res['gcp_cost'] = estimate_gcp_cost(
            train_time_s=res.get('train_time', 0),
            pred_time_s=res.get('pred_time', 0),
            is_gpu_model=is_gpu,
        )

    # Log resource summary
    log.info(f"\n  Resource Summary ({scenario}):")
    log.info(f"  {'Model':<25s} {'GPU Mem (MB)':>12s} {'CPU Mem (MB)':>12s} "
             f"{'GPU Power (W)':>13s} {'Energy (Wh)':>11s} {'GCP $/run':>10s}")
    for name, res in predictions.items():
        tp = res.get('train_profile', {})
        cost = res.get('gcp_cost', {})
        log.info(f"  {name:<25s} {tp.get('gpu_memory_peak_mb', 0):>12.0f} "
                 f"{tp.get('cpu_memory_peak_mb', 0):>12.0f} "
                 f"{tp.get('gpu_power_avg_w', 0):>13.1f} "
                 f"{tp.get('gpu_energy_wh', 0):>11.4f} "
                 f"{cost.get('cost_single_run_usd', 0):>10.4f}")

    # =====================================================================
    # EVALUATION
    # =====================================================================
    log.info(f"\n--- Evaluation & Reports ---")

    # Classification reports and confusion matrices
    # (exclude autoencoder from multi-class reports since it's binary-only)
    supervised_preds = {k: v for k, v in predictions.items() if k != 'Autoencoder'}

    for name, res in supervised_preds.items():
        save_classification_report(data['y_test'], res['y_pred'],
                                   data['class_names'], name, scenario)
        save_confusion_matrix(data['y_test'], res['y_pred'],
                              data['class_names'], name, scenario)
        plot_confusion_matrix(data['y_test'], res['y_pred'],
                              data['class_names'], name, scenario)

    # Comparison table
    generate_comparison_table(supervised_preds, scenario)

    # Full comparison table (ALL models including Autoencoder, ALL metrics)
    generate_full_comparison_table(predictions, scenario)

    # LightGBM variants dedicated table
    generate_lgbm_variants_table(lgbm_results, scenario)

    # Save dedicated resource profiling table (all models including autoencoder)
    _save_resource_table(predictions, scenario)

    # LightGBM tuning comparison plot
    plot_lgbm_tuning_comparison(lgbm_results, scenario)

    # Autoencoder error distribution + threshold sensitivity
    plot_autoencoder_error_distribution(ae_result, scenario)
    plot_autoencoder_threshold_curve(ae_result, scenario)

    # Per-class F1 boxplot and bootstrap distribution boxplot
    plot_f1_per_class_boxplot(data['y_test'], supervised_preds,
                              data['class_names'], scenario)
    plot_bootstrap_boxplot(data['y_test'], supervised_preds, scenario)

    # =====================================================================
    # STATISTICAL VALIDATION
    # =====================================================================
    log.info(f"\n--- Statistical Validation ---")

    boot_df = run_bootstrap_analysis(data['y_test'], supervised_preds, scenario)
    mcnemar_df = run_mcnemar_analysis(data['y_test'], supervised_preds, scenario)
    kappa_df = run_kappa_analysis(data['y_test'], supervised_preds, scenario)

    # =====================================================================
    # EXPLAINABILITY (optional, time-intensive)
    # =====================================================================
    if not skip_xai:
        log.info(f"\n--- Explainability (SHAP + LIME) ---")

        # SHAP for Random Forest (tree-based, fast)
        run_shap_analysis(
            rf_result['model'], data['X_train_flat'], data['X_test_flat'],
            data['y_test'], data['class_names'],
            'Random Forest', scenario, model_type='tree')

        # SHAP for best LightGBM
        if 'LightGBM (Tuned)' in predictions:
            run_shap_analysis(
                predictions['LightGBM (Tuned)']['model'],
                data['X_train_flat'], data['X_test_flat'],
                data['y_test'], data['class_names'],
                'LightGBM (Tuned)', scenario, model_type='tree')

        # SHAP for 1D-CNN (neural, slower)
        run_shap_analysis(
            cnn_result['model'], data['X_train'], data['X_test'],
            data['y_test'], data['class_names'],
            '1D-CNN', scenario, model_type='neural')

        # LIME for Random Forest
        run_lime_analysis(
            rf_result['model'], data['X_train_flat'], data['X_test_flat'],
            data['y_test'], rf_result['y_pred'], data['class_names'],
            'Random Forest', scenario)

        # LIME for 1D-CNN (needs custom predict_fn)
        def cnn_predict_fn(X):
            if X.ndim == 2:
                X = X[..., np.newaxis]
            return cnn_result['model'].predict(X, verbose=0)

        run_lime_analysis(
            cnn_result['model'], data['X_train_flat'], data['X_test_flat'],
            data['y_test'], cnn_result['y_pred'], data['class_names'],
            '1D-CNN', scenario, predict_fn=cnn_predict_fn)

        # LIME for LightGBM (Tuned) — NOVO
        if 'LightGBM (Tuned)' in predictions:
            run_lime_analysis(
                predictions['LightGBM (Tuned)']['model'],
                data['X_train_flat'], data['X_test_flat'],
                data['y_test'], predictions['LightGBM (Tuned)']['y_pred'],
                data['class_names'], 'LightGBM (Tuned)', scenario)

    clear_memory()

    return {
        'predictions': predictions,
        'supervised_preds': supervised_preds,
        'data': data,
        'lgbm_results': lgbm_results,
        'ae_result': ae_result,
        'kappa_df': kappa_df,
        'boot_df': boot_df,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='IoMT IDS Experiment Pipeline')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to CICIoMT2024 dataset')
    parser.add_argument('--scenarios', type=int, nargs='+', default=[2, 6, 19],
                        help='Classification scenarios (2, 6, 19)')
    parser.add_argument('--skip_xai', action='store_true',
                        help='Skip explainability analysis (faster)')
    parser.add_argument('--include_light_cnn', action='store_true',
                        help='Include Light-CNN (depthwise separable) in evaluation')
    parser.add_argument('--only_stats', action='store_true',
                        help='Run only statistical validation (requires existing predictions)')
    parser.add_argument('--include_cnn_focal', action='store_true',
                        help='Include CNN with Focal Loss variant')
    parser.add_argument('--include_light_cnn_v2', action='store_true',
                        help='Include Light-CNN V2 variant')
    parser.add_argument('--include_mlp_baseline', action='store_true',
                        help='Include MLP Baseline (fully-connected, no convolutions)')
    parser.add_argument('--light_cnn_v2_epochs', type=int, default=None,
                        help='Override epochs for Light-CNN V2')
    parser.add_argument('--only_autoencoder', action='store_true',
                        help='Run only the Autoencoder (skip all other models)')
    args = parser.parse_args()

    setup_logging()

    log.info("=" * 70)
    log.info("  IoMT IDS - SBSeg 2026 Experiment Pipeline")
    log.info("=" * 70)
    log.info(f"  Data:      {args.data_dir}")
    log.info(f"  Scenarios: {args.scenarios}")
    log.info(f"  Skip XAI:  {args.skip_xai}")
    log.info(f"  Light-CNN: {args.include_light_cnn}")
    log.info(f"  CNN-Focal: {args.include_cnn_focal}")
    log.info(f"  Light-V2:  {args.include_light_cnn_v2}")
    log.info(f"  MLP Base:  {args.include_mlp_baseline}")
    if args.light_cnn_v2_epochs:
        log.info(f"  V2 Epochs: {args.light_cnn_v2_epochs}")
    log.info(f"  Seed:      {SEED}")
    log.info(f"  Output:    {RESULTS_DIR}")
    log.info(f"  GPU:       {_GPUS}")
    if _GPUS:
        for gpu in _GPUS:
            details = tf.config.experimental.get_device_details(gpu)
            log.info(f"             {gpu.name} - {details.get('device_name', 'unknown')}")
    else:
        log.info(f"             No GPU detected — running on CPU")

    # Set seeds
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    total_start = time.time()

    # Run each scenario
    all_scenario_results = OrderedDict()
    all_kappa_dfs = {}
    all_scenario_accuracies = {}

    for class_config in args.scenarios:
        scenario = f"{class_config}-class"
        result = run_scenario(args.data_dir, class_config,
                              skip_xai=args.skip_xai,
                              include_light_cnn=args.include_light_cnn,
                              include_cnn_focal=args.include_cnn_focal,
                              include_light_cnn_v2=args.include_light_cnn_v2,
                              light_cnn_v2_epochs=args.light_cnn_v2_epochs,
                              include_mlp_baseline=args.include_mlp_baseline,
                              only_autoencoder=args.only_autoencoder)

        all_scenario_results[scenario] = result['supervised_preds']
        all_kappa_dfs[scenario] = result['kappa_df']

        # Collect accuracies for Friedman test
        all_scenario_accuracies[scenario] = {
            name: res['accuracy']
            for name, res in result['supervised_preds'].items()
        }

    # =====================================================================
    # CROSS-SCENARIO ANALYSIS
    # =====================================================================
    if not args.only_autoencoder:
        log.info(f"\n{'='*70}")
        log.info("  CROSS-SCENARIO ANALYSIS")
        log.info(f"{'='*70}")

        # Friedman + Nemenyi
        friedman = None
        if len(args.scenarios) >= 3:
            friedman = run_friedman_nemenyi(all_scenario_accuracies)

        # Cross-scenario figures
        from evaluation.visualization import generate_all_figures
        generate_all_figures(
            all_scenario_results,
            kappa_dfs=all_kappa_dfs,
            friedman=friedman,
        )

        # Comprehensive LaTeX table
        generate_latex_table(all_scenario_results)

    # =====================================================================
    # SUMMARY
    # =====================================================================
    total_time = time.time() - total_start
    log.info(f"\n{'='*70}")
    log.info(f"  EXPERIMENT COMPLETE")
    log.info(f"  Total time: {total_time/60:.1f} minutes")
    log.info(f"{'='*70}")

    # List output files
    log.info(f"\n  Output files:")
    for directory in [TABLES_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, XAI_DIR]:
        files = sorted(Path(directory).glob('*'))
        if files:
            rel_dir = os.path.relpath(directory, RESULTS_DIR)
            log.info(f"\n  {rel_dir}/")
            for f in files:
                size = f.stat().st_size
                log.info(f"    {f.name:<55s} ({size:>10,} bytes)")

    log.info(f"\n  Key files for the paper:")
    log.info(f"    tables/table_comprehensive.tex   -> Main results table")
    log.info(f"    tables/bootstrap_ci_*.csv         -> Statistical validation")
    log.info(f"    tables/mcnemar_*.csv              -> Pairwise significance")
    log.info(f"    tables/lgbm_tuning_*.csv          -> LightGBM improvement")
    log.info(f"    figures/accuracy_heatmap.pdf       -> Accuracy comparison")
    log.info(f"    figures/cd_diagram.pdf             -> Critical Difference")
    log.info(f"    figures/cm_*.pdf                   -> Confusion matrices")
    log.info(f"    xai/shap_global_*.pdf              -> SHAP feature importance")
    log.info(f"    xai/lime_*.pdf                     -> LIME instance explanations")


if __name__ == '__main__':
    main()
