"""
SHAP explainability analysis for IoMT IDS models.

Implements:
  - Global feature importance (mean |SHAP values|)
  - Per-class SHAP summary plots
  - Local instance explanations (force plots)
  - Feature interaction analysis

Supports both tree-based (TreeExplainer) and neural network (GradientExplainer) models.
"""
import logging
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

from config import (
    SEED, SHAP_BACKGROUND_SAMPLES, SHAP_TEST_SAMPLES,
    XAI_DIR, FIG_DPI, FIG_FORMAT, PLOT_STYLE,
    CICIOMT_FEATURE_NAMES,
)

log = logging.getLogger(__name__)
plt.rcParams.update(PLOT_STYLE)

# Suppress SHAP/NumPy FutureWarnings about global RNG
warnings.filterwarnings("ignore", message=".*NumPy global RNG.*")


def _save_fig(fig, name, output_dir=None):
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    stamped = f"{name}_{ts}"
    out = output_dir or XAI_DIR
    os.makedirs(out, exist_ok=True)
    for fmt in FIG_FORMAT:
        fig.savefig(f"{out}/{stamped}.{fmt}", dpi=FIG_DPI,
                    bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def _get_feature_names(n_features: int) -> list[str]:
    """
    Retorna nomes de features para o dataset CICIoMT2024.
    - Se n_features <= 45: retorna os primeiros n_features nomes.
    - Se n_features > 45: complementa com Feature_{i} genéricos.
    """
    known = CICIOMT_FEATURE_NAMES  # 45 nomes
    if n_features <= len(known):
        return known[:n_features]
    return known + [f'Feature_{i}' for i in range(len(known), n_features)]




def run_shap_analysis(model, X_train, X_test, y_test, class_names,
                       model_name, scenario, model_type='tree',
                       output_dir=None):
    """
    Run comprehensive SHAP analysis.

    Args:
        model: trained model
        X_train: 2D array for background data selection
        X_test: 2D array for explanations
        y_test: ground truth labels
        class_names: list of class names
        model_name: string identifier
        scenario: string scenario name
        model_type: 'tree' for RF/LightGBM, 'neural' for CNN
        output_dir: optional output directory (default: XAI_DIR)
    """
    log.info(f"  SHAP analysis for {model_name} ({scenario})...")
    safe = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    n_features = X_train.shape[-1] if X_train.ndim == 2 else X_train.shape[1]
    feature_names = _get_feature_names(n_features)

    # Subsample for efficiency
    n_bg = min(SHAP_BACKGROUND_SAMPLES, len(X_train))
    n_test = min(SHAP_TEST_SAMPLES, len(X_test))
    rng = np.random.RandomState(SEED)
    bg_idx = rng.choice(len(X_train), n_bg, replace=False)
    test_idx = rng.choice(len(X_test), n_test, replace=False)

    try:
        if model_type == 'tree':
            X_bg = X_train[bg_idx] if X_train.ndim == 2 else X_train[bg_idx].reshape(n_bg, -1)
            X_exp = X_test[test_idx] if X_test.ndim == 2 else X_test[test_idx].reshape(n_test, -1)

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_exp)

        elif model_type == 'neural':
            X_bg = X_train[bg_idx]
            X_exp = X_test[test_idx]

            # Ensure 3D input for CNN: (batch, features, 1)
            if X_bg.ndim == 2:
                X_bg = X_bg.reshape(X_bg.shape[0], X_bg.shape[1], 1)
            if X_exp.ndim == 2:
                X_exp = X_exp.reshape(X_exp.shape[0], X_exp.shape[1], 1)

            explainer = shap.GradientExplainer(model, X_bg)
            shap_values = explainer.shap_values(X_exp)

        elif model_type == 'neural_flat':
            # For flat neural networks (MLP) — keep data as 2D
            X_bg = X_train[bg_idx].astype(np.float32)
            X_exp = X_test[test_idx].astype(np.float32)

            explainer = shap.GradientExplainer(model, X_bg)
            shap_values = explainer.shap_values(X_exp)

        else:
            log.warning(f"    Unknown model_type: {model_type}. Skipping.")
            return

        # --- Normalize shap_values to list of 2D arrays ---
        is_binary = len(class_names) == 2
        n_classes = len(class_names)

        def _flatten_sv(sv):
            """Flatten any SHAP value array to 2D (n_samples, n_features)."""
            if sv.ndim == 1:
                return sv.reshape(1, -1)
            if sv.ndim == 2:
                return sv[:, :n_features] if sv.shape[1] > n_features else sv
            # ndim >= 3: squeeze all trailing dims except first two
            # e.g. (500, 45, 1) -> (500, 45)
            # e.g. (500, 1, 45, 1) -> (500, 45)
            s = sv.shape
            flat = sv.reshape(s[0], -1)
            return flat[:, :n_features] if flat.shape[1] > n_features else flat

        # Convert single ndarray to list of per-class arrays
        if not isinstance(shap_values, list):
            sv = np.asarray(shap_values)
            # Try to detect class axis: last dim == n_classes
            if sv.ndim >= 3 and sv.shape[-1] == n_classes:
                shap_values = [_flatten_sv(sv[..., i]) for i in range(n_classes)]
            elif sv.ndim >= 4 and sv.shape[-2] == n_classes:
                shap_values = [_flatten_sv(sv[..., i, :]) for i in range(n_classes)]
            else:
                shap_values = [_flatten_sv(sv)]

        # Flatten each per-class array
        shap_values = [_flatten_sv(sv) for sv in shap_values]

        # Flatten X_exp for plotting
        X_plot = X_exp.reshape(n_test, -1) if X_exp.ndim > 2 else X_exp.copy()
        if X_plot.shape[1] > n_features:
            X_plot = X_plot[:, :n_features]

        # Determine what to plot globally
        if is_binary and len(shap_values) == 2:
            sv_global = shap_values[1]  # positive class
        elif len(shap_values) == 1:
            sv_global = shap_values[0]
        else:
            # Multi-class: mean |SHAP| across all classes
            sv_global = np.mean([np.abs(sv) for sv in shap_values], axis=0)

        # Align shapes (defensive)
        min_cols = min(sv_global.shape[1], X_plot.shape[1], n_features)
        sv_global = sv_global[:, :min_cols]
        X_plot = X_plot[:, :min_cols]
        feature_names = feature_names[:min_cols]
        shap_values = [sv[:, :min_cols] for sv in shap_values]

        # --- Global Summary Plot ---
        fig = plt.figure(figsize=(10, 6))
        is_multiclass_mean = not is_binary and len(shap_values) > 2
        plot_type = 'bar' if is_multiclass_mean else None

        if plot_type == 'bar':
            shap.summary_plot(sv_global, X_plot,
                              feature_names=feature_names,
                              plot_type='bar', show=False, max_display=15)
        else:
            shap.summary_plot(sv_global, X_plot,
                              feature_names=feature_names,
                              show=False, max_display=15)

        plt.title(f'SHAP Global Feature Importance: {model_name} ({scenario})',
                  fontweight='bold', fontsize=11)
        _save_fig(fig, f'shap_global_{safe}_{scenario}', output_dir)

        # --- Per-class summary (for multi-class, <=6 classes) ---
        if len(shap_values) > 1 and not is_binary and len(class_names) <= 6:
            for cls_idx, cls_name in enumerate(class_names):
                if cls_idx >= len(shap_values):
                    break
                fig = plt.figure(figsize=(10, 6))
                sv = shap_values[cls_idx]
                shap.summary_plot(sv, X_plot,
                                  feature_names=feature_names,
                                  show=False, max_display=10)
                plt.title(f'SHAP: {cls_name} ({model_name}, {scenario})',
                          fontweight='bold', fontsize=11)
                _save_fig(fig, f'shap_class_{cls_name}_{safe}_{scenario}', output_dir)

        log.info(f"    SHAP plots saved for {model_name}")

    except Exception as e:
        log.error(f"    SHAP failed for {model_name}: {e}")


