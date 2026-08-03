"""
LIME explainability analysis for IoMT IDS models.

Implements:
  - Instance-level explanations for correctly/incorrectly classified samples
  - Explanations for benign and attack instances
  - Feature contribution visualization
  - Comparative LIME across models for same instance

Reference: Ribeiro et al. (2016). "Why Should I Trust You?"
"""
import logging
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular

from config import (
    SEED, LIME_NUM_FEATURES, LIME_NUM_SAMPLES,
    XAI_DIR, FIG_DPI, FIG_FORMAT, PLOT_STYLE,
    CICIOMT_FEATURE_NAMES,
)

log = logging.getLogger(__name__)
plt.rcParams.update(PLOT_STYLE)
np.random.seed(SEED)


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


def _find_representative_instances(X_test, y_test, y_pred, class_names, n_per_class=1):
    """
    Find representative instances for LIME:
    - One correctly classified per class (if available)
    - One misclassified per class (if available)
    """
    instances = {}
    for cls_idx, cls_name in enumerate(class_names):
        mask = (y_test == cls_idx)
        if not mask.any():
            continue

        # Correctly classified
        correct = mask & (y_pred == cls_idx)
        if correct.any():
            idx = np.where(correct)[0][0]
            instances[f'{cls_name}_correct'] = idx

        # Misclassified
        wrong = mask & (y_pred != cls_idx)
        if wrong.any():
            idx = np.where(wrong)[0][0]
            instances[f'{cls_name}_misclassified'] = idx

    return instances


def run_lime_analysis(model, X_train, X_test, y_test, y_pred, class_names,
                       model_name, scenario, predict_fn=None,
                       output_dir=None):
    """
    Run LIME instance-level explanations.

    Args:
        model: trained model
        X_train: 2D training data (for LIME's background distribution)
        X_test: 2D test data
        y_test: ground truth
        y_pred: model predictions
        class_names: list of class names
        model_name: string identifier
        scenario: scenario name
        predict_fn: optional custom predict function (for CNNs needing reshape)
        output_dir: optional output directory (default: XAI_DIR)
    """
    log.info(f"  LIME analysis for {model_name} ({scenario})...")
    safe = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
    n_features = X_train.shape[1]
    feature_names = _get_feature_names(n_features)

    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train[:min(5000, len(X_train))],
        feature_names=feature_names[:n_features],
        class_names=class_names,
        mode='classification',
        random_state=SEED,
    )

    # Prediction function
    if predict_fn is None:
        if hasattr(model, 'predict_proba'):
            predict_fn = model.predict_proba
        else:
            # For Keras models: reshape to 3D if needed
            def predict_fn(X):
                if X.ndim == 2:
                    X = X[..., np.newaxis]
                return model.predict(X, verbose=0)

    # Find representative instances
    instances = _find_representative_instances(X_test, y_test, y_pred, class_names)

    if not instances:
        log.warning(f"    No representative instances found. Skipping LIME.")
        return

    # Explain key instances
    max_explanations = min(6, len(instances))  # Limit for time
    explained = 0

    for label, idx in instances.items():
        if explained >= max_explanations:
            break

        try:
            instance = X_test[idx]
            true_label = class_names[y_test[idx]]
            pred_label = class_names[y_pred[idx]]

            explanation = explainer.explain_instance(
                instance, predict_fn,
                num_features=LIME_NUM_FEATURES,
                num_samples=LIME_NUM_SAMPLES,
            )

            # Save as figure
            fig = explanation.as_pyplot_figure()
            fig.set_size_inches(10, 5)
            plt.title(
                f'LIME: {label}\n'
                f'True: {true_label} | Predicted: {pred_label}',
                fontweight='bold', fontsize=10,
            )
            _save_fig(fig, f'lime_{safe}_{scenario}_{label}', output_dir)
            explained += 1

        except Exception as e:
            log.warning(f"    LIME failed for {label}: {e}")

    log.info(f"    {explained} LIME explanations saved for {model_name}")
