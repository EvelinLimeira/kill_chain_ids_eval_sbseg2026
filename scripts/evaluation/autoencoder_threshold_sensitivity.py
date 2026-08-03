#!/usr/bin/env python3
"""
autoencoder_threshold_sensitivity.py
=====================================
Reproducible MULTI-SEED experiment for a SINGLE benign-only autoencoder used
as a BINARY anomaly detector over the ORIGINAL 19-class CICIoMT2024 test
partition.

Scientific design 
---------------------------------------------------------------
* One autoencoder, trained on benign traffic only, evaluated as
  benign(0)-vs-attack(1). The 19-class labels are used ONLY to (a) binarize
  and (b) compute per-attack (18 classes) and per-phase (5 groups) recall.
* SPLIT_SEED (default 42) fixes the benign train/validation partition ONCE and
  is reused by every training seed, so the measured variability reflects weight
  initialization, batch order and stochastic training -- NOT changes in data
  composition.
* The SAME benign validation set is used both for early stopping AND for
  threshold calibration. This matches the current paper protocol and is
  recorded explicitly in every metadata.json.
* The primary threshold is P85 of the benign VALIDATION reconstruction errors.
  Thresholds are NEVER selected on the test set.

Reproducibility architecture
----------------------------
* Orchestrator/worker split. The orchestrator iterates over seeds and launches
  one INDEPENDENT SUBPROCESS per seed (fresh TensorFlow state). It then
  aggregates the per-seed raw artefacts.
* Device and determinism environment variables are configured BEFORE the first
  TensorFlow import (TF is imported lazily inside the worker only).
* Scientific functions return FULL-PRECISION floats. Rounding happens only in a
  dedicated ``format_results_for_article`` layer; every artefact is written both
  as ``*_raw.csv`` (full precision) and ``*_article_formatted.csv``.
* Primary statistics across seeds: mean and sample standard deviation
  (ddof=1) with a 95% Student-t confidence interval. Median/IQR are secondary.
  No "best/median/representative" seed is ever chosen as the paper result.
  Seed 42 is only a reference-artefact run (model, errors, predictions).

Usage
-----
Canonical CPU run (5 seeds):
    python autoencoder_threshold_sensitivity.py \\
        --seeds 40 41 42 43 44 --split-seed 42 --device cpu \\
        --run-tag linux_cpu_camera_ready

Reference artefact:
    Do NOT run a separate single-seed job. The reference artefact is seed 42
    INSIDE the canonical multi-seed run, i.e. the directory
        <canonical_multiseed_run>/seed_42/
    (model.keras, errors, predictions, metadata). It is flagged
    "reference_seed": true and is one of the exact models included in the
    reported aggregation. A single-seed invocation (e.g. --seeds 42) is only an
    optional reproduction command, never the artefact cited by the paper.

Smoke test:
    python autoencoder_threshold_sensitivity.py \\
        --seeds 42 --device cpu --smoke-test --run-tag smoke

Internal worker invocation (spawned by the orchestrator; not for manual use):
    python autoencoder_threshold_sensitivity.py --worker-seed 42 --device cpu \\
        --output-dir <run_dir> ...
"""

# NOTE: TensorFlow is intentionally NOT imported at module import time. It is
# imported lazily inside the worker AFTER device/determinism env vars are set.
import argparse
import gc
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup so that "config" and "data_loader" from the project root import.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

log = logging.getLogger("ae_multiseed")

# ===========================================================================
# Frozen experiment constants
# ===========================================================================
SPLIT_SEED_DEFAULT = 42
DEFAULT_SEEDS = [40, 41, 42, 43, 44]
REFERENCE_SEED_DEFAULT = 42

PRIMARY_PERCENTILE = 85
SENSITIVITY_PERCENTILES = [85, 90, 95, 99]

# Architecture / hyperparameters (must match the paper description).
ARCHITECTURE = [45, 64, 32, 16, 32, 64, 45]
ENC_DIMS = [64, 32, 16]
DROPOUT = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 512
MAX_EPOCHS = 20
EARLY_STOPPING_PATIENCE = 5

# Expected dataset invariants (camera-ready corrected distribution).
EXPECTED_N_FEATURES = 45
EXPECTED_N_CLASSES = 19
EXPECTED_N_TEST = 1_614_182
EXPECTED_N_BENIGN = 37_607
EXPECTED_N_ATTACK = 1_576_575

# Inference benchmark defaults.
DEFAULT_BENCH_BATCH_SIZES = [1, 32, 256, 512, 4096]
DEFAULT_REPS_BS1 = 1000
DEFAULT_REPS_LARGE = 100
DEFAULT_FULL_TEST_PASSES = 5
DEFAULT_WARMUP_ITERS = 10

# Kill-chain phase mapping for the 18 attack classes.
PHASE_MAP = {
    "Recon-OS_Scan":             "Reconnaissance",
    "Recon-Ping_Sweep":          "Reconnaissance",
    "Recon-Port_Scan":           "Reconnaissance",
    "Recon-VulScan":             "Reconnaissance",
    "Spoofing":                  "ARP Spoofing",
    "DoS-ICMP":                  "Impact: DoS",
    "DoS-SYN":                   "Impact: DoS",
    "DoS-TCP":                   "Impact: DoS",
    "DoS-UDP":                   "Impact: DoS",
    "DDoS-ICMP":                 "Impact: DDoS",
    "DDoS-SYN":                  "Impact: DDoS",
    "DDoS-TCP":                  "Impact: DDoS",
    "DDoS-UDP":                  "Impact: DDoS",
    "MQTT-DDoS-Connect_Flood":   "Impact: MQTT",
    "MQTT-DDoS-Publish_Flood":   "Impact: MQTT",
    "MQTT-DoS-Connect_Flood":    "Impact: MQTT",
    "MQTT-DoS-Publish_Flood":    "Impact: MQTT",
    "MQTT-Malformed_Data":       "Impact: MQTT",
}
PHASE_ORDER = [
    "Reconnaissance", "ARP Spoofing",
    "Impact: DoS", "Impact: DDoS", "Impact: MQTT",
]

# Metrics reported (in order) in the aggregated main table.
MAIN_METRICS_ORDER = [
    "tau_p85", "recall_attack", "precision_attack", "fpr", "fp",
    "specificity", "balanced_accuracy", "f1_macro", "f1_attack",
    "accuracy", "kappa", "auc_roc", "ap_attack", "ap_benign",
    "ap_attack_minus_prevalence", "ap_benign_minus_prevalence",
]

# Student-t 0.975 critical values (two-sided 95%) for small df; fallback 1.96.
_T_TABLE_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


# ===========================================================================
# Fingerprinting helpers
# ===========================================================================
def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of_array(arr: np.ndarray) -> str:
    """SHA-256 of a numpy array's raw bytes (C-contiguous, dtype/shape aware)."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode("utf-8"))
    h.update(str(a.shape).encode("utf-8"))
    h.update(a.tobytes())
    return h.hexdigest()


def sha256_of_strings(items) -> str:
    h = hashlib.sha256()
    for s in items:
        h.update(str(s).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# ===========================================================================
# Pure metric functions (FULL PRECISION -- never round here)
# ===========================================================================
def confusion_counts(y_true_binary, y_pred_binary):
    """Return (tp, fp, tn, fn) as ints. 1 = attack (positive), 0 = benign."""
    yt = np.asarray(y_true_binary).astype(np.int64)
    yp = np.asarray(y_pred_binary).astype(np.int64)
    tp = int(np.sum((yt == 1) & (yp == 1)))
    fp = int(np.sum((yt == 0) & (yp == 1)))
    tn = int(np.sum((yt == 0) & (yp == 0)))
    fn = int(np.sum((yt == 1) & (yp == 0)))
    return tp, fp, tn, fn


def safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def metrics_at_threshold(y_true_binary, errors, tau, val_errors=None,
                         attack_prevalence=None, benign_prevalence=None):
    """
    Compute the full binary metric set at a given threshold, FULL PRECISION.

    ``tau`` flags a sample as attack when ``errors > tau``.
    ``val_errors`` (optional) enables the calibration-exceedance diagnostics.
    Threshold-independent metrics (AUC-ROC, AP) are included here; do NOT repeat
    them per threshold row in the sensitivity table.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    errors = np.asarray(errors)
    y_true_binary = np.asarray(y_true_binary).astype(np.int64)
    y_pred = (errors > tau).astype(np.int64)

    tp, fp, tn, fn = confusion_counts(y_true_binary, y_pred)
    n_alerts = int(tp + fp)

    recall_attack = safe_div(tp, tp + fn)
    precision_attack = safe_div(tp, tp + fp)
    specificity = safe_div(tn, tn + fp)
    fpr = safe_div(fp, fp + tn)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    balanced_accuracy = (recall_attack + specificity) / 2.0
    f1_attack = safe_div(2 * precision_attack * recall_attack,
                         precision_attack + recall_attack)

    # Macro F1 over the two classes (benign, attack), full precision.
    precision_benign = safe_div(tn, tn + fn)
    recall_benign = safe_div(tn, tn + fp)
    f1_benign = safe_div(2 * precision_benign * recall_benign,
                         precision_benign + recall_benign)
    f1_macro = (f1_attack + f1_benign) / 2.0

    # Cohen's kappa (full precision, computed directly from counts).
    total = tp + fp + tn + fn
    po = safe_div(tp + tn, total)
    p_pred_pos = safe_div(tp + fp, total)
    p_true_pos = safe_div(tp + fn, total)
    pe = p_pred_pos * p_true_pos + (1 - p_pred_pos) * (1 - p_true_pos)
    kappa = safe_div(po - pe, 1 - pe) if (1 - pe) != 0 else 0.0

    try:
        auc_roc = float(roc_auc_score(y_true_binary, errors))
    except ValueError:
        auc_roc = float("nan")
    try:
        ap_attack = float(average_precision_score(y_true_binary, errors))
    except ValueError:
        ap_attack = float("nan")
    try:
        ap_benign = float(average_precision_score(1 - y_true_binary, -errors))
    except ValueError:
        ap_benign = float("nan")

    if attack_prevalence is None:
        attack_prevalence = float(np.mean(y_true_binary))
    if benign_prevalence is None:
        benign_prevalence = 1.0 - attack_prevalence

    out = {
        "tau": float(tau),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn, "n_alerts": n_alerts,
        "accuracy": accuracy,
        "recall_attack": recall_attack,
        "precision_attack": precision_attack,
        "specificity": specificity,
        "fpr": fpr,
        "balanced_accuracy": balanced_accuracy,
        "f1_attack": f1_attack,
        "f1_macro": f1_macro,
        "kappa": kappa,
        "auc_roc": auc_roc,
        "ap_attack": ap_attack,
        "ap_benign": ap_benign,
        "attack_prevalence": float(attack_prevalence),
        "benign_prevalence": float(benign_prevalence),
        "ap_attack_minus_prevalence": ap_attack - float(attack_prevalence),
        "ap_benign_minus_prevalence": ap_benign - float(benign_prevalence),
    }
    if val_errors is not None:
        cal = calibration_exceedance_rate(val_errors, tau)
        out["calibration_exceedance_rate"] = cal
        out["test_benign_fpr_minus_calibration_exceedance"] = fpr - cal
    return out


def calibration_exceedance_rate(val_errors, tau) -> float:
    """Proportion of benign validation errors strictly above tau (full prec.)."""
    val_errors = np.asarray(val_errors)
    return float(np.mean(val_errors > tau))


def compute_thresholds(val_errors, percentiles):
    """Map percentile -> tau computed on benign VALIDATION errors only."""
    val_errors = np.asarray(val_errors)
    return {int(p): float(np.percentile(val_errors, p)) for p in percentiles}


def build_sensitivity_rows(y_true_binary, test_errors, val_errors, percentiles):
    """Per-threshold rows (threshold-DEPENDENT metrics only), full precision."""
    rows = []
    for p in percentiles:
        tau = float(np.percentile(np.asarray(val_errors), p))
        m = metrics_at_threshold(y_true_binary, test_errors, tau,
                                 val_errors=val_errors)
        rows.append({
            "percentile": int(p),
            "tau": m["tau"],
            "calibration_exceedance_rate": m["calibration_exceedance_rate"],
            "test_benign_fpr_minus_calibration_exceedance":
                m["test_benign_fpr_minus_calibration_exceedance"],
            "recall_attack": m["recall_attack"],
            "precision_attack": m["precision_attack"],
            "fpr": m["fpr"],
            "specificity": m["specificity"],
            "balanced_accuracy": m["balanced_accuracy"],
            "f1_attack": m["f1_attack"],
            "f1_macro": m["f1_macro"],
            "kappa": m["kappa"],
            "tp": m["tp"], "fp": m["fp"], "tn": m["tn"], "fn": m["fn"],
            "n_alerts": m["n_alerts"],
        })
    return rows


def check_monotonicity(sensitivity_rows):
    """
    With increasing percentiles: tau non-decreasing; FPR, recall and alert
    count non-increasing. Raises RuntimeError (not assert) on violation.
    """
    rows = sorted(sensitivity_rows, key=lambda r: r["percentile"])
    taus = [r["tau"] for r in rows]
    fprs = [r["fpr"] for r in rows]
    recalls = [r["recall_attack"] for r in rows]
    alerts = [r["n_alerts"] for r in rows]

    def non_decreasing(xs):
        return all(b >= a for a, b in zip(xs, xs[1:]))

    def non_increasing(xs):
        return all(b <= a for a, b in zip(xs, xs[1:]))

    if not non_decreasing(taus):
        raise RuntimeError(f"tau not non-decreasing across percentiles: {taus}")
    if not non_increasing(fprs):
        raise RuntimeError(f"FPR not non-increasing across percentiles: {fprs}")
    if not non_increasing(recalls):
        raise RuntimeError(
            f"Recall not non-increasing across percentiles: {recalls}")
    if not non_increasing(alerts):
        raise RuntimeError(
            f"Alert count not non-increasing across percentiles: {alerts}")


def validate_phase_mapping(class_names):
    """Structural validation of the 18-attack / 5-phase taxonomy."""
    if "Benign" not in class_names:
        raise ValueError(
            f"Expected class 'Benign' not found. Classes: {sorted(class_names)}")
    attack_classes = set(class_names) - {"Benign"}
    unmapped = attack_classes - set(PHASE_MAP)
    if unmapped:
        raise ValueError(
            f"Attack classes without a phase mapping (would be 'Unknown'): "
            f"{sorted(unmapped)}")
    if len(attack_classes) != 18:
        raise ValueError(
            f"Expected 18 attack classes, found {len(attack_classes)}: "
            f"{sorted(attack_classes)}")


def recall_by_class_and_phase(y_test_19class, class_names, errors, tau, n_attack):
    """
    Per-attack and per-phase recall for a single seed, FULL PRECISION.
    Returns (df_class, df_phase). Macro recall of a phase is the simple mean of
    the FULL-PRECISION per-class recalls (never rounded before averaging).
    """
    validate_phase_mapping(class_names)
    y_test_19class = np.asarray(y_test_19class)
    y_pred_binary = (np.asarray(errors) > tau).astype(np.int64)

    rows_class = []
    for idx, name in enumerate(class_names):
        if name == "Benign":
            continue
        mask = y_test_19class == idx
        support = int(np.sum(mask))
        if support == 0:
            continue
        detected = int(np.sum(y_pred_binary[mask]))
        rows_class.append({
            "phase": PHASE_MAP[name],
            "class": name,
            "support": support,
            "detected": detected,
            "missed": support - detected,
            "recall": float(detected) / float(support),
        })
    df_class = pd.DataFrame(rows_class).sort_values(["phase", "class"]) \
        .reset_index(drop=True)

    if int(df_class["support"].sum()) != int(n_attack):
        raise ValueError(
            f"Sum of per-class attack support ({int(df_class['support'].sum())}) "
            f"!= number of test attacks ({int(n_attack)}).")

    rows_phase = []
    for phase in PHASE_ORDER:
        grp = df_class[df_class["phase"] == phase]
        if len(grp) == 0:
            continue
        support = int(grp["support"].sum())
        detected = int(grp["detected"].sum())
        rows_phase.append({
            "phase": phase,
            "n_classes": int(len(grp)),
            "support": support,
            "detected": detected,
            "missed": support - detected,
            "macro_recall": float(np.mean(grp["recall"].to_numpy())),
            "weighted_recall": float(detected) / float(support),
        })
    df_phase = pd.DataFrame(rows_phase).reset_index(drop=True)
    return df_class, df_phase


# ===========================================================================
# Aggregation statistics (mean/SD primary; median/IQR secondary; 95% t-CI)
# ===========================================================================
def t_critical_975(df: int) -> float:
    """Two-sided 95% Student-t critical value. Fallback to 1.96 for large df."""
    if df <= 0:
        return float("nan")
    try:
        from scipy.stats import t as _t
        return float(_t.ppf(0.975, df))
    except Exception:
        return float(_T_TABLE_975.get(df, 1.96))


def aggregate_stats(values):
    """
    Aggregate a list of per-seed values. Primary: mean + sample SD (ddof=1) +
    95% Student-t CI. Secondary: median, Q1, Q3, min, max, n. Full precision.
    """
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    n = int(arr.size)
    mean = float(np.mean(arr)) if n else float("nan")
    sd = float(np.std(arr, ddof=1)) if n >= 2 else float("nan")
    if n >= 2 and np.isfinite(sd):
        tcrit = t_critical_975(n - 1)
        half = tcrit * sd / np.sqrt(n)
        ci_low, ci_high = mean - half, mean + half
    else:
        ci_low = ci_high = float("nan")
    return {
        "mean": mean,
        "sd": sd,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "median": float(np.median(arr)) if n else float("nan"),
        "q1": float(np.percentile(arr, 25)) if n else float("nan"),
        "q3": float(np.percentile(arr, 75)) if n else float("nan"),
        "min": float(np.min(arr)) if n else float("nan"),
        "max": float(np.max(arr)) if n else float("nan"),
        "n_seeds": n,
    }


# ===========================================================================
# Article formatting layer (rounding ONLY here)
# ===========================================================================
def _round_metric(name: str, value):
    """Presentation rounding: metrics 4dp, thresholds/tau 6dp, counts int."""
    if value is None:
        return None
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return value
    if np.isnan(fv):
        return fv
    lname = name.lower()
    if lname.startswith("n_") or lname in {"tp", "fp", "tn", "fn", "support",
                                           "detected", "missed", "n_classes",
                                           "n_seeds", "n_alerts", "epochs_executed"}:
        return int(round(fv))
    if "tau" in lname or "threshold" in lname:
        return round(fv, 6)
    return round(fv, 4)


def format_results_for_article(df: pd.DataFrame) -> pd.DataFrame:
    """Return a rounded copy for article tables. Never mutates the raw frame."""
    out = df.copy()
    for col in out.columns:
        out[col] = out[col].map(lambda v, c=col: _round_metric(c, v))
    return out


def mean_sd_string(mean: float, sd: float, decimals: int = 4) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "n/a"
    if sd is None or (isinstance(sd, float) and np.isnan(sd)):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} $\\pm$ {sd:.{decimals}f}"


# ===========================================================================
# Environment fingerprint & consistency
# ===========================================================================
def library_versions():
    versions = {"python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "pandas_version": pd.__version__}
    try:
        import sklearn
        versions["scikit_learn_version"] = sklearn.__version__
    except Exception:
        versions["scikit_learn_version"] = None
    return versions


def git_commit_or_none():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_PROJECT_ROOT,
            capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


# Fields that MUST be identical for seeds to be aggregated together.
# Includes thread configuration and dataset/split fingerprints, because two
# runs with different threads or a different split must never be combined.
ENV_CONSISTENCY_KEYS = [
    "os_platform", "python_version", "tensorflow_version", "numpy_version",
    "pandas_version", "scikit_learn_version", "device_used", "architecture",
    "n_features", "batch_size", "max_epochs", "early_stopping_patience",
    "intra_op_threads", "inter_op_threads", "determinism_enabled",
    "split_seed", "loader_split_seed",
    "dataset_fingerprint", "x_test_fingerprint", "feature_list_fingerprint",
    "class_list_fingerprint",
    "benign_split_train_idx_fingerprint", "benign_split_val_idx_fingerprint",
    "feature_names",
]


def check_environment_consistency(per_seed_metadata):
    """
    Ensure every field in ENV_CONSISTENCY_KEYS is identical across seeds.
    Returns (ok: bool, report: dict). Never combines CPU/GPU or different envs.
    """
    report = {"consistent": True, "keys": {}, "discrepancies": []}
    if not per_seed_metadata:
        report["consistent"] = False
        report["discrepancies"].append("no per-seed metadata found")
        return False, report

    for key in ENV_CONSISTENCY_KEYS:
        seen = {}
        for md in per_seed_metadata:
            val = md.get(key)
            val_key = json.dumps(val, sort_keys=True, default=str)
            seen.setdefault(val_key, []).append(md.get("training_seed"))
        report["keys"][key] = {"distinct_values": len(seen)}
        if len(seen) > 1:
            report["consistent"] = False
            report["discrepancies"].append({
                "field": key,
                "groups": {k: v for k, v in seen.items()},
            })
    return report["consistent"], report


# ===========================================================================
# Device / determinism (env vars MUST be set before importing TensorFlow)
# ===========================================================================
def set_env_before_tf(device: str, seed: int, tf_log_level: str = "3"):
    """Configure device + determinism environment variables. Call before any
    TensorFlow import (i.e. at the very start of the worker process).

    tf_log_level maps to TF_CPP_MIN_LOG_LEVEL (0=all, 1=no INFO, 2=no INFO/WARN,
    3=no INFO/WARN/ERROR). Default 3 silences a benign C++ tf.data version-
    compatibility notice ("use_unbounded_threadpool ... Unknown attributes will
    be ignored"). Python exceptions still propagate regardless of this setting;
    lower it (e.g. --tf-log-level 0) for debugging.
    """
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = str(tf_log_level)


def quiet_tf_python_logging():
    """Silence the Python-side TensorFlow/Keras deprecation chatter (e.g. the
    'tf.reset_default_graph is deprecated' notice emitted by clear_session).
    Genuine errors still raise Python exceptions."""
    try:
        import absl.logging as _absl
        _absl.set_verbosity(_absl.ERROR)
    except Exception:
        pass
    logging.getLogger("tensorflow").setLevel(logging.ERROR)


def seed_everything(seed: int):
    import tensorflow as tf
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    determinism = True
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception as exc:  # pragma: no cover
        determinism = False
        log.warning("enable_op_determinism failed: %s", exc)
    return determinism


def configure_threads(intra, inter):
    import tensorflow as tf
    if intra is not None:
        tf.config.threading.set_intra_op_parallelism_threads(int(intra))
    if inter is not None:
        tf.config.threading.set_inter_op_parallelism_threads(int(inter))


def resolve_device(requested: str):
    """Return (device_used, detected_gpu_names). Abort if gpu requested but
    none detected -- NO silent CPU fallback."""
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    names = [g.name for g in gpus]
    if requested == "gpu":
        if not gpus:
            raise RuntimeError(
                "--device gpu requested but no GPU detected by TensorFlow. "
                "Refusing to fall back to CPU.")
        for g in gpus:
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except RuntimeError:
                pass
        return "gpu", names
    if requested == "cpu":
        return "cpu", names
    # auto
    return ("gpu" if gpus else "cpu"), names


# ===========================================================================
# Model
# ===========================================================================
def build_autoencoder(n_features: int):
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization

    inp = x = Input(shape=(n_features,))
    for dim in ENC_DIMS:
        x = Dense(dim, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(DROPOUT)(x)
    for dim in reversed(ENC_DIMS[:-1]):
        x = Dense(dim, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(DROPOUT)(x)
    decoded = Dense(n_features, activation="linear")(x)
    model = Model(inp, decoded, name="autoencoder")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss="mse")
    return model


def reconstruction_errors_in_batches(model, x, batch_size=4096):
    """Per-sample MSE, computed in batches (full precision float32)."""
    errors = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        recon = model.predict(x[start:end], batch_size=batch_size, verbose=0)
        errors[start:end] = np.mean(np.square(x[start:end] - recon), axis=1)
    return errors


# ===========================================================================
# Timing helpers (perf_counter_ns; GPU sync via .numpy())
# ===========================================================================
def _now_ns():
    return time.perf_counter_ns()


def benchmark_inference(model, x_test, tau, batch_sizes, reps_bs1, reps_large,
                        warmup_iters, seed, device):
    """
    Steady-state inference benchmark. Two scopes:
      * model_only: reconstructed = model(batch, training=False)
      * end_to_end: + MSE + (errors > tau) threshold decision  [PRIMARY]
    Warm-up iterations are excluded. GPU work is materialised via .numpy().
    Returns a list of raw per-repetition records.
    """
    import tensorflow as tf
    n_test = len(x_test)
    records = []
    ts = datetime.now(timezone.utc).isoformat()

    def run_once(batch, scope):
        if scope == "model_only":
            out = model(batch, training=False)
            _ = np.asarray(out)  # force materialisation / GPU sync
        else:  # end_to_end
            out = model(batch, training=False)
            recon = np.asarray(out)
            err = np.mean(np.square(batch - recon), axis=1)
            _ = (err > tau).astype(np.int8)

    for bs in batch_sizes:
        reps = reps_bs1 if bs == 1 else reps_large
        # Pre-select fixed sample batches OUTSIDE the timed region.
        if bs == 1:
            idxs = [i % n_test for i in range(reps)]
            batches = [x_test[i:i + 1] for i in idxs]
        else:
            take = min(bs, n_test)
            batches = [np.ascontiguousarray(x_test[0:take])]
        for scope in ("model_only", "end_to_end"):
            # Warm-up (excluded from timing).
            warm_batch = batches[0]
            for _ in range(warmup_iters):
                run_once(warm_batch, scope)
            for rep in range(reps):
                batch = batches[rep] if bs == 1 else batches[0]
                n_flows = batch.shape[0]
                t0 = _now_ns()
                run_once(batch, scope)
                elapsed_s = (_now_ns() - t0) / 1e9
                records.append({
                    "seed": seed,
                    "device": device,
                    "benchmark_type": scope,
                    "batch_size": int(bs),
                    "repetition": int(rep),
                    "number_of_flows": int(n_flows),
                    "elapsed_seconds": float(elapsed_s),
                    "batch_latency_ms": float(elapsed_s * 1e3),
                    "effective_time_per_flow_us":
                        float(elapsed_s * 1e6 / n_flows),
                    "throughput_flows_per_second":
                        float(n_flows / elapsed_s) if elapsed_s > 0 else float("nan"),
                    "warmup_iterations": int(warmup_iters),
                    "timestamp_utc": ts,
                })
    return records


def benchmark_full_test(model, x_test, tau, passes, seed, device):
    """End-to-end passes over the FULL test set (no disk writes in timed region).
    An explicit, untimed warm-up pass precedes the measured passes so the result
    does not depend on which batch sizes ran before it."""
    records = []
    ts = datetime.now(timezone.utc).isoformat()
    n_test = len(x_test)

    # Explicit warm-up (always excluded from the timed region).
    warm_size = min(4096, n_test)
    warm = model(x_test[:warm_size], training=False)
    warm_recon = np.asarray(warm)
    warm_err = np.mean(np.square(x_test[:warm_size] - warm_recon), axis=1)
    _ = (warm_err > tau).astype(np.int8)

    for p in range(passes):
        t0 = _now_ns()
        errs = reconstruction_errors_in_batches(model, x_test, batch_size=4096)
        _ = (errs > tau).astype(np.int8)
        elapsed_s = (_now_ns() - t0) / 1e9
        records.append({
            "seed": seed, "device": device,
            "benchmark_type": "full_test_end_to_end",
            "batch_size": 4096, "repetition": int(p),
            "number_of_flows": int(n_test),
            "elapsed_seconds": float(elapsed_s),
            "batch_latency_ms": float(elapsed_s * 1e3),
            "effective_time_per_flow_us": float(elapsed_s * 1e6 / n_test),
            "throughput_flows_per_second":
                float(n_test / elapsed_s) if elapsed_s > 0 else float("nan"),
            "warmup_iterations": 1,
            "warmup_batch_size": int(warm_size),
            "warmup_scope": "end_to_end",
            "timestamp_utc": ts,
        })
    return records


def measure_model_reload(model_path, x_sample):
    """Measure model reload and the first prediction after reload inside an
    already initialized TensorFlow runtime.

    This is NOT a process-level cold start because TensorFlow and the execution
    device have already been initialized by the current worker (training, warm-up
    and inference have already run). Reporting it as a cold start would overclaim.
    """
    import tensorflow as tf
    t0 = _now_ns()
    model = tf.keras.models.load_model(model_path, compile=False)
    reload_seconds = (_now_ns() - t0) / 1e9
    t0 = _now_ns()
    output = model(x_sample[0:1], training=False)
    _ = np.asarray(output)
    first_prediction_seconds = (_now_ns() - t0) / 1e9
    del model
    gc.collect()
    return float(reload_seconds), float(first_prediction_seconds)


# ===========================================================================
# Data preparation (shared across seeds via SPLIT_SEED)
# ===========================================================================
def prepare_data(split_seed: int, smoke: bool):
    """
    Load the dataset, validate invariants, and build the FIXED benign
    train/validation split (indices) using split_seed. Returns a dict with the
    arrays, labels, class names, split indices and fingerprints. Timing of this
    step is measured by the caller and NOT counted as training time.
    """
    from config import DATA_DIR, VAL_SIZE, SEED
    from data_loader import load_data
    from sklearn.model_selection import train_test_split as _split

    data = load_data(DATA_DIR, EXPECTED_N_CLASSES)
    X_train_flat = data["X_train_flat"]
    X_val_flat = data["X_val_flat"]
    X_test_flat = data["X_test_flat"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]
    class_names = list(data["label_encoder"].classes_)
    feature_names = [str(f) for f in data.get("feature_names", [])]

    # Structural validation (always).
    if X_train_flat.shape[1] != EXPECTED_N_FEATURES:
        raise ValueError(f"Expected {EXPECTED_N_FEATURES} features, "
                         f"found {X_train_flat.shape[1]}")
    if len(feature_names) != EXPECTED_N_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_N_FEATURES} feature names from data_loader, "
            f"found {len(feature_names)}: {feature_names}")
    if len(feature_names) != X_train_flat.shape[1]:
        raise ValueError(
            f"Feature-name count ({len(feature_names)}) does not match feature "
            f"matrix width ({X_train_flat.shape[1]}).")
    if len(class_names) != EXPECTED_N_CLASSES:
        raise ValueError(f"Expected {EXPECTED_N_CLASSES} classes, "
                         f"found {len(class_names)}: {class_names}")
    if "Benign" not in class_names:
        raise ValueError(f"'Benign' class missing. Classes: {class_names}")
    validate_phase_mapping(class_names)
    for name, arr in (("X_test", X_test_flat), ("X_train", X_train_flat)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or infinite values")
    if len(X_test_flat) != len(y_test):
        raise ValueError("len(X_test) != len(y_test)")

    benign_idx = class_names.index("Benign")

    # Exact-count invariants (skipped under smoke, which subsamples).
    if not smoke:
        if len(y_test) != EXPECTED_N_TEST:
            raise ValueError(f"Expected {EXPECTED_N_TEST} test samples, "
                             f"found {len(y_test)}")

    # Build benign pool (deterministic; depends only on config.SEED split).
    benign_mask_train = y_train == benign_idx
    benign_mask_val = y_val == benign_idx
    X_benign_all = np.concatenate([X_train_flat[benign_mask_train],
                                   X_val_flat[benign_mask_val]])

    # Fixed benign train/val split by INDEX using split_seed.
    all_idx = np.arange(len(X_benign_all))
    train_idx, val_idx = _split(all_idx, test_size=VAL_SIZE,
                                random_state=split_seed)
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    X_train_benign = X_benign_all[train_idx]
    X_val_benign = X_benign_all[val_idx]

    if smoke:
        # Subsample keeping every class present; recompute test set.
        rng = np.random.RandomState(split_seed)
        X_train_benign = X_train_benign[:3000]
        X_val_benign = X_val_benign[:1000]
        keep = []
        for idx in range(len(class_names)):
            cls_idx = np.where(y_test == idx)[0]
            if len(cls_idx) == 0:
                continue
            take = min(300, len(cls_idx))
            keep.append(rng.choice(cls_idx, size=take, replace=False))
        keep = np.sort(np.concatenate(keep))
        X_test_flat = X_test_flat[keep]
        y_test = y_test[keep]

    y_test_binary = (y_test != benign_idx).astype(np.int64)
    n_benign = int(np.sum(y_test_binary == 0))
    n_attack = int(np.sum(y_test_binary == 1))
    if not smoke:
        if n_benign != EXPECTED_N_BENIGN:
            raise ValueError(f"Expected {EXPECTED_N_BENIGN} benign test, "
                             f"found {n_benign}")
        if n_attack != EXPECTED_N_ATTACK:
            raise ValueError(f"Expected {EXPECTED_N_ATTACK} attack test, "
                             f"found {n_attack}")

    fingerprints = {
        "feature_list_fingerprint": sha256_of_strings(feature_names),
        "class_list_fingerprint": sha256_of_strings(class_names),
        "dataset_fingerprint": sha256_of_array(y_test),   # alias: y_test
        "y_test_fingerprint": sha256_of_array(y_test),
        "x_test_fingerprint": sha256_of_array(X_test_flat),
        "benign_split_train_idx_fingerprint": sha256_of_array(train_idx),
        "benign_split_val_idx_fingerprint": sha256_of_array(val_idx),
    }
    return {
        "X_train_benign": X_train_benign,
        "X_val_benign": X_val_benign,
        "X_test_flat": X_test_flat,
        "y_test": y_test,
        "y_test_binary": y_test_binary,
        "class_names": class_names,
        "feature_names": feature_names,
        "benign_idx": benign_idx,
        "n_benign": n_benign,
        "n_attack": n_attack,
        "attack_prevalence": float(np.mean(y_test_binary)),
        "benign_prevalence": float(1.0 - np.mean(y_test_binary)),
        "train_idx": train_idx,
        "val_idx": val_idx,
        "fingerprints": fingerprints,
        "n_features": int(X_train_flat.shape[1]),
        "val_size": VAL_SIZE,
        "loader_split_seed": SEED,
    }


# ===========================================================================
# Worker: run a single seed end-to-end
# ===========================================================================
def run_worker(args):
    import tensorflow as tf
    quiet_tf_python_logging()

    seed = args.worker_seed
    device_requested = args.device
    smoke = args.smoke_test
    seed_dir = os.path.join(args.output_dir, f"seed_{seed}")
    if os.path.exists(seed_dir) and not args.overwrite:
        # Allow re-entry only if not already completed.
        if os.path.exists(os.path.join(seed_dir, "metadata.json")):
            raise FileExistsError(
                f"Seed directory already completed: {seed_dir} "
                f"(use --overwrite to replace).")
    os.makedirs(seed_dir, exist_ok=True)

    _add_file_logger(os.path.join(seed_dir, "run.log"))
    log.info("=== Worker seed=%s device=%s smoke=%s ===", seed,
             device_requested, smoke)

    configure_threads(args.intra_op_threads, args.inter_op_threads)
    determinism = seed_everything(seed)
    device_used, gpu_names = resolve_device(device_requested)
    log.info("device_requested=%s device_detected=%s device_used=%s",
             device_requested, gpu_names or "none", device_used)

    max_epochs = 2 if smoke else MAX_EPOCHS
    patience = 1 if smoke else EARLY_STOPPING_PATIENCE

    # --- A. Data preparation time ---
    t0 = _now_ns()
    d = prepare_data(args.split_seed, smoke)
    data_prep_s = (_now_ns() - t0) / 1e9

    X_train_benign = d["X_train_benign"]
    X_val_benign = d["X_val_benign"]
    X_test_flat = d["X_test_flat"]

    # --- B. Model build time ---
    t0 = _now_ns()
    tf.keras.backend.clear_session()
    model = build_autoencoder(d["n_features"])
    model_build_s = (_now_ns() - t0) / 1e9

    # --- C. Training time (around model.fit ONLY) ---
    from tensorflow.keras.callbacks import EarlyStopping
    es = EarlyStopping(monitor="val_loss", patience=patience,
                       restore_best_weights=True)
    t0 = _now_ns()
    history = model.fit(
        X_train_benign, X_train_benign,
        validation_data=(X_val_benign, X_val_benign),
        epochs=max_epochs, batch_size=BATCH_SIZE,
        callbacks=[es], verbose=0)
    training_s = (_now_ns() - t0) / 1e9

    val_loss_hist = list(map(float, history.history.get("val_loss", [])))
    epochs_executed = len(history.history.get("loss", []))
    best_val_loss = float(np.min(val_loss_hist)) if val_loss_hist else float("nan")
    selected_epoch = int(np.argmin(val_loss_hist)) + 1 if val_loss_hist else -1
    training_per_epoch = training_s / epochs_executed if epochs_executed else float("nan")

    model_path = os.path.join(seed_dir, "model.keras")
    model.save(model_path)
    model_size_bytes = int(os.path.getsize(model_path))

    # --- D. Calibration time ---
    t0 = _now_ns()
    val_errors = reconstruction_errors_in_batches(model, X_val_benign)
    val_recon_s = (_now_ns() - t0) / 1e9
    t0 = _now_ns()
    thresholds = compute_thresholds(val_errors, SENSITIVITY_PERCENTILES)
    thr_compute_s = (_now_ns() - t0) / 1e9
    total_calibration_s = val_recon_s + thr_compute_s
    tau_p85 = thresholds[PRIMARY_PERCENTILE]

    # --- Test reconstruction errors (functional, not the timed benchmark) ---
    test_errors = reconstruction_errors_in_batches(model, X_test_flat)

    # --- Metrics @ P85 (full precision) ---
    m_p85 = metrics_at_threshold(
        d["y_test_binary"], test_errors, tau_p85, val_errors=val_errors,
        attack_prevalence=d["attack_prevalence"],
        benign_prevalence=d["benign_prevalence"])
    m_p85["percentile"] = PRIMARY_PERCENTILE
    m_p85["tau_p85"] = tau_p85
    m_p85["seed"] = seed

    # --- Threshold sensitivity (P85-P99) ---
    sens_rows = build_sensitivity_rows(
        d["y_test_binary"], test_errors, val_errors, SENSITIVITY_PERCENTILES)
    check_monotonicity(sens_rows)

    # --- Per-attack / per-phase recall @ P85 ---
    df_class, df_phase = recall_by_class_and_phase(
        d["y_test"], d["class_names"], test_errors, tau_p85, d["n_attack"])

    # --- E/F/G. Inference benchmark, full-test passes, and model reload ---
    bench_sizes = args.benchmark_batch_sizes or DEFAULT_BENCH_BATCH_SIZES
    reps_bs1 = 20 if smoke else DEFAULT_REPS_BS1
    reps_large = 5 if smoke else DEFAULT_REPS_LARGE
    full_passes = 1 if smoke else DEFAULT_FULL_TEST_PASSES
    if smoke:
        bench_sizes = [b for b in bench_sizes if b <= 32] or [1, 32]

    timing_records = benchmark_inference(
        model, X_test_flat, tau_p85, bench_sizes, reps_bs1, reps_large,
        args.warmup_iters, seed, device_used)
    timing_records += benchmark_full_test(
        model, X_test_flat, tau_p85, full_passes, seed, device_used)
    # Model reload inside an ALREADY-INITIALISED TF runtime (not a process-level
    # cold start: TF/device are already warm from training).
    model_reload_s, first_pred_after_reload_s = measure_model_reload(
        model_path, X_test_flat)

    # --- Save per-seed artefacts (outside every timed region) ---
    predictions_p85 = (test_errors > tau_p85).astype(np.int8)
    np.save(os.path.join(seed_dir, "validation_errors.npy"), val_errors)
    np.save(os.path.join(seed_dir, "test_errors.npy"), test_errors)
    np.save(os.path.join(seed_dir, "predictions_p85.npy"), predictions_p85)

    pd.DataFrame([m_p85]).to_csv(
        os.path.join(seed_dir, "metrics_p85_raw.csv"), index=False)
    with open(os.path.join(seed_dir, "metrics_p85_raw.json"), "w",
              encoding="utf-8") as fh:
        json.dump(m_p85, fh, indent=2)
    pd.DataFrame(sens_rows).to_csv(
        os.path.join(seed_dir, "threshold_sensitivity_raw.csv"), index=False)
    df_class.to_csv(os.path.join(seed_dir, "class_recall_raw.csv"), index=False)
    df_phase.to_csv(os.path.join(seed_dir, "phase_recall_raw.csv"), index=False)
    pd.DataFrame(timing_records).to_csv(
        os.path.join(seed_dir, "timing_raw.csv"), index=False)

    # training_history.csv
    hist_df = pd.DataFrame(history.history)
    hist_df.insert(0, "epoch", np.arange(1, len(hist_df) + 1))
    hist_df.to_csv(os.path.join(seed_dir, "training_history.csv"), index=False)

    timing_summary = summarize_timing(timing_records)
    timing_summary.update({
        "seed": seed, "device_used": device_used,
        "data_preparation_time_seconds": data_prep_s,
        "model_build_time_seconds": model_build_s,
        "training_time_seconds": training_s,
        "epochs_executed": epochs_executed,
        "selected_epoch": selected_epoch,
        "training_time_per_executed_epoch": training_per_epoch,
        "best_val_loss": best_val_loss,
        "validation_reconstruction_time_seconds": val_recon_s,
        "threshold_computation_time_seconds": thr_compute_s,
        "total_calibration_time_seconds": total_calibration_s,
        "model_reload_time_seconds": model_reload_s,
        "first_prediction_after_reload_seconds": first_pred_after_reload_s,
        "model_size_bytes": model_size_bytes,
    })
    with open(os.path.join(seed_dir, "timing_summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(timing_summary, fh, indent=2)

    # metadata.json
    versions = library_versions()
    metadata = {
        "training_seed": seed,
        "split_seed": args.split_seed,
        "reference_seed": (seed == args.reference_seed),
        "run_tag": args.run_tag,
        "smoke_test": smoke,
        "architecture": ARCHITECTURE,
        "encoding_dims": ENC_DIMS,
        "dropout": DROPOUT,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "primary_percentile": PRIMARY_PERCENTILE,
        "sensitivity_percentiles": SENSITIVITY_PERCENTILES,
        "same_val_for_earlystop_and_calibration": True,
        "device_requested": device_requested,
        "device_detected": gpu_names or [],
        "device_used": device_used,
        "determinism_enabled": determinism,
        "intra_op_threads": args.intra_op_threads,
        "inter_op_threads": args.inter_op_threads,
        "os_platform": platform.platform(),
        "tensorflow_version": tf.__version__,
        **versions,
        "feature_names": d["feature_names"],
        "class_names": d["class_names"],
        "loader_split_seed": d["loader_split_seed"],
        "n_features": d["n_features"],
        "n_test_benign": d["n_benign"],
        "n_test_attack": d["n_attack"],
        "attack_prevalence": d["attack_prevalence"],
        "benign_prevalence": d["benign_prevalence"],
        "tau_p85": tau_p85,
        "validation_exceedance_rate_p85":
            calibration_exceedance_rate(val_errors, tau_p85),
        "selected_epoch": selected_epoch,
        "epochs_executed": epochs_executed,
        "best_val_loss": best_val_loss,
        "training_time_seconds": training_s,
        "model_build_time_seconds": model_build_s,
        "total_calibration_time_seconds": total_calibration_s,
        "model_size_bytes": model_size_bytes,
        **d["fingerprints"],
        "git_commit": git_commit_or_none(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(seed_dir, "metadata.json"), "w",
              encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    del model
    gc.collect()
    tf.keras.backend.clear_session()
    log.info("Seed %s complete: recall@P85=%.6f fpr@P85=%.6f tau=%.6f",
             seed, m_p85["recall_attack"], m_p85["fpr"], tau_p85)
    return seed_dir


def summarize_timing(timing_records):
    """Within-seed timing summary: per (benchmark_type, batch_size) latency and
    throughput distribution (median/mean/SD/percentiles), full precision."""
    df = pd.DataFrame(timing_records)
    summary = {"per_benchmark": []}
    if df.empty:
        return summary
    for (btype, bs), grp in df.groupby(["benchmark_type", "batch_size"]):
        lat = grp["batch_latency_ms"].to_numpy(dtype=np.float64)
        thr = grp["throughput_flows_per_second"].to_numpy(dtype=np.float64)
        eff = grp["effective_time_per_flow_us"].to_numpy(dtype=np.float64)
        summary["per_benchmark"].append({
            "benchmark_type": btype,
            "batch_size": int(bs),
            "repetitions": int(len(grp)),
            "latency_ms_median": float(np.median(lat)),
            "latency_ms_mean": float(np.mean(lat)),
            "latency_ms_sd": float(np.std(lat, ddof=1)) if len(lat) > 1 else float("nan"),
            "latency_ms_p5": float(np.percentile(lat, 5)),
            "latency_ms_p25": float(np.percentile(lat, 25)),
            "latency_ms_p75": float(np.percentile(lat, 75)),
            "latency_ms_p95": float(np.percentile(lat, 95)),
            "latency_ms_p99": float(np.percentile(lat, 99)),
            "latency_ms_min": float(np.min(lat)),
            "latency_ms_max": float(np.max(lat)),
            "effective_time_per_flow_us_median": float(np.median(eff)),
            "throughput_mean": float(np.mean(thr)),
            "throughput_sd": float(np.std(thr, ddof=1)) if len(thr) > 1 else float("nan"),
            "throughput_median": float(np.median(thr)),
            "throughput_min": float(np.min(thr)),
            "throughput_max": float(np.max(thr)),
        })
    return summary


def _add_file_logger(path):
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s",
                                      "%H:%M:%S"))
    log.addHandler(fh)


# ===========================================================================
# Orchestrator
# ===========================================================================
def os_tag():
    system = platform.system().lower()
    if "linux" in system:
        return "wsl" if "microsoft" in platform.release().lower() else "linux"
    if "windows" in system:
        return "windows"
    return system or "os"


def build_run_dir(args):
    label_parts = [p for p in [args.run_tag, os_tag(), args.device] if p]
    label = "_".join(label_parts)
    return os.path.join(args.output_dir, label)


def spawn_worker(seed, args, run_dir):
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = str(seed)          # must be set before process start
    env["TF_DETERMINISTIC_OPS"] = "1"
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if args.device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    cmd = [sys.executable, os.path.abspath(__file__),
           "--worker-seed", str(seed),
           "--split-seed", str(args.split_seed),
           "--device", args.device,
           "--output-dir", run_dir,
           "--reference-seed", str(args.reference_seed),
           "--warmup-iters", str(args.warmup_iters),
           "--tf-log-level", str(args.tf_log_level)]
    if args.run_tag:
        cmd += ["--run-tag", args.run_tag]
    if args.smoke_test:
        cmd += ["--smoke-test"]
    if args.overwrite:
        cmd += ["--overwrite"]
    if args.intra_op_threads is not None:
        cmd += ["--intra-op-threads", str(args.intra_op_threads)]
    if args.inter_op_threads is not None:
        cmd += ["--inter-op-threads", str(args.inter_op_threads)]
    if args.benchmark_batch_sizes:
        cmd += ["--benchmark-batch-sizes"] + [str(b) for b in args.benchmark_batch_sizes]
    log.info("Spawning worker for seed %s", seed)
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"Worker for seed {seed} failed "
                           f"(exit code {proc.returncode}).")


def run_orchestrator(args):
    run_dir = build_run_dir(args)
    os.makedirs(run_dir, exist_ok=True)
    _add_file_logger(os.path.join(run_dir, "orchestrator.log"))
    log.info("Run directory: %s", run_dir)
    log.info("Seeds: %s | split_seed=%s | device=%s | reference_seed=%s",
             args.seeds, args.split_seed, args.device, args.reference_seed)

    if args.aggregate_only:
        log.info("Aggregate-only mode: skipping training, reusing existing "
                 "per-seed artefacts (no previous execution is overwritten).")
        for seed in args.seeds:
            md = os.path.join(run_dir, f"seed_{seed}", "metadata.json")
            if not os.path.exists(md):
                raise FileNotFoundError(
                    f"--aggregate-only requested but seed artefact missing: {md}")
    else:
        for seed in args.seeds:
            spawn_worker(seed, args, run_dir)

    aggregate_dir = os.path.join(run_dir, "aggregate")
    os.makedirs(aggregate_dir, exist_ok=True)
    aggregate_results(args, run_dir, aggregate_dir)
    log.info("Aggregation complete: %s", aggregate_dir)
    print_tree(run_dir)


# ===========================================================================
# Aggregation
# ===========================================================================
def _load_seed_artifacts(run_dir, seeds):
    per_seed = []
    for seed in seeds:
        sd = os.path.join(run_dir, f"seed_{seed}")
        with open(os.path.join(sd, "metadata.json"), encoding="utf-8") as fh:
            md = json.load(fh)
        with open(os.path.join(sd, "metrics_p85_raw.json"), encoding="utf-8") as fh:
            m = json.load(fh)
        per_seed.append({
            "seed": seed,
            "metadata": md,
            "metrics_p85": m,
            "sensitivity": pd.read_csv(os.path.join(sd, "threshold_sensitivity_raw.csv")),
            "class_recall": pd.read_csv(os.path.join(sd, "class_recall_raw.csv")),
            "phase_recall": pd.read_csv(os.path.join(sd, "phase_recall_raw.csv")),
            "timing": pd.read_csv(os.path.join(sd, "timing_raw.csv")),
            "timing_summary": json.load(open(os.path.join(sd, "timing_summary.json"),
                                             encoding="utf-8")),
        })
    return per_seed


def aggregate_results(args, run_dir, aggregate_dir):
    per_seed = _load_seed_artifacts(run_dir, args.seeds)

    # 1) Environment consistency gate (refuse mixing CPU/GPU or differing envs).
    ok, report = check_environment_consistency([p["metadata"] for p in per_seed])
    with open(os.path.join(aggregate_dir, "environment_consistency_report.json"),
              "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    if not ok:
        raise RuntimeError(
            "Environment/dataset mismatch across seeds -- aggregation refused. "
            "See environment_consistency_report.json. CPU and GPU (or differing "
            "library versions / splits) must never be combined in one summary.")

    # 2) Main P85 metrics aggregation.
    agg_rows = []
    for metric in MAIN_METRICS_ORDER:
        vals = [p["metrics_p85"][metric] for p in per_seed]
        st = aggregate_stats(vals)
        agg_rows.append({"metric": metric, **st})
    df_main_raw = pd.DataFrame(agg_rows)
    df_main_raw.to_csv(os.path.join(aggregate_dir, "multiseed_metrics_raw.csv"),
                       index=False)
    format_results_for_article(df_main_raw).to_csv(
        os.path.join(aggregate_dir, "multiseed_metrics_article_formatted.csv"),
        index=False)

    # 3) Threshold sensitivity aggregation (per percentile, per metric).
    sens_metrics = ["tau", "calibration_exceedance_rate", "recall_attack",
                    "precision_attack", "fpr", "specificity",
                    "balanced_accuracy", "f1_attack", "f1_macro", "kappa",
                    "n_alerts", "tp", "fp", "tn", "fn"]
    sens_rows = []
    for pct in SENSITIVITY_PERCENTILES:
        row = {"percentile": pct}
        for metric in sens_metrics:
            vals = [float(p["sensitivity"].set_index("percentile").loc[pct, metric])
                    for p in per_seed]
            st = aggregate_stats(vals)
            # Full statistics (mean, sd, ci95, median, q1, q3, min, max, n).
            for statistic, value in st.items():
                row[f"{metric}_{statistic}"] = value
        sens_rows.append(row)
    df_sens_raw = pd.DataFrame(sens_rows)
    df_sens_raw.to_csv(
        os.path.join(aggregate_dir, "multiseed_threshold_sensitivity_raw.csv"),
        index=False)
    format_results_for_article(df_sens_raw).to_csv(
        os.path.join(aggregate_dir,
                     "multiseed_threshold_sensitivity_article_formatted.csv"),
        index=False)

    # 4) Class & phase recall aggregation (support constant across seeds).
    _aggregate_recall(per_seed, aggregate_dir, level="class")
    _aggregate_recall(per_seed, aggregate_dir, level="phase")

    # 5) Timing aggregation (training across seeds; inference across medians).
    _aggregate_timing(per_seed, aggregate_dir)

    # 6) LaTeX tables + manifest.
    _write_latex_tables(df_main_raw, df_sens_raw, aggregate_dir, per_seed)
    manifest = {
        "seeds": args.seeds,
        "split_seed": args.split_seed,
        "reference_seed": args.reference_seed,
        "device": args.device,
        "run_tag": args.run_tag,
        "smoke_test": args.smoke_test,
        "n_seeds": len(args.seeds),
        "run_dir": run_dir,
        "os_tag": os_tag(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "environment_consistent": ok,
        "note": ("Primary results are mean +/- sample SD (ddof=1) with 95% "
                 "Student-t CI across all seeds. No representative/best seed is "
                 "selected. Seed 42 is a reference-artefact run only. The same "
                 "benign validation set is used for early stopping AND threshold "
                 "calibration. Inference timing starts from pre-extracted and "
                 "preprocessed feature vectors."),
    }
    with open(os.path.join(aggregate_dir, "run_manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


def _aggregate_recall(per_seed, aggregate_dir, level):
    key = "class_recall" if level == "class" else "phase_recall"
    group_col = "class" if level == "class" else "phase"
    recall_col = "recall" if level == "class" else "macro_recall"

    # Per-seed long table.
    long_rows = []
    for p in per_seed:
        df = p[key].copy()
        df["seed"] = p["seed"]
        long_rows.append(df)
    per_seed_long = pd.concat(long_rows, ignore_index=True)
    per_seed_long.to_csv(
        os.path.join(aggregate_dir, f"per_seed_{level}_recall_raw.csv"),
        index=False)

    # Support must be constant across seeds.
    agg_rows = []
    groups = per_seed[0][key][group_col].tolist()
    for g in groups:
        supports = [int(p[key].set_index(group_col).loc[g, "support"])
                    for p in per_seed]
        if len(set(supports)) != 1:
            raise ValueError(
                f"Support for {level} '{g}' varies across seeds: {supports}. "
                f"Support must be identical (test partition is fixed).")
        recalls = [float(p[key].set_index(group_col).loc[g, recall_col])
                   for p in per_seed]
        st = aggregate_stats(recalls)
        agg_rows.append({level: g, "support": supports[0],
                         "recall_metric": recall_col, **st})
    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(os.path.join(aggregate_dir, f"aggregate_{level}_recall_raw.csv"),
                  index=False)
    if level == "phase":
        df_agg.to_csv(os.path.join(aggregate_dir, "multiseed_phase_recall_raw.csv"),
                      index=False)
        format_results_for_article(df_agg).to_csv(
            os.path.join(aggregate_dir,
                         "multiseed_phase_recall_article_formatted.csv"),
            index=False)
    else:
        df_agg.to_csv(os.path.join(aggregate_dir, "multiseed_class_recall_raw.csv"),
                      index=False)


def _aggregate_timing(per_seed, aggregate_dir):
    # Training / calibration / build times across seeds (full statistics).
    train_time_fields = [
        "training_time_seconds", "epochs_executed",
        "training_time_per_executed_epoch", "total_calibration_time_seconds",
        "model_build_time_seconds", "model_reload_time_seconds",
        "first_prediction_after_reload_seconds",
        "data_preparation_time_seconds", "model_size_bytes",
    ]
    train_rows = []
    for field in train_time_fields:
        vals = [p["timing_summary"].get(field) for p in per_seed]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        train_rows.append({"metric": field, **aggregate_stats(vals)})
    df_train = pd.DataFrame(train_rows)
    df_train.to_csv(
        os.path.join(aggregate_dir, "multiseed_training_time_raw.csv"), index=False)
    format_results_for_article(df_train).to_csv(
        os.path.join(aggregate_dir,
                     "multiseed_training_time_article_formatted.csv"), index=False)

    # Inference: aggregate ACROSS medians of each seed (no representative model).
    rows = []
    all_keys = set()
    for p in per_seed:
        for b in p["timing_summary"]["per_benchmark"]:
            all_keys.add((b["benchmark_type"], b["batch_size"]))
    for (btype, bs) in sorted(all_keys):
        seed_latency_medians, seed_throughput_medians = [], []
        seed_latency_p95, seed_latency_p99 = [], []
        for p in per_seed:
            for b in p["timing_summary"]["per_benchmark"]:
                if b["benchmark_type"] == btype and b["batch_size"] == bs:
                    seed_latency_medians.append(b["latency_ms_median"])
                    seed_latency_p95.append(b["latency_ms_p95"])
                    seed_latency_p99.append(b["latency_ms_p99"])
                    seed_throughput_medians.append(b["throughput_median"])
        lat = aggregate_stats(seed_latency_medians)
        lat95 = aggregate_stats(seed_latency_p95)
        lat99 = aggregate_stats(seed_latency_p99)
        thr = aggregate_stats(seed_throughput_medians)
        rows.append({
            "benchmark_type": btype, "batch_size": bs,
            # Precise labels: statistics computed ACROSS per-seed values.
            "latency_ms_mean_of_seed_medians": lat["mean"],
            "latency_ms_sd_of_seed_medians": lat["sd"],
            "latency_ms_median_of_seed_medians": lat["median"],
            "latency_ms_mean_of_seed_p95": lat95["mean"],
            "latency_ms_sd_of_seed_p95": lat95["sd"],
            "latency_ms_mean_of_seed_p99": lat99["mean"],
            "latency_ms_sd_of_seed_p99": lat99["sd"],
            "throughput_mean_of_seed_medians": thr["mean"],
            "throughput_sd_of_seed_medians": thr["sd"],
            "n_seeds": lat["n_seeds"],
        })
    df_inf = pd.DataFrame(rows)
    df_inf.to_csv(
        os.path.join(aggregate_dir, "multiseed_inference_timing_raw.csv"), index=False)
    format_results_for_article(df_inf).to_csv(
        os.path.join(aggregate_dir,
                     "multiseed_inference_timing_article_formatted.csv"), index=False)

    # Concatenate ALL raw repetitions across seeds (preserved in artefact).
    pd.concat([p["timing"] for p in per_seed], ignore_index=True).to_csv(
        os.path.join(aggregate_dir, "multiseed_inference_timing_allreps_raw.csv"),
        index=False)


# ===========================================================================
# LaTeX tables (article-facing; generated from raw aggregates)
# ===========================================================================
def _stat(df_main_raw, metric, field):
    row = df_main_raw[df_main_raw["metric"] == metric]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][field])


def _write_latex_tables(df_main_raw, df_sens_raw, aggregate_dir, per_seed):
    n_seeds = len(per_seed)

    # --- Main table ---
    def ms(metric, dec=4):
        return mean_sd_string(_stat(df_main_raw, metric, "mean"),
                              _stat(df_main_raw, metric, "sd"), dec)
    main = [
        f"% Auto-generated. Mean +/- sample SD (ddof=1) across {n_seeds} seeds.",
        "\\begin{tabular}{lc}",
        "\\hline",
        "Metric & Mean $\\pm$ SD \\\\",
        "\\hline",
        f"$\\tau$ (P85) & {ms('tau_p85', 6)} \\\\",
        f"Attack recall & {ms('recall_attack')} \\\\",
        f"FPR & {ms('fpr')} \\\\",
        f"Specificity & {ms('specificity')} \\\\",
        f"Balanced accuracy & {ms('balanced_accuracy')} \\\\",
        f"F1-macro & {ms('f1_macro')} \\\\",
        f"AUC-ROC & {ms('auc_roc')} \\\\",
        f"AP (Attack) & {ms('ap_attack')} \\\\",
        f"AP (Benign) & {ms('ap_benign')} \\\\",
        f"False positives (\\#FP) & {ms('fp')} \\\\",
        "\\hline",
        "\\end{tabular}",
    ]
    _write(aggregate_dir, "table_autoencoder_main.tex", "\n".join(main))

    # --- Threshold sensitivity table ---
    lines = [
        f"% Auto-generated threshold sensitivity (mean +/- SD across {n_seeds} seeds).",
        "\\begin{tabular}{lccccc}",
        "\\hline",
        "Threshold & $\\tau$ & Recall & FPR & Bal.\\ Acc.\\ & F1-macro \\\\",
        "\\hline",
    ]
    for pct in SENSITIVITY_PERCENTILES:
        r = df_sens_raw[df_sens_raw["percentile"] == pct].iloc[0]
        lines.append(
            f"P{pct} & {mean_sd_string(r['tau_mean'], r['tau_sd'], 6)} & "
            f"{mean_sd_string(r['recall_attack_mean'], r['recall_attack_sd'])} & "
            f"{mean_sd_string(r['fpr_mean'], r['fpr_sd'])} & "
            f"{mean_sd_string(r['balanced_accuracy_mean'], r['balanced_accuracy_sd'])} & "
            f"{mean_sd_string(r['f1_macro_mean'], r['f1_macro_sd'])} \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    _write(aggregate_dir, "table_autoencoder_thresholds.tex", "\n".join(lines))

    # --- Phase recall table ---
    df_phase = pd.read_csv(os.path.join(aggregate_dir,
                                        "aggregate_phase_recall_raw.csv"))
    lines = [
        f"% Auto-generated per-phase macro recall (mean +/- SD across {n_seeds} seeds).",
        "\\begin{tabular}{lccc}",
        "\\hline",
        "Phase & \\#Classes & Support & Macro recall (mean $\\pm$ SD) \\\\",
        "\\hline",
    ]
    for _, r in df_phase.iterrows():
        lines.append(
            f"{r['phase']} & -- & {int(r['support'])} & "
            f"{mean_sd_string(r['mean'], r['sd'])} \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    _write(aggregate_dir, "table_autoencoder_phase_recall.tex", "\n".join(lines))

    # --- Operational table ---
    df_inf = pd.read_csv(os.path.join(aggregate_dir,
                                      "multiseed_inference_timing_raw.csv"))
    df_train = pd.read_csv(os.path.join(aggregate_dir,
                                        "multiseed_training_time_raw.csv"))
    tt = df_train[df_train["metric"] == "training_time_seconds"].iloc[0]
    ee = df_train[df_train["metric"] == "epochs_executed"].iloc[0]
    size_row = df_train[df_train["metric"] == "model_size_bytes"]
    model_kb = (float(size_row.iloc[0]["mean"]) / 1024.0
                if not size_row.empty else float("nan"))
    device_used = per_seed[0]["metadata"]["device_used"]
    lines = [
        f"% Auto-generated operational table ({n_seeds} seeds, device={device_used}).",
        "% Inference scope: pre-extracted and preprocessed features;",
        "% reconstruction, MSE and threshold decision included.",
        "% Latency columns are statistics computed ACROSS per-seed values:",
        "% 'Mean of per-seed median/P95/P99 latency (ms)'.",
        "% Model size is a single exact value (architecture is fixed).",
        "\\begin{tabular}{lcccccccc}",
        "\\hline",
        "Device & Train time (s) & Epochs & Model size (KB) & Batch & "
        "Mean of seed-median lat.\\ (ms) & Mean of seed-P95 lat.\\ (ms) & "
        "Mean of seed-P99 lat.\\ (ms) & Throughput (flows/s) \\\\",
        "\\hline",
    ]
    e2e = df_inf[df_inf["benchmark_type"] == "end_to_end"]
    for _, r in e2e.iterrows():
        lines.append(
            f"{device_used} & {mean_sd_string(tt['mean'], tt['sd'], 2)} & "
            f"{mean_sd_string(ee['mean'], ee['sd'], 1)} & {model_kb:.1f} & "
            f"{int(r['batch_size'])} & "
            f"{r['latency_ms_mean_of_seed_medians']:.4f} & "
            f"{r['latency_ms_mean_of_seed_p95']:.4f} & "
            f"{r['latency_ms_mean_of_seed_p99']:.4f} & "
            f"{mean_sd_string(r['throughput_mean_of_seed_medians'], r['throughput_sd_of_seed_medians'], 1)} \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    _write(aggregate_dir, "table_autoencoder_operational.tex", "\n".join(lines))


def _write(directory, name, text):
    with open(os.path.join(directory, name), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


def print_tree(run_dir):
    log.info("Generated artefact tree under: %s", run_dir)
    for root, dirs, files in os.walk(run_dir):
        dirs.sort()
        depth = root[len(run_dir):].count(os.sep)
        log.info("%s%s/", "  " * depth, os.path.basename(root) or run_dir)
        for f in sorted(files):
            log.info("%s%s", "  " * (depth + 1), f)


# ===========================================================================
# CLI / main
# ===========================================================================
def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Multi-seed reproducible benign-only autoencoder anomaly "
                    "detector over the original 19-class CICIoMT2024 test "
                    "partition.")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                   help="Training seeds (default: 40 41 42 43 44).")
    p.add_argument("--split-seed", type=int, default=SPLIT_SEED_DEFAULT,
                   help="Fixed benign train/val split seed (default: 42).")
    p.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    p.add_argument("--run-tag", type=str, default="")
    p.add_argument("--output-dir", type=str,
                   default=os.path.join(_PROJECT_ROOT, "results",
                                        "autoencoder_multiseed"))
    p.add_argument("--reference-seed", type=int, default=REFERENCE_SEED_DEFAULT)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--overwrite", action="store_true",
                   help="Allow replacing an already-completed seed directory. "
                        "By default previous executions are never overwritten.")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Skip training; only (re)aggregate existing per-seed "
                        "artefacts. Never overwrites previous executions.")
    p.add_argument("--benchmark-batch-sizes", nargs="+", type=int, default=None)
    p.add_argument("--warmup-iters", type=int, default=DEFAULT_WARMUP_ITERS)
    p.add_argument("--tf-log-level", choices=["0", "1", "2", "3"], default="3",
                   help="TF_CPP_MIN_LOG_LEVEL (default 3 = quiet benign C++ "
                        "tf.data notices; use 0 for debugging).")
    p.add_argument("--intra-op-threads", type=int, default=None)
    p.add_argument("--inter-op-threads", type=int, default=None)
    # Internal worker flag (spawned by orchestrator).
    p.add_argument("--worker-seed", type=int, default=None,
                   help=argparse.SUPPRESS)
    return p


def validate_args(args):
    if args.worker_seed is None:  # orchestrator-level checks
        if len(set(args.seeds)) != len(args.seeds):
            raise ValueError("Training seeds must be unique.")
        if args.reference_seed not in args.seeds:
            raise ValueError("--reference-seed must be included in --seeds.")
    if args.warmup_iters < 1:
        raise ValueError("--warmup-iters must be at least 1.")
    if args.benchmark_batch_sizes and any(bs < 1 for bs in args.benchmark_batch_sizes):
        raise ValueError("Benchmark batch sizes must be positive.")


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    validate_args(args)

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)])

    if args.worker_seed is not None:
        # WORKER MODE: env vars must be set before TensorFlow is imported.
        set_env_before_tf(args.device, args.worker_seed, args.tf_log_level)
        run_worker(args)
    else:
        # ORCHESTRATOR MODE: never imports TensorFlow.
        run_orchestrator(args)


if __name__ == "__main__":
    main()
