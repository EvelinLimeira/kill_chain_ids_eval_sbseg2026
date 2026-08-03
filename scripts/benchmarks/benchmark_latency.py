"""
Inference latency benchmark for all IoMT IDS models.

Measures two modes:
  1. Single-sample latency (1000 samples, 100 warmup discarded)
  2. Batch throughput (full test set, 5 repetitions)

All models run in a single process (realistic deployment scenario: model
loaded in memory serving requests). Keras models get aggressive warmup
(10 batch predictions + 50 single predictions) to fully absorb XLA JIT
compilation before timed measurements begin.

Outputs:
  - results/tables/latency_comparison.csv
  - results/tables/latency_table.tex
  - Console summary

Usage:
  python scripts/benchmarks/benchmark_latency.py

IMPORTANT: Run with a "cold" GPU session (no prior training) to avoid
CUDA memory fragmentation and driver instability.
"""
import os
import sys
import gc
import time
import warnings
import logging
import numpy as np
import pandas as pd
import joblib

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from config import DATA_DIR, MODELS_DIR, TABLES_DIR, RESULTS_DIR, SEED
from data_loader import load_data
from datetime import datetime

warnings.filterwarnings("ignore")

# Log to console + timestamped file in results/
_log_fmt = "%(asctime)s [%(levelname)s] %(message)s"
_log_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_path = os.path.join(RESULTS_DIR, f"benchmark_latency_{_log_ts}.log")

logging.basicConfig(
    level=logging.INFO,
    format=_log_fmt,
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(_log_path, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)
log.info(f"Log file: {_log_path}")

# ── Constants ────────────────────────────────────────────────────────────
SCENARIOS = [2, 6, 19]
SINGLE_N_SAMPLES = 1000
SINGLE_WARMUP = 100
BATCH_REPEATS = 5
KERAS_BATCH_SIZE = 512

# Aggressive warmup for Keras: absorbs XLA JIT compilation fully
KERAS_WARMUP_BATCH_RUNS = 10   # batch predictions to trigger all XLA kernels
KERAS_WARMUP_SINGLE_RUNS = 50  # single predictions to stabilize per-sample path

SKLEARN_MODELS = [
    ("Random Forest",       "rf_{scenario}.pkl",            "sklearn"),
    ("Logistic Regression", "lr_{scenario}.pkl",            "sklearn"),
    ("LightGBM (Original)", "lgbm_original_{scenario}.pkl", "sklearn"),
    ("LightGBM (Tuned)",    "lgbm_tuned_{scenario}.pkl",    "sklearn"),
]

KERAS_MODELS = [
    ("1D-CNN",       "cnn_standard_{scenario}.keras", "keras_cnn"),
    ("Light-CNN",    "cnn_light_{scenario}.keras",    "keras_cnn"),
    ("Autoencoder",  "autoencoder_{scenario}.keras",  "keras_ae"),
]


def _load_model(file_pattern: str, scenario: str):
    """Load a model file, return None if missing."""
    fname = file_pattern.format(scenario=scenario)
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        log.warning(f"  Model not found, skipping: {path}")
        return None
    if path.endswith(".pkl"):
        return joblib.load(path)
    else:
        import tensorflow as tf
        return tf.keras.models.load_model(path, compile=False)


def _predict_single(model, x_single, backend: str):
    if backend == "sklearn":
        model.predict(x_single)
    else:
        model.predict(x_single, verbose=0, batch_size=1)


def _predict_batch(model, X, backend: str):
    if backend == "sklearn":
        model.predict(X)
    else:
        model.predict(X, verbose=0, batch_size=KERAS_BATCH_SIZE)


def _get_input(X_test_flat, X_test_3d, backend: str, indices=None):
    if backend == "keras_cnn":
        return X_test_3d if indices is None else X_test_3d[indices]
    else:
        return X_test_flat if indices is None else X_test_flat[indices]


def _warmup(model, X_test_flat, X_test_3d, backend: str):
    """
    Warmup to absorb all first-run overhead before timed measurements.

    For sklearn: 1 batch prediction (negligible JIT).
    For Keras: aggressive warmup — 10 batch + 50 single predictions to
    fully compile and cache all XLA kernels for this specific architecture.
    """
    X_batch = _get_input(X_test_flat, X_test_3d, backend)
    x_single = _get_input(X_test_flat, X_test_3d, backend, indices=[0])

    if backend == "sklearn":
        _predict_batch(model, X_batch[:100], backend)
    else:
        # Batch warmup: triggers XLA compilation for batch path
        for _ in range(KERAS_WARMUP_BATCH_RUNS):
            _predict_batch(model, X_batch[:KERAS_BATCH_SIZE], backend)
        # Single warmup: triggers XLA compilation for single-sample path
        for _ in range(KERAS_WARMUP_SINGLE_RUNS):
            _predict_single(model, x_single, backend)
        log.info(f"    Warmup done: {KERAS_WARMUP_BATCH_RUNS} batch + "
                 f"{KERAS_WARMUP_SINGLE_RUNS} single predictions")


def benchmark_single_sample(model, X_test_flat, X_test_3d, backend: str) -> dict:
    """
    Mode 1: Single-sample latency.
    Send 1 sample at a time, measure each, discard first SINGLE_WARMUP.
    """
    rng = np.random.RandomState(SEED)
    indices = rng.choice(len(X_test_flat), size=SINGLE_N_SAMPLES, replace=False)

    latencies = []
    for idx in indices:
        x_single = _get_input(X_test_flat, X_test_3d, backend, indices=[idx])
        t0 = time.perf_counter()
        _predict_single(model, x_single, backend)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    # Discard warmup samples
    arr = np.array(latencies[SINGLE_WARMUP:])
    return {
        "SingleSample_Mean_ms":   float(np.mean(arr)),
        "SingleSample_Median_ms": float(np.median(arr)),
        "SingleSample_P95_ms":    float(np.percentile(arr, 95)),
        "SingleSample_P99_ms":    float(np.percentile(arr, 99)),
    }


def benchmark_batch(model, X_test_flat, X_test_3d, backend: str) -> dict:
    """
    Mode 2: Batch throughput.
    Send full test set, repeat BATCH_REPEATS times, report mean.
    """
    X = _get_input(X_test_flat, X_test_3d, backend)
    n = len(X)

    times = []
    for _ in range(BATCH_REPEATS):
        t0 = time.perf_counter()
        _predict_batch(model, X, backend)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    mean_time = float(np.mean(times))
    return {
        "Batch_Throughput_samples_s": n / mean_time,
        "Batch_Amortized_ms":        (mean_time / n) * 1000.0,
        "Batch_Total_s":             mean_time,
    }


def _generate_latex(df: pd.DataFrame, path: str):
    """Generate a publication-ready LaTeX table."""
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\caption{Inference Latency Comparison: Single-Sample and Batch Modes}",
        r"\label{tab:latency}",
        r"\small",
        r"\begin{tabular}{ll rrrr rrr}",
        r"\toprule",
        r"Scenario & Model & \multicolumn{4}{c}{Single-Sample Latency (ms)} "
        r"& \multicolumn{3}{c}{Batch Inference} \\",
        r"\cmidrule(lr){3-6} \cmidrule(lr){7-9}",
        r" & & Mean & Median & P95 & P99 "
        r"& Throughput (s/s) & Amortized (ms) & Total (s) \\",
        r"\midrule",
    ]

    for i, scenario in enumerate(SCENARIOS):
        sub = df[df["Scenario"] == f"{scenario}-class"]
        first = True
        for _, row in sub.iterrows():
            sc_col = f"{scenario}-class" if first else ""
            first = False
            lines.append(
                f"{sc_col} & {row['Model']} & "
                f"{row['SingleSample_Mean_ms']:.3f} & "
                f"{row['SingleSample_Median_ms']:.3f} & "
                f"{row['SingleSample_P95_ms']:.3f} & "
                f"{row['SingleSample_P99_ms']:.3f} & "
                f"{row['Batch_Throughput_samples_s']:,.0f} & "
                f"{row['Batch_Amortized_ms']:.4f} & "
                f"{row['Batch_Total_s']:.2f} \\\\"
            )
        if i < len(SCENARIOS) - 1:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log.info(f"LaTeX table saved to {path}")


def main():
    results = []

    # Pre-load all scenario data
    log.info("=" * 60)
    log.info("Loading datasets for all scenarios...")
    datasets = {}
    for sc in SCENARIOS:
        log.info(f"  Loading {sc}-class scenario...")
        datasets[sc] = load_data(DATA_DIR, sc)
        log.info(f"  Test set size: {len(datasets[sc]['X_test_flat']):,}")

    # ── Block 1: sklearn / LightGBM (CPU) ───────────────────────────────
    log.info("=" * 60)
    log.info("BLOCK 1: sklearn / LightGBM models (CPU)")
    log.info("=" * 60)

    for sc in SCENARIOS:
        scenario = f"{sc}-class"
        data = datasets[sc]
        X_flat, X_3d = data["X_test_flat"], data["X_test"]

        for name, pattern, backend in SKLEARN_MODELS:
            log.info(f"  [{scenario}] {name}...")
            model = _load_model(pattern, scenario)
            if model is None:
                continue

            _warmup(model, X_flat, X_3d, backend)
            single = benchmark_single_sample(model, X_flat, X_3d, backend)
            batch = benchmark_batch(model, X_flat, X_3d, backend)

            row = {"Scenario": scenario, "Model": name, **single, **batch}
            results.append(row)
            log.info(
                f"    Single: {single['SingleSample_Mean_ms']:.3f} ms (mean) | "
                f"Batch: {batch['Batch_Throughput_samples_s']:,.0f} samples/s"
            )
            del model

    gc.collect()
    log.info("  gc.collect() between blocks")

    # ── Block 2: Keras models (GPU) ─────────────────────────────────────
    log.info("=" * 60)
    log.info("BLOCK 2: Keras models (GPU) — aggressive warmup per model")
    log.info("=" * 60)

    import tensorflow as tf

    for sc in SCENARIOS:
        scenario = f"{sc}-class"
        data = datasets[sc]
        X_flat, X_3d = data["X_test_flat"], data["X_test"]

        for name, pattern, backend in KERAS_MODELS:
            log.info(f"  [{scenario}] {name}...")
            model = _load_model(pattern, scenario)
            if model is None:
                continue

            _warmup(model, X_flat, X_3d, backend)
            single = benchmark_single_sample(model, X_flat, X_3d, backend)
            batch = benchmark_batch(model, X_flat, X_3d, backend)

            row = {"Scenario": scenario, "Model": name, **single, **batch}
            results.append(row)
            log.info(
                f"    Single: {single['SingleSample_Mean_ms']:.3f} ms (mean) | "
                f"Batch: {batch['Batch_Throughput_samples_s']:,.0f} samples/s"
            )

            del model
            gc.collect()
            tf.keras.backend.clear_session()
            log.info("    clear_session() + gc.collect()")

    # ── Build results DataFrame ─────────────────────────────────────────
    df = pd.DataFrame(results)
    col_order = [
        "Scenario", "Model",
        "SingleSample_Mean_ms", "SingleSample_Median_ms",
        "SingleSample_P95_ms", "SingleSample_P99_ms",
        "Batch_Throughput_samples_s", "Batch_Amortized_ms", "Batch_Total_s",
    ]
    df = df[col_order]

    # ── Save CSV ────────────────────────────────────────────────────────
    csv_path = os.path.join(TABLES_DIR, "latency_comparison.csv")
    df.to_csv(csv_path, index=False, float_format="%.6f")
    log.info(f"CSV saved to {csv_path}")

    # ── Save LaTeX ──────────────────────────────────────────────────────
    tex_path = os.path.join(TABLES_DIR, "latency_table.tex")
    _generate_latex(df, tex_path)

    # ── Console summary ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("LATENCY BENCHMARK RESULTS")
    log.info("=" * 60)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n" + df.to_string(index=False))
    print()


if __name__ == "__main__":
    main()
