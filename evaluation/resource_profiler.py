"""
Resource profiling for IoMT IDS experiments.

Tracks:
  - GPU memory usage (peak VRAM via nvidia-smi)
  - CPU memory usage (peak RAM via psutil)
  - GPU/CPU power draw and energy consumption
  - Model size on disk and in RAM
  - Detailed inference latency with warmup
  - GCP deployment cost estimation

References:
  - nvidia-smi: NVIDIA System Management Interface
  - psutil: Cross-platform process utilities
"""
import logging
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers: GPU queries via nvidia-smi
# ---------------------------------------------------------------------------

def _query_nvidia_smi(*fields: str) -> Optional[str]:
    """Query nvidia-smi for specific fields. Returns CSV string or None."""
    try:
        result = subprocess.run(
            ['nvidia-smi', f'--query-gpu={",".join(fields)}',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def get_gpu_memory_mb() -> float:
    """Get current GPU memory usage in MB."""
    out = _query_nvidia_smi('memory.used')
    if out:
        try:
            return float(out.split('\n')[0].strip())
        except (ValueError, IndexError):
            pass
    return 0.0


def get_gpu_power_w() -> float:
    """Get current GPU power draw in Watts."""
    out = _query_nvidia_smi('power.draw')
    if out:
        try:
            return float(out.split('\n')[0].strip())
        except (ValueError, IndexError):
            pass
    return 0.0


def get_gpu_total_memory_mb() -> float:
    """Get total GPU memory in MB."""
    out = _query_nvidia_smi('memory.total')
    if out:
        try:
            return float(out.split('\n')[0].strip())
        except (ValueError, IndexError):
            pass
    return 0.0


# ---------------------------------------------------------------------------
# CPU memory via psutil (optional dependency)
# ---------------------------------------------------------------------------

def get_cpu_memory_mb() -> float:
    """Get current process RSS memory in MB."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def get_cpu_percent() -> float:
    """Get current process CPU usage percent."""
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        return proc.cpu_percent(interval=None)
    except ImportError:
        return 0.0


# ---------------------------------------------------------------------------
# Power & Energy Sampler (background thread, 1 Hz)
# ---------------------------------------------------------------------------

@dataclass
class EnergySample:
    """Single energy measurement."""
    timestamp: float
    gpu_power_w: float
    gpu_memory_mb: float
    cpu_memory_mb: float


@dataclass
class EnergyProfile:
    """Aggregated energy profile for a training/inference run."""
    duration_s: float = 0.0
    # GPU
    gpu_power_avg_w: float = 0.0
    gpu_power_max_w: float = 0.0
    gpu_energy_wh: float = 0.0
    gpu_memory_peak_mb: float = 0.0
    # CPU
    cpu_memory_peak_mb: float = 0.0
    # Samples
    n_samples: int = 0

    def to_dict(self) -> dict:
        return {
            'duration_s': round(self.duration_s, 2),
            'gpu_power_avg_w': round(self.gpu_power_avg_w, 2),
            'gpu_power_max_w': round(self.gpu_power_max_w, 2),
            'gpu_energy_wh': round(self.gpu_energy_wh, 4),
            'gpu_memory_peak_mb': round(self.gpu_memory_peak_mb, 1),
            'cpu_memory_peak_mb': round(self.cpu_memory_peak_mb, 1),
            'n_energy_samples': self.n_samples,
        }


class ResourceMonitor:
    """
    Background resource monitor that samples GPU power, GPU memory,
    and CPU memory at ~1 Hz.

    Usage:
        monitor = ResourceMonitor()
        monitor.start()
        # ... training / inference ...
        profile = monitor.stop()
    """

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._samples: list[EnergySample] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0

    def start(self):
        """Start background sampling."""
        self._samples = []
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> EnergyProfile:
        """Stop sampling and return aggregated profile."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        duration = time.time() - self._start_time
        return self._aggregate(duration)

    def _sample_loop(self):
        """Sampling loop running in background thread."""
        # Warm up psutil cpu_percent
        try:
            import psutil
            psutil.Process(os.getpid()).cpu_percent(interval=None)
        except ImportError:
            pass

        while self._running:
            sample = EnergySample(
                timestamp=time.time(),
                gpu_power_w=get_gpu_power_w(),
                gpu_memory_mb=get_gpu_memory_mb(),
                cpu_memory_mb=get_cpu_memory_mb(),
            )
            self._samples.append(sample)
            time.sleep(self.interval)

    def _aggregate(self, duration: float) -> EnergyProfile:
        """Aggregate samples into a profile."""
        if not self._samples:
            return EnergyProfile(duration_s=duration)

        gpu_powers = [s.gpu_power_w for s in self._samples if s.gpu_power_w > 0]
        gpu_mems = [s.gpu_memory_mb for s in self._samples]
        cpu_mems = [s.cpu_memory_mb for s in self._samples if s.cpu_memory_mb > 0]

        profile = EnergyProfile(
            duration_s=duration,
            n_samples=len(self._samples),
        )

        if gpu_powers:
            profile.gpu_power_avg_w = np.mean(gpu_powers)
            profile.gpu_power_max_w = np.max(gpu_powers)
            # Energy (Wh) = avg_power(W) * duration(s) / 3600
            profile.gpu_energy_wh = profile.gpu_power_avg_w * duration / 3600

        if gpu_mems:
            profile.gpu_memory_peak_mb = np.max(gpu_mems)

        if cpu_mems:
            profile.cpu_memory_peak_mb = np.max(cpu_mems)

        return profile


# ---------------------------------------------------------------------------
# Inference Latency Profiler (with warmup)
# ---------------------------------------------------------------------------

def profile_inference_latency(model, X_sample, n_warmup: int = 100,
                               n_iterations: int = 1000,
                               batch_size: int = 100,
                               is_keras: bool = False) -> dict:
    """
    Profile inference latency with warmup iterations.

    Args:
        model: Trained model (sklearn or keras)
        X_sample: Test data (2D array for sklearn, 2D or 3D for keras)
        n_warmup: Number of warmup iterations (discarded)
        n_iterations: Number of timed iterations
        batch_size: Batch size for throughput measurement
        is_keras: Whether model is a Keras model

    Returns:
        dict with latency_ms, throughput_samples_s, latency_std_ms, etc.
    """
    # Prepare single sample and batch
    single = X_sample[:1]
    batch = X_sample[:batch_size]

    predict_fn = (lambda x: model.predict(x, verbose=0)) if is_keras else model.predict

    # --- Warmup ---
    for _ in range(n_warmup):
        predict_fn(single)

    # --- Single-sample latency ---
    times = []
    for _ in range(n_iterations):
        t0 = time.perf_counter()
        predict_fn(single)
        times.append(time.perf_counter() - t0)

    times = np.array(times) * 1000  # to ms
    latency_ms = np.mean(times)
    latency_std_ms = np.std(times)
    latency_p50_ms = np.percentile(times, 50)
    latency_p95_ms = np.percentile(times, 95)
    latency_p99_ms = np.percentile(times, 99)

    # --- Batch throughput ---
    batch_times = []
    for _ in range(min(100, n_iterations)):
        t0 = time.perf_counter()
        predict_fn(batch)
        batch_times.append(time.perf_counter() - t0)

    avg_batch_time = np.mean(batch_times)
    throughput = len(batch) / avg_batch_time if avg_batch_time > 0 else 0

    return {
        'latency_mean_ms': round(latency_ms, 4),
        'latency_std_ms': round(latency_std_ms, 4),
        'latency_p50_ms': round(latency_p50_ms, 4),
        'latency_p95_ms': round(latency_p95_ms, 4),
        'latency_p99_ms': round(latency_p99_ms, 4),
        'throughput_samples_s': round(throughput, 1),
        'batch_size': len(batch),
        'n_warmup': n_warmup,
        'n_iterations': n_iterations,
    }


# ---------------------------------------------------------------------------
# Model Size (disk + RAM)
# ---------------------------------------------------------------------------

def get_model_size(model, model_path: Optional[str] = None) -> dict:
    """
    Get model size on disk and estimated RAM footprint.

    Args:
        model: Trained model object
        model_path: Path to saved model file (for disk size)

    Returns:
        dict with disk_size_mb, ram_estimate_mb
    """
    disk_mb = 0.0
    if model_path:
        p = Path(model_path)
        if p.is_file():
            disk_mb = p.stat().st_size / (1024 * 1024)
        elif p.is_dir():
            disk_mb = sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) / (1024 * 1024)

    # RAM estimate via sys.getsizeof (rough)
    import sys
    ram_mb = 0.0
    try:
        # For sklearn models, use joblib to estimate
        import io
        import joblib
        buf = io.BytesIO()
        joblib.dump(model, buf)
        ram_mb = buf.tell() / (1024 * 1024)
    except Exception:
        ram_mb = sys.getsizeof(model) / (1024 * 1024)

    return {
        'model_disk_size_mb': round(disk_mb, 2),
        'model_ram_size_mb': round(ram_mb, 2),
    }


# ---------------------------------------------------------------------------
# GCP Deployment Cost Estimation
# ---------------------------------------------------------------------------

# GCP pricing reference (as of 2025)
GCP_PRICING = {
    'gpu_t4': {
        'name': 'n1-standard-8 + T4 GPU',
        'price_per_hour': 0.35 + 0.19,  # T4 GPU + n1-standard-8
        'description': 'GPU instance for deep learning models',
    },
    'cpu_n1_standard_8': {
        'name': 'n1-standard-8 (CPU only)',
        'price_per_hour': 0.19,
        'description': 'CPU instance for classical ML models',
    },
}


def estimate_gcp_cost(train_time_s: float, pred_time_s: float,
                       is_gpu_model: bool = False,
                       retrain_frequency_per_year: int = 52) -> dict:
    """
    Estimate GCP deployment cost.

    Args:
        train_time_s: Training time in seconds
        pred_time_s: Prediction time in seconds
        is_gpu_model: Whether model requires GPU (CNN, Autoencoder)
        retrain_frequency_per_year: How often to retrain (default: weekly = 52)

    Returns:
        dict with cost estimates
    """
    instance = 'gpu_t4' if is_gpu_model else 'cpu_n1_standard_8'
    pricing = GCP_PRICING[instance]
    price_hr = pricing['price_per_hour']

    train_hours = train_time_s / 3600
    pred_hours = pred_time_s / 3600

    cost_single_train = train_hours * price_hr
    cost_single_pred = pred_hours * price_hr
    cost_single_run = cost_single_train + cost_single_pred

    cost_annual_retrain = cost_single_train * retrain_frequency_per_year
    cost_annual_total = cost_annual_retrain + cost_single_pred * 365  # daily inference

    return {
        'gcp_instance': pricing['name'],
        'gcp_price_per_hour': price_hr,
        'cost_single_train_usd': round(cost_single_train, 4),
        'cost_single_pred_usd': round(cost_single_pred, 6),
        'cost_single_run_usd': round(cost_single_run, 4),
        'cost_annual_retrain_usd': round(cost_annual_retrain, 2),
        'cost_annual_total_usd': round(cost_annual_total, 2),
        'retrain_frequency': f'{retrain_frequency_per_year}x/year',
    }
