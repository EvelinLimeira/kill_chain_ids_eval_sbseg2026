# Runtime Reference

Measured wall-clock times on the reference environment (AMD Ryzen 9 7940HS,
48 GB RAM, NVIDIA RTX 4060 8 GB, Python 3.12). Times are indicative and vary
with hardware, especially for the supervised training pipeline, which is
memory-bound on the ~7.16M-sample training partition.

## Per-target reference times

| Target / command | What it does | Dataset | Approx. time | Peak RAM |
|------------------|--------------|---------|--------------|----------|
| `make smoke` | Synthetic-data install check (pytest) | No | < 3 min | < 2 GB |
| `make verify` | Recompute per-class metrics (Tables 5, 6) from saved confusion matrices | No | < 30 s | < 1 GB |
| `make reproduce-supervised` | Train RF/LR/LightGBM/CNN; Tables 4-7 and 12 | Yes | ~20–30 min | up to ~50 GB |
| `make reproduce-autoencoder` | Five-seed benign-only autoencoder; Tables 8 and 9 | Yes | ~5–10 min (CPU) | ~6 GB |
| `make reproduce-operational` | Inference latency/throughput benchmark; Table 10 | Yes (loads models) | ~30–45 min | ~8 GB |
| `make reproduce-xai` | SHAP + LIME figures; Figures 2, 3 | Yes | ~45–60 min | ~10 GB |
| `make audit` | Cross-partition duplicate integrity check | Yes | ~3–5 min | ~8 GB |

## Component training times (from the reference run)

Measured during model construction (not part of the inference benchmark):

| Model | Train time (s) | Serialized size (MB) |
|-------|----------------|----------------------|
| Random Forest | 192.5 | 783.56 |
| Logistic Regression | 1,154.0 | 0.01 |
| LightGBM (Default) | 270.4 | 3.78 |
| LightGBM (Tuned) | 8,087.6 | 134.21 |
| 1D-CNN | 6,293.4 | 1.62 |
| Light-CNN | 265.8 | 0.14 |
| Autoencoder | 19.77 ± 5.14 (5 seeds) | 0.22 |

## Notes

- The supervised pipeline peaks near 50 GB RAM because Random Forest and
  Logistic Regression train on the combined train+validation partition
  (~7.16M samples). Reviewers with less RAM can run a single scenario
  (`python main.py --data_dir ./data --scenarios 19 --skip_xai`).
- The five-seed autoencoder does not require a GPU. Fixed seeds improve
  repeatability, but TensorFlow results are not expected to be bit-identical
  across Windows and Linux/Docker CPU kernels; see `PROVENANCE.md`.
- Operational timings vary by hardware (±30%); the relative ordering between
  models is the stable, reproducible signal (see Table 10 and PROVENANCE.md).
