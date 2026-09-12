# Beyond Aggregate Accuracy: Kill Chain-Aware Evaluation of IoMT Intrusion Detection Models

## Paper Abstract

Machine learning-based Intrusion Detection Systems (IDSs) have shown promising
results for Internet of Medical Things (IoMT) environments; however, aggregate
performance metrics can conceal failures in detecting early-stage behaviors.
This paper presents a kill chain-aware analysis of Random Forest, LightGBM, a
1D-CNN, and an autoencoder using the CICIoMT2024 dataset, analyzing detection
capability across 18 attack categories and multiple stages of the attack
lifecycle. Results show that Random Forest substantially outperforms the CNN in
reconnaissance and spoofing scenarios, while default LightGBM configurations
exhibit severe degradation when transitioning from binary to multi-class
classification. In addition, the benign-only autoencoder achieves high attack
recall without requiring attack labels, although its false-positive burden
limits direct alert generation. Supported by bootstrap confidence intervals
and McNemar's test, the findings demonstrate that IoMT IDSs should move beyond
aggregate accuracy toward phase-aware, class-disaggregated, and statistically
grounded assessment protocols.

This repository contains the source code, configurations, reference results,
and documentation associated with the SBSeg 2026 accepted paper available in https://sol.sbc.org.br/index.php/sbseg. The camera-ready
manuscript is available at
[`manuscript/27104_Camera-ready_SeyJ88S.pdf`](manuscript/27104_Camera-ready_SeyJ88S.pdf).

## README Structure

This README is organized into the following sections:

1. **Features:** principal capabilities provided by the artifact.
2. **Badges Considered:** quality badges requested from the CTA.
3. **Basic Information:** reference environment and hardware requirements.
4. **Dependencies:** required software and third-party resources.
5. **Security Concerns:** execution risks and mitigations.
6. **Installation:** native Python and Docker setup.
7. **Configuration:** dataset placement, scenarios, and execution parameters.
8. **Minimum Test:** fast checks that do not require CICIoMT2024.
9. **Usage:** principal command-line and Makefile entry points.
10. **Experiments:** workflows for reproducing the paper evidence.
11. **Reference Results:** pre-computed evidence distributed with the artifact.
12. **Code Structure:** organization of the implementation.
13. **Known Variability:** interpretation of cross-platform results.
14. **License:** ownership and permitted use.

## Features

- **Multiple classification granularities:** binary, 6-class, and 19-class
  evaluation on CICIoMT2024.
- **Supervised baselines:** Random Forest, Logistic Regression, default and
  tuned LightGBM, and convolutional neural networks.
- **Benign-only anomaly screening:** five-seed autoencoder evaluation with
  P85, P90, P95, and P99 threshold sensitivity.
- **Kill chain-aware analysis:** class-level results grouped into
  reconnaissance, access/exploitation, volumetric impact, and MQTT impact.
- **Statistical validation:** bootstrap confidence intervals, Cohen's kappa,
  and pairwise McNemar tests.
- **Explainability:** global SHAP and local LIME attribution analyses.
- **Operational profiling:** inference latency, throughput, serialized size,
  and training-time evidence.
- **Integrity auditing:** cross-partition feature-signature and missing-value
  analysis.
- **Reference outputs:** bundled tables, reports, figures, confusion matrices,
  per-seed metrics, and metadata.
- **Reproducible environment:** pinned Python dependencies, Dockerfile,
  Docker Compose, Makefile targets, and synthetic smoke tests.

## Basic Information

### Reference Environment

| Component | Configuration |
|-----------|---------------|
| CPU | AMD Ryzen 9 7940HS |
| RAM | 48 GB |
| GPU | NVIDIA RTX 4060 Laptop, 8 GB VRAM |
| Operating system | Ubuntu/WSL2 on Windows 11; historical AE run metadata records Windows CPU execution |
| Python | 3.12 for the main pipeline |
| Dataset | CICIoMT2024 processed Wi-Fi/MQTT feature-level partitions |

### Hardware Requirements

| Evaluation mode | CPU | RAM | GPU | Disk |
|-----------------|-----|-----|-----|------|
| Minimum test | 2 cores | 2 GB | Not required | 2 GB |
| Reference-result verification | 2 cores | 2 GB | Not required | 2 GB |
| Autoencoder workflow | 4 cores | Approximately 6 GB | Optional | Approximately 10 GB |
| 19-class supervised workflow | 8 cores recommended | Up to approximately 50 GB | Optional | Approximately 20 GB plus dataset/models |
| Full experiment | 8 cores recommended | Up to approximately 50 GB | Recommended for neural timing | Approximately 30 GB plus dataset/models |

A user with limited memory may execute only the 19-class scenario instead
of all three classification granularities. Measured runtimes are documented in
[`docs/runtime_reference.md`](docs/runtime_reference.md).

## Dependencies

All Python packages are pinned in [`requirements.txt`](requirements.txt).

| Package | Version |
|---------|---------|
| Python | 3.12 |
| numpy | 2.4.2 |
| pandas | 3.0.1 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| lightgbm | 4.6.0 |
| tensorflow | 2.20.0 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
| shap | 0.50.0 |
| lime | 0.2.0.1 |
| pytest | 9.0.2 |

### Third-Party Dataset

Dataset-dependent workflows require the processed Wi-Fi/MQTT portion of
**CICIoMT2024**. The source dataset is not redistributed by this repository and
remains subject to its provider's terms. Download and placement instructions
are available in [`data/README.md`](data/README.md).

The minimum test and reference-result verification do not require the dataset.

## Security Concerns

When executing this artifact, consider the following:

- **Isolated environment:** Docker or a dedicated Python virtual environment
  is recommended to prevent dependency conflicts.
- **Serialized models:** Python pickle files may execute arbitrary code when
  loaded. Load only models generated locally or obtained from this trusted
  artifact package.
- **Resource consumption:** full training can use substantial CPU, memory,
  disk, and time. Check the requirements before starting it.
- **Network access:** experimental execution makes no network request after
  dependencies and the dataset are available. The optional dataset downloader
  accesses Kaggle and may require authentication.
- **Credentials:** do not commit Kaggle tokens, API keys, SSH keys, or other
  credentials.
- **Dataset license:** CICIoMT2024 source files must not be redistributed as
  part of this repository without permission from the provider.

No cloud infrastructure, SSH key, paid service, or external API is required
for the experiments themselves.

## Installation

### Option A - Docker (Recommended)

Requirements: Docker Engine or Docker Desktop with Docker Compose support.

```bash
git clone https://github.com/EvelinLimeira/iomt_ids_ai_models_eval.git
cd iomt_ids_ai_models_eval
docker compose build
```

Run the minimum test:

```bash
docker compose run --rm artifact make smoke
```

Verify the bundled confusion-matrix evidence:

```bash
docker compose run --rm artifact make verify
```

### Option B - Native Python

```bash
git clone https://github.com/EvelinLimeira/iomt_ids_ai_models_eval.git
cd iomt_ids_ai_models_eval

python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\Activate.ps1    # Windows PowerShell

python -m pip install --upgrade pip
pip install -r requirements.txt
make smoke
```

If GNU Make is unavailable, execute the equivalent Python commands shown in
the Minimum Test and Experiments sections.

## Docker

The supplied image uses `python:3.12-slim` and provides a CPU-only environment.
The dataset is not included in the image.

Build and check the image:

```bash
docker compose build
docker compose run --rm artifact make smoke
docker compose run --rm artifact make verify
```

For dataset-dependent experiments, place CICIoMT2024 under `data/train/` and
`data/test/`. Docker Compose mounts `data/` read-only and persists generated
outputs under `results/`.

```bash
docker compose run --rm artifact make reproduce-autoencoder
docker compose run --rm artifact make reproduce-supervised
```

GPU timings from the paper cannot be regenerated inside the supplied CPU-only
image. Run the benchmark directly in a CUDA-enabled native environment when
GPU measurements are required.

## Configuration

### Dataset Setup

Automated download from the public Kaggle mirror:

```bash
pip install kagglehub
python scripts/download_dataset.py
# equivalent:
make download-data
```

Manual placement:

```text
data/
|-- train/
|   `-- *_train.pcap.csv
`-- test/
    `-- *_test.pcap.csv
```

The loader retains 45 numeric features and maps labels into 2-, 6-, or
19-class scenarios. Train and test partitions remain separate. Training-only
statistics are used for imputation and standardization.

### Principal Parameters

- Supervised seed: `42`.
- Supervised validation split: stratified 80/20 where early stopping applies.
- Autoencoder training seeds: `40 41 42 43 44`.
- Fixed benign split seed: `42`.
- Autoencoder operating point: P85 of benign-validation reconstruction error;
  P90/P95/P99 are sensitivity points.
- Output root: `results/`.

Model hyperparameters and scenario mappings are defined in `config.py` and the
corresponding modules under `models/`.

## Minimum Test

The smoke test uses synthetic data, requires neither CICIoMT2024 nor a GPU, and
normally completes in less than three minutes.

```bash
make smoke
# equivalent:
python -m pytest tests/test_smoke.py -v --tb=short
```

It checks:

- the expected 45-feature representation;
- leakage-aware preprocessing;
- Random Forest and default LightGBM construction;
- metric computation;
- benign-only autoencoder splitting.

A successful run ends with all smoke tests passing.

The second fast check recomputes the 19-class per-class metrics from bundled
confusion matrices:

```bash
make verify
# equivalent:
python scripts/analysis/compute_per_class_metrics.py
```

Expected output: `results/tables/per_class_19-class.csv`.

## Usage

The main Makefile targets are:

```bash
make help
make smoke
make verify
make download-data
make reproduce-supervised
make reproduce-autoencoder
make reproduce-operational
make reproduce-xai
make audit
make reproduce
make full-experiment
```

The principal supervised command is:

```bash
python main.py --data_dir ./data --scenarios 2 6 19 --skip_xai
```

Reduced-memory 19-class execution:

```bash
python main.py --data_dir ./data --scenarios 19 --skip_xai
```

## Experiments

This section describes the executable workflows that produce the evidence for
the claims declared in the CTA submission form. The exact paper-to-file mapping
is documented in [`PROVENANCE.md`](PROVENANCE.md).

### Experiment 1 - Supervised Detection and Statistical Validation

```bash
make reproduce-supervised
```

- **Dataset:** required.
- **Resources:** up to approximately 50 GB RAM.
- **Reference time:** approximately 20-30 minutes, with substantial variation
  across hardware and model libraries.
- **Paper evidence:** Tables 4-7 and 12.
- **Outputs:** `comparison_*.csv`, `per_class_19-class.csv`,
  `lgbm_variants_*.csv`, `bootstrap_ci_19-class.csv`, and
  `mcnemar_19-class.csv` under `results/tables/`.
- **Verification:** Random Forest remains above the evaluated 1D-CNN in
  19-class macro-F1 and reconnaissance recall; tuned LightGBM remains
  substantially above default LightGBM in 19-class macro-F1.

### Experiment 2 - Five-Seed Benign-Only Autoencoder

```bash
make reproduce-autoencoder
# equivalent:
python scripts/evaluation/autoencoder_threshold_sensitivity.py \
  --seeds 40 41 42 43 44 --split-seed 42 --device cpu --run-tag repro
```

- **Dataset:** required.
- **Resources:** approximately 6 GB RAM; GPU optional.
- **Reference time:** approximately 5-10 minutes.
- **Paper evidence:** Tables 8-9 and Sections 4.3 and 5.
- **Outputs:** per-seed and aggregate files under
  `results/autoencoder_multiseed/repro_<os>_cpu/`.
- **Verification:** all five seeds complete; P85 retains high attack recall and
  explicitly reports its benign false-positive rate; increasing the threshold
  from P85 to P99 reduces false positives.

### Experiment 3 - Operational Benchmark

Run Experiment 1 first because this benchmark loads trained models.

```bash
make reproduce-operational
# equivalent:
python scripts/benchmarks/benchmark_latency.py
```

- **Dataset and trained models:** required.
- **Reference time:** approximately 30-45 minutes.
- **Paper evidence:** Table 10.
- **Output:** `results/tables/latency_comparison.csv`.
- **Verification:** the benchmark completes and records latency and throughput.
  Absolute values are hardware-dependent.

### Experiment 4 - SHAP and LIME

Run Experiment 1 first to produce the trained 19-class models.

```bash
make reproduce-xai
# equivalent:
python scripts/xai/run_xai.py
```

- **Dataset and trained models:** required.
- **Reference time:** approximately 45-60 minutes.
- **Paper evidence:** Figures 2-3.
- **Outputs:** global SHAP plots under `results/xai/` and local LIME plots
  under `results/figures/`.
- **Verification:** all reference figure types are generated without errors.

### Experiment 5 - Dataset Integrity Audit

```bash
make audit
```

- **Dataset:** required.
- **Output:** `audit_reports/crosspartition_audit.json`.
- **Verification:** the report records missing values and cross-partition
  feature-signature overlap.

### Full Reproduction

```bash
make reproduce       # Experiments 1, 2, and 3
make full-experiment # Experiments 1-4
```

## Reference Results

The repository includes the outputs generated during preparation of the
camera-ready paper:

- `results/tables/`: aggregate, per-class, LightGBM, statistical, and
  operational evidence;
- `results/reports/`: class-level model reports;
- `results/autoencoder_multiseed/camera_ready_v1_windows_cpu/`: five-seed
  autoencoder reference run;
- `results/xai/`: reference SHAP figures;
- `results/figures/`: reference LIME figures;
- `audit_reports/`: data-integrity evidence.

The source CICIoMT2024 CSV files and large supervised models are intentionally
excluded. The supervised workflows regenerate the required models.

## Code Structure

```text
.
|-- config.py                 # Hyperparameters, features, labels, paths, seeds
|-- data_loader.py            # Dataset loading and leakage-aware preprocessing
|-- main.py                   # Supervised training/evaluation entry point
|-- models/                   # Classical, CNN, and autoencoder implementations
|-- evaluation/               # Metrics, statistics, profiling, visualization
|-- explainability/           # SHAP and LIME implementations
|-- scripts/
|   |-- analysis/             # Per-class, stability, and control analyses
|   |-- audit/                # Dataset-integrity audit
|   |-- benchmarks/           # Latency and throughput benchmark
|   |-- evaluation/           # Autoencoder multi-seed experiments
|   `-- xai/                  # Explainability orchestration
|-- tests/                    # Synthetic smoke and focused tests
|-- data/                     # Dataset instructions and local placement
|-- results/                  # Reference and regenerated results
|-- manuscript/               # Camera-ready paper
|-- docs/                     # Runtime documentation
|-- Dockerfile
|-- docker-compose.yml
|-- Makefile
|-- requirements.txt
|-- PROVENANCE.md
`-- LICENSE
```

## Extensibility

- Add a supervised model by implementing the common fit/predict interface under
  `models/` and registering its configuration in `config.py`.
- Add a classification granularity by extending the label mapping in
  `config.py` and the corresponding scenario handling in the loader.
- Add an evaluation metric in `evaluation/metrics.py` and include it in the
  exported comparison tables.
- Add a new attribution method under `explainability/` and register its output
  in `scripts/xai/run_xai.py`.
- Keep new paper evidence traceable by updating `PROVENANCE.md`.

## Known Variability

The bundled reference results are the source for exact comparison with the
camera-ready tables. Fresh executions are expected to preserve the scientific
conclusions, not to reproduce every floating-point value bit for bit.

The reference five-seed autoencoder run was produced on Windows CPU, while the
Docker image executes TensorFlow on Linux CPU. TensorFlow/oneDNN kernels,
thread scheduling, and floating-point operation ordering can produce different
optimization trajectories even with identical application-level seeds. Docker
metrics may therefore be slightly lower, and an individual seed may converge
to a different local optimum.

Reviewers should verify dataset/split fingerprints, completion of all seeds,
generation of the documented outputs, preservation of the principal result
direction, and explicit reporting of false positives and variability.
Operational latency and throughput are also hardware-dependent.

## License

This is a proprietary CESAR/CISSA artifact made available under the limited
[`Academic Artifact Evaluation License`](LICENSE). Authorized CTA reviewers and
SBSeg 2026 organizers may access, install, execute, and inspect it solely for
the confidential artifact-evaluation process. Redistribution, publication,
commercial use, and use outside that evaluation require prior written
authorization from CESAR/CISSA.

The license becomes effective only after approval by an authorized
CESAR/CISSA representative.

This work was funded by the projects *Xeque-Mate Agente & Agregador*, supported
by CISSA/CESAR with resources from the PPI IoT/Manufatura 4.0 of the MCTI,
grant numbers CIS-AFCCT-2025-5-1-2 and CIS-AFCCT-2025-5-1-1, signed with
EMBRAPII.

