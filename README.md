# Beyond Aggregate Accuracy: Kill Chain-Aware Evaluation of IoMT Intrusion Detection Models

This artifact accompanies the SBSeg 2026 paper *Beyond Aggregate Accuracy:
Kill Chain-Aware Evaluation of IoMT Intrusion Detection Models*. It contains
the implementation, configurations, reference results, and reproduction
workflows used to evaluate Random Forest, LightGBM, a 1D-CNN, Logistic
Regression, and a benign-only autoencoder on CICIoMT2024.

**Abstract.** Machine learning-based Intrusion Detection Systems (IDSs) have
shown promising results for Internet of Medical Things (IoMT) environments;
however, aggregate performance metrics can conceal failures in detecting
early-stage behaviors. This paper presents a kill chain-aware analysis of
Random Forest, LightGBM, a 1D-CNN, and an autoencoder using the CICIoMT2024
dataset. Results show that Random Forest substantially outperforms the CNN in
reconnaissance and spoofing scenarios, while default LightGBM configurations
degrade severely from binary to multi-class classification. The autoencoder
provides high attack recall at P85 with a substantial benign false-positive
rate. Bootstrap confidence intervals, McNemar's test, and five-seed analysis
support a phase-aware, class-disaggregated, and statistically grounded IDS
evaluation protocol.

The camera-ready manuscript is available under [`manuscript/`](manuscript/).
The exact mapping from the paper's tables and figures to scripts and reference
outputs is maintained in [`PROVENANCE.md`](PROVENANCE.md).

## Structure of this README

1. Badges considered and artifact scope.
2. Repository organization and execution environment.
3. Dependencies, security considerations, and installation.
4. Dataset setup and minimum test.
5. Reproduction workflows and expected outputs.
6. Reference results, known variability, and license.

## Badges Considered

The authors request consideration for:

- **Functional (SeloF):** the artifact provides an isolated environment, a
  synthetic smoke test, and executable analysis workflows.
- **Sustainable (SeloS):** the code is modularized and its configurations,
  model implementations, evaluation routines, and provenance are separated.
- **Reproducible (SeloR):** the main experimental workflows and their expected
  outputs are documented and automated.
- **Available (SeloD):** consideration depends on CTA acceptance of the stable
  repository and the access restrictions described in the License section.

## Artifact Scope

The artifact supports two levels of evaluation:

1. **Fast verification without the dataset:** checks the installation and
   recomputes selected metrics from the bundled reference outputs.
2. **Experimental reproduction with CICIoMT2024:** retrains the models and
   regenerates detection, anomaly-screening, operational, statistical, and
   explainability results.

The scientific claims themselves are declared in the CTA submission form.
This README focuses on the procedures needed to evaluate them. See
[`PROVENANCE.md`](PROVENANCE.md) for claim-to-evidence traceability.

## Repository Structure

```text
.
|-- README.md                  # Artifact entry point
|-- LICENSE                    # Restricted artifact-evaluation license
|-- PROVENANCE.md              # Paper -> command -> result mapping
|-- manuscript/                # Camera-ready paper and manuscript notice
|-- config.py                  # Hyperparameters, labels, paths, and seeds
|-- data_loader.py             # Leakage-aware CICIoMT2024 preprocessing
|-- main.py                    # Supervised training and evaluation pipeline
|-- models/                    # RF/LR/LightGBM/CNN/autoencoder definitions
|-- evaluation/                # Metrics, statistics, profiling, visualization
|-- explainability/            # SHAP and LIME routines
|-- scripts/                   # Reproduction, analysis, audit, and benchmark entry points
|-- tests/                     # Synthetic smoke and focused tests
|-- data/                      # Dataset instructions; source CSVs are not distributed
|-- results/                   # Bundled reference results and generated outputs
|-- docs/                      # Runtime reference
|-- Dockerfile
|-- docker-compose.yml
|-- Makefile
`-- requirements.txt
```

## Basic Information

Reference experiments were performed on the following workstation:

| Component | Reference configuration |
|-----------|-------------------------|
| CPU | AMD Ryzen 9 7940HS |
| RAM | 48 GB |
| GPU | NVIDIA RTX 4060 Laptop, 8 GB |
| Host | Windows 11 with Ubuntu/WSL2 where indicated |
| Python | 3.12 for the main pipeline; historical AE metadata records its exact runtime |

The supplied Docker image is Linux CPU-only. A GPU is not required for the
minimum test or reference-result verification. Full supervised training can
peak near 50 GB RAM. A reviewer with less memory may execute only the 19-class
scenario.

Detailed measured times are listed in
[`docs/runtime_reference.md`](docs/runtime_reference.md).

## Dependencies

Python dependencies are pinned in [`requirements.txt`](requirements.txt).
Principal packages are:

| Package | Version |
|---------|---------|
| Python | 3.12 |
| numpy | 2.4.2 |
| pandas | 3.0.1 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| lightgbm | 4.6.0 |
| tensorflow | 2.20.0 |
| shap | 0.50.0 |
| lime | 0.2.0.1 |
| pytest | 9.0.2 |

The dataset-dependent workflows require the processed Wi-Fi/MQTT portion of
**CICIoMT2024**. The dataset is a third-party resource and is not licensed or
redistributed by this repository. See [`data/README.md`](data/README.md).

## Security Concerns

- Execute the artifact in Docker or another isolated environment.
- Model files created with Python pickle must be loaded only when generated by
  the reviewer or obtained from this trusted artifact package; pickle may
  execute arbitrary code during deserialization.
- Full training can consume substantial CPU, RAM, disk, and time. Check the
  resource estimates before starting it.
- Experimental execution makes no network requests after the dataset and
  dependencies are available. The optional dataset-download command does use
  the network and may require Kaggle authentication.
- Do not commit Kaggle credentials, API tokens, or other secrets.

No cloud infrastructure, SSH key, or paid service is required for the
experiments. Repository-access restrictions and any private access procedure
must be communicated to reviewers through the CTA submission platform.

## Installation

### Option A - Docker (recommended)

Requirements: Docker Engine or Docker Desktop with Compose support.

```bash
git clone <repository-url>
cd repo_sbseg_beyond_accuracy
docker compose build
```

Run the minimum test:

```bash
docker compose run --rm artifact make smoke
```

Verify bundled reference results:

```bash
docker compose run --rm artifact make verify
```

### Option B - Native Python

```bash
git clone <repository-url>
cd repo_sbseg_beyond_accuracy
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
# .venv\Scripts\activate         # Windows PowerShell
pip install -r requirements.txt
make smoke
```

The placeholder `<repository-url>` must be replaced by the stable artifact URL
before submission.

## Dataset Setup

The minimum test and reference-result verification do not require the dataset.
For full reproduction, obtain CICIoMT2024 separately.

Automated download from the public Kaggle mirror:

```bash
pip install kagglehub
python scripts/download_dataset.py
# or: make download-data
```

Manual placement is also supported:

```text
data/
|-- train/
|   `-- *_train.pcap.csv
`-- test/
    `-- *_test.pcap.csv
```

The loader expects 45 numeric flow features plus the class information. See
[`data/README.md`](data/README.md) for filenames and validation details.

## Minimum Test

The smoke test uses synthetic data, does not require a GPU or CICIoMT2024, and
normally completes in less than three minutes:

```bash
make smoke
# equivalent:
python -m pytest tests/test_smoke.py -v --tb=short
```

Successful execution ends with all smoke tests passing. It checks feature
count, leakage-aware preprocessing, model construction, metric computation,
and benign-only splitting.

Reference-result verification recomputes the 19-class per-class metrics from
the bundled confusion matrices:

```bash
make verify
# equivalent:
python scripts/analysis/compute_per_class_metrics.py
```

Expected output: `results/tables/per_class_19-class.csv`.

## Experiments

The workflows below are the executable procedures used to assess the claims
declared in the CTA form. Exact paper-table mappings and reference values are
in [`PROVENANCE.md`](PROVENANCE.md).

### Workflow 1 - Supervised detection and statistical validation

```bash
make reproduce-supervised
# equivalent:
python main.py --data_dir ./data --scenarios 2 6 19 --skip_xai
```

- **Input:** CICIoMT2024 train/test partitions.
- **Resources:** up to approximately 50 GB RAM.
- **Reference time:** approximately 20-30 minutes on the reference system;
  tuned LightGBM and neural training may take longer on other systems.
- **Outputs:** `results/tables/comparison_*.csv`,
  `lgbm_variants_*.csv`, `per_class_19-class.csv`,
  `bootstrap_ci_19-class.csv`, and `mcnemar_19-class.csv`.
- **Success:** all requested scenarios complete; RF remains above the
  evaluated 1D-CNN in 19-class macro-F1 and reconnaissance recall; tuned
  LightGBM remains substantially above its default configuration in 19-class
  macro-F1.

Reduced-memory execution:

```bash
python main.py --data_dir ./data --scenarios 19 --skip_xai
```

### Workflow 2 - Benign-only autoencoder

```bash
make reproduce-autoencoder
# equivalent:
python scripts/evaluation/autoencoder_threshold_sensitivity.py \
  --seeds 40 41 42 43 44 --split-seed 42 --device cpu --run-tag repro
```

- **Input:** CICIoMT2024 train/test partitions.
- **Resources:** approximately 6 GB RAM; GPU not required.
- **Reference time:** approximately 5-10 minutes on the reference system.
- **Outputs:** a new directory under
  `results/autoencoder_multiseed/repro_<os>_cpu/`, containing per-seed models,
  metrics, thresholds, phase recall, histories, metadata, and aggregates.
- **Success:** all five seeds complete; P85 retains high attack recall while
  its benign false-positive rate is explicitly reported; increasing the
  threshold from P85 to P99 reduces the false-positive rate.

### Workflow 3 - Operational benchmark

This workflow requires trained supervised models. Run Workflow 1 first.

```bash
make reproduce-operational
# equivalent:
python scripts/benchmarks/benchmark_latency.py
```

- **Input:** CICIoMT2024 test data and trained models.
- **Reference time:** approximately 30-45 minutes.
- **Output:** `results/tables/latency_comparison.csv`.
- **Success:** the benchmark completes and records latency and throughput.
  Absolute timings are hardware-dependent and are not expected to equal the
  paper values.

### Workflow 4 - Explainability figures

This workflow requires the trained 19-class models from Workflow 1.

```bash
make reproduce-xai
# equivalent:
python scripts/xai/run_xai.py
```

- **Input:** CICIoMT2024 test data and trained 19-class models.
- **Reference time:** approximately 45-60 minutes.
- **Outputs:** SHAP plots under `results/xai/` and LIME plots under
  `results/figures/`.
- **Success:** the scripts generate the global SHAP and local LIME figures
  without runtime errors.

### Workflow 5 - Data-integrity audit

```bash
make audit
```

- **Input:** CICIoMT2024 train/test partitions.
- **Output:** `audit_reports/crosspartition_audit.json`.
- **Success:** the audit completes and reports missing-value and
  cross-partition feature-signature statistics.

### Full reproduction

```bash
make reproduce       # Workflows 1, 2, and 3
make full-experiment # Workflows 1-4
```

## Reference Results

The bundled files under `results/` are the results generated during preparation
of the camera-ready article. They are provided so reviewers can inspect the
reported evidence without retraining every model.

- `results/tables/`: aggregate, per-class, LightGBM, statistical, and
  operational tables.
- `results/reports/`: class-level reports.
- `results/autoencoder_multiseed/camera_ready_v1_windows_cpu/`: five-seed
  autoencoder reference run.
- `results/xai/` and `results/figures/`: SHAP and LIME reference figures.

These are project-generated derived results. They do not include or
redistribute CICIoMT2024 source records. Large supervised model files are not
part of the distributed artifact and can be regenerated with Workflow 1.

## Reproduction Variability

The objective is to reproduce the paper's conclusions and qualitative ordering,
not to obtain bit-identical floating-point values on every platform.

The reference autoencoder run was produced on Windows CPU. The Docker image
runs TensorFlow on Linux CPU. Even with fixed seeds and identical inputs,
TensorFlow/oneDNN kernel selection, operation ordering, and floating-point
rounding can produce different optimization trajectories. Docker results may
therefore be slightly lower than the bundled reference values; individual
seeds may vary more substantially.

Reviewers should evaluate a fresh run by checking:

1. dataset, split, feature, and class fingerprints;
2. completion of all requested seeds;
3. preservation of the principal result direction;
4. explicit reporting of variability and false positives; and
5. generation of the documented outputs.

The bundled reference results are the source for exact comparison with the
camera-ready tables. Fresh Docker runs are reproduction evidence, not a demand
for bitwise replication. Operational latency and throughput are also expected
to vary with hardware. See [`PROVENANCE.md`](PROVENANCE.md) for detailed
interpretation.

## Data Integrity

The bundled audit reports zero missing values in the evaluated partitions and
461 shared feature signatures among 892,268 unique test signatures (0.05%).
The report is available at `audit_reports/crosspartition_audit.json` and can be
regenerated with `make audit`.

## License

This is a proprietary CESAR/CISSA artifact made available under the limited
[`Academic Artifact Evaluation License`](LICENSE). Authorized CTA reviewers and
SBSeg 2026 organizers may access, install, execute, and inspect it solely for
the confidential artifact-evaluation process. Redistribution, publication,
commercial use, and use outside that evaluation require prior written
authorization from CESAR/CISSA.

The evaluation license becomes effective only after approval by an authorized
CESAR/CISSA representative. Restricted availability may make the artifact
ineligible for SeloD; eligibility should be confirmed with the CTA.

This work was funded by the projects *Xeque-Mate Agente & Agregador*, supported
by CISSA/CESAR with resources from the PPI IoT/Manufatura 4.0 of the MCTI,
grant numbers CIS-AFCCT-2025-5-1-2 and CIS-AFCCT-2025-5-1-1, signed with
EMBRAPII.

