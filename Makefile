# ============================================================
# Makefile - IoMT IDS Artifact (SBSeg 2026)
# Paper: Beyond Aggregate Accuracy: Kill Chain-Aware Evaluation
#        of IoMT Intrusion Detection Models
#
# Requirements: Python 3.12 in an active virtualenv.
#               For dataset-dependent targets, place CICIoMT2024
#               in data/train/ and data/test/ (see data/README.md).
#
# Quick start:
#   make smoke      # fast install check (synthetic data, CPU, no dataset)
#   make verify     # recompute Tables 6/7 from bundled results (no dataset)
#
# Reproduce paper claims (require the dataset):
#   make reproduce-supervised    # Tables 4-7 and 12
#   make reproduce-autoencoder   # Tables 8 and 9
#   make reproduce-operational   # Table 10
#   make reproduce-xai           # Figures 2, 3
#   make reproduce               # supervised + autoencoder + operational
#   make full-experiment         # everything above
#
# See PROVENANCE.md for the table/figure -> artifact mapping and
# docs/runtime_reference.md for measured execution times.
# ============================================================

PYTHON := python
DATA   := ./data
AE_TAG := repro

.PHONY: smoke verify download-data \
        reproduce reproduce-supervised reproduce-autoencoder \
        reproduce-operational reproduce-xai audit full-experiment help

# ------------------------------------------------------------
# smoke - installation check with synthetic data (no dataset, CPU)
# ------------------------------------------------------------
smoke:
	@echo "[smoke] Running smoke test (synthetic data, no GPU required)..."
	$(PYTHON) -m pytest tests/test_smoke.py -v --tb=short
	@echo "[smoke] PASSED"

# ------------------------------------------------------------
# download-data - fetch CICIoMT2024 from Kaggle into data/train and data/test.
#   Requires: pip install kagglehub. Defaults to the public mirror
#   limamateus/cic-iomt-2024-wifi-mqtt; override with DATASET=owner/slug.
# ------------------------------------------------------------
download-data:
	@echo "[data] Downloading CICIoMT2024 from Kaggle (kagglehub)..."
	$(PYTHON) scripts/download_dataset.py $(if $(DATASET),--dataset $(DATASET),)
	@echo "[data] Done -> data/train/, data/test/"

# ------------------------------------------------------------
# verify - regenerate per-class metrics (Tables 5, 6) from the bundled
#          confusion matrices. Does NOT require the dataset or retraining.
#          The autoencoder aggregate CSVs (Tables 8-9) are already provided
#          under results/autoencoder_multiseed/ (see PROVENANCE.md).
# ------------------------------------------------------------
verify:
	@echo "[verify] Recomputing per-class metrics from saved confusion matrices..."
	$(PYTHON) scripts/analysis/compute_per_class_metrics.py
	@echo "[verify] Done -> results/tables/per_class_19-class.csv"

# ------------------------------------------------------------
# reproduce-supervised - Tables 4-7 and 12
#   Trains RF, LR, LightGBM (default+tuned), 1D-CNN and runs the
#   bootstrap CI / McNemar statistical validation.
# ------------------------------------------------------------
reproduce-supervised:
	@echo "[supervised] Tables 4-7 and 12 (RF/LR/LightGBM/CNN + statistics)..."
	$(PYTHON) main.py --data_dir $(DATA) --scenarios 2 6 19 --skip_xai
	@echo "[supervised] Done -> results/tables/comparison_*.csv, lgbm_variants_*.csv,"
	@echo "             per_class_19-class.csv, bootstrap_ci_19-class.csv, mcnemar_19-class.csv"

# ------------------------------------------------------------
# reproduce-autoencoder - Tables 8 and 9
#   Five training seeds (40-44), fixed benign split (seed 42), CPU.
#   Writes results/autoencoder_multiseed/<tag>_<os>_<device>/.
# ------------------------------------------------------------
reproduce-autoencoder:
	@echo "[autoencoder] Tables 8 and 9 (five-seed benign-only autoencoder)..."
	$(PYTHON) scripts/evaluation/autoencoder_threshold_sensitivity.py \
	    --seeds 40 41 42 43 44 --split-seed 42 --device cpu --run-tag $(AE_TAG)
	@echo "[autoencoder] Done -> results/autoencoder_multiseed/$(AE_TAG)_<os>_cpu/aggregate/"

# ------------------------------------------------------------
# reproduce-operational - Table 10
#   NOTE: absolute timings are hardware-dependent; the relative ordering
#   between models is the reproducible signal. See docs/runtime_reference.md.
# ------------------------------------------------------------
reproduce-operational:
	@echo "[operational] Table 10 (latency / throughput benchmark)..."
	$(PYTHON) scripts/benchmarks/benchmark_latency.py
	@echo "[operational] Done -> results/tables/latency_comparison.csv"

# ------------------------------------------------------------
# reproduce-xai - Figures 2 and 3
# ------------------------------------------------------------
reproduce-xai:
	@echo "[xai] Figures 2, 3 (SHAP + LIME)..."
	$(PYTHON) scripts/xai/run_xai.py
	@echo "[xai] Done -> results/xai/ and results/figures/"

# ------------------------------------------------------------
# reproduce - main claims (supervised + autoencoder + operational)
# ------------------------------------------------------------
reproduce: reproduce-supervised reproduce-autoencoder reproduce-operational

# ------------------------------------------------------------
# audit - data integrity (cross-partition duplicate analysis)
# ------------------------------------------------------------
audit:
	@echo "[audit] Cross-partition duplicate audit..."
	$(PYTHON) scripts/audit/audit_crosspartition_duplicates.py
	@echo "[audit] Report -> audit_reports/crosspartition_audit.json"

# ------------------------------------------------------------
# full-experiment - everything (supervised + autoencoder + operational + xai)
# ------------------------------------------------------------
full-experiment: reproduce reproduce-xai

# ------------------------------------------------------------
help:
	@echo ""
	@echo "Targets:"
	@echo "  download-data          Download CICIoMT2024 from Kaggle into data/ (needs kaggle + creds)"
	@echo "  smoke                  Install check: synthetic data, CPU, no dataset (<3 min)"
	@echo "  verify                 Recompute Tables 6/7 from bundled results (no dataset)"
	@echo "  reproduce-supervised   Tables 4-7 and 12 (needs dataset)"
	@echo "  reproduce-autoencoder  Tables 8 and 9 (needs dataset)"
	@echo "  reproduce-operational  Table 10 (needs dataset + models)"
	@echo "  reproduce-xai          Figures 2, 3 (needs dataset)"
	@echo "  reproduce              supervised + autoencoder + operational"
	@echo "  full-experiment        reproduce + xai"
	@echo "  audit                  Cross-partition duplicate check"
	@echo ""
	@echo "No 'make' available? Run the command shown inside each target directly."
	@echo "See README.md (Experiments), PROVENANCE.md, docs/runtime_reference.md."
	@echo ""
