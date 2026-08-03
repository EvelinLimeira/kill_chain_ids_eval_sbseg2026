# CICIoMT2024 Dataset

This project uses the **CICIoMT2024** dataset, publicly available from the
Canadian Institute for Cybersecurity (CIC).

## Option A — Automated download from Kaggle (recommended)

A helper script downloads the dataset via
[`kagglehub`](https://github.com/Kaggle/kagglehub) and arranges the CSVs into
`train/` and `test/` automatically.

1. Install dependencies:
   ```bash
   pip install kagglehub pandas
   ```
   The default mirror,
   [`limamateus/cic-iomt-2024-wifi-mqtt`](https://www.kaggle.com/datasets/limamateus/cic-iomt-2024-wifi-mqtt),
   is public and can usually be downloaded without authentication. If your
   environment requires it, authenticate once with `kagglehub.login()`, or set
   `KAGGLE_USERNAME`/`KAGGLE_KEY`, or place a token at `~/.kaggle/kaggle.json`.

2. Run the downloader:
   ```bash
   python scripts/download_dataset.py
   # or via make:
   make download-data
   ```

This mirror ships the dataset as two consolidated tables (`train`/`test`) with
a `label` column whose values are the original per-capture file names (e.g.
`TCP_IP-DDoS-ICMP1_train`, `Benign_test`). The script downloads the snapshot to
the local kagglehub cache (reused on subsequent runs) and splits each table by
`label` into the per-capture CSVs the pipeline expects, reproducing the
original 51 train / 21 test file layout exactly (verified: 7,160,831 train and
1,614,182 test rows, matching the counts below).

**Do not substitute a different mirror without checking class integrity
first.** The alternative mirror `amineipad/cic-iomt-dataset-2024` was
evaluated and rejected: it merges `MQTT-DDoS-Connect_Flood` and
`MQTT-DoS-Connect_Flood` into a single `"DoS Connect Flood"` label, collapsing
the dataset to 18 classes and breaking the paper's 19-class evaluation
(Tables 6, 7, 9-11).

## Option B — Manual download from the official source

1. Visit the official dataset page:
   https://www.unb.ca/cic/datasets/iomt-dataset-2024.html

2. Download the **Network** partition (Wi-Fi + MQTT feature-level CSV files).

3. Place files as follows:

```
data/
├── train/
│   ├── ARP_Spoofing_train.pcap.csv
│   ├── Benign_train.pcap.csv
│   ├── MQTT-DDoS-Connect_Flood_train.pcap.csv
│   └── ... (51 CSV files total)
└── test/
    ├── ARP_Spoofing_test.pcap.csv
    ├── Benign_test.pcap.csv
    ├── MQTT-DDoS-Connect_Flood_test.pcap.csv
    └── ... (21 CSV files total)
```

## Expected partition sizes

| Partition | Rows       | Unique signatures |
|-----------|------------|-------------------|
| Train     | 7,160,831  | 4,515,080         |
| Test      | 1,614,182  | 892,268           |

Total: ~8.8 million records, 45 numeric features per row.

## Validation

After placing the files, run the audit script to verify integrity:

```bash
python scripts/audit/audit_crosspartition_duplicates.py
```

Expected output:
```
Training samples : 7,160,831
Test samples     : 1,614,182
Numeric features : 45
Missing values   : 0
Cross-partition exact duplicates: 461 signatures (0.05% of test)
```

The 461 shared feature signatures are a known dataset property
(repetitive network traffic patterns) and do not affect the
reported results.

## Features

The 45 numeric features used in this study are listed in
`config.py` (`CICIOMT_FEATURE_NAMES`). The feature `Magnitue`
preserves the original spelling from the CICIoMT2024 dataset.

## Label mapping

- **19-class**: all individual attack types + Benign
- **6-class**: Benign, DDoS, DoS, MQTT, Recon, Spoofing
- **2-class**: Benign, Attack

Full mapping is defined in `config.py` (`CATEGORY_MAP`).

## Citation

Dadkhah, S., Neto, E. C. P., Ferreira, R., Molokwu, R. C.,
Sadeghi, S., & Ghorbani, A. A. (2024). CICIoMT2024: A benchmark
dataset for multi-protocol security assessment in IoMT.
*Internet of Things*, 28, 101351.
https://doi.org/10.1016/j.iot.2024.101351
