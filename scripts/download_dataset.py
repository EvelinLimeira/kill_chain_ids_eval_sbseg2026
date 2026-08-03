#!/usr/bin/env python3
"""
Download the CICIoMT2024 dataset from Kaggle and arrange it for this artifact.

The pipeline (data_loader.py) expects one CSV per attack capture under:
    data/train/<label>.pcap.csv   (per-attack + Benign training files)
    data/test/<label>.pcap.csv    (per-attack + Benign test files)
where <label> contains a substring recognized by config.py's category maps
(e.g. "ARP_Spoofing", "TCP_IP-DDoS-ICMP", "MQTT-DDoS-Connect_Flood", "Benign").

The default mirror (limamateus/cic-iomt-2024-wifi-mqtt) ships the dataset as
two consolidated tables (train/test), each with a 'label' column whose values
are exactly the original per-capture file names (e.g. "TCP_IP-DDoS-ICMP1_train",
"Benign_test"). This script downloads the snapshot via kagglehub, and for any
consolidated table it splits rows by 'label' into the per-capture CSVs the
pipeline expects; files that are already per-capture (no 'label' column) are
copied through unchanged. This reproduces the original 51 train / 21 test file
layout exactly.

NOTE: a different Kaggle mirror, amineipad/cic-iomt-dataset-2024, was checked
and rejected as a default: it merges MQTT-DDoS-Connect_Flood and
MQTT-DoS-Connect_Flood into a single "DoS Connect Flood" label, collapsing the
19-class taxonomy to 18 classes and breaking Tables 6, 7, and 9-11 of the
paper. Do not use it as a substitute without re-deriving that split.

Prerequisites
-------------
    pip install kagglehub pandas

The default dataset is public; kagglehub can usually download it without
authentication. If your environment requires it, authenticate once with
kagglehub.login(), or set KAGGLE_USERNAME/KAGGLE_KEY, or place a token at
~/.kaggle/kaggle.json. See https://github.com/Kaggle/kagglehub.

Usage
-----
    python scripts/download_dataset.py
    python scripts/download_dataset.py --dataset <owner>/<dataset-slug>

Options
-------
    --dataset    Kaggle dataset handle "owner/slug"
                 (default: limamateus/cic-iomt-2024-wifi-mqtt).
    --data-dir   Target data directory (default: ./data).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# Confirmed public mirror preserving the full 19-class taxonomy (verified
# against the official per-category row counts reported in the paper).
DEFAULT_DATASET = os.environ.get(
    "CICIOMT_KAGGLE_DATASET", "limamateus/cic-iomt-2024-wifi-mqtt"
)

EXPECTED_TRAIN = 51   # Wi-Fi/MQTT training capture files
EXPECTED_TEST = 21    # Wi-Fi/MQTT test capture files
LABEL_COLUMN = "label"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download CICIoMT2024 from Kaggle.")
    p.add_argument("--dataset", default=DEFAULT_DATASET,
                   help='Kaggle dataset handle "owner/slug" '
                        f"(default: {DEFAULT_DATASET}).")
    p.add_argument("--data-dir", default=str(Path(__file__).resolve().parents[1] / "data"),
                   help="Target data directory (default: ./data).")
    return p.parse_args()


def fail(msg: str, code: int = 1) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def determine_split(filename: str) -> str | None:
    """Infer 'train' or 'test' from a source file name. Prefers the more
    specific '_train'/'_test' markers used by the per-capture layout, then
    falls back to a plain 'train'/'test' substring for consolidated tables."""
    name = filename.lower()
    if "_train" in name or name.startswith("train"):
        return "train"
    if "_test" in name or name.startswith("test"):
        return "test"
    if "train" in name:
        return "train"
    if "test" in name:
        return "test"
    return None


def materialize_source(path: Path, data_dir: Path, pd) -> None:
    """Write one source file into data/train or data/test.

    If the file has a 'label' column, it is a consolidated table: rows are
    split by label into one CSV per capture (matching the pipeline's expected
    layout). Otherwise the file is assumed to already be a per-capture CSV
    and is copied through unchanged.
    """
    split = determine_split(path.name)
    if split is None:
        return  # unrecognized file (e.g. a notebook artifact); skip

    out_dir = data_dir / split
    out_dir.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        return

    if LABEL_COLUMN in df.columns:
        for label, group in df.groupby(LABEL_COLUMN):
            out_name = f"{label}.pcap.csv"
            group.drop(columns=[LABEL_COLUMN]).to_csv(out_dir / out_name, index=False)
    else:
        shutil.copy2(path, out_dir / path.name)


def select_source_files(snapshot_dir: Path) -> list[Path]:
    """Collect source files, de-duplicating same-content csv/parquet pairs
    (kagglehub mirrors sometimes ship both) by preferring parquet."""
    by_stem: dict[str, Path] = {}
    for f in snapshot_dir.rglob("*"):
        if f.suffix.lower() not in (".csv", ".parquet"):
            continue
        stem = f.stem.lower()
        if stem not in by_stem or f.suffix.lower() == ".parquet":
            by_stem[stem] = f
    return list(by_stem.values())


def main() -> None:
    args = parse_args()

    try:
        import kagglehub
    except ImportError:
        fail("The 'kagglehub' package is not installed. Run: pip install kagglehub")
    try:
        import pandas as pd
    except ImportError:
        fail("The 'pandas' package is not installed. Run: pip install pandas")

    data_dir = Path(args.data_dir).resolve()

    print(f"Downloading '{args.dataset}' via kagglehub (cached after first run)...")
    try:
        snapshot_path = Path(kagglehub.dataset_download(args.dataset))
    except Exception as exc:  # noqa: BLE001
        fail(f"Download failed for '{args.dataset}'. If this dataset requires "
             "authentication, configure Kaggle credentials (kagglehub.login(), "
             "KAGGLE_USERNAME/KAGGLE_KEY, or ~/.kaggle/kaggle.json). "
             f"Details: {exc}")

    print(f"Downloaded to local cache: {snapshot_path}")
    print("Arranging CSVs into data/train and data/test ...")

    for src in select_source_files(snapshot_path):
        print(f"  processing {src.name} ...")
        materialize_source(src, data_dir, pd)

    n_train = len(list((data_dir / "train").glob("*.csv"))) if (data_dir / "train").exists() else 0
    n_test = len(list((data_dir / "test").glob("*.csv"))) if (data_dir / "test").exists() else 0

    print(f"  train CSVs: {n_train}")
    print(f"  test  CSVs: {n_test}")

    if n_train == 0 and n_test == 0:
        fail("No train/test CSVs were produced. The mirror layout may differ; "
             f"inspect the downloaded snapshot at {snapshot_path}.")
    if n_train != EXPECTED_TRAIN or n_test != EXPECTED_TEST:
        print(f"WARNING: expected {EXPECTED_TRAIN} train / {EXPECTED_TEST} test "
              "Wi-Fi/MQTT capture files. The counts above differ; verify the "
              "dataset matches the CICIoMT2024 Wi-Fi/MQTT feature-level "
              "partition and preserves all 19 classes (see data/README.md).")
    else:
        print("Dataset is ready. Run `make verify` or a reproduction target next.")


if __name__ == "__main__":
    main()
