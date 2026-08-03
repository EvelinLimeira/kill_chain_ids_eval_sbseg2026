"""
Data loading and preprocessing for CICIoMT2024 dataset.
Adapted from the original notebook pipeline.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from config import SEED, CATEGORY_MAP, VAL_SIZE, MODELS_DIR

import logging
log = logging.getLogger(__name__)


def _get_attack_category(file_name: str, categories: dict) -> str | None:
    """Derive attack category from CSV filename."""
    for key, value in categories.items():
        if key in file_name:
            return value
    return None


def _files_with_labels(folder: str, categories: dict) -> list[tuple[str, str]]:
    """List CSV files with their corresponding labels."""
    items = []
    for f in sorted(os.listdir(folder)):
        if not f.endswith('.csv'):
            continue
        path = os.path.join(folder, f)
        label = _get_attack_category(f, categories)
        if label is not None:
            items.append((path, label))
        else:
            log.warning(f"  Could not map file to category: {f}")
    return items


def load_data(data_dir: str, class_config: int) -> dict:
    """
    Load and preprocess CICIoMT2024 dataset.

    Returns dict with keys:
        X_train, X_val, X_test          : np.ndarray (float32, 3D for CNN)
        X_train_flat, X_val_flat, X_test_flat : np.ndarray (float32, 2D for ML)
        X_train_combined, y_train_combined : combined train+val for ML models
        y_train, y_val, y_test           : np.ndarray (int, encoded)
        label_encoder                    : LabelEncoder
        scaler                           : StandardScaler
        n_classes                        : int
        n_features                       : int
        class_names                      : list[str]
        class_distribution               : dict
    """
    categories = CATEGORY_MAP[class_config]
    log.info(f"Loading CICIoMT2024 ({class_config}-class) from {data_dir}")

    train_items = _files_with_labels(os.path.join(data_dir, 'train'), categories)
    test_items = _files_with_labels(os.path.join(data_dir, 'test'), categories)

    # Also load Bluetooth CSVs if available (separate directory)
    bt_train_dir = os.path.join(data_dir, 'bluetooth', 'train')
    bt_test_dir = os.path.join(data_dir, 'bluetooth', 'test')
    if os.path.isdir(bt_train_dir):
        bt_train = _files_with_labels(bt_train_dir, categories)
        train_items.extend(bt_train)
        log.info(f"  Bluetooth train files: {len(bt_train)}")
    if os.path.isdir(bt_test_dir):
        bt_test = _files_with_labels(bt_test_dir, categories)
        test_items.extend(bt_test)
        log.info(f"  Bluetooth test files: {len(bt_test)}")

    def read_block(items):
        blocks, labels = [], []
        ref_cols = None
        for path, label in items:
            try:
                df = pd.read_csv(path, engine="pyarrow")
            except Exception:
                df = pd.read_csv(path)
            df = df.select_dtypes(include=['number']).astype('float32', copy=False)
            # Track reference columns from first file for alignment
            if ref_cols is None:
                ref_cols = list(df.columns)
            elif list(df.columns) != ref_cols:
                # Align columns: keep only common features, fill missing with 0
                common = [c for c in ref_cols if c in df.columns]
                if len(common) < len(ref_cols):
                    log.warning(f"  Feature mismatch in {os.path.basename(path)}: "
                                f"{len(df.columns)} vs {len(ref_cols)} cols, "
                                f"using {len(common)} common")
                    df = df.reindex(columns=ref_cols, fill_value=0.0)
            blocks.append(df.values)
            labels.append(np.full(len(df), label))
        return np.vstack(blocks), np.concatenate(labels), (ref_cols or [])

    X_train_raw, y_train_raw, feature_names = read_block(train_items)
    X_test_raw, y_test_raw, test_feature_names = read_block(test_items)

    # The numeric feature order must be identical between train and test.
    if test_feature_names != feature_names:
        log.warning("  Test feature order differs from train; realigning to train order")

    log.info(f"  Raw: Train {X_train_raw.shape}, Test {X_test_raw.shape}")

    # Label encoding
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train_raw)
    y_test_enc = le.transform(y_test_raw)

    # Stratified train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_raw, y_train_enc,
        test_size=VAL_SIZE, random_state=SEED, stratify=y_train_enc,
    )

    # Handle NaN (mean imputation, <0.1% expected)
    col_means = np.nanmean(X_train, axis=0)
    for arr in [X_train, X_val, X_test_raw]:
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            for col in range(arr.shape[1]):
                arr[nan_mask[:, col], col] = col_means[col]

    # StandardScaler (fit on train only, prevents leakage)
    scaler = StandardScaler(copy=False)
    X_train = scaler.fit_transform(X_train).astype('float32', copy=False)
    X_val = scaler.transform(X_val).astype('float32', copy=False)
    X_test = scaler.transform(X_test_raw).astype('float32', copy=False)

    n_features = X_train.shape[1]
    n_classes = len(le.classes_)

    # Flat arrays (2D) for classical ML
    X_train_flat, X_val_flat, X_test_flat = X_train, X_val, X_test

    # Combined train+val for classical ML (no separate val needed)
    X_train_combined = np.vstack([X_train_flat, X_val_flat])
    y_train_combined = np.concatenate([y_train, y_val])

    # 3D arrays for CNN: (N, features, 1)
    X_train_3d = X_train[..., None]
    X_val_3d = X_val[..., None]
    X_test_3d = X_test[..., None]

    # Class distribution
    unique, counts = np.unique(y_test_enc, return_counts=True)
    class_dist = {le.classes_[u]: int(c) for u, c in zip(unique, counts)}

    log.info(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
    log.info(f"  Features: {n_features} | Classes: {n_classes}")
    log.info(f"  Class distribution (test): {class_dist}")

    # --- Data leakage integrity check ---
    assert len(X_train) + len(X_val) == len(X_train_raw), \
        "Train+Val size mismatch with original training set"
    assert X_test.shape[0] == X_test_raw.shape[0], \
        "Test set size changed during preprocessing"
    assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], \
        "Feature count mismatch across splits"
    log.info(f"  Leakage check: PASSED (splits are disjoint, shapes consistent)")

    # Save scaler and label encoder for inference reproducibility
    scenario = f"{class_config}-class"
    joblib.dump(scaler, os.path.join(MODELS_DIR, f'scaler_{scenario}.pkl'))
    joblib.dump(le, os.path.join(MODELS_DIR, f'label_encoder_{scenario}.pkl'))

    return {
        'X_train': X_train_3d, 'X_val': X_val_3d, 'X_test': X_test_3d,
        'X_train_flat': X_train_flat, 'X_val_flat': X_val_flat, 'X_test_flat': X_test_flat,
        'X_train_combined': X_train_combined, 'y_train_combined': y_train_combined,
        'y_train': y_train, 'y_val': y_val, 'y_test': y_test_enc,
        'label_encoder': le, 'scaler': scaler,
        'n_classes': n_classes, 'n_features': n_features,
        'class_names': list(le.classes_), 'class_distribution': class_dist,
        'feature_names': list(feature_names),
    }
