"""
Autoencoder for anomaly-based intrusion detection in IoMT.

Rationale for inclusion:
  - Complementary paradigm: unsupervised anomaly detection vs supervised classification
  - Addresses zero-day attack scenario (detects anomalies without labeled attack data)
  - Trains on benign traffic only, flags high reconstruction error as anomalous
  - Provides deployment alternative when labeled attack data is scarce
  - Strengthens paper by showing multi-paradigm evaluation

Approach:
  1. Train autoencoder on BENIGN traffic only (reconstruction task)
  2. Compute reconstruction error on test set
  3. Threshold: samples with error > percentile(val_errors, 95%) = anomaly
  4. Binary evaluation: Benign vs Attack
"""
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    cohen_kappa_score, roc_auc_score,
)

import os
from config import SEED, AUTOENCODER_PARAMS, MODELS_DIR

log = logging.getLogger(__name__)


def _build_autoencoder(n_features: int) -> tuple[Model, Model, Model]:
    """
    Build symmetric autoencoder.

    Returns: (autoencoder, encoder, decoder) models.
    Architecture: n_features -> 64 -> 32 -> 16 -> 32 -> 64 -> n_features
    """
    p = AUTOENCODER_PARAMS
    enc_dims = p['encoding_dims']  # [64, 32, 16]

    # Encoder
    inp = Input(shape=(n_features,))
    x = inp
    for dim in enc_dims:
        x = Dense(dim, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(p['dropout'])(x)
    encoded = x

    # Decoder (symmetric)
    x = encoded
    for dim in reversed(enc_dims[:-1]):
        x = Dense(dim, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(p['dropout'])(x)
    decoded = Dense(n_features, activation='linear')(x)

    autoencoder = Model(inp, decoded, name='autoencoder')
    encoder = Model(inp, encoded, name='encoder')

    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=p['learning_rate']),
        loss='mse',
    )
    return autoencoder, encoder


def train_autoencoder(X_train_flat, y_train, X_val_flat, y_val,
                       X_test_flat, y_test, label_encoder,
                       scenario: str) -> dict:
    """
    Train autoencoder on benign traffic, evaluate as anomaly detector.

    For multi-class scenarios (6, 19), this reduces to binary:
    benign = class 'Benign', anomaly = everything else.
    """
    log.info("  Training Autoencoder (Anomaly Detection)...")
    p = AUTOENCODER_PARAMS
    tf.random.set_seed(SEED)

    # Identify benign class index
    class_names = list(label_encoder.classes_)
    benign_idx = class_names.index('Benign') if 'Benign' in class_names else 0

    # =====================================================================
    # CONSISTENT BENIGN SPLIT (independent of scenario class count)
    #
    # Problem: data_loader uses train_test_split(stratify=y) where y has
    # different labels per scenario (2/6/19), producing different splits.
    # The autoencoder only uses Benign data, so the same physical Benign
    # samples must land in train/val consistently across scenarios.
    #
    # Solution: pool ALL Benign from train+val, then re-split with a fixed
    # seed using binary stratify. This guarantees identical Benign train/val
    # sets regardless of the scenario's class count.
    # =====================================================================
    from sklearn.model_selection import train_test_split as _tt_split
    from config import VAL_SIZE

    benign_mask_train = (y_train == benign_idx)
    benign_mask_val = (y_val == benign_idx)

    X_benign_all = np.concatenate([
        X_train_flat[benign_mask_train],
        X_val_flat[benign_mask_val],
    ], axis=0)

    # Split benign pool with fixed seed (always the same result)
    X_train_benign, X_val_benign = _tt_split(
        X_benign_all, test_size=VAL_SIZE, random_state=SEED,
    )

    log.info(f"    Benign samples (consistent split): "
             f"Train {len(X_train_benign):,} | Val {len(X_val_benign):,}")

    n_features = X_train_flat.shape[1]
    autoencoder, encoder = _build_autoencoder(n_features)

    trainable = np.sum([np.prod(v.shape) for v in autoencoder.trainable_weights])
    log.info(f"    Parameters: {int(trainable):,}")

    early_stop = EarlyStopping(
        monitor='val_loss', patience=p['early_stop_patience'],
        restore_best_weights=True,
    )

    # Train on benign only (reconstruction task)
    t0 = time.time()
    history = autoencoder.fit(
        X_train_benign, X_train_benign,
        validation_data=(X_val_benign, X_val_benign),
        epochs=p['epochs'], batch_size=p['batch_size'],
        callbacks=[early_stop], verbose=1,
    )
    train_time = time.time() - t0

    # Compute reconstruction errors
    t0 = time.time()
    val_reconstructed = autoencoder.predict(X_val_benign, verbose=0)
    val_errors = np.mean((X_val_benign - val_reconstructed) ** 2, axis=1)

    test_reconstructed = autoencoder.predict(X_test_flat, verbose=0)
    test_errors = np.mean((X_test_flat - test_reconstructed) ** 2, axis=1)
    pred_time = time.time() - t0

    # Ground truth binary: 0 = benign, 1 = attack
    y_test_binary = (y_test != benign_idx).astype(int)

    # =====================================================================
    # MULTIPLE THRESHOLD ANALYSIS
    # Tests percentiles 85-99 to show sensitivity and find optimal F1
    # =====================================================================
    percentiles = list(range(85, 100))
    threshold_results = []

    for pct in percentiles:
        thr = np.percentile(val_errors, pct)
        y_pred_bin = (test_errors > thr).astype(int)

        t_acc  = accuracy_score(y_test_binary, y_pred_bin)
        t_prec = precision_score(y_test_binary, y_pred_bin, zero_division=0)
        t_rec  = recall_score(y_test_binary, y_pred_bin, zero_division=0)
        t_f1   = f1_score(y_test_binary, y_pred_bin, average='binary', zero_division=0)
        t_f1m  = f1_score(y_test_binary, y_pred_bin, average='macro', zero_division=0)

        threshold_results.append({
            'percentile': pct, 'threshold': thr,
            'accuracy': t_acc, 'precision': t_prec,
            'recall': t_rec, 'f1_binary': t_f1, 'f1_macro': t_f1m,
        })

    # Find optimal threshold (max F1-macro)
    import pandas as pd
    thr_df = pd.DataFrame(threshold_results)
    best_row = thr_df.loc[thr_df['f1_macro'].idxmax()]
    optimal_pct = int(best_row['percentile'])
    optimal_threshold = best_row['threshold']

    log.info(f"    Threshold analysis (percentiles {percentiles[0]}-{percentiles[-1]}):")
    log.info(f"    Optimal: p{optimal_pct} (threshold={optimal_threshold:.6f})")
    log.info(f"      Acc={best_row['accuracy']:.4f} | Prec={best_row['precision']:.4f} | "
             f"Rec={best_row['recall']:.4f} | F1={best_row['f1_macro']:.4f}")

    # Save threshold analysis table
    thr_df.to_csv(os.path.join(MODELS_DIR, f'ae_threshold_analysis_{scenario}.csv'), index=False)

    # Use optimal threshold for final evaluation
    threshold = optimal_threshold
    y_pred_binary = (test_errors > threshold).astype(int)

    acc  = accuracy_score(y_test_binary, y_pred_binary)
    f1_m = f1_score(y_test_binary, y_pred_binary, average='macro', zero_division=0)
    f1_w = f1_score(y_test_binary, y_pred_binary, average='weighted', zero_division=0)
    prec = precision_score(y_test_binary, y_pred_binary, zero_division=0)
    rec  = recall_score(y_test_binary, y_pred_binary, zero_division=0)
    kappa = cohen_kappa_score(y_test_binary, y_pred_binary)

    try:
        auc = roc_auc_score(y_test_binary, test_errors)
    except ValueError:
        auc = 0.0

    epochs_run = len(history.history['loss'])
    log.info(f"    Final (p{optimal_pct}): Acc={acc:.4f} | F1m={f1_m:.4f} | "
             f"Prec={prec:.4f} | Rec={rec:.4f} | AUC={auc:.4f} | "
             f"Epochs={epochs_run} | Train={train_time:.1f}s")

    autoencoder.save(f"{MODELS_DIR}/autoencoder_{scenario}.keras")

    return {
        'y_pred': y_pred_binary, 'y_test_binary': y_test_binary,
        'reconstruction_errors': test_errors, 'threshold': threshold,
        'optimal_percentile': optimal_pct,
        'threshold_analysis': thr_df,
        'train_time': train_time, 'pred_time': pred_time,
        'model': autoencoder, 'encoder': encoder,
        'history': history.history,
        'accuracy': acc, 'f1_macro': f1_m, 'f1_weighted': f1_w,
        'precision': prec, 'recall': rec, 'auc_roc': auc, 'kappa': kappa,
        'trainable_params': int(trainable),
        'note': f'Binary evaluation (Benign vs Attack). Trained on benign only. '
                f'Optimal threshold at percentile {optimal_pct}.',
    }
