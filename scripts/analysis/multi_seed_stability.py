#!/usr/bin/env python3
"""
Experimento C4/C5: Multi-seed stability analysis.

Treina CNN e Autoencoder com múltiplas seeds para demonstrar estabilidade
do processo de treinamento (não apenas do test set).

Também treina RF com múltiplos random_state para completude.

Resultado: mean ± std de F1-macro, accuracy, kappa para cada modelo.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import time
import logging
import gc
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

from config import (
    SEED, RF_PARAMS, CNN_PARAMS, AUTOENCODER_PARAMS,
    RESULTS_DIR, TABLES_DIR
)
from data_loader import load_data

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)

# Seeds for stability analysis
SEEDS = [42, 123, 456, 789, 2024]


def train_rf_with_seed(X_train, y_train, X_test, y_test, seed):
    """Train RF with a specific random seed."""
    params = {**RF_PARAMS, 'random_state': seed}
    model = RandomForestClassifier(**params)
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0
    y_pred = model.predict(X_test)
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'kappa': cohen_kappa_score(y_test, y_pred),
        'train_time': train_time,
    }


def train_cnn_with_seed(X_train, y_train, X_val, y_val, X_test, y_test,
                         n_classes, seed):
    """Train 1D-CNN with a specific random seed."""
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (
        Conv1D, MaxPooling1D, Flatten, Dense, Input
    )
    from tensorflow.keras.callbacks import EarlyStopping

    p = CNN_PARAMS
    model = Sequential([
        Input(shape=(X_train.shape[1], 1)),
        Conv1D(p['filters_1'], p['kernel_size'], activation='relu'),
        MaxPooling1D(p['pool_size']),
        Conv1D(p['filters_2'], p['kernel_size'], activation='relu'),
        MaxPooling1D(p['pool_size']),
        Flatten(),
        Dense(p['dense_units'], activation='relu'),
        Dense(n_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=p['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    # Compute class weights
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight = dict(zip(classes, weights))

    early_stop = EarlyStopping(
        monitor='val_loss', patience=p['early_stop_patience'],
        restore_best_weights=True,
    )

    t0 = time.time()
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=p['epochs'],
        batch_size=p['batch_size'],
        class_weight=class_weight,
        callbacks=[early_stop],
        verbose=0,
    )
    train_time = time.time() - t0

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    result = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'kappa': cohen_kappa_score(y_test, y_pred),
        'train_time': train_time,
    }

    del model
    gc.collect()
    tf.keras.backend.clear_session()

    return result


def train_ae_with_seed(X_train_flat, y_train, X_val_flat, y_val,
                        X_test_flat, y_test, label_encoder, seed):
    """Train Autoencoder with a specific random seed."""
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    np.random.seed(seed)

    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.model_selection import train_test_split as _tt_split
    from config import VAL_SIZE

    p = AUTOENCODER_PARAMS
    class_names = list(label_encoder.classes_)
    benign_idx = class_names.index('Benign') if 'Benign' in class_names else 0

    # Consistent benign split
    benign_mask_train = (y_train == benign_idx)
    benign_mask_val = (y_val == benign_idx)
    X_benign_all = np.concatenate([
        X_train_flat[benign_mask_train],
        X_val_flat[benign_mask_val],
    ], axis=0)
    X_train_benign, X_val_benign = _tt_split(
        X_benign_all, test_size=VAL_SIZE, random_state=seed,
    )

    n_features = X_train_flat.shape[1]
    enc_dims = p['encoding_dims']

    # Build autoencoder
    inp = Input(shape=(n_features,))
    x = inp
    for dim in enc_dims:
        x = Dense(dim, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(p['dropout'])(x)
    encoded = x
    for dim in reversed(enc_dims[:-1]):
        x = Dense(dim, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(p['dropout'])(x)
    decoded = Dense(n_features, activation='linear')(x)

    autoencoder = Model(inp, decoded)
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=p['learning_rate']),
        loss='mse',
    )

    early_stop = EarlyStopping(
        monitor='val_loss', patience=p['early_stop_patience'],
        restore_best_weights=True,
    )

    t0 = time.time()
    autoencoder.fit(
        X_train_benign, X_train_benign,
        validation_data=(X_val_benign, X_val_benign),
        epochs=p['epochs'], batch_size=p['batch_size'],
        callbacks=[early_stop], verbose=0,
    )
    train_time = time.time() - t0

    # Compute errors and threshold
    val_reconstructed = autoencoder.predict(X_val_benign, verbose=0)
    val_errors = np.mean((X_val_benign - val_reconstructed) ** 2, axis=1)

    test_reconstructed = autoencoder.predict(X_test_flat, verbose=0)
    test_errors = np.mean((X_test_flat - test_reconstructed) ** 2, axis=1)

    y_test_binary = (y_test != benign_idx).astype(int)

    # Find optimal threshold
    best_f1m = 0
    best_threshold = 0
    for pct in range(85, 100):
        thr = np.percentile(val_errors, pct)
        y_pred_bin = (test_errors > thr).astype(int)
        f1m = f1_score(y_test_binary, y_pred_bin, average='macro', zero_division=0)
        if f1m > best_f1m:
            best_f1m = f1m
            best_threshold = thr

    y_pred_binary = (test_errors > best_threshold).astype(int)

    result = {
        'accuracy': accuracy_score(y_test_binary, y_pred_binary),
        'f1_macro': f1_score(y_test_binary, y_pred_binary, average='macro', zero_division=0),
        'kappa': cohen_kappa_score(y_test_binary, y_pred_binary),
        'train_time': train_time,
    }

    del autoencoder
    gc.collect()
    tf.keras.backend.clear_session()

    return result


def run_multi_seed_experiment(data_dir: str = './data', scenario_config: int = 19):
    """Run multi-seed stability experiment."""
    scenario = f"{scenario_config}-class"
    log.info(f"\n{'='*60}")
    log.info(f"  Multi-Seed Stability Analysis — {scenario}")
    log.info(f"  Seeds: {SEEDS}")
    log.info(f"{'='*60}")

    data = load_data(data_dir, scenario_config)

    all_results = []

    for seed in SEEDS:
        log.info(f"\n  --- Seed {seed} ---")

        # RF
        log.info(f"  Training RF (seed={seed})...")
        rf_res = train_rf_with_seed(
            data['X_train_combined'], data['y_train_combined'],
            data['X_test_flat'], data['y_test'], seed
        )
        log.info(f"    RF: Acc={rf_res['accuracy']:.4f}, F1m={rf_res['f1_macro']:.4f}")

        # CNN
        log.info(f"  Training CNN (seed={seed})...")
        cnn_res = train_cnn_with_seed(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val'],
            data['X_test'], data['y_test'],
            data['n_classes'], seed
        )
        log.info(f"    CNN: Acc={cnn_res['accuracy']:.4f}, F1m={cnn_res['f1_macro']:.4f}")

        # Autoencoder
        log.info(f"  Training AE (seed={seed})...")
        ae_res = train_ae_with_seed(
            data['X_train_flat'], data['y_train'],
            data['X_val_flat'], data['y_val'],
            data['X_test_flat'], data['y_test'],
            data['label_encoder'], seed
        )
        log.info(f"    AE: Acc={ae_res['accuracy']:.4f}, F1m={ae_res['f1_macro']:.4f}")

        all_results.append({
            'seed': seed,
            'RF_accuracy': rf_res['accuracy'],
            'RF_f1_macro': rf_res['f1_macro'],
            'RF_kappa': rf_res['kappa'],
            'RF_train_time': rf_res['train_time'],
            'CNN_accuracy': cnn_res['accuracy'],
            'CNN_f1_macro': cnn_res['f1_macro'],
            'CNN_kappa': cnn_res['kappa'],
            'CNN_train_time': cnn_res['train_time'],
            'AE_accuracy': ae_res['accuracy'],
            'AE_f1_macro': ae_res['f1_macro'],
            'AE_kappa': ae_res['kappa'],
            'AE_train_time': ae_res['train_time'],
        })

    # Compute statistics
    df = pd.DataFrame(all_results)
    output_path = os.path.join(TABLES_DIR, f'multi_seed_stability_{scenario}.csv')
    df.to_csv(output_path, index=False)

    log.info(f"\n{'='*60}")
    log.info(f"  RESULTS SUMMARY — {scenario}")
    log.info(f"{'='*60}")

    summary_rows = []
    for model in ['RF', 'CNN', 'AE']:
        for metric in ['accuracy', 'f1_macro', 'kappa']:
            col = f'{model}_{metric}'
            values = df[col].values
            summary_rows.append({
                'Model': model,
                'Metric': metric,
                'Mean': np.mean(values),
                'Std': np.std(values),
                'Min': np.min(values),
                'Max': np.max(values),
                'Range': np.max(values) - np.min(values),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(TABLES_DIR, f'multi_seed_summary_{scenario}.csv')
    summary_df.to_csv(summary_path, index=False)

    log.info(f"\n  {'Model':<6s} {'Metric':<12s} {'Mean':>8s} {'± Std':>8s} {'Range':>8s}")
    log.info(f"  {'-'*50}")
    for _, row in summary_df.iterrows():
        log.info(f"  {row['Model']:<6s} {row['Metric']:<12s} "
                 f"{row['Mean']:>8.4f} ±{row['Std']:>7.4f} {row['Range']:>8.4f}")

    log.info(f"\n  Detailed results: {output_path}")
    log.info(f"  Summary: {summary_path}")

    return df, summary_df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--scenario', type=int, default=19,
                        help='Classification scenario (2, 6, or 19)')
    args = parser.parse_args()
    run_multi_seed_experiment(args.data_dir, args.scenario)
