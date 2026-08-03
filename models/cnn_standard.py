"""
Standard 1D-CNN for IoMT intrusion detection.
Architecture matches the paper (Conv1D-32 -> Pool -> Conv1D-64 -> Pool -> Dense-128).
"""
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import mixed_precision
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight

from config import SEED, CNN_PARAMS, MODELS_DIR

log = logging.getLogger(__name__)


def _build_standard_cnn(n_features: int, n_classes: int) -> tf.keras.Model:
    """Build the standard 1D-CNN architecture."""
    p = CNN_PARAMS
    model = Sequential([
        Input(shape=(n_features, 1)),
        Conv1D(p['filters_1'], p['kernel_size'], activation='relu'),
        MaxPooling1D(p['pool_size']),
        Conv1D(p['filters_2'], p['kernel_size'], activation='relu'),
        MaxPooling1D(p['pool_size']),
        Flatten(),
        Dense(p['dense_units'], activation='relu'),
        Dense(n_classes, activation='softmax', dtype='float32'),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=p['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def _make_tf_dataset(X, y, batch_size, training=True):
    """Create optimized tf.data pipeline."""
    AUTOTUNE = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if training:
        ds = ds.shuffle(min(len(X), CNN_PARAMS['shuffle_buffer']),
                        reshuffle_each_iteration=True)
    ds = ds.batch(batch_size, drop_remainder=training).prefetch(AUTOTUNE)
    return ds


def train_standard_cnn(X_train, y_train, X_val, y_val, X_test, y_test,
                        n_classes: int, scenario: str) -> dict:
    """
    Train the standard 1D-CNN.

    Args:
        X_train, X_val, X_test: 3D arrays (N, features, 1)
        y_train, y_val, y_test: integer encoded labels
    """
    log.info("  Training 1D-CNN (Standard)...")
    p = CNN_PARAMS
    tf.random.set_seed(SEED)

    original_policy = mixed_precision.global_policy().name
    try:
        if p['mixed_precision']:
            mixed_precision.set_global_policy('mixed_float16')

        model = _build_standard_cnn(X_train.shape[1], n_classes)

        # Class weights for imbalanced data
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight = dict(zip(classes.astype(int), weights))

        train_ds = _make_tf_dataset(X_train, y_train, p['batch_size'], training=True)
        val_ds = _make_tf_dataset(X_val, y_val, p['batch_size'], training=False)

        early_stop = EarlyStopping(
            monitor='val_loss', patience=p['early_stop_patience'],
            restore_best_weights=True,
        )

        t0 = time.time()
        history = model.fit(
            train_ds, validation_data=val_ds,
            epochs=p['epochs'], callbacks=[early_stop],
            class_weight=class_weight, verbose=1,
        )
        train_time = time.time() - t0

        t0 = time.time()
        y_pred_probs = model.predict(X_test, verbose=0)
        y_pred = y_pred_probs.argmax(axis=1)
        pred_time = time.time() - t0

        acc = accuracy_score(y_test, y_pred)
        f1_m = f1_score(y_test, y_pred, average='macro', zero_division=0)
        f1_w = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        kappa = cohen_kappa_score(y_test, y_pred)

        epochs_run = len(history.history['loss'])
        log.info(f"    Acc: {acc:.4f} | F1m: {f1_m:.4f} | Epochs: {epochs_run} | "
                 f"Train: {train_time:.1f}s")

        model.save(f"{MODELS_DIR}/cnn_standard_{scenario}.keras")

    finally:
        mixed_precision.set_global_policy(original_policy)

    return {
        'y_pred': y_pred, 'y_pred_probs': y_pred_probs,
        'train_time': train_time, 'pred_time': pred_time,
        'model': model, 'history': history.history,
        'accuracy': acc, 'f1_macro': f1_m, 'f1_weighted': f1_w, 'kappa': kappa,
    }
