"""
Lightweight 1D-CNN using Depthwise Separable Convolutions.

Rationale for inclusion:
  - Reduces parameter count by ~5-8x vs standard Conv1D
  - Targets IoMT gateway/edge deployment with constrained compute
  - Demonstrates accuracy-efficiency trade-off in the paper
  - Addresses reviewer concern about deployment practicality

Architecture:
  SepConv1D-16 -> Pool -> SepConv1D-32 -> Pool -> GlobalAvgPool -> Dense-64 -> Output
  Uses GlobalAveragePooling1D instead of Flatten (fewer parameters).
  Dropout for regularization (important for smaller models).
"""
import time
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    SeparableConv1D, MaxPooling1D, GlobalAveragePooling1D,
    Dense, Dropout, Input, BatchNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import mixed_precision
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.utils.class_weight import compute_class_weight

from config import SEED, LIGHT_CNN_PARAMS, MODELS_DIR

log = logging.getLogger(__name__)


def _build_light_cnn(n_features: int, n_classes: int) -> tf.keras.Model:
    """
    Build lightweight depthwise-separable 1D-CNN.

    Key differences from standard CNN:
      - SeparableConv1D: factorizes convolution into depthwise + pointwise
        reducing parameters from (kernel * in_channels * out_channels) to
        (kernel * in_channels + in_channels * out_channels)
      - GlobalAveragePooling1D: replaces Flatten, eliminating the large
        dense layer that dominates standard CNN parameter count
      - BatchNormalization: stabilizes training with fewer parameters
      - Dropout: prevents overfitting in the smaller architecture
    """
    p = LIGHT_CNN_PARAMS
    model = Sequential([
        Input(shape=(n_features, 1)),

        # Block 1: Depthwise separable convolution
        SeparableConv1D(p['filters_1'], p['kernel_size'], activation='relu',
                        padding='same', depth_multiplier=1),
        BatchNormalization(),
        MaxPooling1D(p['pool_size']),

        # Block 2: Depthwise separable convolution
        SeparableConv1D(p['filters_2'], p['kernel_size'], activation='relu',
                        padding='same', depth_multiplier=1),
        BatchNormalization(),
        MaxPooling1D(p['pool_size']),

        # Global pooling (no Flatten needed, massive param reduction)
        GlobalAveragePooling1D(),

        # Classification head
        Dense(p['dense_units'], activation='relu'),
        Dropout(p['dropout']),
        Dense(n_classes, activation='softmax', dtype='float32'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=p['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def _count_params(model):
    """Count trainable and total parameters."""
    trainable = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    total = np.sum([np.prod(v.shape) for v in model.weights])
    return int(trainable), int(total)


def train_light_cnn(X_train, y_train, X_val, y_val, X_test, y_test,
                     n_classes: int, scenario: str) -> dict:
    """Train the lightweight depthwise-separable CNN."""
    log.info("  Training Light-CNN (Depthwise Separable)...")
    p = LIGHT_CNN_PARAMS
    tf.random.set_seed(SEED)

    original_policy = mixed_precision.global_policy().name
    try:
        if p['mixed_precision']:
            mixed_precision.set_global_policy('mixed_float16')

        model = _build_light_cnn(X_train.shape[1], n_classes)
        trainable, total = _count_params(model)
        log.info(f"    Parameters: {trainable:,} trainable / {total:,} total")

        # Class weights
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight = dict(zip(classes.astype(int), weights))

        # tf.data pipeline
        AUTOTUNE = tf.data.AUTOTUNE

        def make_ds(X, y, training=True):
            ds = tf.data.Dataset.from_tensor_slices((X, y))
            if training:
                ds = ds.shuffle(min(len(X), p['shuffle_buffer']),
                                reshuffle_each_iteration=True)
            return ds.batch(p['batch_size'], drop_remainder=training).prefetch(AUTOTUNE)

        train_ds = make_ds(X_train, y_train, training=True)
        val_ds = make_ds(X_val, y_val, training=False)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=p['early_stop_patience'],
                          restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6),
        ]

        t0 = time.time()
        history = model.fit(
            train_ds, validation_data=val_ds,
            epochs=p['epochs'], callbacks=callbacks,
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
                 f"Train: {train_time:.1f}s | Params: {trainable:,}")

        model.save(f"{MODELS_DIR}/cnn_light_{scenario}.keras")

    finally:
        mixed_precision.set_global_policy(original_policy)

    return {
        'y_pred': y_pred, 'y_pred_probs': y_pred_probs,
        'train_time': train_time, 'pred_time': pred_time,
        'model': model, 'history': history.history,
        'accuracy': acc, 'f1_macro': f1_m, 'f1_weighted': f1_w, 'kappa': kappa,
        'trainable_params': trainable, 'total_params': total,
    }
