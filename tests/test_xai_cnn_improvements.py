"""
Property-based and unit tests for XAI and CNN improvements.

Uses hypothesis for property-based testing to verify universal properties
across many random inputs, complemented by unit tests for specific edge cases.
"""
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from explainability.shap_analysis import _get_feature_names as shap_get_feature_names
from explainability.lime_analysis import _get_feature_names as lime_get_feature_names
from config import CICIOMT_FEATURE_NAMES


# =============================================================================
# Property-Based Tests for _get_feature_names
# =============================================================================

# Feature: xai-and-cnn-improvements, Property 3: Consistência de _get_feature_names entre SHAP e LIME
class TestProperty3ConsistencySHAPLIME:
    """
    **Validates: Requirements 3.2, 3.4, 3.6**

    For any positive integer n_features (1-200), _get_feature_names(n_features)
    from SHAP module must return exactly the same list as from LIME module.
    """

    @given(n_features=st.integers(min_value=1, max_value=200))
    @settings(max_examples=100)
    def test_shap_and_lime_return_identical_lists(self, n_features):
        shap_names = shap_get_feature_names(n_features)
        lime_names = lime_get_feature_names(n_features)
        assert shap_names == lime_names, (
            f"SHAP and LIME returned different feature names for n_features={n_features}"
        )


# Feature: xai-and-cnn-improvements, Property 4: _get_feature_names retorna nomes corretos para qualquer dimensão
class TestProperty4CorrectFeatureNames:
    """
    **Validates: Requirements 3.3, 3.5**

    For any positive integer n_features (1-200):
    - If n_features <= 45: returns list of length n_features with first n_features from CICIOMT_FEATURE_NAMES
    - If n_features > 45: returns list of length n_features where first 45 are CICIOMT_FEATURE_NAMES
      and rest follow pattern Feature_{i}
    """

    @given(n_features=st.integers(min_value=1, max_value=200))
    @settings(max_examples=100)
    def test_feature_names_length_and_content(self, n_features):
        names = shap_get_feature_names(n_features)

        # Length must match n_features
        assert len(names) == n_features, (
            f"Expected {n_features} names, got {len(names)}"
        )

        if n_features <= 45:
            # Must be the first n_features from CICIOMT_FEATURE_NAMES
            assert names == CICIOMT_FEATURE_NAMES[:n_features], (
                f"For n_features={n_features} <= 45, names should be "
                f"first {n_features} of CICIOMT_FEATURE_NAMES"
            )
        else:
            # First 45 must be CICIOMT_FEATURE_NAMES
            assert names[:45] == CICIOMT_FEATURE_NAMES, (
                f"For n_features={n_features} > 45, first 45 names should be CICIOMT_FEATURE_NAMES"
            )
            # Remaining must follow Feature_{i} pattern
            for i in range(45, n_features):
                expected = f"Feature_{i}"
                assert names[i] == expected, (
                    f"For n_features={n_features}, names[{i}] should be '{expected}', "
                    f"got '{names[i]}'"
                )


# =============================================================================
# Unit Tests for _get_feature_names edge cases
# =============================================================================

class TestGetFeatureNamesUnitTests:
    """Unit tests for specific edge cases of _get_feature_names."""

    def test_exact_45_features(self):
        """n=45 returns exactly the 45 known CICIOMT feature names."""
        names = shap_get_feature_names(45)
        assert names == CICIOMT_FEATURE_NAMES
        assert len(names) == 45

    def test_fewer_than_45_features(self):
        """n=10 returns the first 10 known feature names."""
        names = shap_get_feature_names(10)
        assert names == CICIOMT_FEATURE_NAMES[:10]
        assert len(names) == 10

    def test_more_than_45_features(self):
        """n=50 returns 45 known names + 5 generic Feature_{i} names."""
        names = shap_get_feature_names(50)
        assert len(names) == 50
        assert names[:45] == CICIOMT_FEATURE_NAMES
        assert names[45:] == [f"Feature_{i}" for i in range(45, 50)]

    def test_single_feature(self):
        """n=1 returns only the first feature name."""
        names = shap_get_feature_names(1)
        assert names == [CICIOMT_FEATURE_NAMES[0]]

    def test_lime_exact_45_features(self):
        """LIME module: n=45 returns exactly the 45 known feature names."""
        names = lime_get_feature_names(45)
        assert names == CICIOMT_FEATURE_NAMES

    def test_lime_more_than_45_features(self):
        """LIME module: n=50 returns 45 known + 5 generic names."""
        names = lime_get_feature_names(50)
        assert len(names) == 50
        assert names[:45] == CICIOMT_FEATURE_NAMES
        assert names[45:] == [f"Feature_{i}" for i in range(45, 50)]


# =============================================================================
# Property-Based Tests for Focal Loss (Property 5)
# =============================================================================

import tensorflow as tf
import numpy as np
from models.cnn_focal import FocalLoss


def _softmax_from_raw(raw: np.ndarray) -> np.ndarray:
    """Convert raw floats to valid softmax probabilities (row-wise)."""
    # Shift for numerical stability
    shifted = raw - raw.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


# Feature: xai-and-cnn-improvements, Property 5: Focal Loss não-negativa e monotonicamente decrescente com confiança
class TestProperty5FocalLossProperties:
    """
    **Validates: Requirements 4.1**

    For any valid probability distribution (softmax output) and true labels,
    Focal Loss must be:
    (a) non-negative, and
    (b) for a fixed label, smaller when the predicted probability for the
        correct class is higher (easy examples get less weight).
    """

    @given(
        batch_size=st.integers(min_value=1, max_value=16),
        n_classes=st.integers(min_value=2, max_value=10),
        gamma=st.sampled_from([0.0, 0.5, 1.0, 2.0, 5.0]),
        alpha=st.sampled_from([0.1, 0.25, 0.5, 1.0]),
        data=st.data(),
    )
    @settings(max_examples=100)
    def test_focal_loss_non_negative(self, batch_size, n_classes, gamma, alpha, data):
        """Focal Loss must be >= 0 for any valid softmax input and labels."""
        # Generate random raw logits and convert to softmax
        raw = data.draw(
            st.lists(
                st.lists(
                    st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
                    min_size=n_classes, max_size=n_classes,
                ),
                min_size=batch_size, max_size=batch_size,
            )
        )
        y_pred_np = _softmax_from_raw(np.array(raw, dtype=np.float32))

        # Generate random labels in [0, n_classes)
        labels = data.draw(
            st.lists(
                st.integers(min_value=0, max_value=n_classes - 1),
                min_size=batch_size, max_size=batch_size,
            )
        )

        y_true = tf.constant(labels, dtype=tf.int32)
        y_pred = tf.constant(y_pred_np, dtype=tf.float32)

        focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        loss_values = focal_loss(y_true, y_pred).numpy()

        assert np.all(loss_values >= -1e-7), (
            f"Focal Loss has negative values: {loss_values.min():.8f} "
            f"(gamma={gamma}, alpha={alpha})"
        )

    @given(
        n_classes=st.integers(min_value=2, max_value=10),
        true_label=st.data(),
        gamma=st.sampled_from([0.5, 1.0, 2.0, 5.0]),
        alpha=st.sampled_from([0.1, 0.25, 0.5, 1.0]),
    )
    @settings(max_examples=100)
    def test_focal_loss_monotonically_decreasing_with_confidence(
        self, n_classes, true_label, gamma, alpha
    ):
        """
        For a fixed label, Focal Loss must be smaller when the predicted
        probability for the correct class is higher.
        """
        label = true_label.draw(st.integers(min_value=0, max_value=n_classes - 1))

        # Create two predictions: low confidence and high confidence for the correct class
        # Low confidence: probability for correct class is low_p
        # High confidence: probability for correct class is high_p > low_p
        low_p = true_label.draw(
            st.floats(min_value=0.05, max_value=0.45, allow_nan=False, allow_infinity=False)
        )
        high_p = true_label.draw(
            st.floats(min_value=low_p + 0.05, max_value=0.95, allow_nan=False, allow_infinity=False)
        )

        def _make_pred(p_correct, n_cls, correct_idx):
            """Create a probability vector with p_correct for the correct class."""
            remaining = 1.0 - p_correct
            probs = np.full(n_cls, remaining / max(n_cls - 1, 1), dtype=np.float32)
            probs[correct_idx] = p_correct
            # Ensure sums to 1
            probs = probs / probs.sum()
            return probs

        pred_low = _make_pred(low_p, n_classes, label)
        pred_high = _make_pred(high_p, n_classes, label)

        y_true = tf.constant([label], dtype=tf.int32)
        y_pred_low = tf.constant([pred_low], dtype=tf.float32)
        y_pred_high = tf.constant([pred_high], dtype=tf.float32)

        focal_loss = FocalLoss(gamma=gamma, alpha=alpha)
        loss_low = focal_loss(y_true, y_pred_low).numpy()[0]
        loss_high = focal_loss(y_true, y_pred_high).numpy()[0]

        assert loss_high <= loss_low + 1e-6, (
            f"Focal Loss should decrease with higher confidence. "
            f"loss(p={low_p:.4f})={loss_low:.6f} < loss(p={high_p:.4f})={loss_high:.6f} "
            f"(gamma={gamma}, alpha={alpha}, label={label}, n_classes={n_classes})"
        )


# =============================================================================
# Property 6: CNN Focal retorna dicionário com todas as chaves obrigatórias
# =============================================================================

from models.cnn_focal import train_focal_cnn
import config


# Feature: xai-and-cnn-improvements, Property 6: CNN Focal retorna dicionário com todas as chaves obrigatórias
class TestProperty6FocalCNNReturnDict:
    """
    **Validates: Requirements 4.7**

    For any valid execution of train_focal_cnn, the returned dict must contain
    all keys: y_pred, y_pred_probs, train_time, pred_time, model, history,
    accuracy, f1_macro, f1_weighted, kappa.

    Uses small synthetic data (100 samples, 10 features, 3 classes) and 1 epoch.
    """

    REQUIRED_KEYS = {
        'y_pred', 'y_pred_probs', 'train_time', 'pred_time',
        'model', 'history', 'accuracy', 'f1_macro', 'f1_weighted', 'kappa',
    }

    def test_train_focal_cnn_returns_all_required_keys(self, monkeypatch, tmp_path):
        """train_focal_cnn must return a dict with all required keys."""
        np.random.seed(42)

        n_samples_train = 80
        n_samples_val = 10
        n_samples_test = 10
        n_features = 10
        n_classes = 3

        X_train = np.random.randn(n_samples_train, n_features, 1).astype(np.float32)
        X_val = np.random.randn(n_samples_val, n_features, 1).astype(np.float32)
        X_test = np.random.randn(n_samples_test, n_features, 1).astype(np.float32)
        y_train = np.random.randint(0, n_classes, size=n_samples_train)
        y_val = np.random.randint(0, n_classes, size=n_samples_val)
        y_test = np.random.randint(0, n_classes, size=n_samples_test)

        # Override CNN_FOCAL_PARAMS to use 1 epoch and disable mixed precision for test speed
        test_params = dict(CNN_FOCAL_PARAMS)
        test_params['epochs'] = 1
        test_params['mixed_precision'] = False
        test_params['early_stop_patience'] = 1
        monkeypatch.setattr(config, 'CNN_FOCAL_PARAMS', test_params)
        # Also patch in the cnn_focal module which imports CNN_FOCAL_PARAMS at module level
        import models.cnn_focal as cnn_focal_mod
        monkeypatch.setattr(cnn_focal_mod, 'CNN_FOCAL_PARAMS', test_params)

        # Override MODELS_DIR to use tmp_path so we don't pollute real results
        monkeypatch.setattr(cnn_focal_mod, 'MODELS_DIR', str(tmp_path))

        result = train_focal_cnn(
            X_train, y_train, X_val, y_val, X_test, y_test,
            n_classes=n_classes, scenario='test',
        )

        # Check all required keys exist
        missing = self.REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing keys in train_focal_cnn result: {missing}"

        # Check types of key fields
        assert isinstance(result['y_pred'], np.ndarray), "y_pred should be ndarray"
        assert isinstance(result['y_pred_probs'], np.ndarray), "y_pred_probs should be ndarray"
        assert isinstance(result['train_time'], float), "train_time should be float"
        assert isinstance(result['pred_time'], float), "pred_time should be float"
        assert isinstance(result['model'], tf.keras.Model), "model should be tf.keras.Model"
        assert isinstance(result['history'], dict), "history should be dict"
        assert isinstance(result['accuracy'], float), "accuracy should be float"
        assert isinstance(result['f1_macro'], (float, np.floating)), "f1_macro should be float"
        assert isinstance(result['f1_weighted'], (float, np.floating)), "f1_weighted should be float"
        assert isinstance(result['kappa'], float), "kappa should be float"

        # Check shapes
        assert result['y_pred'].shape == (n_samples_test,), (
            f"y_pred shape should be ({n_samples_test},), got {result['y_pred'].shape}"
        )
        assert result['y_pred_probs'].shape == (n_samples_test, n_classes), (
            f"y_pred_probs shape should be ({n_samples_test}, {n_classes}), "
            f"got {result['y_pred_probs'].shape}"
        )

        # Check value ranges
        assert 0.0 <= result['accuracy'] <= 1.0, "accuracy should be in [0, 1]"
        assert result['train_time'] >= 0.0, "train_time should be non-negative"
        assert result['pred_time'] >= 0.0, "pred_time should be non-negative"


# =============================================================================
# Property-Based Tests for Light-CNN V2
# =============================================================================

from models.cnn_light_v2 import train_light_cnn_v2
import models.cnn_light_v2 as cnn_light_v2_mod

# Feature: xai-and-cnn-improvements, Property 7: Light-CNN V2 retorna dicionário com todas as chaves obrigatórias


class TestProperty7LightCNNV2ReturnDict:
    """
    **Validates: Requirements 5.6**

    For any valid execution of train_light_cnn_v2, the returned dict must contain
    all keys: y_pred, y_pred_probs, train_time, pred_time, model, history,
    accuracy, f1_macro, f1_weighted, kappa, trainable_params, total_params.

    Uses small synthetic data (100 samples, 10 features, 3 classes) and 1 epoch.
    """

    REQUIRED_KEYS = {
        'y_pred', 'y_pred_probs', 'train_time', 'pred_time',
        'model', 'history', 'accuracy', 'f1_macro', 'f1_weighted', 'kappa',
        'trainable_params', 'total_params',
    }

    def test_train_light_cnn_v2_returns_all_required_keys(self, monkeypatch, tmp_path):
        """train_light_cnn_v2 must return a dict with all required keys."""
        np.random.seed(42)

        n_samples_train = 80
        n_samples_val = 10
        n_samples_test = 10
        n_features = 10
        n_classes = 3

        X_train = np.random.randn(n_samples_train, n_features, 1).astype(np.float32)
        X_val = np.random.randn(n_samples_val, n_features, 1).astype(np.float32)
        X_test = np.random.randn(n_samples_test, n_features, 1).astype(np.float32)
        y_train = np.random.randint(0, n_classes, size=n_samples_train)
        y_val = np.random.randint(0, n_classes, size=n_samples_val)
        y_test = np.random.randint(0, n_classes, size=n_samples_test)

        # Override LIGHT_CNN_V2_PARAMS to use 1 epoch and disable mixed precision
        test_params = dict(config.LIGHT_CNN_V2_PARAMS)
        test_params['epochs'] = 1
        test_params['mixed_precision'] = False
        test_params['early_stop_patience'] = 1
        monkeypatch.setattr(config, 'LIGHT_CNN_V2_PARAMS', test_params)
        monkeypatch.setattr(cnn_light_v2_mod, 'LIGHT_CNN_V2_PARAMS', test_params)

        # Override MODELS_DIR to use tmp_path so we don't pollute real results
        monkeypatch.setattr(cnn_light_v2_mod, 'MODELS_DIR', str(tmp_path))

        result = train_light_cnn_v2(
            X_train, y_train, X_val, y_val, X_test, y_test,
            n_classes=n_classes, scenario='test',
        )

        # Check all required keys exist
        missing = self.REQUIRED_KEYS - set(result.keys())
        assert not missing, f"Missing keys in train_light_cnn_v2 result: {missing}"

        # Check types of key fields
        assert isinstance(result['y_pred'], np.ndarray), "y_pred should be ndarray"
        assert isinstance(result['y_pred_probs'], np.ndarray), "y_pred_probs should be ndarray"
        assert isinstance(result['train_time'], float), "train_time should be float"
        assert isinstance(result['pred_time'], float), "pred_time should be float"
        assert isinstance(result['model'], tf.keras.Model), "model should be tf.keras.Model"
        assert isinstance(result['history'], dict), "history should be dict"
        assert isinstance(result['accuracy'], float), "accuracy should be float"
        assert isinstance(result['f1_macro'], (float, np.floating)), "f1_macro should be float"
        assert isinstance(result['f1_weighted'], (float, np.floating)), "f1_weighted should be float"
        assert isinstance(result['kappa'], float), "kappa should be float"
        assert isinstance(result['trainable_params'], int), "trainable_params should be int"
        assert isinstance(result['total_params'], int), "total_params should be int"

        # Check shapes
        assert result['y_pred'].shape == (n_samples_test,), (
            f"y_pred shape should be ({n_samples_test},), got {result['y_pred'].shape}"
        )
        assert result['y_pred_probs'].shape == (n_samples_test, n_classes), (
            f"y_pred_probs shape should be ({n_samples_test}, {n_classes}), "
            f"got {result['y_pred_probs'].shape}"
        )

        # Check value ranges
        assert 0.0 <= result['accuracy'] <= 1.0, "accuracy should be in [0, 1]"
        assert result['train_time'] >= 0.0, "train_time should be non-negative"
        assert result['pred_time'] >= 0.0, "pred_time should be non-negative"
        assert result['trainable_params'] > 0, "trainable_params should be positive"
        assert result['total_params'] >= result['trainable_params'], (
            "total_params should be >= trainable_params"
        )


# Feature: xai-and-cnn-improvements, Property 8: epochs_override sobrescreve épocas padrão


class TestProperty8EpochsOverride:
    """
    **Validates: Requirements 5.8**

    For any positive integer epochs_override, when passed to train_light_cnn_v2,
    the number of max epochs configured in training must equal epochs_override.
    Verified by checking that history has at most epochs_override entries.

    Uses small synthetic data and epochs_override values like 1, 2, 3.
    """

    @given(epochs_override=st.integers(min_value=1, max_value=3))
    @settings(max_examples=3, deadline=120_000)
    def test_epochs_override_controls_max_epochs(self, epochs_override, monkeypatch, tmp_path):
        """history length must be <= epochs_override when epochs_override is set."""
        np.random.seed(42)

        n_samples_train = 80
        n_samples_val = 10
        n_samples_test = 10
        n_features = 10
        n_classes = 3

        X_train = np.random.randn(n_samples_train, n_features, 1).astype(np.float32)
        X_val = np.random.randn(n_samples_val, n_features, 1).astype(np.float32)
        X_test = np.random.randn(n_samples_test, n_features, 1).astype(np.float32)
        y_train = np.random.randint(0, n_classes, size=n_samples_train)
        y_val = np.random.randint(0, n_classes, size=n_samples_val)
        y_test = np.random.randint(0, n_classes, size=n_samples_test)

        # Override params: disable mixed precision, keep high patience so early stop
        # doesn't interfere with epoch count verification
        test_params = dict(config.LIGHT_CNN_V2_PARAMS)
        test_params['mixed_precision'] = False
        test_params['early_stop_patience'] = 999  # effectively disable early stopping
        monkeypatch.setattr(config, 'LIGHT_CNN_V2_PARAMS', test_params)
        monkeypatch.setattr(cnn_light_v2_mod, 'LIGHT_CNN_V2_PARAMS', test_params)

        # Override MODELS_DIR to use tmp_path
        monkeypatch.setattr(cnn_light_v2_mod, 'MODELS_DIR', str(tmp_path))

        result = train_light_cnn_v2(
            X_train, y_train, X_val, y_val, X_test, y_test,
            n_classes=n_classes, scenario='test',
            epochs_override=epochs_override,
        )

        # history['loss'] length == number of epochs actually trained
        actual_epochs = len(result['history']['loss'])
        assert actual_epochs <= epochs_override, (
            f"Expected at most {epochs_override} epochs, but trained {actual_epochs}"
        )
        # With early stopping disabled (patience=999), should train exactly epochs_override
        assert actual_epochs == epochs_override, (
            f"Expected exactly {epochs_override} epochs (early stop disabled), "
            f"but trained {actual_epochs}"
        )


# =============================================================================
# Property 1: CNN predict_fn preserva forma e probabilidades
# =============================================================================

from models.cnn_standard import _build_standard_cnn

# Feature: xai-and-cnn-improvements, Property 1: CNN predict_fn preserva forma e probabilidades


class TestProperty1CNNPredictFn:
    """
    **Validates: Requirements 1.3**

    For any 2D array with shape (batch, 45), the custom predict_fn must produce
    a probability array with shape (batch, n_classes) where each row sums to
    approximately 1.0 and all values are in [0, 1].
    """

    N_FEATURES = 10
    N_CLASSES = 3

    @given(batch_size=st.integers(min_value=1, max_value=16))
    @settings(max_examples=10, deadline=120_000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_predict_fn_preserves_shape_and_probabilities(self, batch_size, monkeypatch):
        """predict_fn(2D) must return (batch, n_classes) with valid probabilities."""
        # Disable mixed_precision for test stability
        monkeypatch.setattr(config, 'CNN_PARAMS', {
            **config.CNN_PARAMS, 'mixed_precision': False,
        })
        import models.cnn_standard as cnn_std_mod
        monkeypatch.setattr(cnn_std_mod, 'CNN_PARAMS', {
            **config.CNN_PARAMS, 'mixed_precision': False,
        })

        n_features = self.N_FEATURES
        n_classes = self.N_CLASSES

        # Build a small CNN model
        model = _build_standard_cnn(n_features, n_classes)

        # Define predict_fn exactly as in the standalone script
        def predict_fn(X):
            """Reshape 2D (batch, features) -> 3D (batch, features, 1) for CNN."""
            return model.predict(X[..., np.newaxis], verbose=0)

        # Generate random 2D input (batch, n_features)
        X_2d = np.random.randn(batch_size, n_features).astype(np.float32)

        # Call predict_fn
        probs = predict_fn(X_2d)

        # Verify output shape
        assert probs.shape == (batch_size, n_classes), (
            f"Expected shape ({batch_size}, {n_classes}), got {probs.shape}"
        )

        # Verify all values in [0, 1]
        assert np.all(probs >= 0.0), (
            f"Found negative probability: min={probs.min():.8f}"
        )
        assert np.all(probs <= 1.0), (
            f"Found probability > 1: max={probs.max():.8f}"
        )

        # Verify each row sums to approximately 1.0
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5, err_msg=(
            f"Row sums should be ~1.0, got {row_sums}"
        ))


# =============================================================================
# Property 2: LightGBM predict_proba compatível com LIME
# =============================================================================

import lightgbm as lgb

# Feature: xai-and-cnn-improvements, Property 2: LightGBM predict_proba compatível com LIME


class TestProperty2LightGBMPredictProba:
    """
    **Validates: Requirements 2.2**

    For any 2D array with shape (batch, 45) and a trained LightGBM model,
    model.predict_proba(X) must return array with shape (batch, n_classes)
    where each row sums to approximately 1.0 and all values are in [0, 1].
    """

    N_FEATURES = 10
    N_CLASSES = 3

    @given(batch_size=st.integers(min_value=1, max_value=16))
    @settings(max_examples=10, deadline=120_000)
    def test_predict_proba_compatible_with_lime(self, batch_size):
        """LightGBM predict_proba must return valid probabilities for LIME."""
        n_features = self.N_FEATURES
        n_classes = self.N_CLASSES

        # Train a small LightGBM on synthetic data
        np.random.seed(42)
        X_train = np.random.randn(100, n_features).astype(np.float32)
        y_train = np.random.randint(0, n_classes, size=100)

        model = lgb.LGBMClassifier(
            n_estimators=5,
            num_leaves=4,
            verbose=-1,
        )
        model.fit(X_train, y_train)

        # Generate random 2D input (batch, n_features)
        X_test = np.random.randn(batch_size, n_features).astype(np.float32)

        # Call predict_proba (as LIME would)
        probs = model.predict_proba(X_test)

        # Verify output shape
        assert probs.shape == (batch_size, n_classes), (
            f"Expected shape ({batch_size}, {n_classes}), got {probs.shape}"
        )

        # Verify all values in [0, 1]
        assert np.all(probs >= 0.0), (
            f"Found negative probability: min={probs.min():.8f}"
        )
        assert np.all(probs <= 1.0), (
            f"Found probability > 1: max={probs.max():.8f}"
        )

        # Verify each row sums to approximately 1.0
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5, err_msg=(
            f"Row sums should be ~1.0, got {row_sums}"
        ))
