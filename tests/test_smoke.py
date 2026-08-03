"""
test_smoke.py
=============
Smoke test: verifica que o pipeline completo funciona end-to-end
com dados sintéticos (sem depender do CICIoMT2024).

Gera 500 linhas com o mesmo schema das 45 features numéricas
e executa: pré-processamento → RF → LightGBM default → métricas.

Não usa GPU. Deve completar em < 60 segundos em CPU.

Uso:
    pytest tests/test_smoke.py -v
"""
import os
import sys
import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import CICIOMT_FEATURE_NAMES, SEED


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_data():
    """Gera 500 amostras sintéticas com o schema do CICIoMT2024."""
    rng = np.random.default_rng(SEED)
    n = 500
    n_feat = len(CICIOMT_FEATURE_NAMES)

    X = rng.standard_normal((n, n_feat)).astype("float32")
    # 4 classes: 0=Benign, 1=DDoS, 2=Recon, 3=Spoofing
    y = rng.integers(0, 4, size=n)

    return X, y


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_feature_count():
    """O dataset tem exatamente 45 features."""
    assert len(CICIOMT_FEATURE_NAMES) == 45, (
        f"Expected 45 features, got {len(CICIOMT_FEATURE_NAMES)}"
    )


def test_feature_names_no_duplicates():
    """Nomes de features sem duplicatas."""
    assert len(CICIOMT_FEATURE_NAMES) == len(set(CICIOMT_FEATURE_NAMES))


def test_preprocessing_no_leakage(synthetic_data):
    """StandardScaler fitado só no treino; teste recebe transform sem fit."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    X, y = synthetic_data
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=SEED)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    assert X_train_s.shape[1] == 45
    assert X_test_s.shape[1] == 45
    # Médias do treino devem ser próximas de zero
    assert abs(X_train_s.mean()) < 0.1


def test_random_forest_runs(synthetic_data):
    """RF treina e produz predições para todas as amostras de teste."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    X, y = synthetic_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    rf = RandomForestClassifier(n_estimators=10, random_state=SEED, n_jobs=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    assert len(y_pred) == len(y_test)
    assert set(y_pred).issubset({0, 1, 2, 3})


def test_lightgbm_default_runs(synthetic_data):
    """LightGBM default treina e prediz sem erros."""
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    X, y = synthetic_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    clf = lgb.LGBMClassifier(
        n_estimators=10, learning_rate=0.1, num_leaves=31,
        random_state=SEED, n_jobs=1, verbose=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    assert len(y_pred) == len(y_test)


def test_metrics_computation(synthetic_data):
    """compute_all_metrics retorna dict com as chaves esperadas."""
    from evaluation.metrics import compute_all_metrics
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import train_test_split

    X, y = synthetic_data
    _, _, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )

    dummy = DummyClassifier(strategy="most_frequent", random_state=SEED)
    dummy.fit(X[:400], y_train)
    y_pred = dummy.predict(X[400:])
    # Ensure y_pred and y_test same length
    n = min(len(y_pred), len(y_test))
    y_pred, y_test = y_pred[:n], y_test[:n]

    metrics = compute_all_metrics(y_test, y_pred)
    for key in ("accuracy", "f1_macro", "kappa"):
        assert key in metrics, f"Missing key: {key}"
        assert 0.0 <= metrics[key] <= 1.0, f"Out-of-range: {key}={metrics[key]}"


def test_binary_split_benign_train():
    """
    Verifica que o split benigno para o autoencoder é reprodutível
    com a mesma seed, independente de rótulos de ataque.
    """
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(SEED)
    X_benign = rng.standard_normal((1000, 45)).astype("float32")

    split1_train, split1_val = train_test_split(X_benign, test_size=0.2,
                                                 random_state=SEED)
    split2_train, split2_val = train_test_split(X_benign, test_size=0.2,
                                                 random_state=SEED)

    np.testing.assert_array_equal(split1_train, split2_train)
    np.testing.assert_array_equal(split1_val, split2_val)
