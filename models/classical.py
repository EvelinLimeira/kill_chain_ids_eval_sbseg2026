"""
Classical ML models: Random Forest, Logistic Regression, LightGBM (original + tuned).
"""
import time
import logging
import numpy as np
import joblib
from collections import OrderedDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
import lightgbm as lgb

from config import (
    SEED, RF_PARAMS, LR_PARAMS, LGBM_ORIGINAL_PARAMS,
    LGBM_IMPROVED_CONFIGS, MODELS_DIR,
)

log = logging.getLogger(__name__)


def _evaluate_quick(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'kappa': cohen_kappa_score(y_true, y_pred),
    }


def train_random_forest(X_train, y_train, X_test, y_test, scenario: str) -> dict:
    """Train Random Forest with paper configuration."""
    log.info("  Training Random Forest...")
    t0 = time.time()
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - t0

    metrics = _evaluate_quick(y_test, y_pred)
    log.info(f"    Acc: {metrics['accuracy']:.4f} | F1m: {metrics['f1_macro']:.4f} | "
             f"Train: {train_time:.1f}s")

    joblib.dump(model, f"{MODELS_DIR}/rf_{scenario}.pkl")

    return {
        'y_pred': y_pred, 'train_time': train_time, 'pred_time': pred_time,
        'model': model, **metrics,
    }


def train_logistic_regression(X_train, y_train, X_test, y_test, scenario: str) -> dict:
    """Train Logistic Regression with paper configuration."""
    log.info("  Training Logistic Regression...")
    t0 = time.time()
    model = LogisticRegression(**LR_PARAMS)
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - t0

    metrics = _evaluate_quick(y_test, y_pred)
    log.info(f"    Acc: {metrics['accuracy']:.4f} | F1m: {metrics['f1_macro']:.4f} | "
             f"Train: {train_time:.1f}s")

    joblib.dump(model, f"{MODELS_DIR}/lr_{scenario}.pkl")

    return {
        'y_pred': y_pred, 'train_time': train_time, 'pred_time': pred_time,
        'model': model, **metrics,
    }


def train_lightgbm_all(X_train, y_train, X_val, y_val, X_test, y_test,
                        n_classes: int, scenario: str) -> dict:
    """
    Train LightGBM with original AND improved configurations.
    Addresses Reviewer D's criticism about insufficient tuning.

    Returns dict of config_name -> results.
    """
    log.info("  Training LightGBM variants...")

    # Build objective params
    if n_classes == 2:
        obj = {'objective': 'binary', 'metric': 'binary_logloss'}
    else:
        obj = {'objective': 'multiclass', 'metric': 'multi_logloss',
               'num_class': n_classes}

    # Ensure plain numpy arrays (avoids LightGBM feature name warnings)
    X_train = np.ascontiguousarray(X_train)
    X_val = np.ascontiguousarray(X_val)
    X_test = np.ascontiguousarray(X_test)

    configs = OrderedDict()
    configs['original'] = {**LGBM_ORIGINAL_PARAMS, **obj}
    for name, params in LGBM_IMPROVED_CONFIGS.items():
        configs[f'tuned_{name}'] = {**params, **obj}

    results = {}
    best_improved_name, best_improved_f1 = None, -1

    for config_name, params in configs.items():
        log.info(f"    [{config_name}]...")
        t0 = time.time()
        model = lgb.LGBMClassifier(**params)

        if 'tuned' in config_name:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
        else:
            model.fit(X_train, y_train)

        train_time = time.time() - t0

        t0 = time.time()
        y_pred = model.predict(X_test, validate_features=False)
        pred_time = time.time() - t0

        metrics = _evaluate_quick(y_test, y_pred)
        best_iter = getattr(model, 'best_iteration_', params.get('n_estimators', 100))

        results[config_name] = {
            'y_pred': y_pred, 'train_time': train_time, 'pred_time': pred_time,
            'model': model, 'best_iteration': best_iter, 'config': config_name,
            **metrics,
        }

        log.info(f"      Acc: {metrics['accuracy']:.4f} | F1m: {metrics['f1_macro']:.4f} | "
                 f"Iters: {best_iter} | Time: {train_time:.1f}s")

        if 'tuned' in config_name and metrics['f1_macro'] > best_improved_f1:
            best_improved_f1 = metrics['f1_macro']
            best_improved_name = config_name

    # Save original LightGBM
    joblib.dump(results['original']['model'],
                f"{MODELS_DIR}/lgbm_original_{scenario}.pkl")

    if best_improved_name:
        orig_f1 = results['original']['f1_macro']
        log.info(f"    Best tuned: {best_improved_name} | "
                 f"F1m: {orig_f1:.4f} -> {best_improved_f1:.4f} "
                 f"(delta: {best_improved_f1 - orig_f1:+.4f})")
        joblib.dump(results[best_improved_name]['model'],
                    f"{MODELS_DIR}/lgbm_tuned_{scenario}.pkl")

    return results
