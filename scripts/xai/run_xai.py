"""
Standalone XAI runner — regenera SHAP + LIME sem re-treinar modelos.

Carrega modelos salvos em results/models/ e dados do CICIoMT2024,
depois executa SHAP e LIME para cada cenário.

Arquivos são salvos em results/xai/YYYYMMDD/ (data de execução).

Uso:
    python run_xai.py                        # todos os cenários
    python run_xai.py --scenarios 2          # só 2-class
    python run_xai.py --scenarios 2 6        # 2 e 6-class
    python run_xai.py --no-gpu               # força CPU (sem CUDA)
    python run_xai.py --only-trees           # só RF e LightGBM (sem CNN)
"""
import os
import sys
import time
import logging
import argparse
import warnings
from datetime import datetime
import numpy as np
import joblib

# Add project root to sys.path so imports work from any directory
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*NumPy global RNG.*")

from config import MODELS_DIR, DATA_DIR, XAI_DIR, SEED
from data_loader import load_data
from explainability.shap_analysis import run_shap_analysis
from explainability.lime_analysis import run_lime_analysis

log = logging.getLogger()


def setup_logging():
    fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=fmt, datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout)])


def load_saved_model(name, scenario):
    """Load a saved model from results/models/."""
    ext_map = {
        'rf': '.pkl', 'lr': '.pkl',
        'lgbm_original': '.pkl', 'lgbm_tuned': '.pkl',
        'cnn_standard': '.keras', 'cnn_light': '.keras',
        'autoencoder': '.keras',
    }
    ext = ext_map.get(name, '.pkl')
    path = os.path.join(MODELS_DIR, f"{name}_{scenario}{ext}")
    if not os.path.exists(path):
        log.warning(f"  Model not found: {path}")
        return None
    if ext == '.pkl':
        return joblib.load(path)
    else:
        import tensorflow as tf
        return tf.keras.models.load_model(path, compile=False)


def run_xai_for_scenario(class_config, output_dir, include_cnn=True):
    """Run SHAP + LIME for one scenario."""
    scenario = f"{class_config}-class"
    log.info(f"\n{'='*60}")
    log.info(f"  XAI — {scenario}")
    log.info(f"  Output: {output_dir}")
    log.info(f"{'='*60}")

    t0 = time.time()

    # Load data
    data = load_data(DATA_DIR, class_config)
    X_train_flat = data['X_train_flat']
    X_test_flat = data['X_test_flat']
    X_train_3d = data['X_train']
    X_test_3d = data['X_test']
    y_test = data['y_test']
    class_names = data['class_names']

    # --- Random Forest ---
    rf = load_saved_model('rf', scenario)
    if rf is not None:
        rf_pred = rf.predict(X_test_flat)
        run_shap_analysis(rf, X_train_flat, X_test_flat, y_test,
                          class_names, 'Random Forest', scenario,
                          model_type='tree', output_dir=output_dir)
        run_lime_analysis(rf, X_train_flat, X_test_flat, y_test, rf_pred,
                          class_names, 'Random Forest', scenario,
                          output_dir=output_dir)

    # --- LightGBM (Tuned) ---
    lgbm = load_saved_model('lgbm_tuned', scenario)
    if lgbm is not None:
        lgbm_pred = lgbm.predict(X_test_flat)
        run_shap_analysis(lgbm, X_train_flat, X_test_flat, y_test,
                          class_names, 'LightGBM (Tuned)', scenario,
                          model_type='tree', output_dir=output_dir)
        run_lime_analysis(lgbm, X_train_flat, X_test_flat, y_test, lgbm_pred,
                          class_names, 'LightGBM (Tuned)', scenario,
                          output_dir=output_dir)

    # --- 1D-CNN ---
    if include_cnn:
        cnn = load_saved_model('cnn_standard', scenario)
        if cnn is not None:
            cnn_pred_proba = cnn.predict(X_test_3d, verbose=0)
            cnn_pred = np.argmax(cnn_pred_proba, axis=1)

            run_shap_analysis(cnn, X_train_3d, X_test_3d, y_test,
                              class_names, '1D-CNN', scenario,
                              model_type='neural', output_dir=output_dir)

            def cnn_predict_fn(X):
                if X.ndim == 2:
                    X = X[..., np.newaxis]
                return cnn.predict(X, verbose=0)

            run_lime_analysis(cnn, X_train_flat, X_test_flat, y_test, cnn_pred,
                              class_names, '1D-CNN', scenario,
                              predict_fn=cnn_predict_fn, output_dir=output_dir)

    elapsed = time.time() - t0
    log.info(f"\n  XAI {scenario} concluído em {elapsed:.1f}s ({elapsed/60:.1f} min)")


def main():
    parser = argparse.ArgumentParser(description="Standalone XAI (SHAP + LIME)")
    parser.add_argument('--scenarios', nargs='+', type=int, default=[2, 6, 19],
                        help='Cenários a executar (default: 2 6 19)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Forçar CPU (desabilita CUDA)')
    parser.add_argument('--only-trees', action='store_true',
                        help='Só RF e LightGBM (pula CNN)')
    args = parser.parse_args()

    setup_logging()

    if args.no_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        log.info("GPU desabilitada (--no-gpu)")

    # Subpasta com data de execução: results/xai/20260307/
    date_str = datetime.now().strftime('%Y%m%d')
    output_dir = os.path.join(XAI_DIR, date_str)
    os.makedirs(output_dir, exist_ok=True)

    log.info(f"Cenários: {args.scenarios}")
    log.info(f"CNN: {'não' if args.only_trees else 'sim'}")
    log.info(f"Output: {output_dir}")

    t_total = time.time()
    for sc in args.scenarios:
        run_xai_for_scenario(sc, output_dir, include_cnn=not args.only_trees)

    elapsed = time.time() - t_total
    log.info(f"\nTudo concluído em {elapsed:.1f}s ({elapsed/60:.1f} min)")
    log.info(f"Arquivos salvos em: {output_dir}")


if __name__ == '__main__':
    main()
