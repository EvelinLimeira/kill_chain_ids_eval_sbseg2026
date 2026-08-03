"""
Statistical validation tests for model comparison.

Implements:
  - Bootstrap Confidence Intervals (Efron & Tibshirani, 1993)
  - McNemar's Test (McNemar, 1947)
  - Friedman Test + Nemenyi post-hoc (Demsar, JMLR 2006)
  - Cohen's Kappa (Landis & Koch, 1977)
"""
import json
import logging
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import friedmanchisquare
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score,
)

from config import SEED, N_BOOTSTRAP, CI_LEVEL, TABLES_DIR, FIGURES_DIR

log = logging.getLogger(__name__)


# =============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# =============================================================================

def _compute_metric(y_true, y_pred, metric):
    funcs = {
        'accuracy': lambda yt, yp: accuracy_score(yt, yp),
        'precision_macro': lambda yt, yp: precision_score(yt, yp, average='macro', zero_division=0),
        'recall_macro': lambda yt, yp: recall_score(yt, yp, average='macro', zero_division=0),
        'f1_macro': lambda yt, yp: f1_score(yt, yp, average='macro', zero_division=0),
        'f1_weighted': lambda yt, yp: f1_score(yt, yp, average='weighted', zero_division=0),
        'kappa': lambda yt, yp: cohen_kappa_score(yt, yp),
    }
    return funcs[metric](y_true, y_pred)


def bootstrap_ci(y_true, y_pred, metric, n_boot=N_BOOTSTRAP, ci=CI_LEVEL):
    """Compute bootstrap confidence interval. No retraining needed."""
    n = len(y_true)
    rng = np.random.RandomState(SEED)
    scores = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        if len(np.unique(yt)) < 2:
            scores[i] = np.nan
            continue
        scores[i] = _compute_metric(yt, yp, metric)

    scores = scores[~np.isnan(scores)]
    alpha = (1 - ci) / 2
    mean = np.mean(scores)
    lo = np.percentile(scores, alpha * 100)
    hi = np.percentile(scores, (1 - alpha) * 100)
    return mean, lo, hi


def run_bootstrap_analysis(y_test, predictions, scenario):
    """Bootstrap CI for all models and metrics."""
    log.info(f"  Bootstrap 95% CI ({scenario})...")
    metrics = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro', 'kappa']
    rows = []

    for model_name, data in predictions.items():
        row = {'Model': model_name}
        for m in metrics:
            mean, lo, hi = bootstrap_ci(y_test, data['y_pred'], m)
            row[m] = mean
            row[f'{m}_ci'] = f"{mean:.4f} [{lo:.4f}, {hi:.4f}]"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"{TABLES_DIR}/bootstrap_ci_{scenario}.csv", index=False)

    log.info(f"    {'Model':<25s} {'Accuracy (95% CI)':<30s} {'F1-Macro (95% CI)':<30s}")
    for _, r in df.iterrows():
        log.info(f"    {r['Model']:<25s} {r['accuracy_ci']:<30s} {r['f1_macro_ci']:<30s}")

    return df


# =============================================================================
# McNEMAR'S TEST
# =============================================================================

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    """McNemar's test with continuity correction."""
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)
    n10 = np.sum(correct_a & ~correct_b)
    n01 = np.sum(~correct_a & correct_b)

    if n01 + n10 == 0:
        return 0.0, 1.0

    chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, p_value


def run_mcnemar_analysis(y_test, predictions, scenario):
    """McNemar's test for all model pairs."""
    log.info(f"  McNemar's test ({scenario})...")
    names = list(predictions.keys())
    rows = []

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            chi2, p = mcnemar_test(
                y_test, predictions[names[i]]['y_pred'],
                predictions[names[j]]['y_pred'])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            rows.append({
                'Model_A': names[i], 'Model_B': names[j],
                'Chi2': chi2, 'p_value': p, 'Significance': sig,
            })
            log.info(f"    {names[i]:25s} vs {names[j]:25s} | "
                     f"chi2={chi2:10.2f} | p={p:.2e} | {sig}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{TABLES_DIR}/mcnemar_{scenario}.csv", index=False)
    return df


# =============================================================================
# COHEN'S KAPPA
# =============================================================================

def run_kappa_analysis(y_test, predictions, scenario):
    """Cohen's Kappa with Landis & Koch interpretation."""
    log.info(f"  Cohen's Kappa ({scenario})...")
    rows = []

    for name, data in predictions.items():
        acc = accuracy_score(y_test, data['y_pred'])
        kappa = cohen_kappa_score(y_test, data['y_pred'])
        gap = acc - kappa

        if kappa >= 0.81:     interp = "Almost perfect"
        elif kappa >= 0.61:   interp = "Substantial"
        elif kappa >= 0.41:   interp = "Moderate"
        elif kappa >= 0.21:   interp = "Fair"
        else:                 interp = "Slight/Poor"

        rows.append({'Model': name, 'Accuracy': acc, 'Kappa': kappa,
                     'Gap': gap, 'Interpretation': interp})
        log.info(f"    {name:<25s} | Acc: {acc:.4f} | k: {kappa:.4f} | {interp}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{TABLES_DIR}/kappa_{scenario}.csv", index=False)
    return df


# =============================================================================
# FRIEDMAN + NEMENYI + CRITICAL DIFFERENCE DIAGRAM
# =============================================================================

def run_friedman_nemenyi(all_scenario_accuracies):
    """
    Friedman test + Nemenyi post-hoc across scenarios.
    Reference: Demsar, J. (2006). JMLR, 7, 1-30.
    """
    log.info("  Friedman test + Nemenyi post-hoc...")

    scenarios = list(all_scenario_accuracies.keys())
    model_sets = [set(d.keys()) for d in all_scenario_accuracies.values()]
    common = sorted(set.intersection(*model_sets))

    if len(common) < 3:
        log.warning("    Need >= 3 common models. Skipping.")
        return None

    # Performance matrix (rows=scenarios, cols=models)
    perf = np.array([[all_scenario_accuracies[s][m] for m in common]
                      for s in scenarios])

    # Ranks (1=best)
    ranks = np.zeros_like(perf)
    for i in range(len(scenarios)):
        ranks[i] = stats.rankdata(-perf[i])
    avg_ranks = ranks.mean(axis=0)

    log.info("    Average ranks:")
    for j, m in enumerate(common):
        log.info(f"      {m:<25s}: {avg_ranks[j]:.2f}")

    # Friedman test
    if len(scenarios) >= 3:
        stat, p = friedmanchisquare(*[perf[:, j] for j in range(len(common))])
        log.info(f"    Friedman chi2={stat:.4f}, p={p:.6f}, sig={'YES' if p < 0.05 else 'NO'}")
    else:
        stat, p = np.nan, np.nan
        log.info(f"    Only {len(scenarios)} scenarios. Reporting ranks only.")

    # Critical Difference
    k, N = len(common), len(scenarios)
    q_alpha = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949}
    q = q_alpha.get(k, 2.569)
    CD = q * np.sqrt(k * (k + 1) / (6 * N))
    log.info(f"    CD = {CD:.4f}")

    # Nemenyi
    nemenyi_df = None
    if not np.isnan(p) and p < 0.05:
        try:
            import scikit_posthocs as sp
            nemenyi_df = sp.posthoc_nemenyi_friedman(perf)
            nemenyi_df.index = common
            nemenyi_df.columns = common
            nemenyi_df.to_csv(f"{TABLES_DIR}/nemenyi_pvalues.csv")
            log.info(f"    Nemenyi p-values saved.")
        except ImportError:
            log.warning("    scikit-posthocs not installed.")

    result = {
        'models': common, 'avg_ranks': avg_ranks.tolist(),
        'friedman_stat': float(stat) if not np.isnan(stat) else None,
        'friedman_p': float(p) if not np.isnan(p) else None,
        'CD': CD,
    }

    with open(f"{TABLES_DIR}/friedman_results.json", 'w') as f:
        json.dump(result, f, indent=2)

    return result


# =============================================================================
# ORCHESTRATOR
# =============================================================================

def run_all_statistical_tests(y_test, predictions, scenario,
                               all_scenario_accuracies=None):
    """Run all statistical tests for a scenario."""
    log.info(f"\n{'='*60}")
    log.info(f"  STATISTICAL VALIDATION - {scenario}")
    log.info(f"{'='*60}")

    boot_df = run_bootstrap_analysis(y_test, predictions, scenario)
    mcnemar_df = run_mcnemar_analysis(y_test, predictions, scenario)
    kappa_df = run_kappa_analysis(y_test, predictions, scenario)

    friedman = None
    if all_scenario_accuracies and len(all_scenario_accuracies) >= 3:
        friedman = run_friedman_nemenyi(all_scenario_accuracies)

    return {
        'bootstrap': boot_df,
        'mcnemar': mcnemar_df,
        'kappa': kappa_df,
        'friedman': friedman,
    }
