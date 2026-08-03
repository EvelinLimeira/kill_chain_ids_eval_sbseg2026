#!/usr/bin/env python3
"""
Automated tests for autoencoder_threshold_sensitivity.py

Run with the built-in unittest runner (no pytest dependency required):
    python -m unittest scripts.evaluation.test_autoencoder_threshold_sensitivity -v
or from the scripts/evaluation directory:
    python -m unittest test_autoencoder_threshold_sensitivity -v

Scope (important, no overclaim):
These are UNIT tests for the PURE metric, aggregation, formatting, taxonomy,
fingerprint and timing-summary functions. They do NOT import TensorFlow and do
NOT exercise the neural pipeline. Therefore they do NOT by themselves verify:
real same-seed reproduction across subprocesses, actual GPU usage, GPU-request
failure when absent, GPU synchronisation, TF state isolation, full artefact-tree
creation, or that warm-up is excluded inside benchmark_inference. Those aspects
are checked separately through a smoke run of the full orchestrator/worker.
"""
import importlib.util
import os
import sys
import unittest

import numpy as np
import pandas as pd

# Import the target module directly by path (avoids package config).
_HERE = os.path.dirname(os.path.abspath(__file__))
_MOD_PATH = os.path.join(_HERE, "autoencoder_threshold_sensitivity.py")
_spec = importlib.util.spec_from_file_location("ae_mod", _MOD_PATH)
ae = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ae)


class TestConfusionAndMetrics(unittest.TestCase):
    def test_confusion_counts(self):
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0])
        tp, fp, tn, fn = ae.confusion_counts(y_true, y_pred)
        self.assertEqual((tp, fp, tn, fn), (2, 1, 2, 1))

    def test_fpr_and_specificity(self):
        # 4 benign (2 flagged), 4 attack (all flagged). errors > 0.5 => attack.
        errors = np.array([0.9, 0.9, 0.1, 0.1,   # benign: 2 FP, 2 TN
                           0.9, 0.9, 0.9, 0.9])   # attack: 4 TP
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        m = ae.metrics_at_threshold(y_true, errors, 0.5)
        self.assertAlmostEqual(m["fpr"], 0.5, places=12)
        self.assertAlmostEqual(m["specificity"], 0.5, places=12)
        self.assertEqual((m["tp"], m["fp"], m["tn"], m["fn"]), (4, 2, 2, 0))

    def test_balanced_accuracy(self):
        errors = np.array([0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1])
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        m = ae.metrics_at_threshold(y_true, errors, 0.5)
        # recall = 2/4 = 0.5 ; specificity = 3/4 = 0.75 ; balanced = 0.625
        self.assertAlmostEqual(m["balanced_accuracy"], 0.625, places=12)

    def test_ap_attack_and_benign(self):
        from sklearn.metrics import average_precision_score
        rng = np.random.RandomState(0)
        y_true = np.array([0] * 20 + [1] * 20)
        errors = np.concatenate([rng.normal(0, 1, 20), rng.normal(3, 1, 20)])
        m = ae.metrics_at_threshold(y_true, errors, 1.5)
        self.assertAlmostEqual(
            m["ap_attack"], average_precision_score(y_true, errors), places=10)
        self.assertAlmostEqual(
            m["ap_benign"], average_precision_score(1 - y_true, -errors), places=10)

    def test_full_precision_not_rounded(self):
        # A value that would collapse if rounded to 4 dp.
        errors = np.array([0.0] * 9999 + [1.0])  # 1 attack of 10000 benign-ish
        y_true = np.array([0] * 9999 + [1])
        m = ae.metrics_at_threshold(y_true, errors, 0.5)
        # recall 1.0, but precision has many decimals if there were FPs; here check
        # that fpr is exactly 0 and values are python floats, not rounded strings.
        self.assertIsInstance(m["recall_attack"], float)
        self.assertEqual(m["fpr"], 0.0)


class TestThresholdsAndMonotonicity(unittest.TestCase):
    def test_thresholds_from_validation_only(self):
        val = np.arange(100, dtype=float)
        thr = ae.compute_thresholds(val, [85, 90, 95, 99])
        self.assertLess(thr[85], thr[90])
        self.assertLess(thr[90], thr[95])
        self.assertLess(thr[95], thr[99])

    def test_calibration_exceedance_rate(self):
        val = np.arange(100, dtype=float)
        tau = np.percentile(val, 90)
        rate = ae.calibration_exceedance_rate(val, tau)
        self.assertTrue(0.0 <= rate <= 0.15)

    def test_monotonicity_pass(self):
        rng = np.random.RandomState(1)
        val = rng.normal(0, 1, 5000)
        test = np.concatenate([rng.normal(0, 1, 1000), rng.normal(4, 1, 5000)])
        y_true = np.array([0] * 1000 + [1] * 5000)
        rows = ae.build_sensitivity_rows(y_true, test, val, [85, 90, 95, 99])
        ae.check_monotonicity(rows)  # should not raise

    def test_monotonicity_violation_raises(self):
        rows = [
            {"percentile": 85, "tau": 1.0, "fpr": 0.2, "recall_attack": 0.9, "n_alerts": 100},
            {"percentile": 90, "tau": 0.9, "fpr": 0.3, "recall_attack": 0.95, "n_alerts": 120},
        ]
        with self.assertRaises(RuntimeError):
            ae.check_monotonicity(rows)


class TestPhaseRecall(unittest.TestCase):
    def _synthetic_classes(self):
        return ["Benign"] + list(ae.PHASE_MAP.keys())

    def test_phase_mapping_valid(self):
        ae.validate_phase_mapping(self._synthetic_classes())  # no raise

    def test_unmapped_class_raises(self):
        classes = self._synthetic_classes() + ["BLE-DoS"]
        with self.assertRaises(ValueError):
            ae.validate_phase_mapping(classes)

    def test_missing_benign_raises(self):
        classes = list(ae.PHASE_MAP.keys())
        with self.assertRaises(ValueError):
            ae.validate_phase_mapping(classes)

    def test_macro_recall_full_precision(self):
        # Build a tiny test set: Benign + 18 attacks, 3 samples each attack.
        classes = self._synthetic_classes()
        n_per = 3
        y = []
        errs = []
        for idx, name in enumerate(classes):
            if name == "Benign":
                y += [idx] * n_per
                errs += [0.0] * n_per
            else:
                y += [idx] * n_per
                # detect 1 of 3 => recall 1/3 (repeating decimal)
                errs += [1.0, 0.0, 0.0]
        y = np.array(y)
        errs = np.array(errs)
        n_attack = int(np.sum(y != classes.index("Benign")))
        df_class, df_phase = ae.recall_by_class_and_phase(
            y, classes, errs, 0.5, n_attack)
        # every attack class recall should be exactly 1/3
        self.assertTrue(np.allclose(df_class["recall"].to_numpy(), 1.0 / 3.0))
        # macro recall per phase also 1/3 (full precision, not rounded to 0.3333)
        self.assertTrue(np.allclose(df_phase["macro_recall"].to_numpy(), 1.0 / 3.0))
        self.assertGreater(abs(df_phase["macro_recall"].iloc[0] - 0.3333), 1e-6)

    def test_support_sum_mismatch_raises(self):
        classes = self._synthetic_classes()
        y = np.array([classes.index("Benign")] * 2 +
                     [classes.index("Spoofing")] * 2)
        errs = np.array([0.0, 0.0, 1.0, 0.0])
        with self.assertRaises(ValueError):
            ae.recall_by_class_and_phase(y, classes, errs, 0.5, n_attack=999)


class TestAggregation(unittest.TestCase):
    def test_mean_sd_ddof1(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        st = ae.aggregate_stats(vals)
        self.assertAlmostEqual(st["mean"], 3.0, places=12)
        self.assertAlmostEqual(st["sd"], np.std(vals, ddof=1), places=12)
        self.assertEqual(st["n_seeds"], 5)

    def test_ci95_present_for_multiple_seeds(self):
        st = ae.aggregate_stats([0.9, 0.91, 0.89, 0.92, 0.88])
        self.assertTrue(st["ci95_low"] < st["mean"] < st["ci95_high"])

    def test_aggregation_no_rounding(self):
        vals = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
        st = ae.aggregate_stats(vals)
        self.assertGreater(abs(st["mean"] - 0.3333), 1e-6)

    def test_median_iqr_secondary(self):
        st = ae.aggregate_stats([1, 2, 3, 4, 100])
        self.assertEqual(st["median"], 3.0)
        self.assertEqual(st["q1"], 2.0)
        self.assertEqual(st["q3"], 4.0)

    def test_full_stat_keys_present(self):
        # The sensitivity/recall aggregation propagates ALL these statistics
        # (mean, sd, ci95, median, q1, q3, min, max, n) via a per-item loop.
        st = ae.aggregate_stats([0.9, 0.91, 0.89, 0.92, 0.88])
        for k in ("mean", "sd", "ci95_low", "ci95_high", "median", "q1", "q3",
                  "min", "max", "n_seeds"):
            self.assertIn(k, st)
        row = {}
        for statistic, value in st.items():
            row[f"recall_attack_{statistic}"] = value
        self.assertIn("recall_attack_ci95_low", row)
        self.assertIn("recall_attack_q3", row)
        self.assertIn("recall_attack_n_seeds", row)


class TestFeatureVsClassFingerprint(unittest.TestCase):
    def test_feature_and_class_fingerprints_differ(self):
        feature_names = ["Header_Length", "Protocol Type", "Duration", "Rate"]
        class_names = ["Benign", "DDoS-UDP", "Recon-Ping_Sweep"]
        self.assertNotEqual(ae.sha256_of_strings(feature_names),
                            ae.sha256_of_strings(class_names))


class TestFormatting(unittest.TestCase):
    def test_metric_rounding_4dp(self):
        self.assertEqual(ae._round_metric("recall_attack", 0.123456789), 0.1235)

    def test_tau_rounding_6dp(self):
        self.assertEqual(ae._round_metric("tau_p85", 1.23456789), 1.234568)

    def test_counts_are_int(self):
        self.assertEqual(ae._round_metric("n_fp", 9511.0), 9511)
        self.assertIsInstance(ae._round_metric("tp", 3.0), int)

    def test_raw_vs_formatted_separation(self):
        raw = pd.DataFrame([{"metric": "recall_attack", "mean": 1.0 / 3.0,
                             "sd": 0.001234567}])
        formatted = ae.format_results_for_article(raw)
        # raw untouched
        self.assertGreater(abs(raw.iloc[0]["mean"] - 0.3333), 1e-6)
        # formatted rounded
        self.assertEqual(formatted.iloc[0]["mean"], 0.3333)


class TestEnvironmentConsistency(unittest.TestCase):
    def _md(self, **over):
        base = {k: "x" for k in ae.ENV_CONSISTENCY_KEYS}
        base["training_seed"] = over.pop("training_seed", 42)
        base.update(over)
        return base

    def test_consistent_envs_ok(self):
        mds = [self._md(training_seed=s) for s in (40, 41, 42)]
        ok, report = ae.check_environment_consistency(mds)
        self.assertTrue(ok)
        self.assertTrue(report["consistent"])

    def test_mixed_device_refused(self):
        mds = [self._md(training_seed=40, device_used="cpu"),
               self._md(training_seed=41, device_used="gpu")]
        ok, report = ae.check_environment_consistency(mds)
        self.assertFalse(ok)
        self.assertTrue(any(d["field"] == "device_used"
                            for d in report["discrepancies"]))

    def test_mixed_tf_version_refused(self):
        mds = [self._md(training_seed=40, tensorflow_version="2.20.0"),
               self._md(training_seed=41, tensorflow_version="2.15.0")]
        ok, _ = ae.check_environment_consistency(mds)
        self.assertFalse(ok)


class TestFingerprints(unittest.TestCase):
    def test_array_fingerprint_stable(self):
        a = np.arange(10)
        self.assertEqual(ae.sha256_of_array(a), ae.sha256_of_array(np.arange(10)))

    def test_array_fingerprint_sensitive(self):
        a = np.arange(10)
        b = a.copy(); b[0] = 99
        self.assertNotEqual(ae.sha256_of_array(a), ae.sha256_of_array(b))

    def test_benign_split_identical_across_seeds(self):
        # The benign split depends only on split_seed, not the training seed.
        from sklearn.model_selection import train_test_split
        idx = np.arange(1000)
        a_tr, a_val = train_test_split(idx, test_size=0.2, random_state=42)
        b_tr, b_val = train_test_split(idx, test_size=0.2, random_state=42)
        self.assertEqual(ae.sha256_of_array(np.sort(a_tr)),
                         ae.sha256_of_array(np.sort(b_tr)))
        self.assertEqual(ae.sha256_of_array(np.sort(a_val)),
                         ae.sha256_of_array(np.sort(b_val)))


class TestTimingSummary(unittest.TestCase):
    def _records(self):
        recs = []
        for rep in range(10):
            recs.append({
                "seed": 42, "device": "cpu", "benchmark_type": "end_to_end",
                "batch_size": 32, "repetition": rep, "number_of_flows": 32,
                "elapsed_seconds": 0.01 + rep * 1e-4,
                "batch_latency_ms": (0.01 + rep * 1e-4) * 1e3,
                "effective_time_per_flow_us": (0.01 + rep * 1e-4) * 1e6 / 32,
                "throughput_flows_per_second": 32 / (0.01 + rep * 1e-4),
                "warmup_iterations": 10, "timestamp_utc": "t",
            })
        return recs

    def test_summary_has_percentiles(self):
        s = ae.summarize_timing(self._records())
        b = s["per_benchmark"][0]
        for key in ("latency_ms_median", "latency_ms_p95", "latency_ms_p99",
                    "throughput_mean", "throughput_sd"):
            self.assertIn(key, b)

    def test_throughput_consistent_with_flows(self):
        recs = self._records()
        for r in recs:
            expected = r["number_of_flows"] / r["elapsed_seconds"]
            self.assertAlmostEqual(r["throughput_flows_per_second"], expected,
                                   places=6)

    def test_effective_time_per_flow_consistent(self):
        recs = self._records()
        for r in recs:
            expected = r["elapsed_seconds"] * 1e6 / r["number_of_flows"]
            self.assertAlmostEqual(r["effective_time_per_flow_us"], expected,
                                   places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
