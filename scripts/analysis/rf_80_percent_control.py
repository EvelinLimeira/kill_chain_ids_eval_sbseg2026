#!/usr/bin/env python3
"""
Experimento Controle C1: Treinar RF com 80% dos dados (mesmo volume que CNN/LightGBM Tuned).

Objetivo: Verificar se a diferença de performance RF vs CNN se deve ao volume
de dados extra (RF usa 100% = train+val, CNN usa 80% = train only).

Resultado esperado: Se RF com 80% ainda supera CNN significativamente,
o claim de que "the gap far exceeds what additional data alone could explain"
é empiricamente validado.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import time
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    classification_report, recall_score, precision_score
)

from config import SEED, RF_PARAMS, RESULTS_DIR, TABLES_DIR
from data_loader import load_data

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
log = logging.getLogger(__name__)


def run_rf_control_experiment(data_dir: str = './data'):
    """
    Train RF with 80% data (X_train_flat only) and compare with RF 100% (X_train_combined).
    """
    results = []

    for class_config in [2, 6, 19]:
        scenario = f"{class_config}-class"
        log.info(f"\n{'='*60}")
        log.info(f"  RF Control Experiment — {scenario}")
        log.info(f"{'='*60}")

        data = load_data(data_dir, class_config)

        # --- RF with 100% data (original, train+val combined) ---
        log.info("  Training RF with 100% data (train+val combined)...")
        t0 = time.time()
        rf_100 = RandomForestClassifier(**RF_PARAMS)
        rf_100.fit(data['X_train_combined'], data['y_train_combined'])
        time_100 = time.time() - t0
        y_pred_100 = rf_100.predict(data['X_test_flat'])

        acc_100 = accuracy_score(data['y_test'], y_pred_100)
        f1m_100 = f1_score(data['y_test'], y_pred_100, average='macro', zero_division=0)
        kappa_100 = cohen_kappa_score(data['y_test'], y_pred_100)

        log.info(f"    RF 100%: Acc={acc_100:.4f}, F1m={f1m_100:.4f}, κ={kappa_100:.4f}, "
                 f"N_train={len(data['X_train_combined']):,}, Time={time_100:.1f}s")

        # --- RF with 80% data (X_train_flat only, same as CNN/LightGBM Tuned) ---
        log.info("  Training RF with 80% data (train only, no val)...")
        t0 = time.time()
        rf_80 = RandomForestClassifier(**RF_PARAMS)
        rf_80.fit(data['X_train_flat'], data['y_train'])
        time_80 = time.time() - t0
        y_pred_80 = rf_80.predict(data['X_test_flat'])

        acc_80 = accuracy_score(data['y_test'], y_pred_80)
        f1m_80 = f1_score(data['y_test'], y_pred_80, average='macro', zero_division=0)
        kappa_80 = cohen_kappa_score(data['y_test'], y_pred_80)

        log.info(f"    RF  80%: Acc={acc_80:.4f}, F1m={f1m_80:.4f}, κ={kappa_80:.4f}, "
                 f"N_train={len(data['X_train_flat']):,}, Time={time_80:.1f}s")

        # --- Delta ---
        delta_acc = acc_100 - acc_80
        delta_f1m = f1m_100 - f1m_80
        delta_kappa = kappa_100 - kappa_80

        log.info(f"    Delta (100% - 80%): Acc={delta_acc:+.4f}, F1m={delta_f1m:+.4f}, "
                 f"κ={delta_kappa:+.4f}")

        # --- Per-phase analysis for 19-class ---
        if class_config == 19:
            class_names = list(data['label_encoder'].classes_)

            # Recon classes
            recon_classes = [c for c in class_names if 'Recon' in c]
            recon_indices = [class_names.index(c) for c in recon_classes]

            # Compute per-class recall for recon
            for rf_name, y_pred in [("RF_100%", y_pred_100), ("RF_80%", y_pred_80)]:
                recon_recalls = []
                for idx in recon_indices:
                    mask = data['y_test'] == idx
                    if mask.sum() > 0:
                        rec = recall_score(
                            (data['y_test'][mask] == idx).astype(int),
                            (y_pred[mask] == idx).astype(int),
                            zero_division=0
                        )
                        recon_recalls.append(rec)
                avg_recon_recall = np.mean(recon_recalls) if recon_recalls else 0
                log.info(f"    {rf_name} Recon avg recall: {avg_recon_recall:.4f}")

            # ARP Spoofing F1
            if 'ARP_Spoofing' in class_names:
                arp_idx = class_names.index('ARP_Spoofing')
            elif 'Spoofing' in class_names:
                arp_idx = class_names.index('Spoofing')
            else:
                arp_idx = None

            if arp_idx is not None:
                for rf_name, y_pred in [("RF_100%", y_pred_100), ("RF_80%", y_pred_80)]:
                    mask = (data['y_test'] == arp_idx) | (y_pred == arp_idx)
                    if mask.sum() > 0:
                        arp_f1 = f1_score(
                            (data['y_test'] == arp_idx).astype(int),
                            (y_pred == arp_idx).astype(int),
                            zero_division=0
                        )
                        log.info(f"    {rf_name} ARP Spoofing F1: {arp_f1:.4f}")

        results.append({
            'Scenario': scenario,
            'RF_100%_Acc': acc_100,
            'RF_100%_F1m': f1m_100,
            'RF_100%_Kappa': kappa_100,
            'RF_100%_N_train': len(data['X_train_combined']),
            'RF_100%_Time_s': time_100,
            'RF_80%_Acc': acc_80,
            'RF_80%_F1m': f1m_80,
            'RF_80%_Kappa': kappa_80,
            'RF_80%_N_train': len(data['X_train_flat']),
            'RF_80%_Time_s': time_80,
            'Delta_Acc': delta_acc,
            'Delta_F1m': delta_f1m,
            'Delta_Kappa': delta_kappa,
        })

    # Save results
    df = pd.DataFrame(results)
    output_path = os.path.join(TABLES_DIR, 'rf_80_percent_control.csv')
    df.to_csv(output_path, index=False)
    log.info(f"\n  Results saved to: {output_path}")
    log.info(f"\n{'='*60}")
    log.info("  SUMMARY")
    log.info(f"{'='*60}")
    print(df.to_string(index=False))

    # Interpretation
    log.info("\n  INTERPRETATION:")
    for _, row in df.iterrows():
        if abs(row['Delta_F1m']) < 0.01:
            log.info(f"    {row['Scenario']}: Delta F1m = {row['Delta_F1m']:+.4f} — "
                     f"NEGLIGIBLE difference. Data volume does NOT explain the gap.")
        elif abs(row['Delta_F1m']) < 0.03:
            log.info(f"    {row['Scenario']}: Delta F1m = {row['Delta_F1m']:+.4f} — "
                     f"SMALL difference. Data volume has minor effect.")
        else:
            log.info(f"    {row['Scenario']}: Delta F1m = {row['Delta_F1m']:+.4f} — "
                     f"NOTABLE difference. Data volume contributes to the gap.")

    return df


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    args = parser.parse_args()
    run_rf_control_experiment(args.data_dir)
