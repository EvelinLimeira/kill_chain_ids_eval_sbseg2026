#!/usr/bin/env python3
"""
audit_crosspartition_duplicates.py
===================================
Auditoria segura de duplicatas entre as partições de treino e teste
do CICIoMT2024, usando hashes de linha em vez de merge completo.

Evita OOM: processa cada arquivo separadamente, armazena apenas hashes
(int64), nunca carrega treino e teste juntos na memória.

Saída (stdout + audit_reports/crosspartition_audit.json):
  - unique train signatures
  - unique test signatures
  - shared signatures (exact feature-level duplicates)
  - NaN count in train and test
  - duplicate rows within each partition

Uso:
    python scripts/audit/audit_crosspartition_duplicates.py
    python scripts/audit/audit_crosspartition_duplicates.py --data-dir ./data
"""

import argparse
import glob
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from config import CICIOMT_FEATURE_NAMES, DATA_DIR

log = logging.getLogger(__name__)


def _hash_folder(folder: str, feature_cols: list[str]) -> tuple[set, int, int]:
    """
    Lê todos os CSVs de uma pasta, calcula hash de cada linha
    usando apenas as 45 features numéricas.

    Retorna:
        hash_set   : set de hashes únicos
        total_rows : total de linhas processadas
        nan_count  : total de NaN encontrados
    """
    hash_set: set = set()
    total_rows = 0
    nan_count = 0

    csv_files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not csv_files:
        log.warning(f"Nenhum CSV encontrado em: {folder}")
        return hash_set, total_rows, nan_count

    for fpath in csv_files:
        fname = os.path.basename(fpath)
        try:
            df = pd.read_csv(fpath, engine="pyarrow",
                             usecols=lambda c: c in feature_cols)
        except Exception:
            df = pd.read_csv(fpath, usecols=lambda c: c in feature_cols)

        # Alinhar colunas à ordem canônica
        present = [c for c in feature_cols if c in df.columns]
        df = df[present].astype("float32")

        nan_count += int(df.isna().sum().sum())
        total_rows += len(df)

        row_hashes = pd.util.hash_pandas_object(df, index=False)
        hash_set.update(row_hashes.tolist())

        log.info(f"  {fname}: {len(df):,} rows  "
                 f"unique so far: {len(hash_set):,}")

    return hash_set, total_rows, nan_count


def run_audit(data_dir: str, feature_cols: list[str]) -> dict:
    train_dir = os.path.join(data_dir, "train")
    test_dir  = os.path.join(data_dir, "test")

    log.info("=" * 60)
    log.info("Auditando partição de TREINO...")
    t0 = time.time()
    train_hashes, train_rows, train_nan = _hash_folder(train_dir, feature_cols)
    log.info(f"Treino: {train_rows:,} linhas | "
             f"{len(train_hashes):,} assinaturas únicas | "
             f"NaN: {train_nan:,} | {time.time()-t0:.1f}s")

    log.info("=" * 60)
    log.info("Auditando partição de TESTE...")
    t0 = time.time()
    test_hashes, test_rows, test_nan = _hash_folder(test_dir, feature_cols)
    log.info(f"Teste:  {test_rows:,} linhas | "
             f"{len(test_hashes):,} assinaturas únicas | "
             f"NaN: {test_nan:,} | {time.time()-t0:.1f}s")

    # Duplicatas internas (linhas não-únicas dentro de cada partição)
    train_internal_dup = train_rows - len(train_hashes)
    test_internal_dup  = test_rows  - len(test_hashes)

    # Assinaturas compartilhadas entre treino e teste
    log.info("=" * 60)
    log.info("Calculando intersecção treino ∩ teste...")
    t0 = time.time()
    shared = train_hashes & test_hashes
    log.info(f"Assinaturas compartilhadas: {len(shared):,} | {time.time()-t0:.1f}s")

    result = {
        "train_total_rows":          train_rows,
        "train_unique_signatures":   len(train_hashes),
        "train_internal_duplicates": train_internal_dup,
        "train_nan_values":          train_nan,
        "test_total_rows":           test_rows,
        "test_unique_signatures":    len(test_hashes),
        "test_internal_duplicates":  test_internal_dup,
        "test_nan_values":           test_nan,
        "cross_partition_shared_signatures": len(shared),
        "features_used":             len(feature_cols),
        "feature_list":              feature_cols,
    }

    log.info("=" * 60)
    log.info("RESULTADO FINAL:")
    log.info(f"  Treino : {train_rows:,} linhas | "
             f"{train_internal_dup:,} duplicatas internas | "
             f"NaN: {train_nan}")
    log.info(f"  Teste  : {test_rows:,} linhas | "
             f"{test_internal_dup:,} duplicatas internas | "
             f"NaN: {test_nan}")
    log.info(f"  Cross-partition shared signatures: {len(shared):,}")

    if len(shared) == 0:
        log.info("  → Nenhuma duplicata exata cross-partition encontrada.")
    else:
        log.info(f"  → {len(shared):,} assinaturas de features idênticas "
                 f"aparecem em ambas as partições.")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Auditoria de duplicatas cross-partition CICIoMT2024"
    )
    parser.add_argument("--data-dir", default=DATA_DIR,
                        help=f"Diretório raiz dos dados (default: {DATA_DIR})")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log.info(f"Data dir: {args.data_dir}")
    log.info(f"Features: {len(CICIOMT_FEATURE_NAMES)} colunas")

    result = run_audit(args.data_dir, CICIOMT_FEATURE_NAMES)

    # Salvar relatório
    os.makedirs(os.path.join(_PROJECT_ROOT, "audit_reports"), exist_ok=True)
    out_path = os.path.join(_PROJECT_ROOT, "audit_reports",
                            "crosspartition_audit.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"\nRelatório salvo: {out_path}")

    # Linha para inserir no artigo
    shared = result["cross_partition_shared_signatures"]
    train_nan = result["train_nan_values"]
    test_nan = result["test_nan_values"]
    log.info("\n--- Texto para o artigo (§3.6) ---")
    if shared == 0:
        log.info(
            "No exact cross-partition duplicates were found "
            "(verified by row-hash comparison on 45 numeric features). "
            f"Missing values: {train_nan + test_nan} "
            "(train: {train_nan}, test: {test_nan})."
        )
    else:
        log.info(
            f"{shared:,} exact cross-partition duplicate signatures found "
            f"(row-hash on 45 numeric features)."
        )


if __name__ == "__main__":
    main()
