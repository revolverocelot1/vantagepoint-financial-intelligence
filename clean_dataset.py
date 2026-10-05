"""
clean_dataset.py: Forensic Data Cleaning & Sanitation Engine
VantagePoint Forensic Data Hygiene Pipeline

Applies rigorous data transformations based on the forensic audit:
  1. Deduplicates true double-charge transactions (T0410).
  2. Resolves ID sequence collisions (T0031 -> T0032, T0760 -> T0761).
  3. Imputes missing category (T0702 -> Entertainment) based on description 'Streaming'.
  4. Imputes missing description (T0138 -> Miscellaneous).
  5. Corrects category typos and mismatches (T0202 'Foods' with Car Loan EMI -> 'Debt Payment').
  6. Rectifies rogue future date typo (T0277 '2026-11-15' -> '2025-07-15' in July 2025 sequence).
  7. Normalizes slash date formats (T0643 '2026/06/15' -> '2026-06-15').
  8. Sanitizes negative expense values (T0089 -4500 -> 4500.00).
  9. Preserves valid statistical outliers (T0488: INR 1,85,000; T0556: INR 2,05,000) for anomaly detection.
  10. Exports production-ready cleaned files:
      - transactions_cleaned.csv
      - liabilities_cleaned.csv
      - assets_cleaned.csv
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DataCleaner")


def clean_datasets(dataset_dir: Path, output_dirs: List[Path]) -> Dict[str, Any]:
    txn_path = dataset_dir / "transactions.csv"
    liab_path = dataset_dir / "liabilities.csv"
    asset_path = dataset_dir / "assets.csv"

    logger.info(f"Reading raw files from: {dataset_dir}")
    df_txn = pd.read_csv(txn_path)
    df_liab = pd.read_csv(liab_path)
    df_asset = pd.read_csv(asset_path)

    audit_trail: List[Dict[str, Any]] = []

    # --------------------------------------------------------------------------
    # 1. TRANSACTIONS CLEANING
    # --------------------------------------------------------------------------
    initial_txn_count = len(df_txn)
    cleaned_rows: List[Dict[str, Any]] = []
    seen_exact_hashes = set()

    for idx, row in df_txn.iterrows():
        t_id = str(row["txn_id"]).strip()
        t_date = str(row["date"]).strip()
        t_cat = str(row["category"]).strip() if pd.notna(row["category"]) else ""
        t_desc = str(row["description"]).strip() if pd.notna(row["description"]) else ""
        t_amt = row["amount"]
        t_type = str(row["type"]).strip().lower() if pd.notna(row["type"]) else "expense"

        # A. Detect and drop exact duplicate transactions (e.g. T0410 line 410 & 411)
        row_signature = (t_date, t_cat, t_desc, float(t_amt), t_type)
        if t_id == "T0410" and row_signature in seen_exact_hashes:
            audit_trail.append({
                "txn_id": t_id,
                "action": "DROP_DUPLICATE",
                "details": f"Dropped identical double-charge record: {t_desc} of INR {float(t_amt):,.2f} on {t_date}",
            })
            continue
        seen_exact_hashes.add(row_signature)

        # B. Resolve ID collisions where underlying transactions are distinct
        # T0031 second occurrence was Car loan EMI (should be T0032)
        if t_id == "T0031" and "car loan" in t_desc.lower():
            old_id = t_id
            t_id = "T0032"
            audit_trail.append({
                "txn_id": old_id,
                "action": "RENAME_ID_COLLISION",
                "details": f"Renamed second T0031 (Car loan EMI) to {t_id} restoring skipped sequence number",
            })

        # T0760 second occurrence was Movies in Feb 2026 (should be T0761)
        if t_id == "T0760" and "movies" in t_desc.lower():
            old_id = t_id
            t_id = "T0761"
            audit_trail.append({
                "txn_id": old_id,
                "action": "RENAME_ID_COLLISION",
                "details": f"Renamed second T0760 (Movies in Feb 2026) to {t_id} restoring skipped sequence number",
            })

        # C. Date Normalization
        # Normalize slash separator (e.g. 2026/06/15 -> 2026-06-15)
        if "/" in t_date:
            norm_date = t_date.replace("/", "-")
            audit_trail.append({
                "txn_id": t_id,
                "action": "NORMALIZE_DATE_FORMAT",
                "details": f"Converted slash date '{t_date}' to ISO format '{norm_date}'",
            })
            t_date = norm_date

        # Correct rogue future date (T0277 dated 2026-11-15 situated in July 2025 sequence)
        if t_id == "T0277" and t_date == "2026-11-15":
            corrected_date = "2025-07-15"
            audit_trail.append({
                "txn_id": t_id,
                "action": "CORRECT_FUTURE_DATE_TYPO",
                "details": f"Realigned out-of-bounds date '{t_date}' to July 2025 sequence '{corrected_date}' (restoring July 2025 Healthcare)",
            })
            t_date = corrected_date

        # D. Category Imputation & Typo Correction
        if not t_cat or t_cat.lower() == "nan":
            if "streaming" in t_desc.lower():
                t_cat = "Entertainment"
            else:
                t_cat = "Other"
            audit_trail.append({
                "txn_id": t_id,
                "action": "IMPUTE_MISSING_CATEGORY",
                "details": f"Imputed empty category to '{t_cat}' based on description '{t_desc}'",
            })
        elif t_cat.lower() == "foods":
            if "emi" in t_desc.lower():
                t_cat = "Debt Payment"
            else:
                t_cat = "Food"
            audit_trail.append({
                "txn_id": t_id,
                "action": "CORRECT_CATEGORY_TYPO",
                "details": f"Corrected typo category 'Foods' with description '{t_desc}' to '{t_cat}'",
            })

        # E. Description Imputation
        if not t_desc or t_desc.lower() == "nan":
            t_desc = "Miscellaneous"
            audit_trail.append({
                "txn_id": t_id,
                "action": "IMPUTE_MISSING_DESCRIPTION",
                "details": f"Imputed missing description for category '{t_cat}' to '{t_desc}'",
            })

        # F. Amount Sanitization (Handle negative expense)
        try:
            val_amt = float(t_amt)
        except (ValueError, TypeError):
            val_amt = 0.0

        if val_amt < 0:
            pos_amt = abs(val_amt)
            audit_trail.append({
                "txn_id": t_id,
                "action": "RECTIFY_NEGATIVE_AMOUNT",
                "details": f"Converted negative outflow ({val_amt}) to positive expenditure value ({pos_amt:,.2f})",
            })
            val_amt = pos_amt

        # Format amount to standard 2 decimal places
        cleaned_rows.append({
            "txn_id": t_id,
            "date": t_date,
            "category": t_cat,
            "description": t_desc,
            "amount": round(val_amt, 2),
            "type": t_type,
        })

    df_cleaned_txn = pd.DataFrame(cleaned_rows)

    # --------------------------------------------------------------------------
    # 2. LIABILITIES CLEANING
    # --------------------------------------------------------------------------
    # Enforce standard formatting and schema
    df_cleaned_liab = df_liab.copy()
    df_cleaned_liab["outstanding"] = df_cleaned_liab["outstanding"].astype(float).round(2)
    df_cleaned_liab["interest_rate"] = df_cleaned_liab["interest_rate"].astype(float).round(2)
    df_cleaned_liab["emi"] = df_cleaned_liab["emi"].astype(float).round(2)
    df_cleaned_liab["due_date"] = pd.to_datetime(df_cleaned_liab["due_date"]).dt.strftime("%Y-%m-%d")

    # --------------------------------------------------------------------------
    # 3. ASSETS CLEANING
    # --------------------------------------------------------------------------
    df_cleaned_asset = df_asset.copy()
    df_cleaned_asset["value"] = df_cleaned_asset["value"].astype(float).round(2)
    df_cleaned_asset["as_of_date"] = pd.to_datetime(df_cleaned_asset["as_of_date"]).dt.strftime("%Y-%m-%d")

    # --------------------------------------------------------------------------
    # 4. EXPORT TO TARGET DIRECTORIES
    # --------------------------------------------------------------------------
    for out_dir in output_dirs:
        out_dir.mkdir(parents=True, exist_ok=True)
        txn_out = out_dir / "transactions_cleaned.csv"
        liab_out = out_dir / "liabilities_cleaned.csv"
        asset_out = out_dir / "assets_cleaned.csv"

        df_cleaned_txn.to_csv(txn_out, index=False)
        df_cleaned_liab.to_csv(liab_out, index=False)
        df_cleaned_asset.to_csv(asset_out, index=False)
        logger.info(f"Exported cleaned datasets to: {out_dir}")

    summary = {
        "initial_transaction_count": initial_txn_count,
        "final_cleaned_transaction_count": len(df_cleaned_txn),
        "modifications_applied_count": len(audit_trail),
        "audit_trail": audit_trail,
    }

    return summary


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    dataset_dir = base_dir / "Dataset"
    target_dirs = [base_dir, dataset_dir]

    summary = clean_datasets(dataset_dir, target_dirs)
    print("\n" + "=" * 80)
    print("DATASET FORENSIC CLEANING COMPLETE")
    print("=" * 80)
    print(f"Initial Rows : {summary['initial_transaction_count']}")
    print(f"Cleaned Rows : {summary['final_cleaned_transaction_count']}")
    print(f"Modifications: {summary['modifications_applied_count']}")
    print("\nAudit Trail of Rectifications:")
    for item in summary["audit_trail"]:
        print(f" - [{item['action']}] Txn {item['txn_id']}: {item['details']}")
    print("=" * 80)
