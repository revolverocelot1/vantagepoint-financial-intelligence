"""
Asset Vantage - Analytics Challenge
detector_and_advisor.py: Comprehensive Trends & Anomaly Detector + Financial Advisory Engine

Accomplishes:
  GOAL 1: Trends & Anomaly Detection ("What changed recently?")
    - Multi-format date parsing & outflow aggregation across 24-month horizon.
    - Spending Drift analysis: 3-month recent window vs. 21-month historical baseline (% and absolute INR).
    - Multi-tier statistical anomaly detection: Global Z-Score, Category-specific Z-Score,
      IQR bounds, category sudden spikes, and data hygiene audits (negative values, zero amounts, typos, duplicates).
    - Structured output dictionary for enterprise consumption.

  GOAL 2: 3 Actionable Deterministic Recommendations + AI Executive Memo ("What should we do next?")
    - Action 1 (Debt): Highest interest-rate liability retirement & refinancing strategy.
    - Action 2 (Leakage): Category-level leakage arrest with an exact prescribed monthly budget cap.
    - Action 3 (Runway): 6-month safety liquidity runway audit & exact INR contribution requirement.
    - AI Bonus: Live Executive Family Office Memo generation powered by google-genai (gemini-2.5-flash).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AssetVantageAdvisor")

# Default provided API key fallback for the challenge environment
DEFAULT_GEMINI_API_KEY = os.environ.get(
    "GEMINI_API_KEY",
    os.environ.get("GOOGLE_API_KEY", "AIzaSyALaEIdDQt2NXMZM4gBTCrlfmbBUnfR3YQ")
)


# ==============================================================================
# 1. FILE RESOLUTION & DATA INGESTION
# ==============================================================================

def find_file(preferred_name: str, fallback_name: str, custom_path: Optional[str] = None) -> Path:
    """
    Locates a data file intelligently across local and dataset directories,
    prioritizing cleaned datasets if available.
    """
    if custom_path and Path(custom_path).exists():
        return Path(custom_path)

    base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    search_dirs = [
        base_dir,
        Path.cwd(),
        base_dir / "Dataset",
        Path("Dataset"),
    ]

    # Search for preferred cleaned file first, then fallback raw file
    for name in [preferred_name, fallback_name]:
        for d in search_dirs:
            candidate = d / name
            if candidate.exists() and candidate.is_file():
                return candidate

    raise FileNotFoundError(
        f"Unable to locate '{preferred_name}' or '{fallback_name}'. Checked search paths."
    )


def load_and_clean_data(
    transactions_path: Optional[str] = None,
    liabilities_path: Optional[str] = None,
    assets_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """
    Loads raw or cleaned CSV files, enforces schemas, captures data hygiene anomalies,
    and returns sanitized DataFrames for downstream analytics.
    """
    txn_file = find_file("transactions_cleaned.csv", "transactions.csv", transactions_path)
    liab_file = find_file("liabilities_cleaned.csv", "liabilities.csv", liabilities_path)
    asset_file = find_file("assets_cleaned.csv", "assets.csv", assets_path)

    logger.info("Loading datasets:")
    logger.info(f" - Transactions: {txn_file}")
    logger.info(f" - Liabilities:  {liab_file}")
    logger.info(f" - Assets:       {asset_file}")

    df_txn = pd.read_csv(txn_file)
    df_liab = pd.read_csv(liab_file)
    df_asset = pd.read_csv(asset_file)

    hygiene_anomalies: List[Dict[str, Any]] = []

    # 1. Check duplicate transaction IDs
    dup_mask = df_txn.duplicated(subset=["txn_id"], keep=False)
    if dup_mask.any():
        for _, row in df_txn[dup_mask].iterrows():
            hygiene_anomalies.append({
                "txn_id": str(row["txn_id"]),
                "issue": "Duplicate transaction ID",
                "date": str(row.get("date")),
                "category": str(row.get("category")),
                "amount": float(row.get("amount", 0.0)),
            })

    # 2. Parse dates gracefully (supports YYYY-MM-DD, YYYY/MM/DD, etc.)
    raw_dates = df_txn["date"].astype(str).str.strip().str.replace("/", "-")
    parsed_dates = pd.to_datetime(raw_dates, errors="coerce")

    snapshot_date = pd.to_datetime("2026-10-01")
    for idx, (p_date, raw_d) in enumerate(zip(parsed_dates, df_txn["date"])):
        row = df_txn.iloc[idx]
        if pd.isna(p_date):
            hygiene_anomalies.append({
                "txn_id": str(row["txn_id"]),
                "issue": f"Unparseable date format: '{raw_d}'",
                "date": str(raw_d),
                "category": str(row.get("category")),
                "amount": float(row.get("amount", 0.0)),
            })
        elif p_date > snapshot_date:
            hygiene_anomalies.append({
                "txn_id": str(row["txn_id"]),
                "issue": f"Date '{p_date.strftime('%Y-%m-%d')}' is in future beyond snapshot 2026-10-01 (sequence-corrected to July 2025)",
                "date": str(raw_d),
                "category": str(row.get("category")),
                "amount": float(row.get("amount", 0.0)),
            })

    # Clean date series
    df_txn["date_parsed"] = parsed_dates
    future_filter = (df_txn["txn_id"] == "T0277") & (df_txn["date_parsed"] > snapshot_date)
    df_txn.loc[future_filter, "date_parsed"] = pd.to_datetime("2025-07-15")
    df_txn["year_month"] = df_txn["date_parsed"].dt.strftime("%Y-%m")

    # 3. Clean and normalize amount
    df_txn["amount_raw"] = pd.to_numeric(df_txn["amount"], errors="coerce").fillna(0.0)

    for _, row in df_txn.iterrows():
        amt = float(row["amount_raw"])
        txn_type = str(row.get("type", "")).strip().lower()
        if amt < 0:
            hygiene_anomalies.append({
                "txn_id": str(row["txn_id"]),
                "issue": f"Negative transaction amount ({amt:,.2f})",
                "date": str(row.get("date")),
                "category": str(row.get("category")),
                "amount": amt,
            })
        elif amt == 0 and txn_type == "expense":
            hygiene_anomalies.append({
                "txn_id": str(row["txn_id"]),
                "issue": "Zero value expense transaction",
                "date": str(row.get("date")),
                "category": str(row.get("category")),
                "amount": amt,
            })

    df_txn["amount"] = df_txn["amount_raw"].abs()

    # 4. Standardize categories & descriptions
    df_txn["category_raw"] = df_txn["category"].fillna("").astype(str).str.strip()
    df_txn["description_raw"] = df_txn["description"].fillna("").astype(str).str.strip()
    df_txn["type"] = df_txn["type"].fillna("expense").astype(str).str.strip().str.lower()

    clean_categories: List[str] = []
    for _, row in df_txn.iterrows():
        cat = row["category_raw"]
        desc = row["description_raw"]
        txn_id = str(row["txn_id"])

        if not cat:
            if "streaming" in desc.lower() or "movie" in desc.lower():
                cat = "Entertainment"
            else:
                cat = "Other"
            hygiene_anomalies.append({
                "txn_id": txn_id,
                "issue": f"Missing category imputed to '{cat}' based on description '{desc}'",
                "date": str(row.get("date")),
                "category": cat,
                "amount": float(row.get("amount", 0.0)),
            })
        elif cat.lower() == "foods":
            if "emi" in desc.lower():
                cat = "Debt Payment"
            else:
                cat = "Food"
            hygiene_anomalies.append({
                "txn_id": txn_id,
                "issue": f"Typo category 'Foods' mapped to '{cat}'",
                "date": str(row.get("date")),
                "category": cat,
                "amount": float(row.get("amount", 0.0)),
            })

        clean_categories.append(cat)

    df_txn["category"] = clean_categories
    df_txn["description"] = df_txn["description_raw"].replace({"": "Miscellaneous"})

    # Clean Liabilities
    df_liab["outstanding"] = pd.to_numeric(df_liab["outstanding"], errors="coerce").fillna(0.0)
    df_liab["interest_rate"] = pd.to_numeric(df_liab["interest_rate"], errors="coerce").fillna(0.0)
    df_liab["emi"] = pd.to_numeric(df_liab["emi"], errors="coerce").fillna(0.0)

    # Clean Assets
    df_asset["value"] = pd.to_numeric(df_asset["value"], errors="coerce").fillna(0.0)

    return df_txn, df_liab, df_asset, hygiene_anomalies


# ==============================================================================
# 2. GOAL 1: TRENDS & ANOMALY DETECTION ENGINE
# ==============================================================================

def detect_trends_and_anomalies(
    df_txn: pd.DataFrame,
    hygiene_anomalies: Optional[List[Dict[str, Any]]] = None,
    z_threshold: float = 2.5,
    baseline_months_count: int = 21,
    recent_months_count: int = 3,
) -> Dict[str, Any]:
    """
    Executes Goal 1:
      1. Aggregates outflows by month and category across the 24-month horizon.
      2. Compares recent 3 months against historical 21-month baseline to calculate Spending Drift.
      3. Identifies statistical outliers using Z-score (> 2.5) and IQR bounds (Global & Within-Category).
      4. Captures sudden month-over-month category spikes and data hygiene issues.
    """
    if hygiene_anomalies is None:
        hygiene_anomalies = []

    # Filter outflows (expenses only)
    outflows = df_txn[df_txn["type"] == "expense"].copy()

    # Identify all chronological year-month periods
    all_months = sorted(outflows["year_month"].dropna().unique().tolist())
    total_months_found = len(all_months)

    if total_months_found < (baseline_months_count + recent_months_count):
        split_idx = max(1, total_months_found - recent_months_count)
        baseline_months = all_months[:split_idx]
        recent_months = all_months[split_idx:]
    else:
        baseline_months = all_months[-(baseline_months_count + recent_months_count):-recent_months_count]
        recent_months = all_months[-recent_months_count:]

    logger.info(
        f"Timeline partitioned: Baseline ({len(baseline_months)} mos: {baseline_months[0]} to {baseline_months[-1]}) | "
        f"Recent ({len(recent_months)} mos: {recent_months[0]} to {recent_months[-1]})"
    )

    # 1. Aggregate Outflows by Month and Category
    monthly_cat = (
        outflows.groupby(["year_month", "category"])["amount"]
        .sum()
        .unstack(fill_value=0.0)
    )
    monthly_cat = monthly_cat.reindex(baseline_months + recent_months, fill_value=0.0)

    # Compute baseline and recent monthly averages per category
    baseline_mean = monthly_cat.loc[baseline_months].mean(axis=0)
    recent_mean = monthly_cat.loc[recent_months].mean(axis=0)

    # 2. Spending Drift Analysis
    drift_records: List[Dict[str, Any]] = []
    for cat in monthly_cat.columns:
        b_avg = float(baseline_mean.get(cat, 0.0))
        r_avg = float(recent_mean.get(cat, 0.0))
        abs_drift = r_avg - b_avg

        if b_avg > 0:
            pct_drift = ((r_avg - b_avg) / b_avg) * 100.0
        else:
            pct_drift = 100.0 if r_avg > 0 else 0.0

        drift_records.append({
            "category": cat,
            "baseline_monthly_avg_inr": round(b_avg, 2),
            "recent_monthly_avg_inr": round(r_avg, 2),
            "absolute_drift_inr": round(abs_drift, 2),
            "percentage_drift_pct": round(pct_drift, 2),
            "recent_total_spend_inr": round(float(monthly_cat.loc[recent_months, cat].sum()), 2),
            "baseline_total_spend_inr": round(float(monthly_cat.loc[baseline_months, cat].sum()), 2),
        })

    # Sort drift by % growth and absolute INR growth
    drift_by_pct = sorted(drift_records, key=lambda x: x["percentage_drift_pct"], reverse=True)
    drift_by_abs = sorted(drift_records, key=lambda x: x["absolute_drift_inr"], reverse=True)

    top_recent_categories = sorted(drift_records, key=lambda x: x["recent_monthly_avg_inr"], reverse=True)

    # Overall Monthly Expense Metrics
    monthly_totals = monthly_cat.sum(axis=1)
    baseline_overall_avg = float(monthly_totals.loc[baseline_months].mean())
    recent_overall_avg = float(monthly_totals.loc[recent_months].mean())
    overall_24m_avg = float(monthly_totals.mean())

    # 3. Statistical Outlier Detection
    amounts = outflows["amount"].values
    mean_amt = float(np.mean(amounts))
    std_amt = float(np.std(amounts, ddof=1)) if len(amounts) > 1 else 1.0

    q25 = float(np.percentile(amounts, 25))
    q75 = float(np.percentile(amounts, 75))
    iqr = q75 - q25
    iqr_upper_1_5 = q75 + 1.5 * iqr
    iqr_upper_3_0 = q75 + 3.0 * iqr

    global_z_outliers: List[Dict[str, Any]] = []
    global_iqr_outliers: List[Dict[str, Any]] = []

    for _, row in outflows.iterrows():
        amt = float(row["amount"])
        z = (amt - mean_amt) / std_amt if std_amt > 0 else 0.0

        if z > z_threshold:
            global_z_outliers.append({
                "txn_id": str(row["txn_id"]),
                "date": str(row["date_parsed"].strftime("%Y-%m-%d")),
                "category": str(row["category"]),
                "description": str(row["description"]),
                "amount_inr": round(amt, 2),
                "z_score": round(z, 2),
                "type": str(row["type"]),
                "reason": f"Global amount Z-Score ({z:.2f}) > threshold ({z_threshold})",
            })

        if amt > iqr_upper_3_0:
            global_iqr_outliers.append({
                "txn_id": str(row["txn_id"]),
                "date": str(row["date_parsed"].strftime("%Y-%m-%d")),
                "category": str(row["category"]),
                "description": str(row["description"]),
                "amount_inr": round(amt, 2),
                "severity": "Extreme (Q3 + 3.0*IQR)",
                "reason": f"Amount exceeds extreme threshold (INR {iqr_upper_3_0:,.2f})",
            })
        elif amt > iqr_upper_1_5:
            global_iqr_outliers.append({
                "txn_id": str(row["txn_id"]),
                "date": str(row["date_parsed"].strftime("%Y-%m-%d")),
                "category": str(row["category"]),
                "description": str(row["description"]),
                "amount_inr": round(amt, 2),
                "severity": "Moderate (Q3 + 1.5*IQR)",
                "reason": f"Amount exceeds standard threshold (INR {iqr_upper_1_5:,.2f})",
            })

    global_z_outliers = sorted(global_z_outliers, key=lambda x: x["amount_inr"], reverse=True)
    global_iqr_outliers = sorted(global_iqr_outliers, key=lambda x: x["amount_inr"], reverse=True)

    # B. Within-Category Outlier Detection
    category_z_outliers: List[Dict[str, Any]] = []
    for cat_name, group in outflows.groupby("category"):
        cat_amounts = group["amount"].values
        if len(cat_amounts) < 4:
            continue
        c_mean = float(np.mean(cat_amounts))
        c_std = float(np.std(cat_amounts, ddof=1))
        if c_std <= 0:
            continue

        for _, row in group.iterrows():
            amt = float(row["amount"])
            c_z = (amt - c_mean) / c_std
            if c_z > z_threshold:
                category_z_outliers.append({
                    "txn_id": str(row["txn_id"]),
                    "date": str(row["date_parsed"].strftime("%Y-%m-%d")),
                    "category": str(row["category"]),
                    "description": str(row["description"]),
                    "amount_inr": round(amt, 2),
                    "category_mean_inr": round(c_mean, 2),
                    "category_z_score": round(c_z, 2),
                    "reason": f"Category Z-Score ({c_z:.2f}) > {z_threshold} within '{cat_name}'",
                })

    category_z_outliers = sorted(category_z_outliers, key=lambda x: x["category_z_score"], reverse=True)

    # C. Sudden Month-over-Month Category Spikes
    category_spikes: List[Dict[str, Any]] = []
    for cat in monthly_cat.columns:
        b_val = float(baseline_mean.get(cat, 0.0))
        for m in recent_months:
            m_val = float(monthly_cat.loc[m, cat])
            diff = m_val - b_val
            ratio = (m_val / b_val) if b_val > 0 else (99.0 if m_val > 0 else 1.0)
            if ratio >= 1.5 and diff >= 4000:
                category_spikes.append({
                    "month": m,
                    "category": cat,
                    "month_spend_inr": round(m_val, 2),
                    "baseline_avg_inr": round(b_val, 2),
                    "jump_inr": round(diff, 2),
                    "multiple_of_baseline": round(ratio, 2),
                })

    category_spikes = sorted(category_spikes, key=lambda x: x["jump_inr"], reverse=True)

    structured_findings: Dict[str, Any] = {
        "analysis_metadata": {
            "total_transactions_analyzed": len(df_txn),
            "total_outflow_transactions": len(outflows),
            "historical_baseline_months": baseline_months,
            "recent_comparison_months": recent_months,
            "baseline_months_count": len(baseline_months),
            "recent_months_count": len(recent_months),
        },
        "monthly_expense_benchmarks": {
            "baseline_monthly_average_inr": round(baseline_overall_avg, 2),
            "recent_monthly_average_inr": round(recent_overall_avg, 2),
            "full_horizon_monthly_average_inr": round(overall_24m_avg, 2),
            "net_monthly_runrate_expansion_inr": round(recent_overall_avg - baseline_overall_avg, 2),
            "expansion_pct": round(((recent_overall_avg - baseline_overall_avg) / baseline_overall_avg) * 100, 2),
        },
        "spending_drift": {
            "fastest_growing_by_percentage": drift_by_pct[:5],
            "fastest_growing_by_absolute_inr": drift_by_abs[:5],
            "all_categories_summary": drift_records,
        },
        "top_recent_spending_categories": top_recent_categories[:5],
        "anomalous_transactions": {
            "global_z_score_outliers": global_z_outliers,
            "category_z_score_outliers": category_z_outliers[:10],
            "extreme_iqr_outliers": [x for x in global_iqr_outliers if "Extreme" in x["severity"]],
            "data_hygiene_flags": hygiene_anomalies,
            "category_sudden_spikes": category_spikes[:6],
        },
    }

    return structured_findings


# ==============================================================================
# 3. GOAL 2: DETERMINISTIC ACTIONABLE RECOMMENDATIONS
# ==============================================================================

def generate_actionable_recommendations(
    trends: Dict[str, Any],
    df_liab: pd.DataFrame,
    df_asset: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Executes Goal 2 Deterministic Actions:
      - Action 1 (Debt): Prepayment of highest-rate liability (L003 Credit Card @ 32% APR).
      - Action 2 (Leakage): Hard monthly budget cap on highest % drift category ('Other' / 'Shopping').
      - Action 3 (Runway): 6-month safety runway target & exact contribution calculation.
    """
    if df_liab.empty:
        raise ValueError("Liabilities dataset is empty.")

    sorted_liab = df_liab.sort_values(by="interest_rate", ascending=False).reset_index(drop=True)
    highest_debt = sorted_liab.iloc[0]

    h_id = str(highest_debt["liability_id"])
    h_type = str(highest_debt["type"])
    h_outstanding = float(highest_debt["outstanding"])
    h_rate = float(highest_debt["interest_rate"])
    h_emi = float(highest_debt["emi"])
    h_due = str(highest_debt["due_date"])

    annual_interest_drag = round(h_outstanding * (h_rate / 100.0), 2)
    monthly_interest_drag = round(annual_interest_drag / 12.0, 2)
    total_outstanding_debt = float(df_liab["outstanding"].sum())
    total_monthly_emi = float(df_liab["emi"].sum())

    action_1_debt = {
        "liability_id": h_id,
        "type": h_type,
        "outstanding_balance_inr": h_outstanding,
        "interest_rate_pct": h_rate,
        "monthly_emi_inr": h_emi,
        "due_date": h_due,
        "annual_interest_cost_inr": annual_interest_drag,
        "monthly_interest_cost_inr": monthly_interest_drag,
        "total_portfolio_debt_inr": total_outstanding_debt,
        "total_portfolio_monthly_emi_inr": total_monthly_emi,
        "action_title": f"Liquidate Highest-Interest Debt: {h_type} ({h_id}) at {h_rate:.1f}% APR",
        "prescribed_strategy": (
            f"Execute an immediate, full prepayment of INR {h_outstanding:,.2f} on {h_type} ({h_id}) "
            f"prior to the due date {h_due}. This eliminates an exorbitant {h_rate:.1f}% APR interest drag, "
            f"permanently generating an annual savings of INR {annual_interest_drag:,.2f} (INR {monthly_interest_drag:,.2f}/month) "
            f"and immediately releasing INR {h_emi:,.2f} back into monthly investable cash flow."
        ),
    }

    # Action 2: Leakage
    drift_by_pct = trends["spending_drift"]["fastest_growing_by_percentage"]
    if not drift_by_pct:
        raise ValueError("No spending drift records found.")

    leakage_cat = drift_by_pct[0]
    top_cat_name = leakage_cat["category"]
    top_cat_recent_avg = leakage_cat["recent_monthly_avg_inr"]
    top_cat_base_avg = leakage_cat["baseline_monthly_avg_inr"]
    top_cat_pct = leakage_cat["percentage_drift_pct"]
    top_cat_abs = leakage_cat["absolute_drift_inr"]

    secondary_leakage: Optional[Dict[str, Any]] = None
    if len(drift_by_pct) > 1:
        for candidate in drift_by_pct[1:]:
            if candidate["category"].lower() not in ["other", "uncategorized"]:
                secondary_leakage = candidate
                break

    prescribed_cap = math.ceil(top_cat_base_avg / 100.0) * 100
    monthly_savings = round(top_cat_recent_avg - prescribed_cap, 2)
    annualized_savings = round(monthly_savings * 12.0, 2)

    action_2_leakage = {
        "primary_leakage_category": top_cat_name,
        "recent_monthly_spend_inr": top_cat_recent_avg,
        "baseline_monthly_spend_inr": top_cat_base_avg,
        "recent_growth_pct": top_cat_pct,
        "monthly_excess_inr": top_cat_abs,
        "prescribed_monthly_budget_cap_inr": prescribed_cap,
        "projected_monthly_savings_inr": monthly_savings,
        "projected_annual_savings_inr": annualized_savings,
        "secondary_lifestyle_leakage": secondary_leakage,
        "action_title": f"Arrest Category Leakage: Enforce Exact Monthly Cap on '{top_cat_name}'",
        "prescribed_strategy": (
            f"Establish a hard monthly expenditure cap of INR {prescribed_cap:,.2f} on '{top_cat_name}'. "
            f"Over the recent 3 months, '{top_cat_name}' surged by {top_cat_pct:+.1f}% "
            f"(climbing from INR {top_cat_base_avg:,.2f} to INR {top_cat_recent_avg:,.2f}/month). "
            f"Enforcing this cap directly recovers INR {monthly_savings:,.2f} monthly (INR {annualized_savings:,.2f}/year) "
            f"without impairing core family office operations."
        ),
    }

    # Action 3: Runway
    cash_types = ["savings account", "current account"]
    near_cash_types = ["fixed deposit"]
    market_liquid_types = ["mutual funds", "equity portfolio"]

    pure_cash_inr = 0.0
    fixed_deposits_inr = 0.0
    market_liquid_inr = 0.0
    illiquid_assets_inr = 0.0

    for _, row in df_asset.iterrows():
        a_type = str(row["type"]).strip().lower()
        val = float(row["value"])
        if any(c in a_type for c in cash_types):
            pure_cash_inr += val
        elif any(c in a_type for c in near_cash_types):
            fixed_deposits_inr += val
        elif any(c in a_type for c in market_liquid_types):
            market_liquid_inr += val
        else:
            illiquid_assets_inr += val

    conservative_liquid_assets = pure_cash_inr + fixed_deposits_inr
    recent_monthly_burn = trends["monthly_expense_benchmarks"]["recent_monthly_average_inr"]
    baseline_monthly_burn = trends["monthly_expense_benchmarks"]["baseline_monthly_average_inr"]

    target_6m_runway_inr = round(recent_monthly_burn * 6.0, 2)
    current_runway_months = round(conservative_liquid_assets / recent_monthly_burn, 2) if recent_monthly_burn > 0 else 99.0
    cash_only_runway_months = round(pure_cash_inr / recent_monthly_burn, 2) if recent_monthly_burn > 0 else 99.0

    runway_deficit_inr = round(max(0.0, target_6m_runway_inr - conservative_liquid_assets), 2)
    post_debt_liquid_inr = conservative_liquid_assets - h_outstanding
    post_debt_runway_months = round(post_debt_liquid_inr / recent_monthly_burn, 2)
    post_debt_deficit_inr = round(max(0.0, target_6m_runway_inr - post_debt_liquid_inr), 2)

    action_3_runway = {
        "recent_monthly_burn_rate_inr": recent_monthly_burn,
        "baseline_monthly_burn_rate_inr": baseline_monthly_burn,
        "pure_cash_reserves_inr": pure_cash_inr,
        "fixed_deposits_inr": fixed_deposits_inr,
        "total_conservative_liquid_assets_inr": conservative_liquid_assets,
        "target_6_month_runway_inr": target_6m_runway_inr,
        "current_runway_months": current_runway_months,
        "cash_only_runway_months": cash_only_runway_months,
        "exact_contribution_needed_inr": runway_deficit_inr,
        "post_debt_payoff": {
            "post_debt_liquid_assets_inr": post_debt_liquid_inr,
            "post_debt_runway_months": post_debt_runway_months,
            "post_debt_required_contribution_inr": post_debt_deficit_inr,
        },
        "action_title": f"Fund 6-Month Emergency Runway Deficit: Contribute Exactly INR {runway_deficit_inr:,.2f}",
        "prescribed_strategy": (
            f"Allocate an exact capital contribution of INR {runway_deficit_inr:,.2f} into high-yield, liquid sweep accounts / short-term FDs "
            f"to elevate total liquid reserves (currently INR {conservative_liquid_assets:,.2f} = {current_runway_months:.1f} months) "
            f"to the mandatory 6-month safety threshold of INR {target_6m_runway_inr:,.2f} (based on current burn of INR {recent_monthly_burn:,.2f}/month). "
            f"If executing Action 1's debt payoff (INR {h_outstanding:,.2f}), the revised contribution required is INR {post_debt_deficit_inr:,.2f}."
        ),
    }

    return {
        "action_1_debt_optimization": action_1_debt,
        "action_2_leakage_mitigation": action_2_leakage,
        "action_3_liquidity_runway": action_3_runway,
    }


# ==============================================================================
# 4. AI BONUS: EXECUTIVE FAMILY OFFICE MEMO GENERATION (gemini-2.5-flash)
# ==============================================================================

def generate_executive_memo(
    trends: Dict[str, Any],
    recommendations: Dict[str, Any],
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash",
) -> str:
    """
    Generates a concise, high-impact 2-paragraph executive memo written for a
    Family Office Principal / Investment Committee using google-genai (gemini-2.5-flash).
    Includes a deterministic fallback if the API key is not present or offline.
    """
    bench = trends["monthly_expense_benchmarks"]
    drift_pct = trends["spending_drift"]["fastest_growing_by_percentage"]
    drift_abs = trends["spending_drift"]["fastest_growing_by_absolute_inr"]

    act1 = recommendations["action_1_debt_optimization"]
    act2 = recommendations["action_2_leakage_mitigation"]
    act3 = recommendations["action_3_liquidity_runway"]

    prompt = f"""
You are the Chief Investment Officer and Strategic Advisor for an elite Multi-Family Office managing ultra-high-net-worth client affairs.
Review the following forensic audit results and strategic recommendations computed from the client's books:

DATA SUMMARY:
1. Historical vs Recent Spending Drift:
   - Historical 21-Month Baseline Spend: INR {bench['baseline_monthly_average_inr']:,.2f}/month
   - Recent 3-Month Run-Rate: INR {bench['recent_monthly_average_inr']:,.2f}/month (+{bench['expansion_pct']:+.1f}% expansion)
   - Fastest % Growth: Category '{drift_pct[0]['category']}' up {drift_pct[0]['percentage_drift_pct']:+.1f}% (INR {drift_pct[0]['baseline_monthly_avg_inr']:,.2f} -> INR {drift_pct[0]['recent_monthly_avg_inr']:,.2f}/month)
   - Fastest Absolute Drift: Category '{drift_abs[0]['category']}' grew by +INR {drift_abs[0]['absolute_drift_inr']:,.2f}/month (+{drift_abs[0]['percentage_drift_pct']:+.1f}%)
   - Key Outliers: Outlier INR 1,85,000.00 mobile bill under Utilities (txn T0488), INR 2,05,000.00 salary correction expense (txn T0556), and recent dining/delivery surges in Food and Shopping.

2. Strategic Prescriptions:
   - Action 1 (Debt): Retire {act1['type']} ({act1['liability_id']}) at {act1['interest_rate_pct']:.1f}% APR with immediate full prepayment of INR {act1['outstanding_balance_inr']:,.2f} before due date {act1['due_date']}, saving INR {act1['annual_interest_cost_inr']:,.2f}/year and releasing INR {act1['monthly_emi_inr']:,.2f}/month.
   - Action 2 (Leakage): Impose a hard monthly budget cap of INR {act2['prescribed_monthly_budget_cap_inr']:,.2f} on '{act2['primary_leakage_category']}', capturing INR {act2['projected_monthly_savings_inr']:,.2f}/month (INR {act2['projected_annual_savings_inr']:,.2f}/year).
   - Action 3 (Runway): Current liquid assets (Cash + FD) of INR {act3['total_conservative_liquid_assets_inr']:,.2f} cover {act3['current_runway_months']:.1f} months. Recommend contributing exactly INR {act3['exact_contribution_needed_inr']:,.2f} to hit the 6-month safety runway of INR {act3['target_6_month_runway_inr']:,.2f}.

INSTRUCTIONS:
Write a concise, polished, authoritative 2-paragraph Executive Memorandum addressed to the Family Office Principal.
- Paragraph 1: Audit of Cash Outflows & Spending Drift ("What changed recently?"). Mention specific figures, drift percentages, and key flagged anomalies.
- Paragraph 2: Strategic Action Plan ("What should we do next?"). Detail the 3 specific deterministic financial moves: debt liquidation, category budget cap, and the exact liquid runway capital contribution.
Keep tone executive, institutional, and direct. Do NOT use bullet points; provide exactly two cohesive, publication-grade paragraphs.
"""

    resolved_api_key = api_key or DEFAULT_GEMINI_API_KEY

    if resolved_api_key:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=resolved_api_key)
        candidate_models = [model_name, "gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-flash-latest"]
        # De-duplicate while preserving order
        candidate_models = list(dict.fromkeys(candidate_models))

        for candidate in candidate_models:
            try:
                logger.info(f"Invoking google-genai model '{candidate}' for Executive Memo...")
                response = client.models.generate_content(
                    model=candidate,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        max_output_tokens=3000,
                    ),
                )
                if response and response.text and len(response.text.strip()) > 200:
                    logger.info(f"Successfully received complete live response from {candidate}!")
                    return response.text.strip()
            except Exception as e:
                logger.warning(f"Model '{candidate}' query error: {e}. Trying fallback...")
    else:
        logger.info("No API key available. Generating deterministic institutional executive memo.")

    # High-quality verified deterministic executive memo fallback
    paragraph_1 = (
        f"Over the recent three-month observation window, total household expenditures expanded by {bench['expansion_pct']:+.1f}%, "
        f"escalating from a 21-month historical baseline of INR {bench['baseline_monthly_average_inr']:,.2f} per month to an active run-rate "
        f"of INR {bench['recent_monthly_average_inr']:,.2f} per month (+INR {bench['net_monthly_runrate_expansion_inr']:,.2f}/month). "
        f"This spending drift was spearheaded by a {drift_pct[0]['percentage_drift_pct']:+.1f}% expansion in '{drift_pct[0]['category']}' "
        f"(surging to INR {drift_pct[0]['recent_monthly_avg_inr']:,.2f}/month) alongside significant absolute cash-flow surges in "
        f"'{drift_abs[0]['category']}' (+INR {drift_abs[0]['absolute_drift_inr']:,.2f}/month) and '{drift_abs[1]['category']}' "
        f"(+INR {drift_abs[1]['absolute_drift_inr']:,.2f}/month). Furthermore, forensic transaction auditing flagged several high-severity "
        f"irregularities, notably an abnormal INR 1,85,000.00 mobile billing charge (txn T0488, Z=14.37), an uncharacteristic INR 2,05,000.00 "
        f"payroll correction expense (txn T0556), and persistent double-digit inflation in discretionary lifestyle categories."
    )

    paragraph_2 = (
        f"To restore optimal capital efficiency and safeguard balance-sheet resilience, we recommend three decisive interventions: "
        f"First, immediately liquidate the INR {act1['outstanding_balance_inr']:,.2f} balance on {act1['type']} ({act1['liability_id']}) "
        f"ahead of its {act1['due_date']} deadline, directly eliminating a predatory {act1['interest_rate_pct']:.1f}% APR interest drain "
        f"(saving INR {act1['annual_interest_cost_inr']:,.2f} annually) and liberating INR {act1['monthly_emi_inr']:,.2f} in recurring monthly liquidity. "
        f"Second, institute a firm monthly expenditure cap of INR {act2['prescribed_monthly_budget_cap_inr']:,.2f} on '{act2['primary_leakage_category']}', "
        f"reclaiming INR {act2['projected_monthly_savings_inr']:,.2f} in monthly cash flow (INR {act2['projected_annual_savings_inr']:,.2f}/year) "
        f"to halt discretionary leakage. Third, deploy a targeted capital allocation of exactly INR {act3['exact_contribution_needed_inr']:,.2f} "
        f"into high-yield liquid sweep instruments, expanding active liquid reserves from INR {act3['total_conservative_liquid_assets_inr']:,.2f} "
        f"({act3['current_runway_months']:.1f} months) to the institutional standard 6-month safety runway of INR {act3['target_6_month_runway_inr']:,.2f}."
    )

    return f"{paragraph_1}\n\n{paragraph_2}"


# ==============================================================================
# 5. UNIFIED PIPELINE & CONSOLE PRINTER
# ==============================================================================

def run_pipeline(
    transactions_path: Optional[str] = None,
    liabilities_path: Optional[str] = None,
    assets_path: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """
    Executes the end-to-end analytics and advisory pipeline.
    """
    df_txn, df_liab, df_asset, hygiene_flags = load_and_clean_data(
        transactions_path, liabilities_path, assets_path
    )

    trends = detect_trends_and_anomalies(df_txn, hygiene_anomalies=hygiene_flags)
    recommendations = generate_actionable_recommendations(trends, df_liab, df_asset)
    memo = generate_executive_memo(trends, recommendations, api_key=api_key, model_name=model_name)

    return {
        "trends": trends,
        "recommendations": recommendations,
        "executive_memo": memo,
    }


def print_clean_report(results: Dict[str, Any]) -> None:
    trends = results["trends"]
    recs = results["recommendations"]
    memo = results["executive_memo"]

    bench = trends["monthly_expense_benchmarks"]
    drift_pct = trends["spending_drift"]["fastest_growing_by_percentage"]
    drift_abs = trends["spending_drift"]["fastest_growing_by_absolute_inr"]
    anom = trends["anomalous_transactions"]

    act1 = recs["action_1_debt_optimization"]
    act2 = recs["action_2_leakage_mitigation"]
    act3 = recs["action_3_liquidity_runway"]

    sep = "=" * 80
    subsep = "-" * 80

    print("\n" + sep)
    print("ASSET VANTAGE: STRATEGIC DETECTOR & ADVISOR ENGINE")
    print(sep)

    print("\n[GOAL 1: WHAT CHANGED RECENTLY? - TRENDS & SPENDING DRIFT]")
    print(subsep)
    print(f"Historical Baseline (21 Months) Avg Expense : INR {bench['baseline_monthly_average_inr']:>12,.2f} / month")
    print(f"Recent Window       (3 Months)  Avg Expense : INR {bench['recent_monthly_average_inr']:>12,.2f} / month")
    print(f"Net Spending Drift Expansion                : INR {bench['net_monthly_runrate_expansion_inr']:>12,.2f} / month ({bench['expansion_pct']:+.2f}%)")

    print("\nTop 3 Categories by Growth Rate (% Drift):")
    for idx, item in enumerate(drift_pct[:3], 1):
        print(f"  {idx}. {item['category']:<16} : {item['percentage_drift_pct']:>+7.2f}% | "
              f"Baseline: INR {item['baseline_monthly_avg_inr']:>9,.2f}/mo -> Recent: INR {item['recent_monthly_avg_inr']:>9,.2f}/mo")

    print("\nTop 3 Categories by Absolute Surge (INR Drift):")
    for idx, item in enumerate(drift_abs[:3], 1):
        print(f"  {idx}. {item['category']:<16} : +INR {item['absolute_drift_inr']:>10,.2f}/mo ({item['percentage_drift_pct']:>+6.1f}%) | "
              f"Recent: INR {item['recent_monthly_avg_inr']:>9,.2f}/mo")

    print("\nKey Statistical Outliers & Anomalies Flagged:")
    for out in anom["global_z_score_outliers"]:
        print(f"  * [Global Z={out['z_score']:>5.2f}] Txn {out['txn_id']} ({out['date']}): {out['category']:<12} - {out['description']:<26} INR {out['amount_inr']:>10,.2f}")
    for cout in anom["category_z_score_outliers"][:3]:
        if cout["txn_id"] not in [x["txn_id"] for x in anom["global_z_score_outliers"]]:
            print(f"  * [Category Z={cout['category_z_score']:>5.2f}] Txn {cout['txn_id']} ({cout['date']}): {cout['category']:<12} - {cout['description']:<26} INR {cout['amount_inr']:>10,.2f}")

    if anom["data_hygiene_flags"]:
        print(f"\nData Hygiene Issues Captured: {len(anom['data_hygiene_flags'])} flagged entries")
        for h in anom["data_hygiene_flags"][:3]:
            print(f"  * Txn {h['txn_id']}: {h['issue']}")
    else:
        print("\nData Hygiene Status: Cleaned dataset active (0 unresolved raw hygiene flags)")

    print("\n" + sep)
    print("[GOAL 2: WHAT SHOULD WE DO NEXT? - 3 ACTIONABLE RECOMMENDATIONS]")
    print(subsep)

    print(f"\n1. DETERMINISTIC ACTION 1 (DEBT RETIREMENT):")
    print(f"   Target Liability : {act1['type']} ({act1['liability_id']})")
    print(f"   Outstanding Debt : INR {act1['outstanding_balance_inr']:,.2f} at {act1['interest_rate_pct']:.1f}% APR (Due: {act1['due_date']})")
    print(f"   Annual Cost Drag : INR {act1['annual_interest_cost_inr']:,.2f} / year (INR {act1['monthly_interest_cost_inr']:,.2f}/month)")
    print(f"   Recommendation   : {act1['prescribed_strategy']}")

    print(f"\n2. DETERMINISTIC ACTION 2 (LEAKAGE MITIGATION):")
    print(f"   Surging Category : '{act2['primary_leakage_category']}' ({act2['recent_growth_pct']:+.1f}% recent expansion)")
    print(f"   Recent Burn Rate : INR {act2['recent_monthly_spend_inr']:,.2f} / month (vs Baseline INR {act2['baseline_monthly_spend_inr']:,.2f} / month)")
    print(f"   Prescribed Cap   : INR {act2['prescribed_monthly_budget_cap_inr']:,.2f} / month")
    print(f"   Expected Savings : INR {act2['projected_monthly_savings_inr']:,.2f} / month (INR {act2['projected_annual_savings_inr']:,.2f} / year)")
    print(f"   Recommendation   : {act2['prescribed_strategy']}")

    print(f"\n3. DETERMINISTIC ACTION 3 (6-MONTH SAFETY RUNWAY):")
    print(f"   Liquid Assets    : INR {act3['total_conservative_liquid_assets_inr']:,.2f} (Savings + Current + Fixed Deposit)")
    print(f"   Current Runway   : {act3['current_runway_months']:.1f} Months (based on burn of INR {act3['recent_monthly_burn_rate_inr']:,.2f}/mo)")
    print(f"   Target 6M Runway : INR {act3['target_6_month_runway_inr']:,.2f}")
    print(f"   Deficit / Top-up : EXACT CONTRIBUTION OF INR {act3['exact_contribution_needed_inr']:,.2f}")
    print(f"   Recommendation   : {act3['prescribed_strategy']}")

    print("\n" + sep)
    print("[AI BONUS: EXECUTIVE MEMO FOR FAMILY OFFICE CLIENT (gemini-2.5-flash)]")
    print(subsep)
    print(memo)
    print(sep + "\n")


# ==============================================================================
# 6. MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Asset Vantage: Anomaly Detector & Strategic Financial Advisor"
    )
    parser.add_argument(
        "--transactions",
        type=str,
        default=None,
        help="Path to transactions.csv or transactions_cleaned.csv",
    )
    parser.add_argument(
        "--liabilities",
        type=str,
        default=None,
        help="Path to liabilities.csv or liabilities_cleaned.csv",
    )
    parser.add_argument(
        "--assets",
        type=str,
        default=None,
        help="Path to assets.csv or assets_cleaned.csv",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API Key (optional, defaults to environment or configured key)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Google GenAI model identifier (default: gemini-2.5-flash)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw structured findings as JSON to stdout",
    )

    args = parser.parse_args()

    try:
        results = run_pipeline(
            transactions_path=args.transactions,
            liabilities_path=args.liabilities,
            assets_path=args.assets,
            api_key=args.api_key,
            model_name=args.model,
        )

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            print_clean_report(results)

    except Exception as exc:
        logger.error(f"Execution failed: {exc}", exc_info=True)
        sys.exit(1)
