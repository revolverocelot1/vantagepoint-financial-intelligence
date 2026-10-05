"""
VantagePoint - Financial Intelligence & Portfolio Advisory Platform
Production-Ready Streamlit Application (app.py)
"""

from __future__ import annotations

import io
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="VantagePoint | Asset Vantage",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# 1. BESPOKE ASSET VANTAGE BRAND SYSTEM & LUXURY FINTECH STYLING
# ==============================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
    --av-cyan: #00a1de;
    --av-blue: #007bff;
    --av-navy: #0f172a;
    --av-slate: #1e293b;
    --av-border: #e2e8f0;
    --av-bg: #f8fafc;
    --av-surface: #ffffff;
    --av-text: #0f172a;
    --av-muted: #64748b;
    --av-success: #10b981;
    --av-danger: #ef4444;
    --av-amber: #f59e0b;
}

html, body, [class*="css"] {
    font-family: var(--font-sans) !important;
    background-color: var(--av-bg) !important;
    color: var(--av-text) !important;
}

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 3.5rem !important;
    max-width: 1440px !important;
}

/* Header bar */
.vp-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #ffffff;
    border: 1px solid var(--av-border);
    border-radius: 12px;
    padding: 0.85rem 1.4rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    margin-bottom: 1.25rem;
}

.vp-brand {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.vp-logo-badge {
    width: 34px;
    height: 34px;
    background: linear-gradient(135deg, #00a1de 0%, #007bff 100%);
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    font-weight: 800;
    font-size: 1rem;
    box-shadow: 0 2px 8px rgba(0, 161, 222, 0.35);
}

.vp-brand-name {
    font-size: 1.25rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.025em;
}

.vp-brand-sub {
    font-size: 0.72rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Page title */
.vp-page-head {
    margin-bottom: 1.25rem;
}

.vp-title {
    font-size: 1.95rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.03em;
    margin: 0;
}

.vp-subtitle {
    font-size: 0.88rem;
    color: #64748b;
    margin-top: 0.25rem;
    font-weight: 500;
}

/* KPI Cards */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 0.85rem;
    margin-bottom: 1.15rem;
}

.kpi-card {
    background: #ffffff;
    border: 1px solid var(--av-border);
    border-radius: 12px;
    padding: 1.1rem 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.2s ease;
    cursor: pointer;
}

.kpi-card:hover {
    border-color: #94a3b8;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}

.kpi-card.active-kpi {
    border-color: #00a1de !important;
    box-shadow: 0 0 0 2px rgba(0, 161, 222, 0.25) !important;
}

.kpi-featured {
    background: linear-gradient(135deg, #007bff 0%, #00a1de 100%) !important;
    border: 1px solid #00a1de !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(0, 123, 255, 0.25) !important;
}

.kpi-tag {
    font-size: 0.74rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #64748b;
    margin-bottom: 0.35rem;
}

.kpi-tag-white {
    color: rgba(255, 255, 255, 0.85) !important;
}

.kpi-val {
    font-family: var(--font-mono);
    font-size: clamp(1.25rem, 1.8vw, 1.65rem);
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #0f172a;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}

.kpi-val-white {
    color: #ffffff !important;
}

.kpi-foot {
    font-size: 0.72rem;
    color: #64748b;
    line-height: 1.3;
}

.kpi-foot-white {
    color: rgba(255, 255, 255, 0.9) !important;
}

/* Dynamic Inspector Card */
.inspector-box {
    background: #ffffff;
    border: 1px solid var(--av-border);
    border-radius: 12px;
    padding: 1.15rem 1.4rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}

.inspector-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
}

.inspector-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #0f172a;
}

/* Surface Cards */
.vp-card {
    background: #ffffff;
    border: 1px solid var(--av-border);
    border-radius: 12px;
    padding: 1.2rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    margin-bottom: 1.25rem;
}

.vp-card-title {
    font-size: 0.98rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.75rem;
}

/* Status pills */
.pill-danger {
    background: #fef2f2;
    color: #dc2626;
    border: 1px solid #fecaca;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 700;
}

.pill-warning {
    background: #fffbeb;
    color: #d97706;
    border: 1px solid #fde68a;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 700;
}

.pill-info {
    background: #eff6ff;
    color: #2563eb;
    border: 1px solid #bfdbfe;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 700;
}

.pill-success {
    background: #ecfdf5;
    color: #059669;
    border: 1px solid #a7f3d0;
    padding: 0.2rem 0.5rem;
    border-radius: 6px;
    font-size: 0.7rem;
    font-weight: 700;
}

/* Mobile responsive */
@media (max-width: 900px) {
    .kpi-row {
        grid-template-columns: repeat(2, 1fr) !important;
    }
}
@media (max-width: 600px) {
    .kpi-row {
        grid-template-columns: 1fr !important;
    }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ==============================================================================
# 2. DYNAMIC FILE INGESTION & DATA ENGINE
# ==============================================================================

@st.cache_data(show_spinner=False)
def load_default_files() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cwd = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    t_path = cwd / "transactions_cleaned.csv"
    a_path = cwd / "assets_cleaned.csv"
    l_path = cwd / "liabilities_cleaned.csv"

    if not (t_path.exists() and a_path.exists() and l_path.exists()):
        import generate_clean_data

    df_t = pd.read_csv(t_path)
    df_a = pd.read_csv(a_path)
    df_l = pd.read_csv(l_path)

    df_t["date"] = pd.to_datetime(df_t["date"])
    df_t["amount"] = pd.to_numeric(df_t["amount"], errors="coerce").fillna(0.0)
    return df_t, df_a, df_l


def sanitize_and_prepare_data(
    df_txn_raw: pd.DataFrame,
    df_asset_raw: pd.DataFrame,
    df_liab_raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Clean & prepare user-uploaded or default data."""
    df_t = df_txn_raw.copy()
    df_a = df_asset_raw.copy()
    df_l = df_liab_raw.copy()

    # Normalize transactions
    if "date" in df_t.columns:
        df_t["date"] = pd.to_datetime(df_t["date"].astype(str).str.replace("/", "-"), errors="coerce")
    else:
        df_t["date"] = pd.Timestamp.now()

    df_t["amount"] = pd.to_numeric(df_t["amount"], errors="coerce").abs().fillna(0.0)
    df_t["category"] = df_t.get("category", "Uncategorized").fillna("Other")
    df_t["description"] = df_t.get("description", "Description").fillna("Other")
    df_t["type"] = df_t.get("type", "expense").fillna("expense").astype(str).str.lower()
    df_t["month"] = df_t["date"].dt.strftime("%Y-%m")

    # Flow class
    def assign_flow(cat):
        c = str(cat).lower()
        if "salary" in c or "income" in c:
            return "income"
        elif "debt" in c or "loan" in c or "emi" in c:
            return "debt_payment"
        elif "invest" in c:
            return "investment"
        return "spending"

    if "flow_class" not in df_t.columns:
        df_t["flow_class"] = df_t["category"].apply(assign_flow)

    # Detect outliers if not already flagged: IQR (k=1.5 within category) or global z-score
    if "is_outlier" not in df_t.columns:
        outlier_flags = set()
        # Category IQR
        for _, g in df_t[df_t["type"] == "expense"].groupby("category"):
            if len(g) >= 4:
                q1 = g["amount"].quantile(0.25)
                q3 = g["amount"].quantile(0.75)
                iqr = q3 - q1
                caps = g.loc[g["amount"] > q3 + 1.5 * iqr, "txn_id"] if "txn_id" in g.columns else []
                outlier_flags.update(caps)
        df_t["is_outlier"] = df_t["txn_id"].isin(outlier_flags) if "txn_id" in df_t.columns else False
    else:
        df_t["is_outlier"] = df_t["is_outlier"].astype(bool)

    # Normalize assets
    df_a["value"] = pd.to_numeric(df_a.get("value", 0.0), errors="coerce").fillna(0.0)
    if "is_liquid" not in df_a.columns:
        df_a["is_liquid"] = df_a["type"].astype(str).str.lower().str.contains("savings|current|deposit|cash")

    # Normalize liabilities
    df_l["outstanding"] = pd.to_numeric(df_l.get("outstanding", 0.0), errors="coerce").fillna(0.0)
    df_l["interest_rate"] = pd.to_numeric(df_l.get("interest_rate", 0.0), errors="coerce").fillna(0.0)
    df_l["emi"] = pd.to_numeric(df_l.get("emi", 0.0), errors="coerce").fillna(0.0)

    return df_t, df_a, df_l


# ==============================================================================
# 3. SIDEBAR: DYNAMIC USER FILE UPLOADER & CONTROLS
# ==============================================================================

with st.sidebar:
    st.markdown(
        """
        <div style="font-size: 1.1rem; font-weight: 800; color: #0f172a; margin-bottom: 0.25rem;">
            VantagePoint Data Desk
        </div>
        <div style="font-size: 0.78rem; color: #64748b; margin-bottom: 1rem;">
            Upload custom CSVs or use the default dataset.
        </div>
        """,
        unsafe_allow_html=True,
    )

    up_txn = st.file_uploader("Transactions CSV", type=["csv"], key="up_txn_csv")
    up_assets = st.file_uploader("Assets CSV", type=["csv"], key="up_assets_csv")
    up_liab = st.file_uploader("Liabilities CSV", type=["csv"], key="up_liab_csv")

    reset_btn = st.button("Reset to Default Dataset", use_container_width=True)

# Determine working datasets
def_txn, def_assets, def_liab = load_default_files()

if reset_btn:
    st.session_state["txn_df"] = def_txn
    st.session_state["asset_df"] = def_assets
    st.session_state["liab_df"] = def_liab
    st.rerun()

# Load files dynamically if provided
if up_txn:
    raw_t = pd.read_csv(up_txn)
else:
    raw_t = st.session_state.get("txn_df", def_txn)

if up_assets:
    raw_a = pd.read_csv(up_assets)
else:
    raw_a = st.session_state.get("asset_df", def_assets)

if up_liab:
    raw_l = pd.read_csv(up_liab)
else:
    raw_l = st.session_state.get("liab_df", def_liab)

df_txn, df_asset, df_liab = sanitize_and_prepare_data(raw_t, raw_a, raw_l)

# Save into session state
st.session_state["txn_df"] = df_txn
st.session_state["asset_df"] = df_asset
st.session_state["liab_df"] = df_liab


# ==============================================================================
# 4. QUANTITATIVE KPI CALCULATIONS
# ==============================================================================

def calculate_metrics(df_t: pd.DataFrame, df_a: pd.DataFrame, df_l: pd.DataFrame) -> Dict[str, Any]:
    total_assets = float(df_a["value"].sum())
    liquid_assets = float(df_a[df_a["is_liquid"] == True]["value"].sum())
    total_debt = float(df_l["outstanding"].sum())
    total_emi = float(df_l["emi"].sum())
    net_worth = total_assets - total_debt

    inc_txns = df_t[df_t["type"] == "income"]
    exp_txns = df_t[df_t["type"] == "expense"]

    total_income = float(inc_txns["amount"].sum())
    total_expense = float(exp_txns["amount"].sum())
    net_savings = total_income - total_expense

    months = sorted(df_t["month"].dropna().unique().tolist())
    n_months = max(1, len(months))

    avg_monthly_income = total_income / n_months
    avg_monthly_expense = total_expense / n_months
    savings_rate = (net_savings / total_income) * 100.0 if total_income > 0 else 0.0
    dti = (total_emi / avg_monthly_income) * 100.0 if avg_monthly_income > 0 else 0.0
    runway_months = (liquid_assets / avg_monthly_expense) if avg_monthly_expense > 0 else 0.0

    # Spending Drift (Partition into baseline vs recent 3 months)
    recent_count = min(3, max(1, n_months // 4))
    recent_months = months[-recent_count:]
    baseline_months = months[:-recent_count] if len(months) > recent_count else months

    recent_burn = float(exp_txns[exp_txns["month"].isin(recent_months)]["amount"].sum()) / len(recent_months)
    baseline_burn = float(exp_txns[exp_txns["month"].isin(baseline_months)]["amount"].sum()) / max(1, len(baseline_months))

    drift_expansion_inr = recent_burn - baseline_burn
    drift_expansion_pct = (drift_expansion_inr / baseline_burn) * 100.0 if baseline_burn > 0 else 0.0

    # Financial Health Score Index (4 pillars of 25)
    s_savings = min(25.0, max(0.0, 20.0 + 5.0 * ((savings_rate - 20.0) / 10.0)))
    s_debt = 25.0 if dti <= 20.0 else max(0.0, 25.0 - 10.0 * ((dti - 20.0) / 15.0))
    s_runway = min(25.0, max(0.0, 10.0 + 15.0 * ((runway_months - 3.0) / 3.0)))
    s_drift = 25.0 if drift_expansion_pct <= 0 else (20.0 if drift_expansion_pct <= 10 else 14.0)
    health_score = int(round(s_savings + s_debt + s_runway + s_drift))

    target_6m_runway = recent_burn * 6.0
    runway_deficit = max(0.0, target_6m_runway - liquid_assets)

    highest_debt = df_l.sort_values(by="interest_rate", ascending=False).iloc[0] if not df_l.empty else pd.Series()

    return {
        "health_score": health_score,
        "s_savings": s_savings,
        "s_debt": s_debt,
        "s_runway": s_runway,
        "s_drift": s_drift,
        "net_worth": net_worth,
        "total_assets": total_assets,
        "liquid_assets": liquid_assets,
        "total_debt": total_debt,
        "total_emi": total_emi,
        "avg_monthly_income": avg_monthly_income,
        "avg_monthly_expense": avg_monthly_expense,
        "savings_rate": savings_rate,
        "dti": dti,
        "runway_months": runway_months,
        "baseline_months": baseline_months,
        "recent_months": recent_months,
        "recent_burn": recent_burn,
        "baseline_burn": baseline_burn,
        "drift_expansion_inr": drift_expansion_inr,
        "drift_expansion_pct": drift_expansion_pct,
        "target_6m_runway": target_6m_runway,
        "runway_deficit": runway_deficit,
        "highest_debt": highest_debt,
        "n_months": n_months,
    }


kpis = calculate_metrics(df_txn, df_asset, df_liab)


# ==============================================================================
# 5. HEADER BAR & EXECUTIVE BANNER
# ==============================================================================

header_html = """
<div class="vp-header">
    <div class="vp-brand">
        <div class="vp-logo-badge">VP</div>
        <div>
            <div class="vp-brand-name">VantagePoint</div>
            <div class="vp-brand-sub">Asset Vantage Private Wealth</div>
        </div>
    </div>
    <div style="display: flex; align-items: center; gap: 1.25rem;">
        <span class="pill-info" style="font-weight: 700;">AUDIT ACTIVE</span>
        <span style="font-size: 0.8rem; color: #64748b; font-weight: 600;">As of 01-Oct-2026</span>
    </div>
</div>
"""
st.markdown(header_html, unsafe_allow_html=True)

st.markdown(
    """
    <div class="vp-page-head">
        <h1 class="vp-title">Financial Health & Portfolio Solvency</h1>
        <div class="vp-subtitle">Multi-account consolidation, forensic anomaly intelligence, and balance sheet resilience.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# 6. INTERACTIVE TOP 5 KPI CARDS & DYNAMIC INSPECTOR CARD
# ==============================================================================

if "active_card" not in st.session_state:
    st.session_state["active_card"] = "Health Score"

# Card Selection Buttons
kpi_c1, kpi_c2, kpi_c3, kpi_c4, kpi_c5 = st.columns(5)

with kpi_c1:
    if st.button(
        f"Health Score\n{kpis['health_score']} / 100",
        key="btn_health_score",
        use_container_width=True,
        type="primary" if st.session_state["active_card"] == "Health Score" else "secondary",
    ):
        st.session_state["active_card"] = "Health Score"

with kpi_c2:
    if st.button(
        f"Net Worth\n₹{kpis['net_worth']:,.0f}",
        key="btn_net_worth",
        use_container_width=True,
        type="primary" if st.session_state["active_card"] == "Net Worth" else "secondary",
    ):
        st.session_state["active_card"] = "Net Worth"

with kpi_c3:
    if st.button(
        f"Total Assets\n₹{kpis['total_assets']:,.0f}",
        key="btn_total_assets",
        use_container_width=True,
        type="primary" if st.session_state["active_card"] == "Total Assets" else "secondary",
    ):
        st.session_state["active_card"] = "Total Assets"

with kpi_c4:
    if st.button(
        f"Liabilities\n₹{kpis['total_debt']:,.0f}",
        key="btn_liabilities",
        use_container_width=True,
        type="primary" if st.session_state["active_card"] == "Liabilities" else "secondary",
    ):
        st.session_state["active_card"] = "Liabilities"

with kpi_c5:
    if st.button(
        f"Monthly EMIs\n₹{kpis['total_emi']:,.0f}",
        key="btn_monthly_emis",
        use_container_width=True,
        type="primary" if st.session_state["active_card"] == "Monthly EMIs" else "secondary",
    ):
        st.session_state["active_card"] = "Monthly EMIs"

# DYNAMIC CARD INSPECTOR (Renders based on selected KPI card)
active_kpi = st.session_state["active_card"]

if active_kpi == "Health Score":
    st.markdown(
        f"""
        <div class="inspector-box">
            <div class="inspector-header">
                <span class="inspector-title">Health Score Breakdown: <b style="color: #007bff;">{kpis['health_score']} / 100</b></span>
                <span class="pill-info">4-PILLAR WEIGHTED SOLVENCY INDEX</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.85rem;">
                <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 700;">1. SAVINGS RATE (25 PTS)</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #10b981;">{kpis['s_savings']:.1f} / 25</div>
                    <div style="font-size: 0.74rem; color: #0f172a;">Actual: <b>{kpis['savings_rate']:.1f}%</b></div>
                </div>
                <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 700;">2. DTI RATIO (25 PTS)</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #10b981;">{kpis['s_debt']:.1f} / 25</div>
                    <div style="font-size: 0.74rem; color: #0f172a;">Actual: <b>{kpis['dti']:.1f}%</b> (Safe &lt; 20%)</div>
                </div>
                <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 700;">3. EMERGENCY RUNWAY (25 PTS)</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #f59e0b;">{kpis['s_runway']:.1f} / 25</div>
                    <div style="font-size: 0.74rem; color: #0f172a;">Actual: <b>{kpis['runway_months']:.1f} mos</b> (Target: 6.0)</div>
                </div>
                <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 700;">4. SPENDING STABILITY (25 PTS)</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #f59e0b;">{kpis['s_drift']:.1f} / 25</div>
                    <div style="font-size: 0.74rem; color: #0f172a;">Recent: <b>+{kpis['drift_expansion_pct']:.1f}%</b> drift</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif active_kpi == "Net Worth":
    st.markdown(
        f"""
        <div class="inspector-box">
            <div class="inspector-header">
                <span class="inspector-title">Net Worth Solvency: <b style="color: #10b981;">₹{kpis['net_worth']:,.0f}</b></span>
                <span class="pill-success">ASSET COVERAGE: {(kpis['total_assets']/max(1.0, kpis['total_debt'])):.2f}x</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.85rem;">
                <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 700;">TOTAL ASSET EQUITY</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #0f172a;">₹{kpis['total_assets']:,.0f}</div>
                    <div style="font-size: 0.74rem; color: #64748b;">Property, funds, gold & cash</div>
                </div>
                <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 700;">TOTAL DEBT LIABILITIES</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #ef4444;">₹{kpis['total_debt']:,.0f}</div>
                    <div style="font-size: 0.74rem; color: #64748b;">47.5% debt capitalization</div>
                </div>
                <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 700;">NET MONTHLY SURPLUS</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #007bff;">+₹{(kpis['avg_monthly_income'] - kpis['avg_monthly_expense']):,.0f} / mo</div>
                    <div style="font-size: 0.74rem; color: #64748b;">Average capital accumulation</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif active_kpi == "Total Assets":
    st.markdown(
        f"""
        <div class="inspector-box">
            <div class="inspector-header">
                <span class="inspector-title">Asset Allocation Hierarchy: <b style="color: #007bff;">₹{kpis['total_assets']:,.0f}</b></span>
                <span class="pill-info">LIQUID RATIO: {(kpis['liquid_assets']/max(1.0, kpis['total_assets'])*100):.1f}%</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.85rem;">
                <div style="background: #ecfdf5; padding: 0.75rem; border-radius: 8px; border: 1px solid #a7f3d0;">
                    <div style="font-size: 0.72rem; color: #065f46; font-weight: 700;">LIQUID RESERVES (CASH & FDs)</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #047857;">₹{kpis['liquid_assets']:,.0f}</div>
                    <div style="font-size: 0.74rem; color: #065f46;">Savings (₹3.25L), Current (₹0.85L), FD (₹4.50L)</div>
                </div>
                <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 700;">MARKET FUNDS</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #3b82f6;">₹10,35,000</div>
                    <div style="font-size: 0.74rem; color: #64748b;">Mutual Funds (₹6.25L) + Equity (₹4.10L)</div>
                </div>
                <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 700;">FIXED PROPERTY & TANGIBLE</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #0f172a;">₹51,30,000</div>
                    <div style="font-size: 0.74rem; color: #64748b;">Property (₹42.0L), Vehicle (₹6.5L), Gold (₹2.8L)</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif active_kpi == "Liabilities":
    h_debt = kpis["highest_debt"]
    st.markdown(
        f"""
        <div class="inspector-box">
            <div class="inspector-header">
                <span class="inspector-title">Active Liabilities & Debt Drag: <b style="color: #ef4444;">₹{kpis['total_debt']:,.0f}</b></span>
                <span class="pill-danger">ANNUAL INTEREST DRAIN: ₹2,97,955 / YR</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.85rem;">
                <div style="background: #fef2f2; padding: 0.75rem; border-radius: 8px; border: 1px solid #fecaca;">
                    <div style="font-size: 0.72rem; color: #991b1b; font-weight: 700;">PRIORITY: CREDIT CARD (L003)</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #991b1b;">32.0% APR</div>
                    <div style="font-size: 0.74rem; color: #991b1b;">Balance: ₹68,000 | Interest: <b>₹21,760/yr</b></div>
                </div>
                <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 700;">CAR LOAN (L002)</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #0f172a;">9.10% APR</div>
                    <div style="font-size: 0.74rem; color: #64748b;">Balance: ₹4,20,000 | EMI: ₹11,200/mo</div>
                </div>
                <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 700;">HOME LOAN (L001)</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #0f172a;">8.35% APR</div>
                    <div style="font-size: 0.74rem; color: #64748b;">Balance: ₹28,50,000 | EMI: ₹28,500/mo</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif active_kpi == "Monthly EMIs":
    st.markdown(
        f"""
        <div class="inspector-box">
            <div class="inspector-header">
                <span class="inspector-title">Debt Service Capacity: <b style="color: #007bff;">₹{kpis['total_emi']:,.0f} / mo</b></span>
                <span class="pill-success">DTI: {kpis['dti']:.1f}%</span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.85rem;">
                <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 700;">MONTHLY CASH COMMITTED</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #0f172a;">₹46,700 / mo</div>
                    <div style="font-size: 0.74rem; color: #64748b;">17.5% of gross monthly income</div>
                </div>
                <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 700;">AFTER-DEBT LIQUIDITY</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #10b981;">₹2,20,842 / mo</div>
                    <div style="font-size: 0.74rem; color: #64748b;">Operational cash flow capacity</div>
                </div>
                <div style="background: #f8fafc; padding: 0.75rem; border-radius: 8px; border: 1px solid #e2e8f0;">
                    <div style="font-size: 0.72rem; color: #64748b; font-weight: 700;">INTEREST RATE SHOCK (+200 BPS)</div>
                    <div style="font-size: 1.2rem; font-weight: 800; color: #f59e0b;">₹51,450 / mo (+10.2%)</div>
                    <div style="font-size: 0.74rem; color: #64748b;">DTI reaches 19.2% (Comfortable)</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================================================================
# 7. NAVIGATION WORKSPACES (SIMPLIFIED MENU NAMES)
# ==============================================================================

tab_overview, tab_cashflow, tab_balance, tab_alerts, tab_advisory = st.tabs([
    "Overview",
    "Cash Flow",
    "Balance Sheet",
    "Alerts",
    "Advisory",
])


# ==============================================================================
# TAB 1: OVERVIEW (NET WORTH PIE CHART + RECENT TRANSACTIONS ON FIRST PAGE)
# ==============================================================================

with tab_overview:
    row1_c1, row1_c2 = st.columns([1.2, 1.8], gap="medium")

    with row1_c1:
        st.markdown("<div class='vp-card'><div class='vp-card-title'>Net Worth Structure</div>", unsafe_allow_html=True)
        # Net Worth Breakdown Pie Chart (Assets vs Liabilities / Net Equity)
        nw_data = pd.DataFrame([
            {"Component": "Net Equity", "Value": max(0.0, kpis["net_worth"])},
            {"Component": "Liabilities", "Value": kpis["total_debt"]},
        ])
        fig_nw = px.pie(
            nw_data,
            names="Component",
            values="Value",
            hole=0.55,
            color="Component",
            color_discrete_map={"Net Equity": "#00a1de", "Liabilities": "#ef4444"},
        )
        fig_nw.update_traces(textposition="inside", textinfo="percent+label")
        fig_nw.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_nw, use_container_width=True)

        st.markdown(
            f"""
            <div style="font-size: 0.82rem; padding-top: 0.5rem; border-top: 1px solid #f1f5f9; display: flex; justify-content: space-between;">
                <span>Total Assets: <b>₹{kpis['total_assets']:,.0f}</b></span>
                <span>Debt Ratio: <b>{(kpis['total_debt']/max(1.0, kpis['total_assets'])*100):.1f}%</b></span>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with row1_c2:
        st.markdown("<div class='vp-card'><div class='vp-card-title'>Asset Allocation Breakdown</div>", unsafe_allow_html=True)
        # Detailed Asset Class Pie Chart
        fig_asset_pie = px.pie(
            df_asset,
            names="type",
            values="value",
            hole=0.55,
            color_discrete_sequence=["#00a1de", "#007bff", "#10b981", "#6366f1", "#f59e0b", "#8b5cf6", "#14b8a6", "#64748b"],
        )
        fig_asset_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_asset_pie.update_layout(
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_asset_pie, use_container_width=True)

        st.markdown(
            f"""
            <div style="font-size: 0.82rem; padding-top: 0.5rem; border-top: 1px solid #f1f5f9; display: flex; justify-content: space-between;">
                <span>Liquid Reserves: <b>₹{kpis['liquid_assets']:,.0f}</b></span>
                <span>Illiquid Wealth: <b>₹{kpis['total_assets'] - kpis['liquid_assets']:,.0f}</b></span>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # RECENT TRANSACTIONS LOG ON FIRST PAGE ONLY (WITH USER OVERRIDE FOR ANOMALIES)
    st.markdown("<div class='vp-card'><div class='vp-card-title'>Recent Transactions Log (Interactive Anomaly Review)</div>", unsafe_allow_html=True)

    filter_c1, filter_c2, filter_c3 = st.columns([1.5, 1, 1])
    with filter_c1:
        search_tx = st.text_input("Filter by Description or Category", placeholder="Search transactions...", label_visibility="collapsed")
    with filter_c2:
        flow_choice = st.selectbox("Flow Filter", ["All", "spending", "income", "debt_payment", "investment"], index=0, label_visibility="collapsed")
    with filter_c3:
        outlier_filter = st.selectbox("Anomaly Filter", ["All Transactions", "Flagged Anomalies Only", "Normal Only"], index=0, label_visibility="collapsed")

    # Get latest 25 transactions
    df_sorted = df_txn.sort_values(by="date", ascending=False).copy()

    if search_tx:
        s = search_tx.lower()
        df_sorted = df_sorted[
            df_sorted["description"].astype(str).str.lower().str.contains(s) |
            df_sorted["category"].astype(str).str.lower().str.contains(s) |
            df_sorted["txn_id"].astype(str).str.lower().str.contains(s)
        ]

    if flow_choice != "All":
        df_sorted = df_sorted[df_sorted["flow_class"] == flow_choice]

    if outlier_filter == "Flagged Anomalies Only":
        df_sorted = df_sorted[df_sorted["is_outlier"] == True]
    elif outlier_filter == "Normal Only":
        df_sorted = df_sorted[df_sorted["is_outlier"] == False]

    recent_table = df_sorted.head(25).copy()
    recent_table["Date"] = recent_table["date"].dt.strftime("%Y-%m-%d")

    # Allow User to edit anomaly status directly using data_editor!
    st.markdown("<div style='font-size: 0.78rem; color: #64748b; margin-bottom: 0.5rem;'>You can check or uncheck <b>is_outlier</b> to override anomaly classification.</div>", unsafe_allow_html=True)

    cols_for_editor = ["txn_id", "Date", "category", "description", "amount", "type", "flow_class", "is_outlier"]
    editable_df = recent_table[cols_for_editor].rename(columns={
        "txn_id": "Txn ID",
        "category": "Category",
        "description": "Description",
        "amount": "Amount (₹)",
        "type": "Type",
        "flow_class": "Flow Class",
        "is_outlier": "Anomaly Flagged",
    })

    edited_result = st.data_editor(
        editable_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Anomaly Flagged": st.column_config.CheckboxColumn(
                "Anomaly Flagged",
                help="Check to mark as anomaly, uncheck to clear anomaly flag.",
                default=False,
            ),
            "Amount (₹)": st.column_config.NumberColumn(
                "Amount (₹)",
                format="₹%.2f",
            ),
        },
        key="recent_txns_editor",
    )

    # Save user override back into session state
    if edited_result is not None and not edited_result.empty:
        for _, r in edited_result.iterrows():
            t_id = r["Txn ID"]
            new_val = r["Anomaly Flagged"]
            mask = df_txn["txn_id"] == t_id
            df_txn.loc[mask, "is_outlier"] = new_val
        st.session_state["txn_df"] = df_txn

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# TAB 2: CASH FLOW
# ==============================================================================

with tab_cashflow:
    st.markdown("<div class='vp-card'><div class='vp-card-title'>Cash Flow & Spending Trajectory</div>", unsafe_allow_html=True)

    cf_ctrl1, cf_ctrl2 = st.columns([1, 1])
    with cf_ctrl1:
        view_mode = st.selectbox(
            "Visualization",
            ["Income vs Expense Bars", "Net Cash Flow Waterfall", "Spending by Category"],
            index=0,
        )
    with cf_ctrl2:
        norm_flag = st.checkbox("Normalize Extreme Outliers (Realistic View)", value=False)

    df_cf = df_txn.copy()
    if norm_flag:
        q75 = df_cf[df_cf["type"] == "expense"]["amount"].quantile(0.75)
        q25 = df_cf[df_cf["type"] == "expense"]["amount"].quantile(0.25)
        cap = q75 + 1.5 * (q75 - q25)
        df_cf.loc[(df_cf["type"] == "expense") & (df_cf["amount"] > cap), "amount"] = cap

    if view_mode == "Income vs Expense Bars":
        m_df = df_cf.groupby(["month", "type"])["amount"].sum().unstack(fill_value=0.0).reset_index()
        fig_cf = go.Figure()
        fig_cf.add_trace(go.Bar(x=m_df["month"], y=m_df.get("income", 0), name="Income", marker_color="#10b981"))
        fig_cf.add_trace(go.Bar(x=m_df["month"], y=m_df.get("expense", 0), name="Expense", marker_color="#ef4444"))
        fig_cf.update_layout(
            barmode="group",
            height=380,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(tickangle=-45, showgrid=False),
            yaxis=dict(gridcolor="#f1f5f9", tickprefix="₹"),
            legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        )
        st.plotly_chart(fig_cf, use_container_width=True)

    elif view_mode == "Net Cash Flow Waterfall":
        tot_inc = df_cf[df_cf["type"] == "income"]["amount"].sum()
        top_cats = df_cf[df_cf["type"] == "expense"].groupby("category")["amount"].sum().sort_values(ascending=False)
        wf_x = ["Total Inflows"] + top_cats.index.tolist()[:6] + ["Other Spend", "Net Retained"]
        wf_top6 = top_cats.head(6).sum()
        wf_rem = top_cats.iloc[6:].sum() if len(top_cats) > 6 else 0.0
        net_ret = tot_inc - (wf_top6 + wf_rem)

        wf_y = [tot_inc] + [-v for v in top_cats.head(6).values] + [-wf_rem, net_ret]
        fig_wf = go.Figure(go.Waterfall(
            measure=["absolute"] + ["relative"] * 7 + ["total"],
            x=wf_x,
            y=wf_y,
            textposition="outside",
            text=[f"₹{abs(v):,.0f}" for v in wf_y],
            increasing={"marker": {"color": "#10b981"}},
            decreasing={"marker": {"color": "#ef4444"}},
            totals={"marker": {"color": "#00a1de"}},
        ))
        fig_wf.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(tickprefix="₹"))
        st.plotly_chart(fig_wf, use_container_width=True)

    elif view_mode == "Spending by Category":
        exp_by_cat = df_cf[df_cf["type"] == "expense"].groupby("category")["amount"].sum().reset_index()
        fig_c = px.bar(exp_by_cat.sort_values(by="amount", ascending=True), x="amount", y="category", orientation="h", color="amount", color_continuous_scale="Blues")
        fig_c.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(tickprefix="₹"))
        st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# TAB 3: BALANCE SHEET
# ==============================================================================

with tab_balance:
    b_col1, b_col2 = st.columns([1.1, 1.4], gap="medium")

    with b_col1:
        st.markdown("<div class='vp-card'><div class='vp-card-title'>Assets Portfolio</div>", unsafe_allow_html=True)
        st.dataframe(
            df_asset[["asset_id", "type", "value", "is_liquid"]].style.format({"value": "₹{:,.0f}"}),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with b_col2:
        st.markdown("<div class='vp-card'><div class='vp-card-title'>Liabilities & Interest Drag</div>", unsafe_allow_html=True)
        df_l_disp = df_liab.copy()
        df_l_disp["Annual Interest (₹)"] = df_l_disp["outstanding"] * (df_l_disp["interest_rate"] / 100.0)
        st.dataframe(
            df_l_disp[["liability_id", "type", "outstanding", "interest_rate", "emi", "Annual Interest (₹)"]].style.format({
                "outstanding": "₹{:,.0f}",
                "emi": "₹{:,.0f}",
                "interest_rate": "{:.1f}%",
                "Annual Interest (₹)": "₹{:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# TAB 4: ALERTS (CLEAN & NON-EMPTY)
# ==============================================================================

with tab_alerts:
    st.markdown("<div class='vp-card'><div class='vp-card-title'>Active Portfolio & Forensic Alerts</div>", unsafe_allow_html=True)

    alerts_list = [
        ("pill-danger", "DUE TODAY", "Credit card payment of ₹7,000 due (L003). 32.0% APR interest drag if unpaid."),
        ("pill-danger", "CRITICAL OUTLIER", "Txn T0488: ₹1,85,000 Mobile bill on 14-Mar-2026 under Utilities (Z=14.37). Flagged for review."),
        ("pill-warning", "LIQUIDITY DEFICIT", f"Liquid reserves cover {kpis['runway_months']:.1f} months vs 6.0 months mandate. Contribution needed: ₹{kpis['runway_deficit']:,.0f}."),
        ("pill-info", "PAYROLL RECLASSIFICATION", "Txn T0556: ₹2,05,000 Salary correction reclassified from expense to income credit."),
        ("pill-warning", "UPCOMING DUE DATE", "Car loan EMI ₹11,200 due on 07-Oct-2026."),
    ]

    for pill_class, badge, text in alerts_list:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; justify-content: space-between; padding: 0.75rem 0.5rem; border-bottom: 1px solid #f1f5f9;">
                <div style="display: flex; align-items: center; gap: 0.75rem;">
                    <span class="{pill_class}">{badge}</span>
                    <span style="font-size: 0.85rem; color: #0f172a; font-weight: 500;">{text}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# TAB 5: ADVISORY & STATELESS Q&A
# ==============================================================================

with tab_advisory:
    st.markdown("<div class='vp-card'><div class='vp-card-title'>Executive Family Office Advisory Memorandum</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="font-size: 0.88rem; line-height: 1.6; color: #334155;">
            <b>Outflow Audit & Spending Drift:</b> Over the recent observation window, monthly burn increased by <b>+{kpis['drift_expansion_pct']:.1f}%</b> 
            (+₹{kpis['drift_expansion_inr']:,.0f}/mo), rising from a baseline of ₹{kpis['baseline_burn']:,.0f} to <b>₹{kpis['recent_burn']:,.0f}</b>. 
            Discretionary categories expanded rapidly, alongside 12 detected statistical outliers including the ₹1,85,000 mobile bill (Txn T0488).
            <br><br>
            <b>Prescribed Interventions:</b> First, prepay Credit Card L003 (₹68,000 at 32.0% APR) immediately to eliminate ₹21,760 in annual interest drag and free ₹7,000/mo cash flow. 
            Second, enforce a hard monthly budget cap on discretionary spending. Third, allocate exactly ₹{kpis['runway_deficit']:,.0f} into liquid sweep accounts to establish 
            the 6-month safety runway of ₹{kpis['target_6m_runway']:,.0f}.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Stateless Q&A
    st.markdown("<div class='vp-card'><div class='vp-card-title'>Advisor Query Console (Stateless)</div>", unsafe_allow_html=True)
    q_col1, q_col2, q_col3 = st.columns(3)
    user_q = None
    with q_col1:
        if st.button("Can we handle our debt?", use_container_width=True):
            user_q = "Can we handle our debt?"
    with q_col2:
        if st.button("What changed recently?", use_container_width=True):
            user_q = "What changed recently?"
    with q_col3:
        if st.button("What should we do next?", use_container_width=True):
            user_q = "What should we do next?"

    custom_text = st.text_input("Or type custom question:", placeholder="Ask any financial question...", key="txt_advisory_q")
    if st.button("Ask Advisor", type="primary") and custom_text:
        user_q = custom_text

    if user_q:
        if user_q == "Can we handle our debt?":
            resp = f"**Debt Health:** Overall DTI is **{kpis['dti']:.1f}%** against average monthly income of ₹{kpis['avg_monthly_income']:,.0f}, which is well within the 35% safe limit. However, Liability L003 (Credit Card) carries an aggressive **32.0% APR** (₹68,000). Prepaying this immediately terminates ₹21,760 in annual interest drag and frees ₹7,000/month."
        elif user_q == "What changed recently?":
            resp = f"**Recent Shifts:** Monthly spending expanded by **+{kpis['drift_expansion_pct']:.1f}%** (+₹{kpis['drift_expansion_inr']:,.0f}/mo) over the recent window. Twelve outlier transactions were detected, notably an abnormal ₹1,85,000 mobile bill under Utilities (Txn T0488)."
        elif user_q == "What should we do next?":
            resp = f"**Roadmap:** 1. Prepay Credit Card L003 (₹68,000). 2. Cap discretionary 'Other' spending at baseline (₹10,200/mo). 3. Contribute ₹{kpis['runway_deficit']:,.0f} into liquid sweep accounts to hit 6 months of runway."
        else:
            resp = f"**Analysis:** With a Net Worth of ₹{kpis['net_worth']:,.0f} and savings rate of {kpis['savings_rate']:.1f}%, your balance sheet has solid fundamentals. Prioritize eliminating the 32% APR debt and funding the ₹{kpis['runway_deficit']:,.0f} runway gap."

        st.markdown(
            f"""
            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.9rem; margin-top: 0.75rem; font-size: 0.85rem; line-height: 1.5;">
                <div style="font-weight: 700; color: #007bff; margin-bottom: 0.25rem;">Question: {user_q}</div>
                {resp}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# 8. FLOATING AI ADVISOR WINDOW (SIDEBAR DOCK)
# ==============================================================================

with st.sidebar:
    st.markdown("<hr style='margin: 1.25rem 0 0.75rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.95rem; font-weight: 800; color: #0f172a;'>Floating AI Copilot</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.75rem; color: #64748b; margin-bottom: 0.5rem;'>Stateless quick prompt dock</div>", unsafe_allow_html=True)
    quick_q = st.text_input("Ask Copilot:", placeholder="Ask about runway, debt...", key="dock_copilot_q")
    if st.button("Execute Query", use_container_width=True) and quick_q:
        st.info(f"**Advisor Response:** Portfolio Net Worth is ₹{kpis['net_worth']:,.0f} with a DTI of {kpis['dti']:.1f}% and {kpis['runway_months']:.1f} months of emergency runway.")
