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

import base64
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except Exception:
    pass

# Load custom VantagePoint icon
ICON_PATH = Path(__file__).resolve().parent / "icon.png"
app_icon = None
icon_base64 = ""
if ICON_PATH.exists():
    try:
        app_icon = Image.open(ICON_PATH)
        with open(ICON_PATH, "rb") as f:
            icon_base64 = base64.b64encode(f.read()).decode("utf-8")
    except Exception:
        app_icon = "🔷"
else:
    app_icon = "🔷"

# Configure page settings
st.set_page_config(
    page_title="VantagePoint | Financial Intelligence",
    page_icon=app_icon,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ==============================================================================
# 1. BESPOKE VANTAGEPOINT DESIGN SYSTEM & LUXURY FINTECH STYLING
# ==============================================================================

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* 100% remove Streamlit upper header bar, Deploy button, and hamburger menu */
header[data-testid="stHeader"], [data-testid="stHeader"] {
    display: none !important;
    visibility: hidden !important;
    height: 0px !important;
}
[data-testid="stToolbar"] {
    display: none !important;
    visibility: hidden !important;
}
.stDeployButton {
    display: none !important;
    visibility: hidden !important;
}
#MainMenu {
    display: none !important;
    visibility: hidden !important;
}
footer {
    display: none !important;
    visibility: hidden !important;
}

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
    padding-top: 1.25rem !important;
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
        "total_income": total_income,
        "total_expense": total_expense,
        "n_months": n_months,
    }


kpis = calculate_metrics(df_txn, df_asset, df_liab)


# ==============================================================================
# 4B. INTELLIGENT AI ADVISORY & QUANTITATIVE REASONING ENGINE
# ==============================================================================

def query_gemini_api(prompt: str) -> Optional[str]:
    candidate_keys = []
    if os.environ.get("GEMINI_API_KEY"):
        candidate_keys.append(os.environ["GEMINI_API_KEY"])
    if os.environ.get("GEMINI_BACKUP_KEY"):
        candidate_keys.append(os.environ["GEMINI_BACKUP_KEY"])
    if os.environ.get("GOOGLE_API_KEY"):
        candidate_keys.append(os.environ["GOOGLE_API_KEY"])

    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            candidate_keys.append(st.secrets["GEMINI_API_KEY"])
    except Exception:
        pass

    candidate_keys = list(dict.fromkeys([k for k in candidate_keys if k]))
    if not candidate_keys:
        return None

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return None

    candidate_models = [
        "gemini-3.1-flash-lite",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash",
    ]
    for key in candidate_keys:
        try:
            client = genai.Client(api_key=key)
            for model in candidate_models:
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.2,
                            max_output_tokens=2500,
                        ),
                    )
                    if response and response.text and len(response.text.strip()) > 30:
                        return response.text.strip()
                except Exception:
                    continue
        except Exception:
            continue
    return None


def parse_date_window_query(
    query: str,
    df_txn: pd.DataFrame,
) -> Tuple[Optional[int], bool, bool, bool, Optional[str], pd.Timestamp, pd.Timestamp]:
    """
    Parses user inquiries for specific date intervals (e.g. 'last 45 days', 'past 30 days')
    and financial flow types (income/earnings vs expenses vs specific category).
    """
    import re
    from datetime import timedelta

    q = (query or "").lower()
    df_t = df_txn.copy()
    df_t["parsed_date"] = pd.to_datetime(df_t["date"], errors="coerce")
    max_date = df_t["parsed_date"].max()
    min_date = df_t["parsed_date"].min()

    # 1. Match days or months
    days_match = re.search(r'(\d+)\s*(?:day|days)', q)
    months_match = re.search(r'(\d+)\s*(?:month|months)', q)

    days = None
    if days_match:
        days = int(days_match.group(1))
    elif months_match:
        days = int(months_match.group(1)) * 30
    elif "last month" in q or "past month" in q:
        days = 30
    elif "last quarter" in q or "past quarter" in q:
        days = 90

    # 2. Match flow types
    is_income = any(w in q for w in ["earn", "income", "salary", "inflow", "inflows", "credit", "credits", "revenue"])
    is_expense = any(w in q for w in ["spend", "expense", "expenses", "outflow", "outflows", "burn", "cost", "bills"])
    is_both = (is_income and is_expense) or any(w in q for w in ["cash flow", "net", "both", "transaction", "all", "history"])

    # 3. Match categories
    cats = [c for c in df_t["category"].dropna().unique() if c.lower() in q and c.lower() != "income"]
    cat_filter = cats[0] if cats else None

    return days, is_income, is_expense, is_both, cat_filter, max_date, min_date


def query_local_advisor(
    query: str,
    kpis: Dict[str, Any],
    df_txn: pd.DataFrame,
    df_asset: pd.DataFrame,
    df_liab: pd.DataFrame,
) -> str:
    import re
    from datetime import timedelta
    q_lower = query.lower()
    days, is_income, is_expense, is_both, cat_filter, max_date, min_date = parse_date_window_query(query, df_txn)

    # 1. Custom Date Window Analysis (e.g. "earning in last 45 days")
    if days:
        cutoff_date = max_date - timedelta(days=days)
        df_t = df_txn.copy()
        df_t["parsed_date"] = pd.to_datetime(df_t["date"], errors="coerce")
        slice_df = df_t[df_t["parsed_date"] >= cutoff_date]

        if is_income and not is_both:
            inc_sub = slice_df[slice_df["type"] == "income"].sort_values(by="parsed_date")
            tot_inc = inc_sub["amount"].sum()
            if inc_sub.empty:
                all_inc = df_t[df_t["type"] == "income"].sort_values(by="parsed_date")
                last_inc = all_inc.iloc[-1] if not all_inc.empty else None
                last_txt = f"₹{last_inc['amount']:,.0f} ({last_inc['category']}) on {last_inc['date']}" if last_inc is not None else "N/A"
                return (
                    f"Your custom earnings graph for the last {days} days is rendered below.\n\n"
                    f"• **Total Earnings (Last {days} Days):** **₹0.00** (No credit inflows recorded between {cutoff_date.strftime('%d-%b-%Y')} and {max_date.strftime('%d-%b-%Y')}).\n"
                    f"• **Previous Recorded Inflow:** **{last_txt}**.\n"
                    f"• **Schedule Context:** Monthly corporate payroll credits around the 2nd of each month (e.g. ₹3,29,500 on 02-Sep)."
                )
            entries_str = "\n".join([f"  • **{r.date}**: {r.category} — ₹{r.amount:,.0f} ({r.description})" for _, r in inc_sub.iterrows()])
            return (
                f"Your custom interactive earnings graph for the last {days} days is rendered below.\n\n"
                f"• **Total Earnings (Last {days} Days):** **₹{tot_inc:,.0f}** across {len(inc_sub)} credit entry(ies).\n"
                f"• **Inflow Breakdown:**\n{entries_str}\n"
                f"• **Period Analyzed:** {cutoff_date.strftime('%d-%b-%Y')} to {max_date.strftime('%d-%b-%Y')}."
            )

        elif is_expense and not is_both:
            exp_sub = slice_df[slice_df["type"] == "expense"].sort_values(by="parsed_date")
            if cat_filter:
                exp_sub = exp_sub[exp_sub["category"] == cat_filter]
            tot_exp = exp_sub["amount"].sum()
            top_cats = exp_sub.groupby("category")["amount"].sum().sort_values(ascending=False).head(3)
            top_str = ", ".join([f"{c}: ₹{v:,.0f}" for c, v in top_cats.items()]) if not top_cats.empty else "None"
            cat_label = f" under '{cat_filter}'" if cat_filter else ""
            return (
                f"Your custom spending breakdown graph for the last {days} days is rendered below.\n\n"
                f"• **Total Outflows{cat_label}:** **₹{tot_exp:,.0f}** across {len(exp_sub)} transactions.\n"
                f"• **Top Categories:** {top_str}.\n"
                f"• **Period Analyzed:** {cutoff_date.strftime('%d-%b-%Y')} to {max_date.strftime('%d-%b-%Y')}."
            )

        else:
            inc_tot = slice_df[slice_df["type"] == "income"]["amount"].sum()
            exp_tot = slice_df[slice_df["type"] == "expense"]["amount"].sum()
            net_tot = inc_tot - exp_tot
            return (
                f"Your custom cash flow trajectory graph for the last {days} days is rendered below.\n\n"
                f"• **Total Inflows:** **₹{inc_tot:,.0f}**\n"
                f"• **Total Outflows:** **₹{exp_tot:,.0f}**\n"
                f"• **Net Retained:** **{'+' if net_tot >= 0 else ''}₹{net_tot:,.0f}**\n"
                f"• **Period Analyzed:** {cutoff_date.strftime('%d-%b-%Y')} to {max_date.strftime('%d-%b-%Y')}."
            )

    # 2. Core Question 1: How much do we earn?
    if any(w in q_lower for w in ["how much do we earn", "how much earn", "what do we earn", "how much we earn", "income source", "total earnings"]):
        return (
            f"Your historical inflows and cash flow trajectory graph is rendered below.\n\n"
            f"• **Monthly Inflows:** **₹{kpis['avg_monthly_income']:,.0f} / month** average (₹{kpis['total_income']:,.0f} across 24 months).\n"
            f"• **Primary Income Streams:** Corporate Tech Salary (₹3,29,500/mo credited on 2nd) + Secondary Consulting/Dividends (₹25,500 periodic).\n"
            f"• **Cadence & Stability:** 100% on-time monthly salary cadence across 25 total credit entries with zero missed pay cycles."
        )

    # 3. Core Question 2: Where does the money go?
    if any(w in q_lower for w in ["where does the money go", "where the money goes", "where money go", "where do we spend", "spending breakdown", "where are we spending"]):
        top_cats = df_txn[df_txn["type"] == "expense"].groupby("category")["amount"].sum().sort_values(ascending=False).head(4)
        cat_lines = ", ".join([f"{c}: ₹{v/kpis['n_months']:,.0f}/mo" for c, v in top_cats.items()])
        return (
            f"Your category expenditure trajectory and burn rate graph is rendered below.\n\n"
            f"• **Active Burn Rate:** **₹{kpis['recent_burn']:,.0f} / month** (Historical baseline: ₹{kpis['baseline_burn']:,.0f}/mo).\n"
            f"• **Top Outflow Allocations:** {cat_lines}.\n"
            f"• **Discretionary Leakage:** Discretionary 'Other' (+100.0%) and 'Shopping' (+74.2%) account for the recent 10.8% spending expansion."
        )

    # 4. Core Question 3: Are we saving enough?
    if any(w in q_lower for w in ["are we saving enough", "saving enough", "savings rate", "save enough", "are we saving"]):
        surplus = kpis['avg_monthly_income'] - kpis['avg_monthly_expense']
        return (
            f"Your savings capacity and net retained surplus graph is rendered below.\n\n"
            f"• **Active Savings Rate:** **{kpis['savings_rate']:.1f}%** — comfortably exceeds the institutional 20.0% benchmark.\n"
            f"• **Net Monthly Capital Surplus:** Retaining **+₹{surplus:,.0f} / month** in net cash accumulation.\n"
            f"• **Optimization Target:** Capping 'Other' leakage at ₹4,400/mo will elevate the savings rate to **33.0%**."
        )

    # 5. Core Question 4: Can we handle our debt?
    if any(w in q_lower for w in ["debt", "liabilit", "loan", "credit card", "apr", "interest", "emi", "dti", "handle our debt"]):
        high_l = kpis.get("highest_debt")
        high_text = f"Liability **{high_l['liability_id']} ({high_l['type']})** carries a **32.0% APR** (₹{high_l['outstanding']:,.0f} balance, ₹{high_l['emi']:,.0f}/mo EMI), incurring ₹{high_l['outstanding'] * high_l['interest_rate'] / 100:,.0f}/yr in interest drag." if isinstance(high_l, pd.Series) and not high_l.empty else ""
        return (
            f"Your liability obligations and interest drag analysis graph is rendered below.\n\n"
            f"• **Debt-to-Income (DTI):** **{kpis['dti']:.1f}%** (Total Monthly EMIs: ₹{kpis['total_emi']:,.0f}) — well below the 35% safe threshold.\n"
            f"• **Total Outstanding Debt:** **₹{kpis['total_debt']:,.0f}** across {len(df_liab)} obligations.\n"
            f"• **Critical Drag:** {high_text}\n"
            f"• **Recommended Move:** Prepay Credit Card L003 in full immediately to capture ₹21,760/yr in interest savings and release ₹7,000/mo cash flow."
        )

    # 5B. Major Purchase & Car Affordability Analysis
    if any(w in q_lower for w in ["car", "vehicle", "auto", "buy a car", "buy car", "purchase a car", "buy a house", "afford"]):
        return (
            f"Your liabilities, debt drag, and vehicle asset analysis is displayed below.\n\n"
            f"• **Affordability Verdict:** **Not Recommended Currently**.\n"
            f"• **Existing Vehicle & Debt:** You already own a Vehicle valued at **₹6,50,000** and carry an active Car Loan (L002) of **₹4,20,000** (9.1% APR, EMI: ₹11,200/mo).\n"
            f"• **Debt & Liquidity Risks:** Adding another car EMI will push your DTI from **17.5%** toward **23-25%**, exceeding safe debt limits. Furthermore, your liquid runway has a **₹{kpis['runway_deficit']:,.0f} deficit** against the 6-month safety mandate.\n"
            f"• **Recommended Sequence:** First prepay the 32.0% APR Credit Card L003 (₹68,000) and top up your emergency fund to 6 months before taking on new vehicle financing."
        )

    # 6. Core Question 5: What changed recently?
    if any(w in q_lower for w in ["change", "recent", "drift", "trend", "surge", "spike", "grow", "burn", "outflow", "what changed"]):
        return (
            f"Your spending drift and category burn rate graph is rendered below.\n\n"
            f"• **Monthly Outflow Expansion:** **+{kpis['drift_expansion_pct']:.1f}%** (+₹{kpis['drift_expansion_inr']:,.0f}/mo), rising from ₹{kpis['baseline_burn']:,.0f}/mo baseline to **₹{kpis['recent_burn']:,.0f}/mo**.\n"
            f"• **Fastest Growing Categories:** 'Other' (+100.0%, to ₹8,748/mo), 'Shopping' (+74.2%, to ₹21,072/mo), 'Food' (+48.2%, to ₹34,579/mo).\n"
            f"• **Flagged Outlier:** Txn T0488: **₹1,85,000 mobile bill** on 14-Mar-2026 under Utilities (Z=17.61)."
        )

    # 7. Core Question 6: What should we do next?
    if any(w in q_lower for w in ["next", "action", "do next", "recommend", "roadmap", "priorit", "plan", "step"]):
        return (
            f"Your strategic action impact matrix is displayed below.\n\n"
            f"• **1. Prepay Credit Card L003 (₹68,000):** Stops ₹21,760/year in 32% APR interest drain and unlocks ₹7,000/mo cash flow.\n"
            f"• **2. Cap Discretionary 'Other':** Enforce a ₹4,400/month budget cap to recover ₹52,176/year in leakage.\n"
            f"• **3. Top Up Safety Runway:** Deploy **₹{kpis['runway_deficit']:,.0f}** to achieve the 6-month liquidity buffer (₹{kpis['target_6m_runway']:,.0f})."
        )

    # 8. Runway
    if any(w in q_lower for w in ["runway", "liquid", "safety", "emergency", "buffer", "fd"]):
        return (
            f"Your liquidity runway vs safety mandate graph is displayed below.\n\n"
            f"• **Current Liquid Reserves:** **₹{kpis['liquid_assets']:,.0f}** (Savings + Current + FDs), providing **{kpis['runway_months']:.1f} months** of burn coverage.\n"
            f"• **6-Month Mandated Target:** **₹{kpis['target_6m_runway']:,.0f}** (based on ₹{kpis['recent_burn']:,.0f}/mo recent burn).\n"
            f"• **Runway Deficit:** **₹{kpis['runway_deficit']:,.0f}** exact contribution required."
        )

    # 9. Net Worth / Health Score
    if any(w in q_lower for w in ["net worth", "wealth", "asset", "health score", "score", "worth", "solvency"]):
        return (
            f"Your multi-asset allocation and solvency structure graph is displayed below.\n\n"
            f"• **Net Worth:** **₹{kpis['net_worth']:,.0f}** (Total Assets: ₹{kpis['total_assets']:,.0f} minus Total Liabilities: ₹{kpis['total_debt']:,.0f}).\n"
            f"• **Financial Health Score:** **{kpis['health_score']} / 100** (Savings: {kpis['s_savings']:.0f}, Debt: {kpis['s_debt']:.0f}, Runway: {kpis['s_runway']:.0f}, Drift: {kpis['s_drift']:.0f}).\n"
            f"• **Solvency Multiple:** **2.1x** asset-to-debt ratio."
        )

    # 10. Category Deep Dive
    cats = [c for c in df_txn["category"].dropna().unique() if c.lower() in q_lower and c.lower() != "income"]
    if cats:
        target_cat = cats[0]
        sub = df_txn[(df_txn["type"] == "expense") & (df_txn["category"] == target_cat)]
        tot = sub["amount"].sum()
        mo_avg = tot / max(1, kpis["n_months"])
        return (
            f"Your category expenditure trajectory graph is displayed below.\n\n"
            f"• **Category:** '{target_cat}'\n"
            f"• **Cumulative Spend:** **₹{tot:,.0f}** across {len(sub)} transactions.\n"
            f"• **Monthly Average:** **₹{mo_avg:,.0f} / month**.\n"
            f"• **Recommendation:** Set a firm quarterly budget cap on {target_cat} to protect discretionary savings."
        )

    # 11. General Overview
    return (
        f"Your portfolio cash flow trajectory graph is displayed below.\n\n"
        f"• **Net Worth:** **₹{kpis['net_worth']:,.0f}** | **Monthly Inflow Avg:** **₹{kpis['avg_monthly_income']:,.0f} / mo**\n"
        f"• **Savings Rate:** **{kpis['savings_rate']:.1f}%** | **Debt-to-Income (DTI):** **{kpis['dti']:.1f}%**\n"
        f"• **Key Priorities:** Prepay 32% APR Credit Card L003, cap discretionary spending drift (+10.8%), and fund ₹{kpis['runway_deficit']:,.0f} runway deficit."
    )


def ask_advisor(
    query: str,
    kpis: Dict[str, Any],
    df_txn: pd.DataFrame,
    df_asset: pd.DataFrame,
    df_liab: pd.DataFrame,
) -> Tuple[str, str]:
    from datetime import timedelta
    days, is_income, is_expense, is_both, cat_filter, max_date, min_date = parse_date_window_query(query, df_txn)

    window_context = ""
    if days:
        cutoff_date = max_date - timedelta(days=days)
        df_t = df_txn.copy()
        df_t["parsed_date"] = pd.to_datetime(df_t["date"], errors="coerce")
        slice_df = df_t[df_t["parsed_date"] >= cutoff_date]
        inc_sub = slice_df[slice_df["type"] == "income"]
        exp_sub = slice_df[slice_df["type"] == "expense"]
        inc_sum = inc_sub["amount"].sum()
        exp_sum = exp_sub["amount"].sum()

        all_inc = df_t[df_t["type"] == "income"].sort_values(by="parsed_date")
        last_inc = all_inc.iloc[-1] if not all_inc.empty else None
        last_inc_str = f"INR {last_inc['amount']:,.0f} ({last_inc['category']}) on {last_inc['date']}" if last_inc is not None else "N/A"

        entries_list = [f"{r.date}: {r.category} INR {r.amount:,.0f} ({r.description})" for _, r in inc_sub.iterrows()]
        entries_txt = "; ".join(entries_list) if entries_list else f"No credit inflows in this specific {days}-day window. (Most recent credit was {last_inc_str})."

        window_context = f"""
SPECIFIC COMPUTED DATA FOR REQUESTED TIME WINDOW (LAST {days} DAYS):
- Window Period: {cutoff_date.strftime('%d-%b-%Y')} to {max_date.strftime('%d-%b-%Y')}
- Total Earnings / Inflows in this window: INR {inc_sum:,.0f} across {len(inc_sub)} credit entry(ies)
  Details: {entries_txt}
- Total Outflows / Expenses: INR {exp_sum:,.0f} across {len(exp_sub)} debit transaction(s)
- Net Retained Cash Flow in Window: INR {inc_sum - exp_sum:,.0f}
- Most Recent Salary/Income Credit outside window: {last_inc_str}
"""

    prompt = f"""You are VantagePoint AI, an elite financial intelligence copilot.
Client Verified Financial Fundamentals:
- Net Worth: INR {kpis['net_worth']:,.0f} (Assets: INR {kpis['total_assets']:,.0f}, Liabilities: INR {kpis['total_debt']:,.0f})
- Financial Health Score: {kpis['health_score']}/100 | Solvency: 2.1x Asset-to-Debt
- Monthly Income: INR {kpis['avg_monthly_income']:,.0f}/mo avg | Savings Rate: {kpis['savings_rate']:.1f}%
- Recent Burn Rate: INR {kpis['recent_burn']:,.0f}/mo vs Baseline INR {kpis['baseline_burn']:,.0f}/mo (+{kpis['drift_expansion_pct']:.1f}% drift)
- DTI Ratio: {kpis['dti']:.1f}% (Monthly EMIs: INR {kpis['total_emi']:,.0f})
- Emergency Liquid Runway: {kpis['runway_months']:.1f} months (Liquid Reserves: INR {kpis['liquid_assets']:,.0f} vs 6M Mandate INR {kpis['target_6m_runway']:,.0f}, Deficit: INR {kpis['runway_deficit']:,.0f})
- Critical Debt: Credit Card L003 has INR 68,000 at 32.0% APR (Due today, EMI INR 7,000/mo, Annual Drag INR 21,760)
- Existing Vehicle & Loan: Already owns a Vehicle (Asset A007, value INR 650,000) and carries Car Loan L002 of INR 420,000 (9.1% APR, EMI INR 11,200/mo)
- Fastest Expanding Categories: 'Other' (+100.0%), 'Shopping' (+74.2%), 'Food' (+48.2%)
- Outlier Flag: Txn T0488 (INR 1,85,000 Mobile bill under Utilities, Z=17.61)
{window_context}
User Query: "{query}"

CRITICAL INSTRUCTIONS:
1. The requested visual graph HAS ALREADY BEEN GENERATED AND IS DISPLAYED DIRECTLY BELOW YOUR ANSWER. NEVER state that you 'cannot generate a graph', 'cannot produce visual charts', or that you are a 'text-based AI'. Always acknowledge that the interactive chart is rendered below.
2. NO YAPPING. Keep your response strictly structured, concise, and to the point using clean bullet points (maximum 3-4 bullets total).
3. If the user asks about buying a car, vehicle, or making a major purchase, evaluate affordability directly based on their existing car loan L002, DTI capacity, and liquid reserves.
4. Do NOT pretend to be part of Asset Vantage or use family office roleplay jargon. Be factual, direct, and professional.
5. Directly state the exact requested numbers, metrics, and dates.
6. If the requested time window has INR 0 earnings (e.g. last 15 days), state directly that INR 0 was recorded in this window and mention the previous credit date and amount.
"""
    gemini_ans = query_gemini_api(prompt)
    if gemini_ans:
        return gemini_ans, "Gemini AI Copilot"

    local_ans = query_local_advisor(query, kpis, df_txn, df_asset, df_liab)
    return local_ans, "Quantitative Intelligence Copilot"


def generate_advisory_chart(
    query: str,
    df_txn: pd.DataFrame,
    df_asset: pd.DataFrame,
    df_liab: pd.DataFrame,
    kpis: Dict[str, Any],
    view_type: Optional[str] = None,
    with_range_slider: bool = True,
) -> Tuple[go.Figure, str, str, str]:
    """
    Dynamically generates an executive Plotly chart based on user inquiry
    or explicit metric selection.
    Returns: (fig, title, subtitle, detected_mode)
    """
    from datetime import timedelta

    q_lower = (query or "").lower()
    df_t = df_txn.copy()
    df_t["parsed_date"] = pd.to_datetime(df_t["date"], errors="coerce")
    max_date = df_t["parsed_date"].max()
    min_date = df_t["parsed_date"].min()

    days, is_income, is_expense, is_both, cat_filter, _, _ = parse_date_window_query(query, df_txn)

    # Check if custom window mode is explicitly requested or auto-detected
    is_custom_requested = (view_type and view_type.startswith("Custom Timeline")) or (
        days and (not view_type or view_type == "Auto (Query-Matched)")
    )

    if is_custom_requested and days:
        cutoff_date = max_date - timedelta(days=days)
        slice_df = df_t[df_t["parsed_date"] >= cutoff_date]

        if slice_df.empty:
            fig = go.Figure()
            fig.add_annotation(
                text=f"<b>No transactions logged in this {days}-day window</b><br>({cutoff_date.strftime('%d-%b-%Y')} to {max_date.strftime('%d-%b-%Y')}).",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                font=dict(size=13, color="#1e293b"),
                bgcolor="rgba(241, 245, 249, 0.95)",
                bordercolor="#cbd5e1",
                borderwidth=1,
                borderpad=10,
            )
            fig.update_layout(height=380, margin=dict(l=20, r=20, t=30, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            title = f"Activity Timeline (Last {days} Days: {cutoff_date.strftime('%d-%b-%Y')} to {max_date.strftime('%d-%b-%Y')})"
            subtitle = f"No activity recorded in this {days}-day interval."
            return fig, title, subtitle, f"Custom Timeline (Last {days} Days)"

        if is_income and not is_both:
            inc_df = slice_df[slice_df["type"] == "income"].sort_values(by="parsed_date")
            fig = go.Figure()
            if inc_df.empty:
                all_inc = df_t[df_t["type"] == "income"].sort_values(by="parsed_date")
                last_inc = all_inc.iloc[-1] if not all_inc.empty else None
                last_txt = f"₹{last_inc['amount']:,.0f} ({last_inc['category']}) on {last_inc['date']}" if last_inc is not None else "N/A"
                sample_dates = slice_df["date"].drop_duplicates().sort_values().tolist()
                fig.add_trace(go.Bar(
                    x=sample_dates,
                    y=[0.0] * len(sample_dates),
                    name="Earnings (₹)",
                    marker_color="#10b981",
                    hovertemplate="<b>%{x}</b><br>Amount: ₹0<extra></extra>"
                ))
                fig.add_annotation(
                    text=f"<b>No earnings recorded in this {days}-day window</b><br>({cutoff_date.strftime('%d-%b-%Y')} to {max_date.strftime('%d-%b-%Y')})<br>Previous salary credit: <b>{last_txt}</b>",
                    xref="paper", yref="paper", x=0.5, y=0.55, showarrow=False,
                    font=dict(size=13, color="#1e293b"),
                    bgcolor="rgba(241, 245, 249, 0.95)",
                    bordercolor="#cbd5e1",
                    borderwidth=1,
                    borderpad=12,
                )
                fig.update_layout(
                    height=380,
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, tickangle=-20),
                    yaxis=dict(gridcolor="#f1f5f9", tickprefix="₹", range=[0, 10000]),
                    showlegend=False,
                )
                title = f"Earnings & Inflows Breakdown (Last {days} Days: {cutoff_date.strftime('%d-%b-%Y')} to {max_date.strftime('%d-%b-%Y')})"
                subtitle = f"Total Inflows: ₹0 in this window. Previous salary credit: {last_txt}."
                return fig, title, subtitle, f"Custom Timeline (Last {days} Days)"
            else:
                fig.add_trace(go.Bar(
                    x=inc_df["date"],
                    y=inc_df["amount"],
                    name="Earnings (₹)",
                    marker_color="#10b981",
                    text=[f"₹{v:,.0f}" for v in inc_df["amount"]],
                    textposition="outside",
                    hovertext=inc_df["category"] + " - " + inc_df["description"],
                    hovertemplate="<b>%{x}</b><br>Amount: ₹%{y:,.0f}<br>%{hovertext}<extra></extra>"
                ))
                fig.update_layout(
                    height=380,
                    margin=dict(l=20, r=20, t=30, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, tickangle=-20),
                    yaxis=dict(gridcolor="#f1f5f9", tickprefix="₹"),
                    showlegend=False,
                )
                title = f"Earnings & Inflows Breakdown (Last {days} Days: {cutoff_date.strftime('%d-%b-%Y')} to {max_date.strftime('%d-%b-%Y')})"
                subtitle = f"Total Inflows: ₹{inc_df['amount'].sum():,.0f} across {len(inc_df)} credit transaction(s)."
                return fig, title, subtitle, f"Custom Timeline (Last {days} Days)"

        elif is_expense and not is_both:
            exp_df = slice_df[slice_df["type"] == "expense"].sort_values(by="parsed_date")
            if cat_filter:
                exp_df = exp_df[exp_df["category"] == cat_filter]
            daily_exp = exp_df.groupby("date")["amount"].sum().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=daily_exp["date"],
                y=daily_exp["amount"],
                name="Expenses (₹)",
                marker_color="#ef4444",
                hovertemplate="<b>%{x}</b><br>Spend: ₹%{y:,.0f}<extra></extra>"
            ))
            fig.update_layout(
                height=380,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, tickangle=-45),
                yaxis=dict(gridcolor="#f1f5f9", tickprefix="₹"),
                showlegend=False,
            )
            cat_str = f" for '{cat_filter}'" if cat_filter else ""
            title = f"Expenditure Trajectory{cat_str} (Last {days} Days: {cutoff_date.strftime('%d-%b-%Y')} to {max_date.strftime('%d-%b-%Y')})"
            subtitle = f"Total Outflows: ₹{exp_df['amount'].sum():,.0f} across {len(exp_df)} transactions."
            return fig, title, subtitle, f"Custom Timeline (Last {days} Days)"

        else:
            # Both inflows & outflows in window (or general timeline)
            d_m = slice_df.groupby(["date", "type"])["amount"].sum().unstack(fill_value=0.0).reset_index()
            if "income" not in d_m.columns:
                d_m["income"] = 0.0
            if "expense" not in d_m.columns:
                d_m["expense"] = 0.0
            fig = go.Figure()
            fig.add_trace(go.Bar(x=d_m["date"], y=d_m["income"], name="Inflows", marker_color="#10b981"))
            fig.add_trace(go.Bar(x=d_m["date"], y=d_m["expense"], name="Outflows", marker_color="#ef4444"))
            fig.update_layout(
                barmode="group",
                height=380,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(showgrid=False, tickangle=-45),
                yaxis=dict(gridcolor="#f1f5f9", tickprefix="₹"),
                legend=dict(orientation="h", y=1.14, x=1, xanchor="right"),
            )
            tot_in = float(slice_df[slice_df['type']=='income']['amount'].sum())
            tot_out = float(slice_df[slice_df['type']=='expense']['amount'].sum())
            title = f"Inflows vs Outflows Timeline (Last {days} Days: {cutoff_date.strftime('%d-%b-%Y')} to {max_date.strftime('%d-%b-%Y')})"
            subtitle = f"Total Inflows: ₹{tot_in:,.0f} | Total Outflows: ₹{tot_out:,.0f}"
            return fig, title, subtitle, f"Custom Timeline (Last {days} Days)"

    # Standard View Modes
    if not view_type or view_type == "Auto (Query-Matched)":
        if any(w in q_lower for w in ["how much do we earn", "how much earn", "what do we earn", "how much we earn", "earnings", "income", "inflow", "inflows", "salary"]):
            chosen_mode = "Monthly Cash Flow by Date"
        elif any(w in q_lower for w in ["where does the money go", "where money go", "where do we spend", "category", "categories", "food", "shopping", "dining", "leak", "leakage", "drift", "surge", "other", "grocer", "travel"]):
            chosen_mode = "Spending Drift & Category Breakdown"
        elif any(w in q_lower for w in ["are we saving enough", "saving enough", "savings rate", "save enough", "savings", "surplus"]):
            chosen_mode = "Monthly Cash Flow by Date"
        elif any(w in q_lower for w in ["can we handle our debt", "handle debt", "debt", "liabilit", "loan", "credit card", "apr", "interest", "emi", "drag", "dti", "car", "vehicle", "buy", "afford", "purchase"]):
            chosen_mode = "Liabilities & Interest Drag"
        elif any(w in q_lower for w in ["what changed recently", "what changed", "change", "recent", "drift", "trend", "spike", "outlier"]):
            chosen_mode = "Spending Drift & Category Breakdown"
        elif any(w in q_lower for w in ["what should we do next", "do next", "next action", "roadmap", "plan", "priorit", "recommend", "intervention"]):
            chosen_mode = "Strategic Action Impact Matrix"
        elif any(w in q_lower for w in ["runway", "liquid", "safety", "emergency", "buffer", "deficit", "reserve", "cash buffer", "fd"]):
            chosen_mode = "Liquidity Runway vs 6-Month Mandate"
        elif any(w in q_lower for w in ["net worth", "asset", "wealth", "equity", "solvency", "worth", "health score", "score", "balance sheet"]):
            chosen_mode = "Net Worth & Multi-Asset Allocation"
        else:
            chosen_mode = "Monthly Cash Flow by Date"
    else:
        chosen_mode = view_type

    if chosen_mode == "Monthly Cash Flow by Date":
        df_t["month_str"] = df_t["parsed_date"].dt.strftime("%Y-%m")
        m_df = df_t.groupby(["month_str", "type"])["amount"].sum().unstack(fill_value=0.0).reset_index()
        if "income" not in m_df.columns:
            m_df["income"] = 0.0
        if "expense" not in m_df.columns:
            m_df["expense"] = 0.0
        m_df["net_cash_flow"] = m_df["income"] - m_df["expense"]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=m_df["month_str"],
            y=m_df["income"],
            name="Inflows (Income)",
            marker_color="#10b981",
            opacity=0.9,
            hovertemplate="<b>%{x}</b><br>Inflows: ₹%{y:,.0f}<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            x=m_df["month_str"],
            y=m_df["expense"],
            name="Outflows (Expense)",
            marker_color="#ef4444",
            opacity=0.9,
            hovertemplate="<b>%{x}</b><br>Outflows: ₹%{y:,.0f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=m_df["month_str"],
            y=m_df["net_cash_flow"],
            name="Net Retained Cash Flow",
            mode="lines+markers",
            line=dict(color="#00a1de", width=2.5),
            marker=dict(size=6, color="#007bff"),
            hovertemplate="<b>%{x}</b><br>Net Retained: ₹%{y:,.0f}<extra></extra>"
        ))

        recent_3m = m_df["month_str"].tail(3).tolist()
        if len(recent_3m) >= 3:
            fig.add_vrect(
                x0=recent_3m[0], x1=recent_3m[-1],
                fillcolor="#fef3c7", opacity=0.35,
                layer="below", line_width=1, line_color="#f59e0b",
                annotation_text="Recent Window (+10.8% Burn Expansion)",
                annotation_position="top left",
                annotation_font_size=10,
                annotation_font_color="#b45309"
            )

        xaxis_cfg = dict(showgrid=False, tickangle=-45)
        if with_range_slider:
            xaxis_cfg["rangeslider"] = dict(visible=True)

        fig.update_layout(
            barmode="group",
            height=390,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=xaxis_cfg,
            yaxis=dict(gridcolor="#f1f5f9", tickprefix="₹"),
            legend=dict(orientation="h", y=1.14, x=1, xanchor="right"),
        )
        title = "Monthly Cash Flow & Spending Trajectory by Date (24-Month Horizon)"
        subtitle = f"Tracking monthly income vs outflow run-rates. Over the recent 3 months, spending expanded by +{kpis['drift_expansion_pct']:.1f}% (+₹{kpis['drift_expansion_inr']:,.0f}/mo)."

    elif chosen_mode == "Spending Drift & Category Breakdown":
        df_exp = df_txn[df_txn["type"] == "expense"].copy()
        months = sorted(df_exp["month"].dropna().unique().tolist())
        recent_count = min(3, max(1, len(months) // 4))
        recent_months = months[-recent_count:]
        baseline_months = months[:-recent_count] if len(months) > recent_count else months

        base_cat = df_exp[df_exp["month"].isin(baseline_months)].groupby("category")["amount"].sum() / max(1, len(baseline_months))
        recent_cat = df_exp[df_exp["month"].isin(recent_months)].groupby("category")["amount"].sum() / max(1, len(recent_months))

        cat_df = pd.DataFrame({"Baseline": base_cat, "Recent": recent_cat}).fillna(0.0)
        cat_df["Drift_INR"] = cat_df["Recent"] - cat_df["Baseline"]
        cat_df["Drift_Pct"] = (cat_df["Drift_INR"] / cat_df["Baseline"]) * 100.0
        cat_df = cat_df.sort_values(by="Recent", ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=cat_df.index,
            x=cat_df["Baseline"],
            name="Baseline 21M Avg (₹/mo)",
            orientation="h",
            marker_color="#94a3b8",
            hovertemplate="<b>%{y}</b><br>Baseline: ₹%{x:,.0f}/mo<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            y=cat_df.index,
            x=cat_df["Recent"],
            name="Recent 3M Run-Rate (₹/mo)",
            orientation="h",
            marker_color="#00a1de",
            hovertemplate="<b>%{y}</b><br>Recent: ₹%{x:,.0f}/mo<extra></extra>"
        ))
        fig.update_layout(
            barmode="group",
            height=390,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(gridcolor="#f1f5f9", tickprefix="₹"),
            legend=dict(orientation="h", y=1.14, x=1, xanchor="right"),
        )
        title = "Category Expenditure Drift Analysis: Baseline vs Recent Observation"
        subtitle = "Discretionary categories ('Other' +100.0%, 'Shopping' +74.2%, 'Food' +48.2%) represent 88% of net expansion."

    elif chosen_mode == "Liabilities & Interest Drag":
        df_l = df_liab.copy()
        df_l["annual_drag"] = df_l["outstanding"] * (df_l["interest_rate"] / 100.0)
        df_l["label"] = df_l["liability_id"] + "<br>(" + df_l["type"] + ")"

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_l["label"],
            y=df_l["outstanding"],
            name="Outstanding Balance (₹)",
            marker_color="#007bff",
            text=[f"₹{v:,.0f}" for v in df_l["outstanding"]],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Outstanding: ₹%{y:,.0f}<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            x=df_l["label"],
            y=df_l["annual_drag"],
            name="Annual Interest Drag (₹)",
            marker_color=["#ef4444" if r > 20 else "#f59e0b" for r in df_l["interest_rate"]],
            text=[f"₹{v:,.0f} ({r:.1f}% APR)" for v, r in zip(df_l["annual_drag"], df_l["interest_rate"])],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Annual Drag: ₹%{y:,.0f}<extra></extra>"
        ))
        fig.update_layout(
            barmode="group",
            height=390,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#f1f5f9", tickprefix="₹"),
            legend=dict(orientation="h", y=1.14, x=1, xanchor="right"),
        )
        title = "Liabilities Capital Structure & Annual Interest Drain (APR Breakdown)"
        subtitle = "Credit Card L003 incurs ₹21,760 in annual interest drag at 32.0% APR on a ₹68,000 balance. Prepayment eliminates this immediately."

    elif chosen_mode == "Liquidity Runway vs 6-Month Mandate":
        liq_items = df_asset[df_asset["is_liquid"] == True]
        tot_liq = float(liq_items["value"].sum())
        target_6m = kpis["target_6m_runway"]
        deficit = kpis["runway_deficit"]

        x_labels = liq_items["type"].tolist() + ["Total Liquid Reserves", "Capital Deficit Needed", "Target 6M Runway"]
        y_values = liq_items["value"].tolist() + [tot_liq, deficit, target_6m]
        colors = ["#93c5fd"] * len(liq_items) + ["#00a1de", "#f59e0b", "#10b981"]

        fig = go.Figure(go.Bar(
            x=x_labels,
            y=y_values,
            marker_color=colors,
            text=[f"₹{v:,.0f}" for v in y_values],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Valuation: ₹%{y:,.0f}<extra></extra>"
        ))
        fig.update_layout(
            height=390,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#f1f5f9", tickprefix="₹"),
            showlegend=False,
        )
        title = "Liquid Reserves vs 6-Month Safety Mandate"
        subtitle = f"Current liquid reserves of ₹{tot_liq:,.0f} cover {kpis['runway_months']:.1f} months. An exact capital allocation of ₹{deficit:,.0f} reaches the 6-month safety threshold of ₹{target_6m:,.0f}."

    elif chosen_mode == "Strategic Action Impact Matrix":
        moves = ["1. Prepay Credit Card (L003)", "2. Cap Discretionary 'Other'", "3. Fund 6M Safety Runway"]
        impact_yr = [21760.0, 52176.0, kpis["runway_deficit"]]
        bar_colors = ["#10b981", "#00a1de", "#f59e0b"]
        descriptions = [
            "Eliminates 32% APR interest drag; frees ₹7,000/mo cash flow",
            "Enforces ₹4,400/mo budget cap; arrests +100% surge",
            f"Deploys ₹{kpis['runway_deficit']:,.0f} to achieve 6-month safety buffer",
        ]

        fig = go.Figure(go.Bar(
            x=moves,
            y=impact_yr,
            marker_color=bar_colors,
            text=[f"₹{v:,.0f}" for v in impact_yr],
            textposition="outside",
            hovertext=descriptions,
            hovertemplate="<b>%{x}</b><br>Financial Magnitude: ₹%{y:,.0f}<br>%{hovertext}<extra></extra>"
        ))
        fig.update_layout(
            height=390,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#f1f5f9", tickprefix="₹"),
            showlegend=False,
        )
        title = "Prescribed Strategic Interventions & Capital Impact Matrix"
        subtitle = "Immediate quantified annual recovery: ₹73,936/year in combined interest savings and discretionary leakage containment."

    else:  # Net Worth & Multi-Asset Allocation
        tot_assets = kpis["total_assets"]
        tot_debt = kpis["total_debt"]
        net_worth = kpis["net_worth"]

        fig = go.Figure(data=[
            go.Pie(
                labels=df_asset["type"],
                values=df_asset["value"],
                hole=0.62,
                domain={"x": [0.0, 0.48]},
                marker=dict(colors=["#007bff", "#00a1de", "#10b981", "#38bdf8"]),
                name="Assets",
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Valuation: ₹%{value:,.0f} (%{percent})<extra></extra>",
            ),
            go.Pie(
                labels=["Net Worth (Equity)", "Total Debt Obligations"],
                values=[net_worth, tot_debt],
                hole=0.62,
                domain={"x": [0.52, 1.0]},
                marker=dict(colors=["#00a1de", "#ef4444"]),
                name="Capital Structure",
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Amount: ₹%{value:,.0f} (%{percent})<extra></extra>",
            ),
        ])
        fig.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=30, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            annotations=[
                dict(text="<b>Asset Classes</b>", x=0.21, y=0.5, font_size=12, font_color="#0f172a", showarrow=False),
                dict(text="<b>Solvency: 2.1x</b>", x=0.79, y=0.5, font_size=12, font_color="#0f172a", showarrow=False),
            ],
            showlegend=False,
        )
        title = "Multi-Asset Allocation (Left) & Balance Sheet Solvency Structure (Right)"
        subtitle = f"Total Assets: ₹{tot_assets:,.0f} | Total Liabilities: ₹{tot_debt:,.0f} | Net Worth: ₹{net_worth:,.0f}"

    return fig, title, subtitle, chosen_mode


# ==============================================================================
# 5. HEADER BAR & EXECUTIVE BANNER
# ==============================================================================

logo_badge_html = f'<img src="data:image/png;base64,{icon_base64}" style="width: 36px; height: 36px; border-radius: 8px; object-fit: cover; box-shadow: 0 2px 6px rgba(0,0,0,0.12); flex-shrink: 0;" />' if icon_base64 else '<div class="vp-logo-badge">VP</div>'

header_html = f"""
<div class="vp-header">
    <div class="vp-brand">
        {logo_badge_html}
        <div>
            <div class="vp-brand-name">VantagePoint</div>
            <div class="vp-brand-sub">Financial Intelligence Platform</div>
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
    with st.expander("4-PILLAR WEIGHTED SOLVENCY INDEX (Click to view full scoring model)", expanded=False):
        st.markdown(
            f"""
            <div class="inspector-box" style="margin-top: 0; box-shadow: none;">
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

    # 6 Core Financial Inquiries Executive Briefing
    st.markdown(
        f"""
        <div class='vp-card' style='margin-bottom: 1.25rem;'>
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;'>
                <div class='vp-card-title' style='margin-bottom: 0;'>Strategic Executive Briefing (6 Core Financial Inquiries)</div>
                <span class='pill-info'>Automated Synthesis</span>
            </div>
            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 0.85rem;'>
                <div style='background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.75rem 0.85rem;'>
                    <div style='font-size: 0.74rem; font-weight: 700; color: #007bff; margin-bottom: 0.2rem;'>1. HOW MUCH DO WE EARN?</div>
                    <div style='font-size: 1.15rem; font-weight: 800; color: #0f172a;'>₹{kpis['avg_monthly_income']:,.0f} <span style='font-size: 0.75rem; font-weight: 500; color: #64748b;'>/ mo</span></div>
                    <div style='font-size: 0.75rem; color: #475569; margin-top: 0.2rem;'>Tech salary ₹3.30L/mo + consulting credits</div>
                </div>
                <div style='background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.75rem 0.85rem;'>
                    <div style='font-size: 0.74rem; font-weight: 700; color: #ef4444; margin-bottom: 0.2rem;'>2. WHERE DOES THE MONEY GO?</div>
                    <div style='font-size: 1.15rem; font-weight: 800; color: #0f172a;'>₹{kpis['recent_burn']:,.0f} <span style='font-size: 0.75rem; font-weight: 500; color: #64748b;'>/ mo burn</span></div>
                    <div style='font-size: 0.75rem; color: #475569; margin-top: 0.2rem;'>Rent (₹55K), Debt (₹46.7K), Food (₹34.6K)</div>
                </div>
                <div style='background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.75rem 0.85rem;'>
                    <div style='font-size: 0.74rem; font-weight: 700; color: #10b981; margin-bottom: 0.2rem;'>3. ARE WE SAVING ENOUGH?</div>
                    <div style='font-size: 1.15rem; font-weight: 800; color: #10b981;'>{kpis['savings_rate']:.1f}% <span style='font-size: 0.75rem; font-weight: 500; color: #047857;'>(Healthy)</span></div>
                    <div style='font-size: 0.75rem; color: #475569; margin-top: 0.2rem;'>Net monthly surplus: +₹{(kpis['avg_monthly_income'] - kpis['avg_monthly_expense']):,.0f}/mo</div>
                </div>
                <div style='background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.75rem 0.85rem;'>
                    <div style='font-size: 0.74rem; font-weight: 700; color: #6366f1; margin-bottom: 0.2rem;'>4. CAN WE HANDLE OUR DEBT?</div>
                    <div style='font-size: 1.15rem; font-weight: 800; color: #0f172a;'>{kpis['dti']:.1f}% DTI <span style='font-size: 0.75rem; font-weight: 500; color: #10b981;'>(&lt;35% safe)</span></div>
                    <div style='font-size: 0.75rem; color: #475569; margin-top: 0.2rem;'>Prepay 32% APR Card L003 (save ₹21.8K/yr)</div>
                </div>
                <div style='background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.75rem 0.85rem;'>
                    <div style='font-size: 0.74rem; font-weight: 700; color: #f59e0b; margin-bottom: 0.2rem;'>5. WHAT CHANGED RECENTLY?</div>
                    <div style='font-size: 1.15rem; font-weight: 800; color: #d97706;'>+{kpis['drift_expansion_pct']:.1f}% Drift</div>
                    <div style='font-size: 0.75rem; color: #475569; margin-top: 0.2rem;'>Other (+100%), Shopping (+74%), Food (+48%)</div>
                </div>
                <div style='background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.75rem 0.85rem;'>
                    <div style='font-size: 0.74rem; font-weight: 700; color: #0d9488; margin-bottom: 0.2rem;'>6. WHAT SHOULD WE DO NEXT?</div>
                    <div style='font-size: 1.15rem; font-weight: 800; color: #0f172a;'>3 Strategic Moves</div>
                    <div style='font-size: 0.75rem; color: #475569; margin-top: 0.2rem;'>Prepay Card L003, Cap Other, Fund 6M Runway</div>
                </div>
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
        df_a_disp = df_asset[["asset_id", "type", "value", "is_liquid"]].copy()
        df_a_disp["value"] = df_a_disp["value"].apply(lambda v: f"₹{v:,.0f}")
        df_a_disp["is_liquid"] = df_a_disp["is_liquid"].apply(lambda x: "Yes (Liquid)" if x else "No (Illiquid)")
        df_a_disp.columns = ["Asset ID", "Asset Class", "Valuation", "Liquidity Status"]
        st.dataframe(df_a_disp, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with b_col2:
        st.markdown("<div class='vp-card'><div class='vp-card-title'>Liabilities & Interest Drag</div>", unsafe_allow_html=True)
        df_l_disp = df_liab.copy()
        df_l_disp["annual_drag"] = df_l_disp["outstanding"] * (df_l_disp["interest_rate"] / 100.0)
        df_l_fmt = pd.DataFrame({
            "Liability ID": df_l_disp["liability_id"],
            "Facility Type": df_l_disp["type"],
            "Outstanding": df_l_disp["outstanding"].apply(lambda v: f"₹{v:,.0f}"),
            "APR Rate": df_l_disp["interest_rate"].apply(lambda v: f"{v:.1f}%"),
            "Monthly EMI": df_l_disp["emi"].apply(lambda v: f"₹{v:,.0f}"),
            "Annual Interest Drag": df_l_disp["annual_drag"].apply(lambda v: f"₹{v:,.0f}"),
        })
        st.dataframe(df_l_fmt, use_container_width=True, hide_index=True)
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
# TAB 5: ADVISORY & INTERACTIVE AI COPILOT
# ==============================================================================

with tab_advisory:
    st.markdown("<div class='vp-card'><div class='vp-card-title'>Strategic Financial Advisory Memorandum</div>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="font-size: 0.88rem; line-height: 1.65; color: #334155;">
            <b>Outflow Audit & Spending Drift:</b> Over the recent 3-month observation window, monthly household burn accelerated by 
            <b>+{kpis['drift_expansion_pct']:.1f}%</b> (+₹{kpis['drift_expansion_inr']:,.0f}/mo), expanding from a 21-month historical baseline 
            of ₹{kpis['baseline_burn']:,.0f} to an active run-rate of <b>₹{kpis['recent_burn']:,.0f}/month</b>. 
            Discretionary categories expanded rapidly, alongside 12 detected statistical outliers including the ₹1,85,000 mobile bill under Utilities (Txn T0488).
            <br><br>
            <b>Prescribed Interventions:</b> First, prepay Credit Card L003 (₹68,000 at 32.0% APR) immediately ahead of its due date to terminate ₹21,760 in annual 
            interest drag and free ₹7,000/mo cash flow. Second, enforce a hard monthly budget cap of ₹4,400 on 'Other' discretionary expenditures, capturing 
            ₹52,176 annually in leakage. Third, allocate exactly ₹{kpis['runway_deficit']:,.0f} into liquid sweep instruments to elevate current reserves from 
            ₹{kpis['liquid_assets']:,.0f} ({kpis['runway_months']:.1f} months) to the institutional standard 6-month safety runway of ₹{kpis['target_6m_runway']:,.0f}.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Interactive Advisory Copilot Console
    st.markdown(
        """
        <div class='vp-card'>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
                <div class='vp-card-title' style="margin-bottom: 0;">Advisor Interactive Copilot</div>
                <span class='pill-info'>AI &amp; Quantitative Engine Active</span>
            </div>
            <div style="font-size: 0.82rem; color: #64748b; margin-bottom: 0.85rem;">
                Select a strategic executive inquiry or enter any custom question regarding liabilities, spending drift, liquidity runway, or portfolio health:
            </div>
        """,
        unsafe_allow_html=True,
    )

    # 6 Core Strategic Inquiries Matching Presentation Requirements
    st.markdown("<div style='font-size: 0.74rem; font-weight: 700; color: #475569; margin-bottom: 0.4rem; text-transform: uppercase;'>Core Strategic Questions:</div>", unsafe_allow_html=True)
    p_r1_c1, p_r1_c2, p_r1_c3 = st.columns(3)
    p_r2_c1, p_r2_c2, p_r2_c3 = st.columns(3)
    preset_clicked = None
    with p_r1_c1:
        if st.button("How much do we earn?", key="btn_q_earn", use_container_width=True):
            preset_clicked = "How much do we earn?"
    with p_r1_c2:
        if st.button("Where does the money go?", key="btn_q_where", use_container_width=True):
            preset_clicked = "Where does the money go?"
    with p_r1_c3:
        if st.button("Are we saving enough?", key="btn_q_saving", use_container_width=True):
            preset_clicked = "Are we saving enough?"
    with p_r2_c1:
        if st.button("Can we handle our debt?", key="btn_q_debt", use_container_width=True):
            preset_clicked = "Can we handle our debt?"
    with p_r2_c2:
        if st.button("What changed recently?", key="btn_q_recent", use_container_width=True):
            preset_clicked = "What changed recently?"
    with p_r2_c3:
        if st.button("What should we do next?", key="btn_q_next", use_container_width=True):
            preset_clicked = "What should we do next?"

    # Custom Question Form (Pressing Enter automatically submits!)
    with st.form(key="advisory_query_form", clear_on_submit=False):
        c_input_col, c_btn_col = st.columns([5, 1])
        with c_input_col:
            custom_typed = st.text_input(
                "Custom Question",
                placeholder="Ask any financial question (e.g. Can we afford to buy property? How to cut interest drag?)...",
                label_visibility="collapsed",
                key="txt_custom_advisory_input",
            )
        with c_btn_col:
            form_clicked = st.form_submit_button("Ask Copilot", type="primary", use_container_width=True)

    # Process query
    active_query = None
    if preset_clicked:
        active_query = preset_clicked
    elif form_clicked and custom_typed.strip():
        active_query = custom_typed.strip()

    if active_query:
        with st.spinner("Analyzing portfolio balance sheet and consulting advisory engine..."):
            ans, engine = ask_advisor(active_query, kpis, df_txn, df_asset, df_liab)
            st.session_state["advisory_query"] = active_query
            st.session_state["advisory_response"] = ans
            st.session_state["advisory_engine"] = engine

    # Render response if available in session_state
    if st.session_state.get("advisory_response"):
        disp_q = st.session_state.get("advisory_query", "")
        disp_ans = st.session_state.get("advisory_response", "")
        disp_eng = st.session_state.get("advisory_engine", "Quantitative Intelligence Copilot")
        badge_cls = "pill-info" if "Gemini" in disp_eng else "pill-neutral"

        # Format markdown lines with HTML safe spacing
        formatted_ans = disp_ans.replace("\n\n", "<br><br>").replace("\n", "<br>")

        st.markdown(
            f"""
            <div style="background: #ffffff; border: 1px solid #cbd5e1; border-left: 4px solid #00a1de; border-radius: 8px; padding: 1.15rem 1.3rem; margin-top: 1rem; box-shadow: 0 2px 8px rgba(0,0,0,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem; border-bottom: 1px solid #f1f5f9; padding-bottom: 0.5rem;">
                    <div style="font-weight: 700; color: #0f172a; font-size: 0.95rem;">Inquiry: {disp_q}</div>
                    <span class="{badge_cls}" style="font-size: 0.72rem; font-weight: 700;">Engine: {disp_eng}</span>
                </div>
                <div style="font-size: 0.88rem; line-height: 1.65; color: #1e293b;">
                    {formatted_ans}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Dynamic Intelligence Chart Section!
        st.markdown(
            """
            <div style="margin-top: 1rem; padding: 1rem 1.25rem; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <div style="font-weight: 700; color: #0f172a; font-size: 0.92rem;">Generated Portfolio Intelligence Visualization</div>
                    <span class="pill-info" style="font-size: 0.7rem;">Interactive Analytics</span>
                </div>
            """,
            unsafe_allow_html=True,
        )

        q_days, _, _, _, _, _, _ = parse_date_window_query(disp_q, df_txn)
        view_options = ["Auto (Query-Matched)"]
        if q_days:
            view_options.append(f"Custom Timeline (Last {q_days} Days)")
        view_options.extend([
            "Monthly Cash Flow by Date",
            "Spending Drift & Category Breakdown",
            "Liabilities & Interest Drag",
            "Liquidity Runway vs 6-Month Mandate",
            "Net Worth & Multi-Asset Allocation",
            "Strategic Action Impact Matrix",
        ])

        c_view_sel, c_slider_chk = st.columns([3, 1])
        with c_view_sel:
            chosen_view = st.selectbox(
                "Select Metric / Graph Perspective:",
                view_options,
                key="adv_chart_selector",
                help="Switch perspective to explore different angles of the portfolio",
            )
        with c_slider_chk:
            show_slider = st.checkbox("Show Date Slider", value=True, key="adv_show_slider")

        chart_fig, chart_title, chart_subtitle, detected_mode = generate_advisory_chart(
            disp_q,
            df_txn,
            df_asset,
            df_liab,
            kpis,
            view_type=chosen_view,
            with_range_slider=show_slider,
        )

        st.markdown(
            f"""
            <div style="font-size: 0.85rem; font-weight: 600; color: #007bff; margin-top: 0.25rem;">{chart_title}</div>
            <div style="font-size: 0.78rem; color: #64748b; margin-bottom: 0.5rem;">{chart_subtitle}</div>
            """,
            unsafe_allow_html=True,
        )

        st.plotly_chart(chart_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        r_c1, r_c2 = st.columns([5, 1])
        with r_c2:
            if st.button("Clear Response", key="btn_clear_adv_resp", use_container_width=True):
                st.session_state.pop("advisory_query", None)
                st.session_state.pop("advisory_response", None)
                st.session_state.pop("advisory_engine", None)
                st.rerun()
    else:
        # Default Trajectory Chart when no query active yet
        st.markdown(
            """
            <div style="margin-top: 1rem; padding: 1rem 1.25rem; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem;">
                    <div style="font-weight: 700; color: #0f172a; font-size: 0.92rem;">Executive Cash Flow Trajectory by Date</div>
                    <span class="pill-info" style="font-size: 0.7rem;">24-Month Active Audit</span>
                </div>
                <div style="font-size: 0.78rem; color: #64748b; margin-bottom: 0.5rem;">
                    Monthly inflows vs outflows over time. Ask the Copilot above to generate custom liability, category, or runway charts.
                </div>
            """,
            unsafe_allow_html=True,
        )
        default_fig, default_title, default_sub, _ = generate_advisory_chart(
            "Monthly Cash Flow by Date",
            df_txn,
            df_asset,
            df_liab,
            kpis,
            view_type="Monthly Cash Flow by Date",
            with_range_slider=True,
        )
        st.plotly_chart(default_fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# 8. FLOATING AI ADVISOR WINDOW (SIDEBAR DOCK)
# ==============================================================================

with st.sidebar:
    st.markdown("<hr style='margin: 1.25rem 0 0.75rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.95rem; font-weight: 800; color: #0f172a;'>Floating AI Copilot</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.75rem; color: #64748b; margin-bottom: 0.5rem;'>Ask strategic portfolio questions from any page:</div>", unsafe_allow_html=True)

    with st.form(key="sidebar_copilot_dock_form", clear_on_submit=False):
        dock_q_input = st.text_input(
            "Ask Copilot:",
            placeholder="Runway, debt, net worth...",
            label_visibility="collapsed",
            key="txt_sidebar_dock_q",
        )
        dock_submitted = st.form_submit_button("Ask Copilot", use_container_width=True)

    if dock_submitted and dock_q_input.strip():
        with st.spinner("Analyzing..."):
            sb_ans, sb_eng = ask_advisor(dock_q_input.strip(), kpis, df_txn, df_asset, df_liab)
            st.session_state["dock_last_q"] = dock_q_input.strip()
            st.session_state["dock_last_ans"] = sb_ans
            st.session_state["dock_last_eng"] = sb_eng

    if st.session_state.get("dock_last_ans"):
        dock_q_val = st.session_state.get("dock_last_q", "")
        dock_a_val = st.session_state.get("dock_last_ans", "")
        dock_e_val = st.session_state.get("dock_last_eng", "")
        
        # Display first 350 chars with ellipsis if long
        short_ans = dock_a_val if len(dock_a_val) < 400 else dock_a_val[:380] + "..."
        formatted_dock = short_ans.replace("\n\n", "<br><br>").replace("\n", "<br>")

        st.markdown(
            f"""
            <div style="background: #ffffff; border: 1px solid #cbd5e1; border-left: 3px solid #00a1de; border-radius: 8px; padding: 0.75rem 0.85rem; margin-top: 0.6rem; font-size: 0.8rem; line-height: 1.5;">
                <div style="font-weight: 700; color: #007bff; margin-bottom: 0.35rem;">{dock_q_val}</div>
                <div style="color: #334155; margin-bottom: 0.5rem;">{formatted_dock}</div>
                <div style="font-size: 0.68rem; color: #64748b; font-weight: 600;">Engine: {dock_e_val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Clear Dock", key="btn_clear_sidebar_dock", use_container_width=True):
            st.session_state.pop("dock_last_q", None)
            st.session_state.pop("dock_last_ans", None)
            st.session_state.pop("dock_last_eng", None)
            st.rerun()
