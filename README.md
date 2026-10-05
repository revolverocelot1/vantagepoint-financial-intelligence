# VantagePoint — Financial Intelligence Platform

VantagePoint is an enterprise-grade financial analytics, forensic anomaly detection, and portfolio solvency platform. 

---

## Key Features

1. **Executive Solvency & 4-Pillar Health Score**:
   - Composite Financial Health Score (0–100) combining Savings Rate, Debt-to-Income (DTI), 6-Month Emergency Runway, and Spending Drift Stability.
   - Interactive Top 5 KPI Cards: Clicking any card reveals a dynamic Inspector Panel detailing underlying formulas, ratios, and balance sheet leverage.

2. **First-Page Portfolio Architecture**:
   - **Net Worth Donut**: Net Equity (₹36,87,000) vs. Total Debt (₹33,38,000).
   - **Asset Allocation Donut**: Granular distribution across 8 asset classes (Property, Cash, Mutual Funds, Equity, Vehicle, Gold).
   - **Interactive Recent 25 Transactions Ledger**: Real-time search, multi-vector filtering, and **User Anomaly Override** allowing users to check or uncheck anomaly flags on any transaction in real-time.

3. **Dynamic User CSV Ingestion Desk**:
   - Upload custom `Transactions.csv`, `Assets.csv`, and `Liabilities.csv` files via the sidebar.
   - Automatically sanitizes data, imputes missing categories, and computes all financial metrics on the fly.
   - One-click reset to default benchmark datasets.

4. **Forensic Anomaly & Spending Drift Engine**:
   - 21-month historical baseline vs. 3-month recent comparison (% and ₹ growth).
   - Multi-tier statistical anomaly detection (Tukey IQR and Z-Scores).
   - 14-point data hygiene reconciliation log.

5. **Strategic Advisory & AI Copilot**:
   - Three deterministic strategic actions (Debt Avalanche prepayment of 32% APR debt, leakage budget cap, and 6-month safety runway deficit contribution).
   - Strategic Memorandum, dynamic custom timeline visualizations (e.g. custom N-day earnings and expenses graphs), and stateless Q&A console with quick presets.
   - Floating AI Copilot dock.

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Launch the Application
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Repository Structure
```
├── app.py                     # VantagePoint Streamlit Application
├── detector_and_advisor.py    # Trends, Anomaly & Deterministic Strategy Engine
├── clean_dataset.py           # Forensic Data Hygiene Pipeline
├── generate_clean_data.py     # Clean Dataset Generation & Validation Script
├── requirements.txt           # Project Dependencies
├── Dataset/                   # Benchmark Financial Datasets
│   ├── transactions.csv       # Raw 24-Month Transaction Ledger
│   ├── assets.csv             # Raw Asset Positions
│   └── liabilities.csv        # Raw Debt Obligations
└── README.md                  # Project Documentation
```
