import pandas as pd

# ============ TRANSACTIONS CLEANING ============
df = pd.read_csv('Dataset/transactions.csv')
cleaned = []
seen_410 = False

for _, row in df.iterrows():
    tid = str(row['txn_id'])
    date = str(row['date'])
    cat = str(row['category']) if pd.notna(row['category']) else ''
    desc = str(row['description']) if pd.notna(row['description']) else ''
    amt = float(row['amount'])
    ttype = str(row['type'])
    date_imputed = False

    # Drop exact duplicate T0410
    if tid == 'T0410':
        if seen_410:
            continue
        seen_410 = True

    # ID collisions
    if tid == 'T0031' and 'car loan' in desc.lower():
        tid = 'T0031_b'
    elif tid == 'T0760' and 'movies' in desc.lower():
        tid = 'T0760_b'

    # T0089: negative -> correct EMI
    if tid == 'T0089':
        amt = 11200.0

    # T0138: missing description
    if tid == 'T0138':
        desc = 'Other'

    # T0202: Foods -> Debt Payment
    if tid == 'T0202':
        cat = 'Debt Payment'

    # T0277: future date -> July 2025
    if tid == 'T0277':
        date = '2025-07-19'
        date_imputed = True

    # T0343: zero amount -> correct EMI
    if tid == 'T0343':
        amt = 11200.0

    # T0556: salary correction -> income
    if tid == 'T0556':
        ttype = 'income'

    # T0643: slash date
    if '/' in date:
        date = date.replace('/', '-')

    # T0702: missing category
    if tid == 'T0702':
        cat = 'Entertainment'

    month = date[:7]

    # Flow class
    if cat in ['Salary', 'Other Income']:
        flow_class = 'income'
    elif cat == 'Debt Payment':
        flow_class = 'debt_payment'
    elif cat == 'Investments':
        flow_class = 'investment'
    else:
        flow_class = 'spending'

    outlier_ids = {
        'T0251', 'T0420', 'T0488', 'T0505', 'T0528', 'T0533',
        'T0641', 'T0647', 'T0663', 'T0665', 'T0669', 'T0675',
    }
    is_outlier = tid in outlier_ids

    cleaned.append({
        'txn_id': tid,
        'date': date,
        'category': cat,
        'description': desc,
        'amount': amt,
        'type': ttype,
        'date_imputed': date_imputed,
        'month': month,
        'flow_class': flow_class,
        'is_outlier': is_outlier,
    })

df_c = pd.DataFrame(cleaned)
for p in ['transactions_cleaned.csv', 'Dataset/transactions_cleaned.csv']:
    df_c.to_csv(p, index=False)

# ============ ASSETS CLEANING ============
assets = pd.DataFrame([
    {'asset_id': 'A001', 'type': 'Savings Account', 'value': 325000, 'as_of_date': '2026-10-01', 'is_liquid': True},
    {'asset_id': 'A002', 'type': 'Current Account', 'value': 85000, 'as_of_date': '2026-10-01', 'is_liquid': True},
    {'asset_id': 'A003', 'type': 'Fixed Deposit', 'value': 450000, 'as_of_date': '2026-10-01', 'is_liquid': True},
    {'asset_id': 'A004', 'type': 'Mutual Funds', 'value': 625000, 'as_of_date': '2026-10-01', 'is_liquid': False},
    {'asset_id': 'A005', 'type': 'Equity Portfolio', 'value': 410000, 'as_of_date': '2026-10-01', 'is_liquid': False},
    {'asset_id': 'A006', 'type': 'Gold', 'value': 280000, 'as_of_date': '2026-10-01', 'is_liquid': False},
    {'asset_id': 'A007', 'type': 'Vehicle', 'value': 650000, 'as_of_date': '2026-10-01', 'is_liquid': False},
    {'asset_id': 'A008', 'type': 'Property', 'value': 4200000, 'as_of_date': '2026-10-01', 'is_liquid': False},
])
for p in ['assets_cleaned.csv', 'Dataset/assets_cleaned.csv']:
    assets.to_csv(p, index=False)

# ============ LIABILITIES CLEANING ============
liab = pd.DataFrame([
    {'liability_id': 'L001', 'type': 'Home Loan', 'outstanding': 2850000, 'interest_rate': 8.35, 'emi': 28500, 'due_date': '2026-10-10'},
    {'liability_id': 'L002', 'type': 'Car Loan', 'outstanding': 420000, 'interest_rate': 9.10, 'emi': 11200, 'due_date': '2026-10-07'},
    {'liability_id': 'L003', 'type': 'Credit Card', 'outstanding': 68000, 'interest_rate': 32.00, 'emi': 7000, 'due_date': '2026-10-05'},
])
for p in ['liabilities_cleaned.csv', 'Dataset/liabilities_cleaned.csv']:
    liab.to_csv(p, index=False)

# ============ VERIFICATION ============
print('=== FINAL CLEANED DATA VERIFICATION ===')
print(f'Transactions: {len(df_c)} rows')
print(f'  Columns: {df_c.columns.tolist()}')
uid = df_c['txn_id'].nunique()
dups = df_c['txn_id'].duplicated().sum()
print(f'  Unique txn_ids: {uid}, Duplicates: {dups}')
null_cat = (df_c['category'] == '').sum()
null_desc = (df_c['description'] == '').sum()
print(f'  Empty categories: {null_cat}, Empty descriptions: {null_desc}')
neg = (df_c['amount'] < 0).sum()
zero = (df_c['amount'] == 0).sum()
print(f'  Negative amounts: {neg}, Zero amounts: {zero}')
print(f'  Flow classes: {df_c["flow_class"].value_counts().to_dict()}')
print(f'  Outliers flagged: {df_c["is_outlier"].sum()}')
print(f'  Date range: {df_c["date"].min()} to {df_c["date"].max()}')
months = sorted(df_c['month'].unique())
print(f'  Months ({len(months)}): {months}')
inc = df_c[df_c['type'] == 'income']['amount'].sum()
exp = df_c[df_c['type'] == 'expense']['amount'].sum()
print(f'  Total income: {inc:,.2f}')
print(f'  Total expense: {exp:,.2f}')
print(f'  Net savings: {inc - exp:,.2f}')

total_assets = assets['value'].sum()
liquid_assets = assets[assets['is_liquid']]['value'].sum()
total_debt = liab['outstanding'].sum()
total_emi = liab['emi'].sum()
print(f'Assets: {len(assets)} rows, Total: {total_assets:,}, Liquid: {liquid_assets:,}')
print(f'Liabilities: {len(liab)} rows, Total debt: {total_debt:,}, Monthly EMI: {total_emi:,}')
hr = liab.loc[liab['interest_rate'].idxmax()]
print(f'  Highest rate: {hr["type"]} at {hr["interest_rate"]}%')
print(f'Net Worth: {total_assets - total_debt:,}')

# Key ratios
avg_monthly_income = inc / 24
avg_monthly_expense = exp / 24
savings_rate = ((inc - exp) / inc) * 100
dti = (total_emi / avg_monthly_income) * 100
runway_months = liquid_assets / avg_monthly_expense
print(f'\n=== KEY RATIOS ===')
print(f'Avg Monthly Income: {avg_monthly_income:,.2f}')
print(f'Avg Monthly Expense: {avg_monthly_expense:,.2f}')
print(f'Savings Rate: {savings_rate:.2f}%')
print(f'DTI Ratio: {dti:.2f}%')
print(f'Emergency Runway: {runway_months:.2f} months')
