"""
Run this script once from the project root to generate indicator_flags.csv.

For each indicator, it finds the trusts with the highest and lowest values,
then looks up their corresponding scores by matching on Metric_description
(where lower score = better).
If the trust with the highest value also has the highest score (i.e. worst),
then higher value = worse, so flip_scale is set to 1.
If the trust with the highest value has the lowest score (i.e. best),
then higher value = better, so flip_scale is set to 0.
"""

import pandas as pd

excluded_units = ['flag', 'score', 'segment']

sectors = {
    'acute':     'data/nof_data_acute.csv',
    'mhcom':     'data/nof_data_mhcom.csv',
    'ambulance': 'data/nof_data_ambulance.csv',
}

def infer_flip(df, metric_description):
    """
    Returns 1 if higher value = worse (scale should be flipped), else 0.
    Matches value and score rows by Metric_description.
    Uses the most recent quarter available for the indicator.
    Returns None if direction cannot be determined.
    """
    metric_rows = df[df['Metric_description'] == metric_description]

    # Use the most recent quarter available
    latest_quarter = metric_rows['Quarter'].max()
    latest = metric_rows[metric_rows['Quarter'] == latest_quarter]

    # Get value rows and score rows, matched by Metric_description
    value_rows = latest[~latest['Units'].isin(excluded_units)][['Trust_code', 'Value']].dropna()
    score_rows = latest[latest['Units'] == 'score'][['Trust_code', 'Value']].dropna()
    score_rows = score_rows.rename(columns={'Value': 'Score'})

    if value_rows.empty or score_rows.empty:
        return None

    merged = value_rows.merge(score_rows, on='Trust_code')
    if len(merged) < 2:
        return None

    # Trust with highest and lowest indicator value
    idx_max_val = merged['Value'].idxmax()
    idx_min_val = merged['Value'].idxmin()

    score_at_max_val = merged.loc[idx_max_val, 'Score']
    score_at_min_val = merged.loc[idx_min_val, 'Score']

    # Lower score = better, so if highest value has highest score → higher is worse → flip
    if score_at_max_val > score_at_min_val:
        return 1   # higher value = worse → flip so better is on right
    else:
        return 0   # higher value = better → no flip

rows = []
seen = set()

for sector, path in sectors.items():
    df = pd.read_csv(path)
    in_scope = df[~df['Units'].isin(excluded_units)]

    for _, row in in_scope[['Metric_ID', 'Metric_description', 'Units']].drop_duplicates().iterrows():
        key = row['Metric_ID']
        if key not in seen:
            seen.add(key)
            flip = infer_flip(df, row['Metric_description'])
            rows.append({
                'Metric_ID':          row['Metric_ID'],
                'Metric_description': row['Metric_description'],
                'Units':              row['Units'],
                'sector':             sector,
                'flip_scale':         flip if flip is not None else 0,
                'flip_inferred':      'yes' if flip is not None else 'no - check manually',
            })

flags_df = pd.DataFrame(rows).sort_values(['sector', 'Metric_description'])
flags_df.to_csv('data/indicator_flags.csv', index=False)

total    = len(flags_df)
inferred = (flags_df['flip_inferred'] == 'yes').sum()
manual   = (flags_df['flip_inferred'] == 'no - check manually').sum()
flipped  = (flags_df['flip_scale'] == 1).sum()

print(f"Done — {total} indicators written to data/indicator_flags.csv")
print(f"  {inferred} auto-inferred ({flipped} flagged as flip)")
print(f"  {manual} could not be inferred — review 'no - check manually' rows")
