"""
Regenerate indicator_flags.csv to cover the full multi-period, multi-sector
indicator set (25/26 + 26/27).

- Every existing (Metric_ID, sector) entry from the old file is preserved
  unchanged (flip_scale, flip_inferred) -- flip direction is a property of
  the indicator, and old entries must stay available for historical period
  views even if the indicator was later dropped.
- For any (Metric_ID, sector) combination present anywhere in the current
  data but NOT in the old file, infer flip_scale from whichever quarter
  (preferring the most recent) has both value and score rows for it.
"""
import pandas as pd
from test_loader import load_all_sector_data, quarter_sort_key

EXCLUDED_UNITS_FOR_INDICATOR = ['flag', 'score', 'segment']

df, _ = load_all_sector_data()
old_flags = pd.read_csv('data/indicator_flags_old.csv')
old_lookup = old_flags.set_index(['Metric_ID', 'sector'])

def infer_flip(value_rows, score_rows, mid, desc):
    vals = value_rows[value_rows['Metric_ID'] == mid][['Trust_code', 'Value']].dropna()
    scores = score_rows[score_rows['Metric_description'] == desc][['Trust_code', 'Value']].dropna()
    if len(vals) < 2 or scores.empty:
        return None
    merged = vals.merge(scores, on='Trust_code', suffixes=('_val', '_score'))
    if len(merged) < 2:
        return None
    hi_val = merged.loc[merged['Value_val'].idxmax(), 'Trust_code']
    lo_val = merged.loc[merged['Value_val'].idxmin(), 'Trust_code']
    hi_score = merged.loc[merged['Value_score'].idxmax(), 'Trust_code']
    lo_score = merged.loc[merged['Value_score'].idxmin(), 'Trust_code']
    if hi_val == hi_score and lo_val == lo_score:
        return 1, 'yes'
    elif hi_val == lo_score and lo_val == hi_score:
        return 0, 'yes'
    else:
        return 0, 'no - check manually'

new_rows = []
for sector in ['acute', 'mhcom', 'ambulance']:
    sector_df = df[df['_sector'] == sector]
    quarters = sorted(sector_df['Quarter'].unique(), key=quarter_sort_key, reverse=True)

    # All distinct value-row Metric_IDs ever seen for this sector, with their
    # most recent description/units.
    all_value_rows = sector_df[~sector_df['Units'].isin(EXCLUDED_UNITS_FOR_INDICATOR)]
    latest_desc = (
        all_value_rows.sort_values('Quarter', key=lambda s: s.map(quarter_sort_key))
        .drop_duplicates('Metric_ID', keep='last')
        .set_index('Metric_ID')[['Metric_description', 'Units']]
    )

    for mid in all_value_rows['Metric_ID'].unique():
        key = (mid, sector)
        if key in old_lookup.index:
            continue  # preserved as-is below
        desc = latest_desc.loc[mid, 'Metric_description']
        units = latest_desc.loc[mid, 'Units']

        result = None
        for q in quarters:
            qdf = sector_df[sector_df['Quarter'] == q]
            vrows = qdf[~qdf['Units'].isin(EXCLUDED_UNITS_FOR_INDICATOR)]
            srows = qdf[qdf['Units'] == 'score']
            result = infer_flip(vrows, srows, mid, desc)
            if result is not None:
                break
        if result is None:
            flip, inferred = 0, 'no - check manually'
        else:
            flip, inferred = result

        new_rows.append({
            'Metric_ID': mid, 'Metric_description': desc, 'Units': units,
            'sector': sector, 'flip_scale': flip, 'flip_inferred': inferred,
        })

carried_over = old_flags.to_dict('records')
result = pd.DataFrame(carried_over + new_rows).sort_values(['sector', 'Metric_ID']).reset_index(drop=True)
result.to_csv('data/indicator_flags.csv', index=False)

print(f"Total rows: {len(result)}  (carried over: {len(carried_over)}, newly added: {len(new_rows)})")
needs_review = result[result['flip_inferred'] == 'no - check manually']
print(f"Rows needing manual review: {len(needs_review)}")
print(needs_review[['Metric_ID','Metric_description','Units','sector','flip_scale']].to_string(index=False))

# Sanity: every (Metric_ID, sector) that's ever a value row anywhere should now have an entry
all_combos = set()
for sector in ['acute','mhcom','ambulance']:
    sd = df[df['_sector']==sector]
    vr = sd[~sd['Units'].isin(EXCLUDED_UNITS_FOR_INDICATOR)]
    for mid in vr['Metric_ID'].unique():
        all_combos.add((mid, sector))
covered = set(zip(result['Metric_ID'], result['sector']))
missing = all_combos - covered
print()
print("Coverage check -- missing combos:", missing if missing else "NONE (fully covered)")
