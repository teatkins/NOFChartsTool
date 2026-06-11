import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ── 1. Page config ────────────────────────────────────────────────────────────

st.set_page_config(layout='wide')
st.title('NOF Charts Tool')

# ── 2. Data loading ───────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    acute     = pd.read_csv('data/nof_data_acute.csv')
    mhcom     = pd.read_csv('data/nof_data_mhcom.csv')
    ambulance = pd.read_csv('data/nof_data_ambulance.csv')

    acute['_sector']     = 'acute'
    mhcom['_sector']     = 'mhcom'
    ambulance['_sector'] = 'ambulance'

    return pd.concat([acute, mhcom, ambulance], ignore_index=True)

df = load_data()

# ── 3. Trust selector ─────────────────────────────────────────────────────────

# Build lookup: Trust_name -> (Trust_code, _sector)
trust_lookup = (
    df[['Trust_name', 'Trust_code', '_sector']]
    .drop_duplicates(subset=['Trust_name'])
    .set_index('Trust_name')
)

trust_name = st.selectbox('Select a trust', sorted(trust_lookup.index))
trust_code = trust_lookup.loc[trust_name, 'Trust_code']
sector     = trust_lookup.loc[trust_name, '_sector']

# ── 4. Sector-scoped dataframe ────────────────────────────────────────────────
# All indicator filtering and benchmarks use only this trust's sector dataset

sector_df = df[df['_sector'] == sector]

# ── 5. Indicator selector ─────────────────────────────────────────────────────

excluded_units = ['flag', 'score', 'segment']

trust_df = sector_df[sector_df['Trust_code'] == trust_code]

indicators = sorted(
    trust_df[~trust_df['Units'].isin(excluded_units)]['Metric_description'].unique()
)

selected_indicator = st.selectbox('Select an indicator', indicators)

# ── 6. Chart generation ───────────────────────────────────────────────────────

quarters_order = ['Q1 2025/26', 'Q2 2025/26', 'Q3 2025/26', 'Q4 2025/26']

chart_df = sector_df[
    (sector_df['Trust_code'] == trust_code) &
    (sector_df['Metric_description'] == selected_indicator) &
    (~sector_df['Units'].isin(excluded_units))
]

if chart_df.empty:
    st.warning('No data available for this indicator and trust.')
else:
    grouped = (
        chart_df
        .groupby('Quarter')[['Value', 'Median_value', 'Lower_quartile', 'Upper_quartile']]
        .mean()
        .reindex([q for q in quarters_order if q in chart_df['Quarter'].unique()])
    )

    units = chart_df['Units'].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 5))

    # Benchmark lines (less prominent) — only plotted if not entirely null
    if grouped['Median_value'].notna().any():
        ax.plot(grouped.index, grouped['Median_value'],
                marker='o', color='grey', linewidth=1,
                linestyle='--', alpha=0.6, label='Median')

    if grouped['Lower_quartile'].notna().any():
        ax.plot(grouped.index, grouped['Lower_quartile'],
                marker='o', color='green', linewidth=1,
                linestyle='--', alpha=0.6, label='Lower Quartile')

    if grouped['Upper_quartile'].notna().any():
        ax.plot(grouped.index, grouped['Upper_quartile'],
                marker='o', color='orange', linewidth=1,
                linestyle='--', alpha=0.6, label='Upper Quartile')

    # Trust value line (prominent)
    ax.plot(grouped.index, grouped['Value'],
            marker='o', color='blue', linewidth=2.5, label='Trust Value')

    ax.set_title(selected_indicator, wrap=True)
    ax.set_ylabel(units)
    ax.set_xlabel('Quarter')
    plt.xticks(rotation=45)
    ax.legend()
    plt.tight_layout()

    st.pyplot(fig)
