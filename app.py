import streamlit as st
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── 1. Page config ────────────────────────────────────────────────────────────

st.set_page_config(layout='wide', page_title='NOF Overview and Trend Visualisations')

# ── 2. Constants ──────────────────────────────────────────────────────────────

EXCLUDED_UNITS = ['flag', 'score', 'segment']
LATEST_QUARTER = 'Q4 2025/26'
QUARTERS_ORDER = ['Q1 2025/26', 'Q2 2025/26', 'Q3 2025/26', 'Q4 2025/26']

# ── 3. Data loading ───────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    acute     = pd.read_csv('data/nof_data_acute.csv')
    mhcom     = pd.read_csv('data/nof_data_mhcom.csv')
    ambulance = pd.read_csv('data/nof_data_ambulance.csv')
    acute['_sector']     = 'acute'
    mhcom['_sector']     = 'mhcom'
    ambulance['_sector'] = 'ambulance'
    return pd.concat([acute, mhcom, ambulance], ignore_index=True)

@st.cache_data
def load_flags():
    return pd.read_csv('data/indicator_flags.csv').set_index('Metric_description')

df    = load_data()
flags = load_flags()

# ── 4. Session state ──────────────────────────────────────────────────────────

if 'page' not in st.session_state:
    st.session_state.page = 'table'
if 'selected_indicator' not in st.session_state:
    st.session_state.selected_indicator = None
if 'table_key' not in st.session_state:
    st.session_state.table_key = 0

# ── 5. Header & trust selector ────────────────────────────────────────────────

st.title('NOF Overview and Trend Visualisations')

if st.session_state.get('page') == 'chart':
    if st.button('← Back to indicator list'):
        st.session_state.page = 'table'
        st.session_state.selected_indicator = None
        st.session_state.table_key += 1
        st.rerun()

trust_lookup = (
    df[['Trust_name', 'Trust_code', '_sector']]
    .drop_duplicates(subset=['Trust_name'])
    .set_index('Trust_name')
)

trust_options = sorted(trust_lookup.index)

# Restore trust selection after back navigation
if 'current_trust' not in st.session_state:
    st.session_state.current_trust = trust_options[0]
trust_index = trust_options.index(st.session_state.current_trust) if st.session_state.current_trust in trust_options else 0

st.markdown('**Select a trust**')
trust_name = st.selectbox('Select a trust', trust_options, index=trust_index, label_visibility='collapsed', key='trust_selector')
st.session_state.current_trust = trust_name
trust_code = trust_lookup.loc[trust_name, 'Trust_code']
sector     = trust_lookup.loc[trust_name, '_sector']
sector_df  = df[df['_sector'] == sector]
trust_df   = sector_df[sector_df['Trust_code'] == trust_code]

# ── 6. Score colour ───────────────────────────────────────────────────────────

def score_colour(score):
    if pd.isna(score):
        return '#aaaaaa'
    if score <= 2:
        return '#2e7d32'
    if score <= 3:
        return '#e65100'
    return '#c62828'

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CHART
# ══════════════════════════════════════════════════════════════════════════════

# Build indicator list here so it's available on both pages
q4_trust_indicators = trust_df[
    (trust_df['Quarter'] == LATEST_QUARTER) &
    (~trust_df['Units'].isin(EXCLUDED_UNITS))
]
all_indicators = sorted(q4_trust_indicators['Metric_description'].unique().tolist())

if st.session_state.page == 'chart':
    st.markdown('**Select an indicator to view its trend chart**')
    chart_selected = st.selectbox(
        label='Select an indicator to view its trend chart',
        options=['— select —'] + all_indicators,
        index=(['— select —'] + all_indicators).index(st.session_state.selected_indicator)
              if st.session_state.selected_indicator in all_indicators else 0,
        label_visibility='collapsed',
        key='indicator_selector'
    )
    if chart_selected != '— select —' and chart_selected != st.session_state.selected_indicator:
        st.session_state.selected_indicator = chart_selected
        st.rerun()

    indicator = st.session_state.selected_indicator
    st.subheader(indicator)

    chart_df = sector_df[
        (sector_df['Trust_code'] == trust_code) &
        (sector_df['Metric_description'] == indicator) &
        (~sector_df['Units'].isin(EXCLUDED_UNITS))
    ]

    if chart_df.empty:
        st.warning('No data available for this indicator and trust.')
    else:
        grouped = (
            chart_df
            .groupby('Quarter')[['Value', 'Median_value', 'Lower_quartile', 'Upper_quartile']]
            .mean()
            .reindex([q for q in QUARTERS_ORDER if q in chart_df['Quarter'].unique()])
        )
        units = chart_df['Units'].iloc[0]
        fig, ax = plt.subplots(figsize=(10, 5))
        if grouped['Median_value'].notna().any():
            ax.plot(grouped.index, grouped['Median_value'], marker='o', color='grey',
                    linewidth=1, linestyle='--', alpha=0.6, label='Median')
        if grouped['Lower_quartile'].notna().any():
            ax.plot(grouped.index, grouped['Lower_quartile'], marker='o', color='green',
                    linewidth=1, linestyle='--', alpha=0.6, label='Lower Quartile')
        if grouped['Upper_quartile'].notna().any():
            ax.plot(grouped.index, grouped['Upper_quartile'], marker='o', color='orange',
                    linewidth=1, linestyle='--', alpha=0.6, label='Upper Quartile')
        ax.plot(grouped.index, grouped['Value'], marker='o', color='blue',
                linewidth=2.5, label='Trust Value')
        ax.set_title(indicator, wrap=True)
        ax.set_ylabel(units)
        ax.set_xlabel('Quarter')
        plt.xticks(rotation=45)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: INDICATOR TABLE
# ══════════════════════════════════════════════════════════════════════════════

# ── Build indicator rows ──────────────────────────────────────────────────────

q4_trust = trust_df[
    (trust_df['Quarter'] == LATEST_QUARTER) &
    (~trust_df['Units'].isin(EXCLUDED_UNITS))
]
q4_sector = sector_df[
    (sector_df['Quarter'] == LATEST_QUARTER) &
    (~sector_df['Units'].isin(EXCLUDED_UNITS))
]
q4_scores = sector_df[
    (sector_df['Quarter'] == LATEST_QUARTER) &
    (sector_df['Units'] == 'score')
]

indicators = sorted(q4_trust['Metric_description'].unique())

if not indicators:
    st.warning('No Q4 data available for this trust.')
    st.stop()

rows = []
for ind in indicators:
    trust_row = q4_trust[q4_trust['Metric_description'] == ind]
    if trust_row.empty:
        continue
    trust_val   = trust_row['Value'].iloc[0]
    q1          = trust_row['Lower_quartile'].iloc[0]
    median      = trust_row['Median_value'].iloc[0]
    q3          = trust_row['Upper_quartile'].iloc[0]
    units_str   = trust_row['Units'].iloc[0]

    sector_vals = q4_sector[q4_sector['Metric_description'] == ind]['Value'].dropna()
    if sector_vals.empty or pd.isna(trust_val):
        continue

    score_row = q4_scores[
        (q4_scores['Metric_description'] == ind) &
        (q4_scores['Trust_code'] == trust_code)
    ]
    score_val = score_row['Value'].iloc[0] if not score_row.empty else None
    flip      = bool(flags.loc[ind, 'flip_scale']) if ind in flags.index else False

    rows.append({
        'indicator': ind,
        'trust_val': trust_val,
        'units':     units_str,
        'q1': q1, 'median': median, 'q3': q3,
        'min_val': sector_vals.min(),
        'max_val': sector_vals.max(),
        'score_val': score_val,
        'flip': flip,
        'summary': False,
    })

# ── Build summary rows ────────────────────────────────────────────────────────

q4_trust_all = trust_df[trust_df['Quarter'] == LATEST_QUARTER]

# Average metric score — look up directly by Metric_description
avg_score_row = q4_trust_all[
    q4_trust_all["Metric_description"] == "Average metric score"
]
avg_score_val = avg_score_row["Value"].iloc[0] if not avg_score_row.empty else None

# Adjusted segment — look up directly by Metric_description
segment_row = q4_trust_all[
    q4_trust_all["Metric_description"] == "Adjusted segment"
]
segment_val = segment_row["Value"].iloc[0] if not segment_row.empty else None

summary_rows = [
    {
        'indicator': 'Average metric score',
        'trust_val': avg_score_val,
        'units':     'score',
        'q1': None, 'median': None, 'q3': None,
        'min_val': None, 'max_val': None,
        'score_val': None,
        'flip': False,
        'summary': True,
    },
    {
        'indicator': 'Adjusted segment',
        'trust_val': segment_val,
        'units':     'segment',
        'q1': None, 'median': None, 'q3': None,
        'min_val': None, 'max_val': None,
        'score_val': None,
        'flip': False,
        'summary': True,
    },
]

all_rows = rows + summary_rows

# ── Indicator drill-down selector ────────────────────────────────────────────

st.markdown("**Select an indicator to view its trend chart**")
selected = st.selectbox(
    label="Select an indicator to view its trend chart",
    options=["— select —"] + all_indicators,
    index=0,
    label_visibility="collapsed",
    key=f"indicator_selector_{st.session_state.table_key}"
)
if selected != "— select —":
    st.session_state.selected_indicator = selected
    st.session_state.page = "chart"
    st.rerun()

# ── Draw the single combined figure ──────────────────────────────────────────

ROW_H      = 0.45
HEADER_H   = 0.40
SUMMARY_H  = 0.45
NAME_W     = 0.42
VALUE_W    = 0.07
SCORE_W    = 0.07
SCALE_W    = 0.44
FIG_W      = 14

n_main    = len(rows)
n_summary = len(summary_rows)
fig_h     = HEADER_H + n_main * ROW_H + 0.15 + n_summary * SUMMARY_H

fig, ax = plt.subplots(figsize=(FIG_W, fig_h))
ax.set_xlim(0, 1)
ax.set_ylim(0, fig_h)
ax.axis('off')
fig.patch.set_facecolor('white')

x_name  = 0.01
x_value = NAME_W
x_score = NAME_W + VALUE_W
x_scale = NAME_W + VALUE_W + SCORE_W
sr_edge = 0.99   # right edge of scale

# ── Header ────────────────────────────────────────────────────────────────────

header_y = fig_h - HEADER_H * 0.55
ax.text(x_name,         header_y, 'Indicator',  fontsize=9, fontweight='bold', va='center')
ax.text(x_value + 0.01, header_y, 'Value',      fontsize=9, fontweight='bold', va='center')
ax.text(x_score + 0.015,header_y, 'Score',      fontsize=9, fontweight='bold', va='center')

# Scale header with worst/best labels
scale_mid = (x_scale + sr_edge) / 2
ax.text(scale_mid, header_y + 0.085,
        'Performance vs sector (Q4)',
        fontsize=9, fontweight='bold', va='center', ha='center')
ax.text(x_scale + 0.005, header_y - 0.065,
        '◀ Worst', fontsize=7, color='#888888', va='center', ha='left')
ax.text(sr_edge - 0.005, header_y - 0.065,
        'Best ▶', fontsize=7, color='#888888', va='center', ha='right')

# Legend
legend_x = scale_mid - 0.06
legend_y  = header_y - 0.065
ax.plot(legend_x, legend_y, marker='D', color='#d32f2f', markersize=5, zorder=5)
ax.text(legend_x + 0.012, legend_y, 'Trust', fontsize=7, color='#d32f2f', va='center')
iqr_swatch = mpatches.FancyBboxPatch(
    (legend_x + 0.065, legend_y - 0.03), 0.025, 0.06,
    boxstyle='square,pad=0', linewidth=0, facecolor='#90caf9'
)
ax.add_patch(iqr_swatch)
ax.text(legend_x + 0.095, legend_y, 'IQR', fontsize=7, color='#1565c0', va='center')
ax.plot([legend_x + 0.135, legend_x + 0.135], [legend_y - 0.04, legend_y + 0.04],
        color='#1565c0', linewidth=1.2)
ax.text(legend_x + 0.148, legend_y, 'Median', fontsize=7, color='#1565c0', va='center')

ax.axhline(fig_h - HEADER_H, color='#999999', linewidth=1.0)

# ── Helper: draw one scale bar ────────────────────────────────────────────────

def draw_scale_row(row_y, row, track_h=0.09):
    sl = x_scale + 0.005
    sr = sr_edge - 0.005
    sw = sr - sl
    mn, mx = row['min_val'], row['max_val']
    span = mx - mn if mx != mn else 1
    flip = row['flip']

    def to_x(v):
        frac = (v - mn) / span
        if flip:
            frac = 1 - frac
        return sl + frac * sw

    # Track
    ax.add_patch(mpatches.FancyBboxPatch(
        (sl, row_y - track_h / 2), sw, track_h,
        boxstyle='square,pad=0', linewidth=0, facecolor='#e0e0e0', zorder=1))

    # IQR
    if not (pd.isna(row['q1']) or pd.isna(row['q3'])):
        xq1, xq3 = to_x(row['q1']), to_x(row['q3'])
        ax.add_patch(mpatches.FancyBboxPatch(
            (min(xq1, xq3), row_y - track_h / 2), abs(xq3 - xq1), track_h,
            boxstyle='square,pad=0', linewidth=0, facecolor='#90caf9', zorder=2))

    # Median
    if not pd.isna(row['median']):
        xm = to_x(row['median'])
        ax.plot([xm, xm], [row_y - track_h * 1.6, row_y + track_h * 1.6],
                color='#1565c0', linewidth=1.2, zorder=3)

    # Trust marker
    if not pd.isna(row['trust_val']):
        xt = to_x(row['trust_val'])
        ax.plot([xt, xt], [row_y - track_h * 1.9, row_y + track_h * 1.9],
                color='#d32f2f', linewidth=1.5, zorder=4)
        ax.plot(xt, row_y, marker='D', color='#d32f2f', markersize=4, zorder=5)

# ── Main indicator rows ───────────────────────────────────────────────────────

for i, row in enumerate(rows):
    row_y = fig_h - HEADER_H - (i + 0.5) * ROW_H

    # Alternating background
    if i % 2 == 0:
        ax.add_patch(mpatches.FancyBboxPatch(
            (0, row_y - ROW_H / 2), 1, ROW_H,
            boxstyle='square,pad=0', linewidth=0, facecolor='#f7f7f7', zorder=0))

    # Indicator name
    ax.text(x_name, row_y, row['indicator'], fontsize=7.5, va='center', ha='left',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                      edgecolor='#dddddd', linewidth=0.6))

    # Value
    val_str = f"{row['trust_val']:.1f}" if not pd.isna(row['trust_val']) else '—'
    ax.text(x_value + 0.035, row_y, val_str, fontsize=8, va='center', ha='center')

    # Score badge
    sv = row['score_val']
    if sv is not None and not pd.isna(sv):
        colour = score_colour(sv)
        ax.add_patch(mpatches.FancyBboxPatch(
            (x_score + 0.002, row_y - 0.085), 0.062, 0.17,
            boxstyle='round,pad=0.01', linewidth=0, facecolor=colour, zorder=2))
        ax.text(x_score + 0.033, row_y, f'{sv:.1f}',
                fontsize=8, fontweight='bold', color='white',
                va='center', ha='center', zorder=3)

    # Scale
    if row['min_val'] != row['max_val']:
        draw_scale_row(row_y, row)

    ax.axhline(row_y - ROW_H / 2, color='#e8e8e8', linewidth=0.5)

# ── Summary section separator ─────────────────────────────────────────────────

sep_y = fig_h - HEADER_H - n_main * ROW_H
# Double rule to signal summary section
ax.axhline(sep_y,        color='#555555', linewidth=1.2)
ax.axhline(sep_y - 0.04, color='#555555', linewidth=0.5)

# ── Summary rows ──────────────────────────────────────────────────────────────

for j, row in enumerate(summary_rows):
    row_y = fig_h - HEADER_H - n_main * ROW_H - 0.15 - (j + 0.5) * SUMMARY_H

    # Summary background (slightly darker)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, row_y - SUMMARY_H / 2), 1, SUMMARY_H,
        boxstyle='square,pad=0', linewidth=0, facecolor='#ececec', zorder=0))

    # Indicator name — bold
    ax.text(x_name, row_y, row['indicator'], fontsize=8, fontweight='bold',
            va='center', ha='left', style='italic')

    # Value
    val = row['trust_val']
    if val is not None and not pd.isna(val):
        val_str = f"{val:.1f}"
    else:
        val_str = '—'
    ax.text(x_value + 0.035, row_y, val_str,
            fontsize=8, fontweight='bold', va='center', ha='center')

    ax.axhline(row_y - SUMMARY_H / 2, color='#cccccc', linewidth=0.5)

plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
st.pyplot(fig, width='stretch')
plt.close(fig)

