import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from data_logic import (
    load_all_sector_data, get_available_quarters, previous_quarter,
    latest_description_lookup, entries_by_quarter, build_domain_items,
    domain_totals, summary_totals, domain_delta_allowed,
    chart_eligible_keys, data_quarter_span_by_key,
    build_chart_series_by_key,
)
from render import (
    render_header, render_row_html, row_chart_status,
    render_domain_total_banner, render_summary_banner,
)

ROW_COLUMNS = [0.035, 0.895, 0.07]  # arrow (group toggle) | row content | chart button

st.set_page_config(layout='wide', page_title='NOF Overview and Trend Visualisations')

# st.dialog's own `width` param only offers fixed presets (small/medium/
# large, up to 1280px), not a percentage of the viewport -- this is a
# documented, community-verified CSS override for that container. Plain CSS
# in a <style> block, not a click handler or javascript: URI, so it isn't
# subject to the sanitization that ruled out the onclick approach earlier.
st.markdown(
    """
    <style>
    div[data-testid="stDialog"] div[role="dialog"] {
        width: 75vw !important;
        max-width: 75vw !important;
        height: 75vh !important;
        max-height: 75vh !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# The indicator table is built from many separate row-level elements (each
# row pairs an HTML block with a real st.button side by side, so clicks work
# reliably -- see project notes), which makes a single synchronised
# horizontal scroll across the whole table impractical. A phone in portrait
# is simply too narrow for the row's six fixed-width data columns plus a
# readable indicator name, so rather than fight that, prompt for landscape
# instead. Pure CSS (a fixed full-viewport overlay + a media query keyed on
# orientation and a phone-width cutoff) -- no JS, so it degrades safely if
# it doesn't apply: worst case is just the normal (squeezed) table, not a
# blank page. The 600px cutoff targets phone portrait widths specifically
# (most run 360-430px) without catching tablets in portrait, which are
# usually wide enough already.
st.markdown(
    """
    <style>
    .nof-rotate-overlay { display: none; }
    @media (orientation: portrait) and (max-width: 600px) {
        .nof-rotate-overlay {
            display: flex;
            position: fixed;
            inset: 0;
            z-index: 9999;
            background: white;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
            padding: 2rem;
            font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif;
        }
    }
    </style>
    <div class="nof-rotate-overlay">
        <div style="font-size:48px;">📱↻</div>
        <div style="font-size:18px; font-weight:600; margin-top:1rem;">Please rotate your device</div>
        <div style="color:#666; margin-top:0.5rem; max-width:280px;">
            This app's table needs a bit more width than portrait mode allows —
            turn your phone sideways to view it.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    return load_all_sector_data()


df = load_data()
available_quarters = get_available_quarters(df)

# ── Session state ─────────────────────────────────────────────────────────

if 'dialog_open' not in st.session_state:
    st.session_state.dialog_open = False
if 'dialog_entry_key' not in st.session_state:
    st.session_state.dialog_entry_key = None
if 'current_period' not in st.session_state:
    st.session_state.current_period = available_quarters[-1]
if 'current_trust' not in st.session_state:
    st.session_state.current_trust = None

st.title('NOF Overview and Trend Visualisations')

# ── Period + trust selectors ─────────────────────────────────────────────

st.markdown('**Select a period**')
selected_period = st.select_slider(
    'Select a period', options=available_quarters,
    value=st.session_state.current_period, label_visibility='collapsed', key='period_selector',
)
st.session_state.current_period = selected_period
prev_period = previous_quarter(selected_period, available_quarters)

trust_lookup = (
    df[['Trust_name', 'Trust_code', '_sector']]
    .drop_duplicates(subset=['Trust_name'])
    .set_index('Trust_name')
)
trust_options = sorted(trust_lookup.index)
if st.session_state.current_trust not in trust_options:
    st.session_state.current_trust = trust_options[0]
trust_index = trust_options.index(st.session_state.current_trust)

st.markdown('**Select a trust**')
trust_name = st.selectbox('Select a trust', trust_options, index=trust_index,
                           label_visibility='collapsed', key='trust_selector')
st.session_state.current_trust = trust_name
trust_code = trust_lookup.loc[trust_name, 'Trust_code']
sector = trust_lookup.loc[trust_name, '_sector']
sector_df = df[df['_sector'] == sector]
trust_df = sector_df[sector_df['Trust_code'] == trust_code]

latest_desc = latest_description_lookup(sector_df)

# ── Chart-eligible entries (global, across ALL periods for this trust) ────

entries_by_q = entries_by_quarter(trust_df, available_quarters)
eligible_keys = chart_eligible_keys(entries_by_q, available_quarters)


def display_desc_for_key(key, entries_by_q):
    """Newest description seen for this entry key across all periods."""
    for q in reversed(available_quarters):
        e = entries_by_q.get(q)
        if e and key in e:
            return e[key]['description']
    return key[1]


eligible_options = sorted(
    ((display_desc_for_key(k, entries_by_q), k) for k in eligible_keys),
    key=lambda t: t[0]
)
desc_to_key = {desc: k for desc, k in eligible_options}
eligible_descs = [desc for desc, _ in eligible_options]

# ── Chart dialog ──────────────────────────────────────────────────────────
# A dialog only stays open across reruns if the code path that opens it is
# re-entered every time (see st.dialog's own docs/example) -- hence the
# persistent dialog_open flag, checked unconditionally below, rather than
# only inside the button's own if-block. on_dismiss resets that flag so an
# X/outside-click/Esc close doesn't leave it stuck open for the next rerun.

def _close_dialog():
    st.session_state.dialog_open = False


def _render_chart_dialog():
    entry_key = st.session_state.dialog_entry_key
    st.caption(trust_name)

    first_q, last_q = data_quarter_span_by_key(trust_df, available_quarters, entry_key)
    range_start, range_end = st.select_slider(
        'Select quarter range', options=available_quarters, value=(first_q, last_q),
        key=f'dialog_range_{entry_key}',
    )
    start_idx, end_idx = available_quarters.index(range_start), available_quarters.index(range_end)
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx
    quarters_range = available_quarters[start_idx:end_idx + 1]

    grouped, display_desc, display_units = build_chart_series_by_key(trust_df, entry_key, quarters_range)

    if grouped['Value'].notna().sum() == 0:
        st.warning('No data available for this indicator and trust in the selected range.')
    else:
        fig, ax = plt.subplots(figsize=(13, 6.5))
        if grouped['Median_value'].notna().any():
            ax.plot(grouped.index, grouped['Median_value'], marker='o', color='grey',
                    linewidth=1, linestyle='--', alpha=0.6, label='Median')
        if grouped['Lower_quartile'].notna().any():
            ax.plot(grouped.index, grouped['Lower_quartile'], marker='o', color='green',
                    linewidth=1, linestyle='--', alpha=0.6, label='Lower Quartile')
        if grouped['Upper_quartile'].notna().any():
            ax.plot(grouped.index, grouped['Upper_quartile'], marker='o', color='orange',
                    linewidth=1, linestyle='--', alpha=0.6, label='Upper Quartile')
        ax.plot(grouped.index, grouped['Value'], marker='o', color='blue', linewidth=2.5, label='Trust Value')
        ax.set_title(display_desc, wrap=True)
        ax.set_ylabel(display_units)
        ax.set_xlabel('Quarter')
        plt.xticks(rotation=45)
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


if st.session_state.dialog_open:
    if st.session_state.dialog_entry_key in eligible_keys:
        dialog_title = display_desc_for_key(st.session_state.dialog_entry_key, entries_by_q)
        st.dialog(dialog_title, width='large', on_dismiss=_close_dialog)(_render_chart_dialog)()
    else:
        # Stale reference (e.g. trust changed underneath) -- don't reopen.
        st.session_state.dialog_open = False

# ══════════════════════════════════════════════════════════════════════════
# DOMAIN-GROUPED INDICATOR TABLE
# ══════════════════════════════════════════════════════════════════════════

trust_has_period_data = not trust_df[trust_df['Quarter'] == selected_period].empty
if not trust_has_period_data:
    st.info(f'{trust_name} has no data for {selected_period}. Try a different period or trust.')
    st.stop()

entries_this = entries_by_q[selected_period]
entries_prev = entries_by_q.get(prev_period) if prev_period else None

if not entries_this:
    st.warning(f'No indicator data available for this trust in {selected_period}.')
    st.stop()

st.caption('Use the 📈 button beside an indicator to see its trend chart.')

# ── Overall summary banner ───────────────────────────────────────────────

summary_rows = summary_totals(trust_df, selected_period)
avg_score_row = summary_rows[summary_rows['Metric_description'] == 'Average metric score']
avg_score = avg_score_row['Value'].iloc[0] if not avg_score_row.empty else None
segment_row = summary_rows[summary_rows['Metric_description'].str.contains('Segment', case=False, na=False)]
segment_val = segment_row['Value'].iloc[0] if not segment_row.empty else None

st.markdown(render_summary_banner(avg_score, segment_val), unsafe_allow_html=True)

# ── One expander per domain ──────────────────────────────────────────────

domain_scores, domain_segments = domain_totals(trust_df, selected_period)
domains = sorted({v['domain'] for v in entries_this.values() if v['domain'] != 'Summary'})

if prev_period:
    prev_domain_scores, _ = domain_totals(trust_df, prev_period)
else:
    prev_domain_scores = None

for domain in domains:
    score = domain_scores.loc[domain, 'Value'] if domain in domain_scores.index else None
    segment = domain_segments.loc[domain, 'Value'] if domain in domain_segments.index else None

    delta_suppressed = True
    delta_html = None
    if entries_prev and domain_delta_allowed(entries_this, entries_prev, domain):
        if prev_domain_scores is not None and domain in prev_domain_scores.index:
            prev_score = prev_domain_scores.loc[domain, 'Value']
            if score is not None and not pd.isna(score) and not pd.isna(prev_score):
                d = score - prev_score
                arrow = '▲' if d > 0 else ('▼' if d < 0 else '=')
                delta_html = f'<span style="font-size:12px;color:#555;">{arrow} {abs(d):.2f}</span>'
                delta_suppressed = False

    label = f"{domain}"
    if score is not None and not pd.isna(score):
        label += f"  —  score {score:.2f}"
    if segment is not None and not pd.isna(segment):
        label += f"  ·  segment {segment:.0f}"

    with st.expander(label, expanded=True):
        st.markdown(
            render_domain_total_banner(domain, score, segment, delta_html, delta_suppressed),
            unsafe_allow_html=True,
        )
        items = build_domain_items(entries_this, domain)

        col_arrow, col_main, col_btn = st.columns(ROW_COLUMNS)
        with col_main:
            st.markdown(render_header(), unsafe_allow_html=True)

        def render_one_row(entry, row_class='', arrow=None, arrow_key=None, open_key=None):
            prev = (entries_prev or {}).get(entry['key'])
            status = row_chart_status(entry, entries_by_q, available_quarters)
            row_html = render_row_html(entry, prev, sector_df, selected_period, status, row_class=row_class)
            col_a, col_main, col_btn = st.columns(ROW_COLUMNS)
            with col_a:
                if arrow is not None:
                    if st.button(arrow, key=arrow_key):
                        st.session_state[open_key] = not st.session_state.get(open_key, False)
                        st.rerun()
            with col_main:
                st.markdown(row_html, unsafe_allow_html=True)
            with col_btn:
                if status == 'eligible':
                    btn_key = f"chartbtn_{trust_code}_{selected_period}_{entry['key'][0]}_{entry['key'][1]}"
                    if st.button('📈', key=btn_key, help='View trend chart'):
                        st.session_state.dialog_entry_key = entry['key']
                        st.session_state.dialog_open = True
                        st.rerun()
                # else: render nothing at all for non-chartable rows

        for item in items:
            if item[0] == 'entry':
                render_one_row(item[1])
            else:
                _, group_entry, components = item
                open_key = f"grpopen_{trust_code}_{domain}_{group_entry['description']}"
                is_open = st.session_state.get(open_key, False)
                render_one_row(
                    group_entry, row_class='nof-group-total',
                    arrow=('▾' if is_open else '▸'),
                    arrow_key=f"toggle_{trust_code}_{domain}_{group_entry['description']}",
                    open_key=open_key,
                )
                if is_open:
                    for c in components:
                        render_one_row(c, row_class='nof-group-child')
