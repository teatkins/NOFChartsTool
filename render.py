"""
HTML/CSS rendering for the domain-grouped indicator table.

Each row's rich content (value/delta/rank/distribution/score/delta-score)
is still rendered as one HTML/CSS grid snippet -- but the "click to view
trend chart" affordance is a REAL Streamlit button placed alongside it
(see app_v2.py), not an HTML <a>/onclick. Two things about Streamlit rule
out HTML-based click handling here:
  - its markdown renderer forces target="_blank" on every rendered <a
    href> link, so a plain navigable href opens a new tab -- and a new
    tab is a brand-new session, losing all st.session_state (the trust
    selection, etc.)
  - its HTML sanitizer strips onclick attributes and javascript: URIs
    from unsafe_allow_html content, so a JS-driven same-window navigation
    doesn't fire either.
A native st.button has neither problem: no new tab, session state intact,
and it renders its own hover tooltip via `help=`.
"""
import html
import pandas as pd

from data_logic import score_band_colour, apply_units_override, sector_value_range, chart_status

STATUS_TITLES = {
    'eligible': 'Click to see trend chart',
    'new': 'New indicator - no trend chart available yet',
    'no_value': 'No trend chart for metrics without indicator values',
}

GRID_TEMPLATE = "minmax(0,3fr) 70px 70px 60px 110px 60px 60px"

STYLE = """
<style>
.nof-table { font-family: -apple-system, Segoe UI, Roboto, Arial, sans-serif; font-size: 13px; }
.nof-row {
  display: grid;
  grid-template-columns: %s;
  align-items: center;
  gap: 6px;
  padding: 5px 4px;
  border-bottom: 1px solid #eee;
}
.nof-header { font-weight: 600; color: #333; border-bottom: 2px solid #999; font-size: 12px; }
.nof-header .nof-col-num { text-align: center; }
.nof-indicator { line-height: 1.3; }
.nof-indicator .nof-sub { display:block; font-size: 11px; color: #888; }
.nof-excl { color: #999; font-size: 10px; font-style: italic; margin-left: 4px; }
.nof-ind-eligible { color: #333; }
.nof-ind-plain { color: #333; }
.nof-group-total .nof-indicator { font-weight: 600; }
.nof-group-child { background: #fafafa; }
.nof-group-child .nof-indicator { padding-left: 14px; border-left: 2px solid #e0e0e0; }
.nof-num { text-align: center; font-variant-numeric: tabular-nums; }
.nof-delta { text-align: center; font-variant-numeric: tabular-nums; color: #555; font-size: 12px; }
.nof-score-badge {
  display: inline-block; min-width: 34px; padding: 2px 6px; border-radius: 4px;
  color: white; font-weight: 600; text-align: center; font-size: 12px;
}
.nof-domain-total {
  background: #f0f4f8; font-weight: 600; border-radius: 6px; padding: 8px 6px; margin-bottom: 4px;
}
div[data-testid="stVerticalBlockBorderWrapper"] .nof-chart-btn button {
  padding: 0.15rem 0.4rem; min-height: 0; font-size: 12px;
}
</style>
""" % GRID_TEMPLATE


def _esc(s):
    return html.escape(str(s)) if s is not None else ''


def _fmt_value(v, units):
    """Raw Value for '%'-unit rows is always already in 0-100 percentage-point
    scale in this data (confirmed against the full dataset: genuine values run
    from -19.6 to 138.6, and small ones like 0.03 are real tiny percentages,
    not fractions) -- never divide by/multiply by 100 here."""
    if v is None or pd.isna(v):
        return '—'
    if units == '%':
        return f'{v:.2f}%' if abs(v) < 10 else f'{v:.1f}%'
    if isinstance(v, float) and v == int(v):
        return f'{int(v)}'
    return f'{v:.2f}' if abs(v) < 100 else f'{v:.1f}'


def _fmt_delta(curr, prev, units):
    if curr is None or prev is None or pd.isna(curr) or pd.isna(prev):
        return ''
    d = curr - prev
    if abs(d) < 1e-9:
        return '<span class="nof-delta">= 0</span>'
    arrow = '▲' if d > 0 else '▼'
    mag = abs(d)
    if units == '%':
        txt = f'{mag:.2f}pp' if mag < 10 else f'{mag:.1f}pp'
    else:
        txt = f'{mag:.2f}' if mag < 100 else f'{mag:.1f}'
    return f'<span class="nof-delta">{arrow} {txt}</span>'


def _score_badge(score):
    if score is None or pd.isna(score):
        return '—'
    colour = score_band_colour(score)
    return f'<span class="nof-score-badge" style="background:{colour}">{score:.2f}</span>'


def _distribution_svg(value, lower_q, upper_q, min_v, max_v):
    if any(x is None or pd.isna(x) for x in (value, min_v, max_v)) or max_v == min_v:
        return ''
    W, H = 100, 14
    def to_x(v):
        return max(0, min(W, (v - min_v) / (max_v - min_v) * W))
    parts = [f'<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}">']
    parts.append(f'<rect x="0" y="{H/2-2}" width="{W}" height="4" fill="#e0e0e0" rx="2"/>')
    if lower_q is not None and upper_q is not None and not pd.isna(lower_q) and not pd.isna(upper_q):
        xl, xu = to_x(lower_q), to_x(upper_q)
        parts.append(f'<rect x="{min(xl,xu):.1f}" y="{H/2-2}" width="{abs(xu-xl):.1f}" height="4" fill="#90caf9" rx="2"/>')
    xv = to_x(value)
    parts.append(f'<line x1="{xv:.1f}" y1="1" x2="{xv:.1f}" y2="{H-1}" stroke="#d32f2f" stroke-width="2"/>')
    parts.append('</svg>')
    return ''.join(parts)


def render_header():
    """The header's grid content only -- rendered inside the SAME column
    layout as every row (see app_v2.py) so the columns actually line up.
    A plain-width header caused misalignment before, since every data row
    sits inside a narrower st.columns() cell than the full-width header did."""
    return (
        STYLE +
        '<div class="nof-table"><div class="nof-row nof-header">'
        '<div>Indicator</div>'
        '<div class="nof-col-num">Value</div>'
        '<div class="nof-col-num">&Delta; Value</div>'
        '<div class="nof-col-num">Rank</div>'
        '<div class="nof-col-num">Distribution</div>'
        '<div class="nof-col-num">Score</div>'
        '<div class="nof-col-num">&Delta; Score</div>'
        '</div></div>'
    )


def row_chart_status(entry, entries_by_q, available_quarters):
    """
    'eligible'  -- has a value in >1 quarter overall AND this quarter's own
                   value is present, so a click-through chart makes sense
    'new'       -- appeared in only one quarter total (any data)
    'no_value'  -- has appeared in multiple quarters but never has a value
                   (pure score-only indicator/combined-group entry), OR is
                   otherwise eligible overall but THIS quarter's own value
                   is missing
    """
    status = chart_status(entries_by_q, available_quarters, entry['key']) if 'key' in entry else 'no_value'
    has_value_this_period = entry['value'] is not None and not pd.isna(entry['value'])
    if status == 'eligible' and not has_value_this_period:
        status = 'no_value'
    return status


def render_row_html(entry, prev_entry, sector_df, this_q, status, row_class=''):
    """
    The row's rich content (indicator name/sub-domain, value, delta, rank,
    distribution, score, delta score) as one HTML/CSS grid snippet -- no
    click affordance here; that's a separate real st.button rendered
    alongside this in app_v2.py.
    """
    units = apply_units_override(entry['metric_id'], entry['units'])
    value_str = _fmt_value(entry['value'], units)
    delta_val = _fmt_delta(entry['value'], prev_entry['value'] if prev_entry else None, units)
    rank_str = f"{int(entry['rank'])}" if entry['rank'] is not None and not pd.isna(entry['rank']) else '—'
    min_v, max_v = sector_value_range(sector_df, entry['metric_id'], this_q) if entry['value'] is not None else (None, None)
    dist_svg = _distribution_svg(entry['value'], entry['lower_q'], entry['upper_q'], min_v, max_v)
    score_badge = _score_badge(entry['score'])
    delta_score = _fmt_delta(entry['score'], prev_entry['score'] if prev_entry else None, 'score')

    excl_marker = '' if entry['counts_in_average'] else '<span class="nof-excl">not in average</span>'
    sub_line = f'<span class="nof-sub">{_esc(entry["sub_domain"])}</span>' if entry.get('sub_domain') else ''

    desc_esc = _esc(entry['description'])
    ind_class = 'nof-ind-eligible' if status == 'eligible' else 'nof-ind-plain'
    indicator_html = f'<span class="{ind_class}">{desc_esc}</span>'

    return (
        STYLE +
        f'<div class="nof-table"><div class="nof-row {row_class}">'
        f'<div class="nof-indicator">{indicator_html}{excl_marker}{sub_line}</div>'
        f'<div class="nof-num">{value_str}</div>'
        f'<div class="nof-num">{delta_val}</div>'
        f'<div class="nof-num">{rank_str}</div>'
        f'<div class="nof-num">{dist_svg}</div>'
        f'<div class="nof-num">{score_badge}</div>'
        f'<div class="nof-num">{delta_score}</div>'
        '</div></div>'
    )


def render_domain_total_banner(domain_name, score, segment, delta_score, delta_suppressed):
    delta_html = ''
    if delta_suppressed:
        delta_html = '<span style="color:#999; font-size:11px;">composition changed since last quarter</span>'
    elif delta_score is not None:
        delta_html = delta_score
    score_html = _score_badge(score) if score is not None else '—'
    segment_html = f'{segment:.0f}' if segment is not None and not pd.isna(segment) else '—'
    return (
        f'<div class="nof-domain-total">'
        f'Domain score: {score_html} &nbsp; {delta_html} &nbsp;&nbsp; Segment: {segment_html}'
        f'</div>'
    )


def render_summary_banner(avg_score, segment):
    score_html = _score_badge(avg_score) if avg_score is not None else '—'
    segment_html = f'{segment:.0f}' if segment is not None and not pd.isna(segment) else '—'
    return (
        STYLE +
        '<div class="nof-table">'
        f'<div class="nof-domain-total" style="font-size:15px;">'
        f'Average metric score: {score_html} &nbsp;&nbsp; Segment: {segment_html}'
        f'</div></div>'
    )
