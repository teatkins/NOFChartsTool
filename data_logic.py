"""
Core data loading and matching logic for the multi-period NOF Charts Tool.

Row-type classification is done by Metric_ID prefix (not the Units column,
which has known data-quality issues in the 26/27 data -- see project notes):

    OF0xxx  -> value (the metric as measured)
    OF1xxx  -> score (25/26 scheme)
    OF2xxx  -> score (26/27 scheme)
    OF4[02]xx -> domain score
    OF4[13]xx -> domain segment
    OF5xxx  -> trust-level summary (Average metric score, Segment, etc.)

A value row and its score row share the same LAST THREE DIGITS of their
Metric_ID (e.g. OF0011 <-> OF2011). This suffix is also used as the stable
identity for cross-period (retained/new/dropped) tracking, since it stays
constant even though the leading digits' offset changes between years
(OF1035 in 25/26 -> OF2035 in 26/27, both suffix '035').

Kept separate from app_v2.py (the Streamlit UI) so it can be unit-tested
without needing a Streamlit runtime.
"""
import glob
import re
import pandas as pd

SECTORS = ['acute', 'mhcom', 'ambulance']

# Known Units labelling errors in the 26/27 data. Kept for the displayed
# unit only; underlying Value figures are correct.
UNITS_OVERRIDE = {
    'OF0011': '%', 'OF0016': '%', 'OF0017': 'mins', 'OF0020': 'count',
    'OF0023': '%', 'OF0047': 'out of 10', 'OF0061': 'out of 10',
    'OF0084': 'out of 10', 'OF0086': '%',
}

_ID_RE = re.compile(r'^OF(\d)(\d)(\d{2})$')


def classify_metric_id(mid):
    """Return 'value' | 'score' | 'domain_score' | 'domain_segment' | 'summary' | 'unknown'."""
    m = _ID_RE.match(mid)
    if not m:
        return 'unknown'
    d1, d2, _ = m.groups()
    if d1 == '0':
        return 'value'
    if d1 in ('1', '2'):
        return 'score'
    if d1 == '4' and d2 in ('0', '2'):
        return 'domain_score'
    if d1 == '4' and d2 in ('1', '3'):
        return 'domain_segment'
    if d1 == '5':
        return 'summary'
    return 'unknown'


def metric_suffix(mid):
    return mid[-3:]


# Known combined-score groups. Each has a fixed combined score ID (varies
# by year, same suffix) whose score substitutes for its components' scores
# in any average. Components are still shown as their own rows
# (informational) and nested under the combined row in the UI, but
# excluded from averaging individually.
COMBINED_GROUPS = [
    {
        'name': 'Combined finance',
        'combined_score_ids': {'OF1080', 'OF2080'},
        'component_value_ids': {'OF0079', 'OF0081'},   # both have their own value+score too
        'component_score_only_ids': set(),
        'suffix_collision': False,
    },
    {
        'name': 'Combined CYP and adult CMH paired outcome completeness score',
        'combined_score_ids': {'OF2029'},               # 26/27 only
        'component_value_ids': {'OF0029', 'OF0030'},     # value-only, never individually scored
        'component_score_only_ids': set(),
        'suffix_collision': True,   # OF0029's suffix ('029') collides with the combined score's
                                     # own ID (OF2029) -- must NOT auto-pair them as value+score
    },
    {
        'name': 'Temporary staffing cost relative band',
        'combined_score_ids': {'OF2108'},                # 26/27 only
        'component_value_ids': set(),
        'component_score_only_ids': {'OF2207', 'OF2208'},  # orphan scores, no value at all
        'suffix_collision': False,
    },
]

_SUFFIX_COLLISION_VALUE_IDS = set().union(
    *(g['component_value_ids'] for g in COMBINED_GROUPS if g['suffix_collision'])
)


def quarter_sort_key(label):
    """Parse 'Qn YYYY/YY' into a sortable (financial_year_start, quarter_number) tuple."""
    m = re.match(r'Q(\d)\s+(\d{4})/(\d{2})', str(label).strip())
    if not m:
        raise ValueError(f"Unrecognised quarter label: {label!r}")
    return (int(m.group(2)), int(m.group(1)))


def load_sector_data(sector, data_dir='data'):
    pattern = f'{data_dir}/nof_data_{sector}_*.csv'
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No data files found matching {pattern}")
    frames = [pd.read_csv(f) for f in files]
    combined = pd.concat(frames, ignore_index=True)
    combined['_sector'] = sector
    return combined


def load_all_sector_data(data_dir='data'):
    frames = [load_sector_data(s, data_dir) for s in SECTORS]
    df = pd.concat(frames, ignore_index=True)
    df['_row_type'] = df['Metric_ID'].map(classify_metric_id)
    df['_suffix'] = df['Metric_ID'].map(metric_suffix)
    return df


def get_available_quarters(df):
    return sorted(df['Quarter'].unique(), key=quarter_sort_key)


def previous_quarter(selected, available_quarters):
    idx = available_quarters.index(selected)
    return available_quarters[idx - 1] if idx > 0 else None


def apply_units_override(metric_id, raw_units):
    return UNITS_OVERRIDE.get(metric_id, raw_units)


def latest_description_lookup(sector_df):
    """Newest wording for every Metric_ID (value or score) across all periods
    in this sector's data, indexed by Metric_ID."""
    ordered = sector_df.sort_values('Quarter', key=lambda s: s.map(quarter_sort_key))
    return ordered.drop_duplicates('Metric_ID', keep='last').set_index('Metric_ID')[
        ['Metric_description', 'Units']
    ]


# ─────────────────────────────────────────────────────────────────────────
# Indicator-row assembly for one trust + one period
# ─────────────────────────────────────────────────────────────────────────

def build_period_indicators(trust_df, this_q):
    """
    For a trust and a single quarter, build "indicator entries" per the
    score-anchored model: every value row and every score row, paired by
    suffix where possible, with the three known combined groups handled
    specially (their combined score is its own entry; components are
    separate entries flagged with their parent group).

    Returns a dict keyed by a stable "entry key":
      - normal value/score pair, value-only, or score-only: ('metric', suffix)
      - a combined group's own entry: ('combined', group_name)
    """
    q_df = trust_df[trust_df['Quarter'] == this_q]
    value_rows = q_df[q_df['_row_type'] == 'value']
    score_rows = q_df[q_df['_row_type'] == 'score']

    entries = {}
    claimed_score_ids = set()

    # 1. Combined-group entries first (claim their suffixes/IDs)
    for group in COMBINED_GROUPS:
        cid_row = score_rows[score_rows['Metric_ID'].isin(group['combined_score_ids'])]
        present_components_value = group['component_value_ids'] & set(value_rows['Metric_ID'])
        present_components_score = group['component_score_only_ids'] & set(score_rows['Metric_ID'])
        if cid_row.empty and not present_components_value and not present_components_score:
            continue  # group not applicable this period for this trust

        if not cid_row.empty:
            r = cid_row.iloc[0]
            entries[('combined', group['name'])] = {
                'metric_id': r['Metric_ID'], 'suffix': metric_suffix(r['Metric_ID']),
                'description': group['name'], 'units': 'score',
                'domain': r['Domain'], 'sub_domain': r['Sub-domain'],
                'value': None, 'score': r['Value'],
                'rank': None, 'median': None, 'lower_q': None, 'upper_q': None,
                'group': None, 'is_group_total': True, 'counts_in_average': True,
            }
            claimed_score_ids.add(r['Metric_ID'])

        for sid in present_components_score:
            claimed_score_ids.add(sid)
            sr = score_rows[score_rows['Metric_ID'] == sid].iloc[0]
            entries[('metric', metric_suffix(sid))] = {
                'metric_id': sid, 'suffix': metric_suffix(sid),
                'description': sr['Metric_description'], 'units': sr['Units'],
                'domain': sr['Domain'], 'sub_domain': sr['Sub-domain'],
                'value': None, 'score': sr['Value'],
                'rank': sr['Rank'], 'median': None, 'lower_q': None, 'upper_q': None,
                'group': group['name'], 'is_group_total': False,
                'counts_in_average': False,
            }

    # 2. Remaining value rows, paired by suffix with remaining score rows
    remaining_scores = score_rows[~score_rows['Metric_ID'].isin(claimed_score_ids)]
    score_by_suffix = {}
    for _, r in remaining_scores.iterrows():
        score_by_suffix.setdefault(metric_suffix(r['Metric_ID']), r)

    for _, vr in value_rows.iterrows():
        vid = vr['Metric_ID']
        suffix = metric_suffix(vid)
        group_name = None
        for group in COMBINED_GROUPS:
            if vid in group['component_value_ids']:
                group_name = group['name']
                break

        score_row = None
        if not (group_name and vid in _SUFFIX_COLLISION_VALUE_IDS):
            score_row = score_by_suffix.get(suffix)

        entries[('metric', suffix)] = {
            'metric_id': vid, 'suffix': suffix,
            'description': vr['Metric_description'], 'units': vr['Units'],
            'domain': vr['Domain'], 'sub_domain': vr['Sub-domain'],
            'value': vr['Value'],
            'score': score_row['Value'] if score_row is not None else None,
            'rank': score_row['Rank'] if score_row is not None else None,
            'median': vr['Median_value'], 'lower_q': vr['Lower_quartile'], 'upper_q': vr['Upper_quartile'],
            'group': group_name, 'is_group_total': False,
            'counts_in_average': (score_row is not None) and (group_name is None),
        }
        if score_row is not None:
            claimed_score_ids.add(score_row['Metric_ID'])

    # 3. Any score rows still unclaimed = pure "orphan" scores (no value at all)
    for _, sr in score_rows.iterrows():
        sid = sr['Metric_ID']
        if sid in claimed_score_ids:
            continue
        suffix = metric_suffix(sid)
        group_name = None
        for group in COMBINED_GROUPS:
            if sid in group['component_score_only_ids']:
                group_name = group['name']
                break
        entries[('metric', suffix)] = {
            'metric_id': sid, 'suffix': suffix,
            'description': sr['Metric_description'], 'units': sr['Units'],
            'domain': sr['Domain'], 'sub_domain': sr['Sub-domain'],
            'value': None, 'score': sr['Value'],
            'rank': sr['Rank'], 'median': None, 'lower_q': None, 'upper_q': None,
            'group': group_name, 'is_group_total': False,
            'counts_in_average': group_name is None,
        }

    for k, v in entries.items():
        v['key'] = k
    return entries


def entry_key_set(entries):
    """Entry keys that COUNT IN THE AVERAGE (contribute a score) -- used for
    computing an average and for comparing domain composition."""
    return {k for k, v in entries.items() if v['counts_in_average']}


# ─────────────────────────────────────────────────────────────────────────
# Domain grouping, domain scores/segments, composition-discontinuity check
# ─────────────────────────────────────────────────────────────────────────

def domain_totals(trust_df, this_q):
    """Domain score + segment rows for this trust/period, indexed by Domain."""
    q_df = trust_df[trust_df['Quarter'] == this_q]
    scores = q_df[q_df['_row_type'] == 'domain_score'].set_index('Domain')
    segments = q_df[q_df['_row_type'] == 'domain_segment'].set_index('Domain')
    return scores, segments


def summary_totals(trust_df, this_q):
    """Trust-level summary rows (Average metric score, Segment, etc.)."""
    q_df = trust_df[trust_df['Quarter'] == this_q]
    return q_df[q_df['_row_type'] == 'summary']


def domain_composition_key(entries, domain):
    """Entry keys counting in the average, restricted to one domain."""
    return {k for k, v in entries.items() if v['counts_in_average'] and v['domain'] == domain}


def domain_delta_allowed(entries_this, entries_prev, domain):
    """
    True only if the domain's included indicator composition (by entry
    key) is IDENTICAL between the two periods -- otherwise a
    quarter-on-quarter domain score comparison would be comparing
    different things and must be suppressed.
    """
    this_comp = domain_composition_key(entries_this, domain)
    prev_comp = domain_composition_key(entries_prev, domain)
    if not this_comp or not prev_comp:
        return False
    return this_comp == prev_comp


# ─────────────────────────────────────────────────────────────────────────
# Cross-period retained/new/dropped + chart eligibility (suffix-based)
# ─────────────────────────────────────────────────────────────────────────

def retained_new_dropped(entries_this, entries_prev):
    keys_this = set(entries_this.keys())
    keys_prev = set(entries_prev.keys())
    if not keys_prev:
        return set(), keys_this, set()
    return keys_this & keys_prev, keys_this - keys_prev, keys_prev - keys_this


def entries_by_quarter(trust_df, available_quarters):
    """Precompute build_period_indicators() once per quarter for this trust."""
    return {
        q: build_period_indicators(trust_df, q)
        for q in available_quarters if not trust_df[trust_df['Quarter'] == q].empty
    }


def chart_eligible_keys(entries_by_q, available_quarters):
    """
    Entry keys with an actual VALUE (not just a score) in more than one
    quarter overall -- a chart with no Value line at all isn't useful, since
    the trend chart plots Value against the sector median/quartile band.
    """
    counts = {}
    for q in available_quarters:
        for k, v in entries_by_q.get(q, {}).items():
            if v['value'] is not None and not pd.isna(v['value']):
                counts[k] = counts.get(k, 0) + 1
    return {k for k, c in counts.items() if c > 1}


def chart_status(entries_by_q, available_quarters, entry_key):
    """
    'eligible'  -- has a value in more than one quarter, chartable
    'new'       -- has appeared in only one quarter total (any data)
    'no_value'  -- has appeared in multiple quarters, but never has a value
                   (a pure score-only indicator or combined-group entry)
    """
    quarters_with_any = [q for q in available_quarters if entry_key in entries_by_q.get(q, {})]
    quarters_with_value = [
        q for q in quarters_with_any
        if entries_by_q[q][entry_key]['value'] is not None and not pd.isna(entries_by_q[q][entry_key]['value'])
    ]
    if len(quarters_with_value) > 1:
        return 'eligible'
    if len(quarters_with_any) <= 1:
        return 'new'
    return 'no_value'


# ─────────────────────────────────────────────────────────────────────────
# Score colour banding: range 1-4 split into four equal 0.75-wide bands
# ─────────────────────────────────────────────────────────────────────────

SCORE_BAND_COLOURS = ['#2e7d32', '#9e9d24', '#ef6c00', '#c62828']  # best -> worst


def build_domain_items(entries, domain):
    """
    Assemble the ordered display items for one domain: standalone entries
    and combined-group entries (with their nested components), sorted by
    (sub_domain, description) alphabetically -- matching the reference
    spreadsheet's ordering. A component whose combined-group total isn't
    present this period (partial group) is shown standalone instead of
    being silently dropped.
    """
    domain_entries = {k: v for k, v in entries.items() if v['domain'] == domain}

    group_totals = {v['description']: (k, v) for k, v in domain_entries.items() if v['is_group_total']}
    components_by_group = {}
    standalone = []

    for k, v in domain_entries.items():
        if v['is_group_total']:
            continue
        if v['group'] and v['group'] in group_totals:
            components_by_group.setdefault(v['group'], []).append(v)
        else:
            standalone.append(v)

    items = []
    for v in standalone:
        items.append(('entry', v))
    for group_name, (k, group_entry) in group_totals.items():
        components = sorted(components_by_group.get(group_name, []), key=lambda e: e['description'])
        items.append(('group', group_entry, components))

    def sort_key(item):
        entry = item[1]
        return (entry.get('sub_domain') or '', entry['description'])

    items.sort(key=sort_key)
    return items


def sector_value_range(sector_df, metric_id, this_q):
    """Min/max of this metric's raw Value across every trust in the sector
    this quarter -- used for the compact 'sector distribution' bar."""
    d = sector_df[(sector_df['Metric_ID'] == metric_id) & (sector_df['Quarter'] == this_q)]
    vals = d['Value'].dropna()
    if vals.empty:
        return None, None
    return vals.min(), vals.max()


def score_band_colour(score):
    if score is None or pd.isna(score):
        return '#aaaaaa'
    if score < 1.75:
        return SCORE_BAND_COLOURS[0]
    if score < 2.5:
        return SCORE_BAND_COLOURS[1]
    if score < 3.25:
        return SCORE_BAND_COLOURS[2]
    return SCORE_BAND_COLOURS[3]


# ─────────────────────────────────────────────────────────────────────────
# Chart-page series builder (suffix/entry-key based)
# ─────────────────────────────────────────────────────────────────────────

def data_quarter_span_by_key(trust_df, available_quarters, entry_key):
    """
    Default quarter range for the chart page: spans the quarters where this
    entry has an actual VALUE (not just a score) -- so a trailing quarter
    where only a score has been published (e.g. NHS hasn't released the raw
    value for e-coli/c-diff yet this quarter) doesn't extend the default
    view into a quarter with no plottable Value point, which reads as a
    confusing gap at the edge of the chart. The range slider itself still
    offers every available quarter, so a person can widen it manually to
    see that a score-only quarter exists.
    """
    found = []
    for q in available_quarters:
        if trust_df[trust_df['Quarter'] == q].empty:
            continue
        entries = build_period_indicators(trust_df, q)
        e = entries.get(entry_key)
        if e is not None and e['value'] is not None and not pd.isna(e['value']):
            found.append(q)
    if not found:
        return None, None
    return found[0], found[-1]


def build_chart_series_by_key(trust_df, entry_key, quarters_range):
    """Gap-aware series (Value + Median/Lower/Upper quartile) for one entry
    key over a chosen quarter range."""
    records = []
    display_desc = None
    display_units = None
    for q in quarters_range:
        entries = build_period_indicators(trust_df, q) if not trust_df[trust_df['Quarter'] == q].empty else {}
        e = entries.get(entry_key)
        if e is None:
            records.append({'Quarter': q, 'Value': None, 'Median_value': None,
                             'Lower_quartile': None, 'Upper_quartile': None})
            continue
        records.append({
            'Quarter': q, 'Value': e['value'],
            'Median_value': e['median'], 'Lower_quartile': e['lower_q'], 'Upper_quartile': e['upper_q'],
        })
        if display_desc is None:
            display_desc = e['description']
            display_units = apply_units_override(e['metric_id'], e['units'])

    grouped = pd.DataFrame(records).set_index('Quarter').astype(float)
    if display_desc is None:
        display_desc = entry_key[1]
        display_units = ''
    return grouped, display_desc, display_units
