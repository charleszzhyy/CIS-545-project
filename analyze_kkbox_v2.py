#!/usr/bin/env python3
"""Development-only EDA; preserve the source parquet and never train a model."""

import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kkbox_matplotlib"))
import duckdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data/processed/kkbox_v2_asof_20170228.parquet"
OUTPUT = ROOT / "reports/kkbox_v2_eda"
REPORT = ROOT / "docs/kkbox_v2_eda_report.md"
SEED = 42

# Decisions are based on feature meaning and development-only EDA, never test outcomes.
PREDICTORS = [
    "city", "age", "registered_via", "has_demographics",
    "membership_tenure_days", "has_listening_logs", "days_since_last_active",
    "active_days_7d", "active_days_30d", "listening_secs_30d",
    "listening_secs_change_30d", "song_completion_rate",
    "has_transactions", "transaction_count", "transaction_count_30d",
    "days_since_last_transaction", "last_amount_paid", "last_plan_days",
    "auto_renew_rate", "last_is_auto_renew", "cancel_count_90d", "last_is_cancel",
    "days_since_last_cancel", "days_until_expiration",
]
DECISIONS = {
    "msno": ("Remove", "Identifier and split/join key; never a predictor."),
    "is_churn": ("Remove", "Target, retained separately from predictors."),
    "city": ("Keep", "Location category; do not interpret numeric codes as ordered."),
    "age": ("Keep", "Interpretable demographic; retain NULL rather than invent an age."),
    "gender": ("Optional", "About 60% missing; observed male/female churn rates nearly identical in development."),
    "registered_via": ("Keep", "Acquisition channel category, not an ordered number."),
    "registration_date": ("Transform", "Replace raw date with membership_tenure_days."),
    "has_demographics": ("Keep", "Distinguishes usable demographics from absent/masked records."),
    "age_missing": ("Remove", "Exactly derivable from age NULL status."),
    "gender_missing": ("Remove", "Exactly derivable from gender NULL status."),
    "has_listening_logs": ("Keep", "Distinguishes missing history from observed zero activity."),
    "listening_record_count": ("Remove", "Row-count audit; use active dates for engagement."),
    "invalid_listening_rows": ("Remove", "Quality audit, not customer behavior; used to guard completion ratio."),
    "active_days": ("Optional", "Historical exposure; recent activity is easier to compare across users."),
    "total_songs": ("Transform", "Use as completion-rate denominator; omit raw historical volume."),
    "total_songs_100": ("Transform", "Use as completion-rate numerator; omit raw historical volume."),
    "total_unique_songs_daily": ("Remove", "Repeated daily counts, not distinct lifetime songs; another volume measure."),
    "total_listening_secs": ("Optional", "Historical volume/exposure; overlaps recent and average listening."),
    "avg_secs_per_active_day": ("Optional", "Implausible positive outliers contaminate historical intensity; omit until audited."),
    "last_active_date": ("Remove", "Exactly represented by days_since_last_active at fixed cutoff."),
    "days_since_last_active": ("Keep", "Interpretable inactivity/recency."),
    "active_days_7d": ("Keep", "Very recent activity; preserve alongside the longer window initially."),
    "active_days_30d": ("Keep", "Recent frequency distinct from listening intensity."),
    "listening_secs_7d": ("Optional", "Short-window volume; overlaps 30-day volume and 7-day frequency."),
    "listening_secs_30d": ("Keep", "Recent listening volume; skew/outliers require attention later."),
    "listening_secs_previous_30d": ("Remove", "Exactly recoverable from recent volume minus its change."),
    "songs_30d": ("Optional", "Alternative recent-volume measure to seconds."),
    "listening_secs_change_30d": ("Keep", "Absolute change between equal 30-day windows; negative is meaningful."),
    "listening_secs_ratio_30d": ("Remove", "Redundant with recent volume/change; undefined at zero prior activity."),
    "has_transactions": ("Keep", "Distinguishes no records from observed no recent transactions."),
    "transaction_count": ("Keep", "Observed history depth; counts events, not successful purchases."),
    "transaction_count_30d": ("Keep", "Recent transaction frequency."),
    "transaction_count_90d": ("Optional", "Intermediate window overlaps history and recent frequency."),
    "invalid_transaction_rows": ("Remove", "Data-quality audit; investigate invalid fields rather than modeling the count."),
    "last_transaction_date": ("Remove", "Exactly represented by days_since_last_transaction."),
    "days_since_last_transaction": ("Keep", "Payment/subscription interaction recency."),
    "total_amount_paid": ("Remove", "History-length driven and largely represented by frequency/payment size."),
    "avg_amount_paid": ("Optional", "Historical payment level; latest amount better reflects current plan."),
    "last_amount_paid": ("Keep", "Latest known payment, including legitimate zero-cost plans."),
    "avg_plan_days": ("Optional", "Historical duration; latest duration reflects current subscription."),
    "last_plan_days": ("Keep", "Current recorded plan duration."),
    "last_membership_expire_date": ("Transform", "Use signed days_until_expiration, masking the likely 1970-01-01 sentinel."),
    "auto_renew_rate": ("Keep", "Historical renewal preference."),
    "last_is_auto_renew": ("Keep", "Latest status may differ from historical preference."),
    "ever_cancel": ("Remove", "Derivable from observed cancel_count when flags are valid."),
    "cancel_count": ("Optional", "Historical cancellations; prefer recent 90-day count for comparability."),
    "cancel_ratio": ("Optional", "History-normalized cancellations; not equivalent to actual churn."),
    "last_is_cancel": ("Keep", "Latest recorded cancellation state; not the target itself."),
    "days_since_last_cancel": ("Keep", "Cancellation recency; NULL means no known cancellation or no history."),
    "cancel_count_30d": ("Optional", "Shorter version of retained 90-day cancellation count."),
    "cancel_count_90d": ("Keep", "Recent cancellation frequency."),
    "transactions_since_last_cancel": ("Optional", "Recovery signal overlaps cancellation/transaction recency."),
}


def markdown(frame):
    """Small Markdown table without an extra tabulate dependency."""
    def cell(x):
        if pd.isna(x):
            return "—"
        if isinstance(x, (float, np.floating)):
            return f"{x:,.3g}"
        return str(x).replace("|", "/")
    return "\n".join([
        "| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join("---" for _ in frame.columns) + " |",
        *["| " + " | ".join(map(cell, row)) + " |" for row in frame.itertuples(index=False, name=None)],
    ])


def rate_plot(con, ax, expression, title):
    groups = con.execute(f"""SELECT {expression} AS segment, COUNT(*) AS users,
        AVG(is_churn) * 100 AS churn_pct FROM dev GROUP BY 1 ORDER BY 1""").df()
    ax.bar(np.arange(len(groups)), groups.churn_pct, color="#3074a6")
    ax.set_xticks(np.arange(len(groups)),
                  [f"{s.split(': ', 1)[-1]}\n(n={n:,})" for s, n in zip(groups.segment, groups.users)],
                  rotation=25, ha="right", fontsize=8)
    ax.set(title=title, ylabel="Churn (%)")
    ax.grid(axis="y", alpha=.2)
    return groups.assign(analysis=title)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    original_stat = (SOURCE.stat().st_size, SOURCE.stat().st_mtime_ns)
    con = duckdb.connect()
    con.execute("SET memory_limit='1GB'")
    con.execute("SET threads=1")
    con.read_parquet(str(SOURCE)).create_view("source")
    schema = con.execute("DESCRIBE source").df()
    rows, users = con.execute("SELECT COUNT(*), COUNT(DISTINCT msno) FROM source").fetchall()[0]
    if rows != users:
        raise ValueError("Duplicate users: establish a user-level split before analysis")

    # Read only IDs/labels to establish the boundary; never inspect test feature values.
    labels = con.execute("SELECT msno, is_churn FROM source ORDER BY msno").df()
    if not labels.is_churn.isin([0, 1]).all():
        raise ValueError("Invalid churn labels")
    rng = np.random.default_rng(SEED)
    test_indices = np.concatenate([
        rng.permutation(labels.index[labels.is_churn == label])[:round(.2 * (labels.is_churn == label).sum())]
        for label in (0, 1)
    ])
    test_ids = labels.loc[test_indices, ["msno"]].sort_values("msno").reset_index(drop=True)
    test_path = OUTPUT / "test_user_ids.csv"
    if test_path.exists():
        if not pd.read_csv(test_path).equals(test_ids):
            raise ValueError("Existing test IDs differ; refusing to change the evaluation split")
    else:
        test_ids.to_csv(test_path, index=False)
    dev_ids = labels.drop(index=test_indices)[["msno"]]
    con.register("dev_ids", dev_ids)
    con.execute("""CREATE VIEW dev AS
        SELECT s.*,
               DATE_DIFF('day', registration_date, DATE '2017-02-28') AS membership_tenure_days,
               CASE WHEN invalid_listening_rows = 0 AND total_songs > 0
                          AND total_songs_100 BETWEEN 0 AND total_songs
                    THEN total_songs_100 / total_songs END AS song_completion_rate,
               CASE WHEN last_membership_expire_date <> DATE '1970-01-01'
                    THEN DATE_DIFF('day', DATE '2017-02-28', last_membership_expire_date)
               END AS days_until_expiration
        FROM source s INNER JOIN dev_ids USING (msno)
    """)
    del labels, test_indices
    print(f"Development: {len(dev_ids):,}; test: {len(test_ids):,}. Test rows excluded from feature analysis.", flush=True)

    numeric = schema.loc[schema.column_type.isin(["INTEGER", "BIGINT", "DOUBLE", "HUGEINT"]), "column_name"].tolist()
    numeric = [c for c in numeric if c != "is_churn"]
    derived = ["membership_tenure_days", "song_completion_rate", "days_until_expiration"]
    quality = []
    for column in schema.column_name.tolist() + derived:
        dtype = schema.set_index("column_name").column_type.get(column, "derived")
        if column in numeric + derived:
            result = con.execute(f'''SELECT COUNT(*) - COUNT("{column}"),
                MIN("{column}"), AVG("{column}"),
                APPROX_QUANTILE("{column}", [0.5, 0.99]), MAX("{column}"),
                COUNT(*) FILTER (WHERE "{column}" IS NOT NULL AND NOT ISFINITE("{column}"))
                FROM dev''').fetchall()[0]
            missing, low, mean, quantiles, high, nonfinite = result
            median, p99 = quantiles if quantiles else [None, None]
        elif dtype == "DATE":
            missing, low, high = con.execute(f'''SELECT COUNT(*) - COUNT("{column}"),
                MIN("{column}"), MAX("{column}") FROM dev''').fetchall()[0]
            mean = median = p99 = None
            nonfinite = 0
        else:
            missing = con.execute(f'SELECT COUNT(*) - COUNT("{column}") FROM dev').fetchall()[0][0]
            low = mean = median = p99 = high = None
            nonfinite = 0
        quality.append([column, dtype, missing / len(dev_ids) * 100, low, mean, median, p99, high, nonfinite])
    quality = pd.DataFrame(quality, columns=["Column", "Type", "Missing %", "Min", "Mean", "Median", "P99", "Max", "Nonfinite"])

    nonnegative = [c for c in numeric if c not in ["listening_secs_change_30d", "listening_secs_ratio_30d"]]
    checks = {
        "Negative nonnegative-valued fields (row count, possibly overlapping)": " OR ".join(f'"{c}" < 0' for c in nonnegative),
        "Age outside 1–100": "age NOT BETWEEN 1 AND 100",
        "Rate outside 0–1": "auto_renew_rate NOT BETWEEN 0 AND 1 OR cancel_ratio NOT BETWEEN 0 AND 1 OR song_completion_rate NOT BETWEEN 0 AND 1",
        "Post-cutoff recorded behavior/registration": "last_active_date > DATE '2017-02-28' OR last_transaction_date > DATE '2017-02-28' OR registration_date > DATE '2017-02-28'",
        "Listening recency disagrees with date": "days_since_last_active <> DATE_DIFF('day', last_active_date, DATE '2017-02-28')",
        "Transaction recency disagrees with date": "days_since_last_transaction <> DATE_DIFF('day', last_transaction_date, DATE '2017-02-28')",
        "Inconsistent listening windows": "active_days_7d > 7 OR active_days_30d > 30 OR active_days_7d > active_days_30d OR active_days_30d > active_days OR active_days > listening_record_count",
        "Inconsistent transaction windows": "transaction_count_30d > transaction_count_90d OR transaction_count_90d > transaction_count OR cancel_count_30d > cancel_count_90d OR cancel_count_90d > cancel_count",
        "No-log indicator with nonmissing measurements": "has_listening_logs=0 AND (active_days IS NOT NULL OR total_listening_secs IS NOT NULL)",
        "No-transaction indicator with nonmissing count": "has_transactions=0 AND transaction_count IS NOT NULL",
        "Masked demographics with nonmissing attributes": "has_demographics=0 AND (registration_date IS NOT NULL OR city IS NOT NULL OR age IS NOT NULL OR gender IS NOT NULL OR registered_via IS NOT NULL)",
        "Users with invalid listening rows": "invalid_listening_rows > 0",
        "Users with invalid transaction rows": "invalid_transaction_rows > 0",
        "Raw completion ratio unavailable/unsafe despite logs": "has_listening_logs=1 AND song_completion_rate IS NULL",
        "Historical average listening above 24 hours per recorded day": "avg_secs_per_active_day > 86400",
        "Recent listening above 24 hours per recorded day": "listening_secs_30d > active_days_30d * 86400",
        "Likely epoch-date expiration sentinel (masked in derived feature)": "last_membership_expire_date = DATE '1970-01-01'",
    }
    checks.update({f"Invalid binary {c}": f"{c} NOT IN (0,1)" for c in
                   ["has_demographics", "has_listening_logs", "has_transactions", "age_missing", "gender_missing", "last_is_auto_renew", "last_is_cancel", "ever_cancel"]})
    checks_frame = pd.DataFrame([(name, con.execute(f"SELECT COUNT(*) FROM dev WHERE {condition}").fetchall()[0][0])
                                 for name, condition in checks.items()], columns=["Check", "Development users"])

    # All feature correlations use at most 50,000 development users.
    sample_ids = dev_ids.sample(n=min(50_000, len(dev_ids)), random_state=SEED)
    con.register("sample_ids", sample_ids)
    sample = con.execute('SELECT ' + ','.join(f'd."{c}"' for c in numeric + derived) +
                         ' FROM dev d INNER JOIN sample_ids USING (msno) ORDER BY msno').df()
    correlations = sample.corr(method="spearman", min_periods=500)
    pairs = []
    for i, a in enumerate(correlations.columns):
        for b in correlations.columns[i+1:]:
            paired = sample[[a, b]].dropna()
            # Missingness indicators can become constant after pairwise deletion.
            if len(paired) < 500 or paired.nunique().min() < 2:
                correlations.loc[a, b] = correlations.loc[b, a] = np.nan
            elif abs(correlations.loc[a, b]) >= .85:
                pairs.append((a, b, correlations.loc[a, b], len(paired)))
    pairs = pd.DataFrame(pairs, columns=["Feature A", "Feature B", "Spearman rho", "Pairwise n"])
    if len(pairs):
        pairs = pairs.iloc[np.argsort(-pairs["Spearman rho"].abs().to_numpy())]
    balance = con.execute("SELECT is_churn, COUNT(*) AS users FROM dev GROUP BY 1 ORDER BY 1").df()
    medians = con.execute("""SELECT is_churn, COUNT(*) AS users,
        MEDIAN(days_since_last_active) AS listening_recency,
        MEDIAN(active_days_30d) AS active_days_30d,
        MEDIAN(listening_secs_30d)/3600 AS listening_hours_30d,
        MEDIAN(transaction_count) AS transactions,
        MEDIAN(last_amount_paid) AS last_payment,
        AVG(last_is_auto_renew)*100 AS latest_auto_renew_pct,
        AVG(last_is_cancel)*100 AS latest_cancel_pct,
        MEDIAN(days_until_expiration) AS days_until_expiration
        FROM dev GROUP BY 1 ORDER BY 1""").df()

    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    groups = []
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(["Renewed", "Churned"], balance.users, color=["#3074a6", "#d87738"])
    for i, n in enumerate(balance.users):
        ax.text(i, n, f"{n:,} ({100*n/len(dev_ids):.2f}%)", ha="center", va="bottom")
    ax.set(title="Development cohort: churn balance", ylabel="Users", ylim=(0, max(balance.users)*1.15))
    fig.tight_layout(); fig.savefig(OUTPUT/'01_churn_balance.png'); plt.close(fig)

    chart_specs = [
        ("02_demographics.png", [
            ("CASE WHEN age IS NULL THEN '6: Missing' WHEN age < 20 THEN '1: 1–19' WHEN age < 30 THEN '2: 20–29' WHEN age < 40 THEN '3: 30–39' WHEN age < 50 THEN '4: 40–49' ELSE '5: 50–100' END", "Age"),
            ("COALESCE(registered_via, 'Missing')", "Registration channel")]),
        ("03_listening.png", [
            ("CASE WHEN days_since_last_active IS NULL THEN '5: No history' WHEN days_since_last_active <= 7 THEN '1: 0–7 days' WHEN days_since_last_active <= 30 THEN '2: 8–30 days' WHEN days_since_last_active <= 90 THEN '3: 31–90 days' ELSE '4: >90 days' END", "Days since listening"),
            ("CASE WHEN active_days_30d IS NULL THEN '5: No history' WHEN active_days_30d=0 THEN '1: 0 days' WHEN active_days_30d<=7 THEN '2: 1–7 days' WHEN active_days_30d<=20 THEN '3: 8–20 days' ELSE '4: 21–30 days' END", "Recorded listening days in last 30")]),
        ("04_subscription_behavior.png", [
            ("CASE WHEN last_is_auto_renew=1 THEN '1: Auto-renew on' WHEN last_is_auto_renew=0 THEN '2: Auto-renew off' ELSE '3: Unknown' END", "Latest auto-renewal"),
            ("CASE WHEN last_is_cancel=1 THEN '1: Cancellation' WHEN last_is_cancel=0 THEN '2: Non-cancellation' ELSE '3: Unknown' END", "Latest transaction cancellation")]),
        ("05_tenure_and_expiration.png", [
            ("CASE WHEN membership_tenure_days IS NULL THEN '5: Unknown' WHEN membership_tenure_days<=90 THEN '1: 0–90 days' WHEN membership_tenure_days<=365 THEN '2: 91–365 days' WHEN membership_tenure_days<=1095 THEN '3: 1–3 years' ELSE '4: >3 years' END", "Membership tenure"),
            ("CASE WHEN days_until_expiration IS NULL THEN '6: Unknown' WHEN days_until_expiration<0 THEN '1: Already expired' WHEN days_until_expiration=0 THEN '2: Cutoff day' WHEN days_until_expiration<=7 THEN '3: In 1–7 days' WHEN days_until_expiration<=30 THEN '4: In 8–30 days' ELSE '5: In >30 days' END", "Known expiration relative to cutoff")]),
    ]
    for filename, specs in chart_specs:
        fig, axes = plt.subplots(1,2,figsize=(12,4.8))
        for ax, (expression, title) in zip(axes, specs):
            groups.append(rate_plot(con, ax, expression, title))
        fig.suptitle("Development users only; descriptive associations, not causal effects", fontsize=10)
        fig.tight_layout(); fig.savefig(OUTPUT/filename); plt.close(fig)
    heat_features = [c for c in PREDICTORS if c in correlations and sample[c].nunique() > 1]
    matrix = correlations.loc[heat_features, heat_features]
    fig, ax = plt.subplots(figsize=(12,10))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#dddddd")
    im = ax.imshow(np.ma.masked_invalid(matrix), vmin=-1, vmax=1, cmap=cmap)
    ax.set_xticks(range(len(matrix)),matrix.columns,rotation=90,fontsize=8)
    ax.set_yticks(range(len(matrix)),matrix.index,fontsize=8)
    ax.set_title("Selected numerical predictors: Spearman correlation\n50,000 development users; gray = insufficient/constant paired values")
    fig.colorbar(im, ax=ax, shrink=.7, label="Spearman rho")
    fig.tight_layout(); fig.savefig(OUTPUT/'06_correlations.png'); plt.close(fig)

    categories = con.execute("""SELECT 'city' AS attribute, COALESCE(city,'Missing') AS value,
        COUNT(*) AS users, AVG(is_churn)*100 AS churn_pct FROM dev GROUP BY city
        UNION ALL SELECT 'gender', COALESCE(gender,'Missing'), COUNT(*), AVG(is_churn)*100
        FROM dev GROUP BY gender ORDER BY attribute, users DESC""").df()
    group_table = pd.concat(groups, ignore_index=True)[["analysis", "segment", "users", "churn_pct"]]
    group_table.segment = group_table.segment.str.replace(r"^\d+: ", "", regex=True)
    decisions = pd.DataFrame([(c, *DECISIONS[c]) for c in schema.column_name],
                             columns=["Original column", "Decision", "Reason"])
    display_quality = quality.drop(columns="Nonfinite").copy()
    selected_missing = quality.set_index("Column").loc[PREDICTORS, "Missing %"].sort_values(ascending=False)
    metadata = json.loads(SOURCE.with_suffix('.metadata.json').read_text())
    renewal_rates = group_table[group_table.analysis.eq("Latest auto-renewal")].set_index("segment").churn_pct
    cancellation_rates = group_table[group_table.analysis.eq("Latest transaction cancellation")].set_index("segment").churn_pct
    check_counts = checks_frame.set_index("Check")["Development users"]
    report = f"""# KKBox v2: development EDA and compact feature set

Source: `{SOURCE.relative_to(ROOT)}`; {rows:,} rows, {users:,} unique IDs, {len(schema)} columns.
Source values are unchanged. Metadata version {metadata['preprocessing_version']} records original
transactions/listening sources and inclusive cutoff {metadata['cutoff_inclusive']}.

## Evaluation boundary

Sorted user IDs; NumPy `default_rng(42)` independently permutes each target class;
20% of each class is held out. Development: **{len(dev_ids):,}**; test: **{len(test_ids):,}**.
Only IDs/labels were read to establish the split. All quality statistics, plots,
correlations, and feature decisions below use development data only. The saved
`reports/kkbox_v2_eda/test_user_ids.csv` fixes the final test boundary. Do not tune
features, imputers, thresholds, or models against this held-out set. This is a
within-cohort split, not a future-month deployment test. No models have been trained.

## Main comparisons

Churn rate: **{100*balance.loc[balance.is_churn.eq(1),'users'].iloc[0]/len(dev_ids):.3f}%**.
Medians omit missing measurements; status percentages use nonmissing flags.

- Auto-renew off: **{renewal_rates['Auto-renew off']:.1f}%** churn versus
  **{renewal_rates['Auto-renew on']:.1f}%** with auto-renew on. Latest cancellation:
  **{cancellation_rates['Cancellation']:.1f}%** versus **{cancellation_rates['Non-cancellation']:.1f}%**
  on non-cancellation records. These are much clearer descriptive differences than listening volume.
- Both outcome groups have a median **15 recorded listening days** in the last 30.
  Churners do not simply listen less in this snapshot. Retain a small behavioral set
  for later incremental evaluation, rather than claiming these are proven predictors.
- Historical seconds reach approximately **9.22e15**, and historical average seconds
  per day approximately **4.25e13**. There are **{check_counts['Historical average listening above 24 hours per recorded day']:,}**
  development users above 24 hours per recorded day. Omit the historical average and
  raw historical seconds from the initial predictors; these values need a raw-log audit.
- **{check_counts['Recent listening above 24 hours per recorded day']:,}** users also exceed that bound in recent listening.
  This can include overlapping/device activity, so it is a diagnostic rather than an
  automatic deletion rule. Investigate or use training-only robust treatment later.
- Age/gender are about 60% missing. Male/female observed churn rates are nearly
  identical; leave gender optional. Missing demographic groups have different rates,
  so absence must not be encoded as a real age or location.

{markdown(medians)}

{markdown(group_table)}

Category detail (city: largest 8 groups; gender: all groups):

{markdown(pd.concat([categories[categories.attribute.eq('city')].head(8),categories[categories.attribute.eq('gender')]]))}

These are descriptive associations. Missing-history segments are separate from
observed zero activity; rare groups have less reliable rates. Differences in
exposure/history and subscription policies can confound these relationships.

## Data quality and potential leakage

Duplicate IDs in source: **{rows-users}**. Nonfinite numerical values in development:
**{int(quality.Nonfinite.sum())}** (count across fields).

{markdown(checks_frame)}

Dates were filtered upstream before aggregation. The development date checks and
source metadata are consistent with this rule; this analysis does not rescan raw
behavioral events. Expiration is a planned future date recorded in an eligible past
transaction, so an expiry after cutoff is not itself future-event leakage. Keep signed
`days_until_expiration` provisionally, with the documented latest-state ordering
assumption. The **{check_counts['Likely epoch-date expiration sentinel (masked in derived feature)']}** development expiry values equal to
1970-01-01 are treated as a likely placeholder and masked in the derived feature;
this is an explicit quality assumption, not proof of the original encoding. Other
negative expiration offsets remain valid. Strong association is expected for a subscription-gap target;
later modeling should compare performance with and without it. Supplied labels have
not been independently regenerated, and their exact horizon is not certified.
Demographics are a later unversioned snapshot; masking future registrations does
not establish historical availability of every attribute. Consider a no-demographics
sensitivity comparison later.

Negative listening changes and negative days until expiration are legitimate.
The table below shows outlier ranges; do not interpret an extreme as automatically
invalid. No rows are dropped or winsorized. For skewed counts/seconds/payment amounts,
consider log1p or training-fold-only clipping during modeling. Never learn clipping
or imputation thresholds from the final test set. Invalid-row counts are audit
attributes, and sums from partially invalid histories may understate real activity.

## Three derived features

- `membership_tenure_days`: cutoff minus usable registration date; NULL if unavailable.
- `song_completion_rate`: `total_songs_100 / total_songs`; NULL for zero/missing
  denominator, inconsistent numerator, or any flagged invalid listening history.
  This conservative guard avoids ratios of partial totals. `num_100` means over
  98.5% completion, not necessarily a literal 100% play.
- `days_until_expiration`: latest recorded expiry minus cutoff; retain negative
  values for expired subscriptions and NULL for unavailable/1970-01-01 expiry.

These are calculated in a development SQL view, not written into the source parquet.

## Recommended predictors: {len(PREDICTORS)}

`{', '.join(PREDICTORS)}`

This is a compact initial specification, not a model-performance claim. Categories
are city and registered_via. Keep NULL semantics until a training-only
modeling pipeline chooses imputation/encoding. Explicit history indicators distinguish
missing sources from observed zero activity. Age/gender missing flags are omitted
because missingness can be encoded with the underlying fields later. Different
time windows and latest/historical auto-renewal are intentionally retained initially.

Selected features with the most missing values:

{markdown(selected_missing.head(8).rename('Missing %').reset_index())}

All original attributes are classified below. Transform means replace/use to derive
the stated selected feature, rather than retain another raw predictor. Optional
attributes are excluded from this initial {len(PREDICTORS)}-predictor specification.

{markdown(decisions)}

Strongest numerical redundancies (absolute Spearman rho >= .85, at least 500
pairwise nonmissing observations in a sample of up to 50,000 development users):

{markdown(pairs.head(12))}

Correlation alone is not a removal rule. Exact structural redundancy, meaning,
missingness, and the compact specification drive the decisions. Sample pair counts
matter especially for rare cancellations. No target-based importance model was fit.

## Figures

"""
    for path in sorted(OUTPUT.glob('*.png')):
        report += f"![{path.stem}](../reports/kkbox_v2_eda/{path.name})\n\n"
    report += "## Column audit (development only)\n\nApproximate median/P99; exact min/max. Target excluded from numerical summaries.\n\n" + markdown(display_quality) + "\n"
    REPORT.write_text(report)
    con.close()
    if (SOURCE.stat().st_size, SOURCE.stat().st_mtime_ns) != original_stat:
        raise RuntimeError("Source file changed during the analysis")
    print(f"Saved {REPORT.relative_to(ROOT)}, six figures, and test IDs. Source parquet unchanged.")
    print(markdown(medians))
    print(markdown(checks_frame[checks_frame['Development users'].ne(0)]))
    print(markdown(pairs.head(12)))


if __name__ == "__main__":
    main()
