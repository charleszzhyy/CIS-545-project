# KKBox v2 snapshot at February 28, 2017

`build_kkbox_v2_asof.py` version 1.0.1 uses Python, DuckDB, and the standard library.
It creates `data/processed/kkbox_v2_asof_20170228.parquet` and a
`kkbox_v2_asof_20170228.metadata.json` provenance sidecar. Existing destinations
are refused, including symlinks. Existing scripts and processed datasets are not changed.

## Sources and execution

| Role | Default source |
| --- | --- |
| Labels | `kkbox_work/data/churn_comp_refresh/train_v2.csv` |
| Demographics | `kkbox_work/members_v3.csv` |
| Transactions | `kkbox_work/v1/transactions.csv` (original release) |
| Listening | `kkbox_work/user_logs.csv` (original release) |

Neither `transactions_v2.csv` nor `user_logs_v2.csv` contributes features.
Run from the project directory, only after approval to process the full sources:

```bash
python3 build_kkbox_v2_asof.py --memory-limit 3GB --threads 1 --max-temp-directory-size 50GB
```

Paths can be overridden with `--labels`, `--members`, `--transactions`, `--logs`,
`--output`, and `--work-dir`. The cutoff and required 970,960 users are fixed.
Use a filesystem supporting hard links for the output directory (the local project
filesystem does). Publication uses a validated staged parquet and no-overwrite links.

DuckDB reads CSVs as strings, parses dates with `TRY_STRPTIME`, and filters to labeled
users and dates **on or before 2017-02-28 before any behavioral aggregation or ordering**.
Unparseable event dates are excluded, not interpreted as zero activity.
Each run uses a fresh work database and spill directory, removed when the run exits.
There is no cache reuse. The sidecar records absolute source paths, byte sizes,
nanosecond modification times, cutoff, preprocessing/DuckDB versions, and validation
counts. Source size/mtime changes during processing abort publication. These identities
are not cryptographic checksums or evidence of historical source availability.

### Running on an 8 GB Mac

The original single-query listening aggregation failed with both 3 GB and 5 GB limits.
Its historical and filtered `COUNT(DISTINCT activity_date)` states track user/date
pairs across the entire eligible history. CSV streaming and `NOT MATERIALIZED` CTEs
do not bound those states, and spilling is not sufficient for every query/operator.

The fix scans the original CSV **once**, joining labeled users and filtering dates
before writing only the six cleaned listening fields to 64 temporary Parquet buckets
using `HASH(msno) % 64`. It aggregates one bucket at a time with the unchanged SQL.
All records for a user belong to one bucket, so distinct dates, duplicate-day sums,
invalid-row counts, and NULL/empty-window behavior retain their definitions. No
global user/day intermediate table is built. Floating sums may differ in their last
bits due to execution order, as with other parallel/reordered floating arithmetic.

DuckDB's partitioned writer is limited to eight open files with 16,384-row groups.
Each bucket's files are removed after its successful aggregation; the temporary
directory is also cleaned on exit. These are per-run scratch files, never reused caches.

Use `--memory-limit 3GB --threads 1` to leave room for macOS, Python, and allocations
outside DuckDB's buffer manager. `--max-temp-directory-size 50GB` caps DuckDB spill
files under the fresh work directory. **It does not cap total scratch disk use**:
the work database and temporary Parquet buckets consume additional space. With the
reported 86 GiB free, this leaves roughly 39 GiB outside that spill cap for staging,
the database, output, and free-space headroom; actual disk use is not yet measured.
Use `--work-dir` to select a local disk with sufficient space. No RAM-sized pandas
dataframes or additional dependencies are used. Bucket sizes depend on the source;
full-data success, memory, disk use, and runtime still require the approved full run.

## Date windows

All recency values use February 28, never the source's maximum date. All windows
are `(cutoff - N days, cutoff]`, inclusive of the cutoff:

| Window | Inclusive dates |
| --- | --- |
| Last 7 days | February 22–28, 2017 |
| Last 30 days | January 30–February 28, 2017 |
| Previous 30 days | December 31, 2016–January 29, 2017 |
| Last 90 days | December 1, 2016–February 28, 2017 |

Historical totals cover all parseable source dates up to the cutoff, not lifetime
behavior before collection began (the original behavioral files begin in 2015).

## Labels and demographics

| Column | Definition |
| --- | --- |
| `msno` | User ID; join key, exclude from model inputs |
| `is_churn` | Supplied v2 binary target, unchanged; never used for feature calculations |
| `city`, `gender`, `registered_via` | Trimmed categories; blanks become NULL |
| `age` | Integer `bd` in 1–100; otherwise NULL |
| `registration_date` | Parsed registration date |
| `has_demographics` | 1 only when a member record has a parseable registration date at/before cutoff |
| `age_missing`, `gender_missing` | 1 when the final cleaned field is NULL |

Missing, invalid-registration, and post-cutoff registration rows have all demographic
attributes masked to NULL and `has_demographics=0`; labeled users remain. The count
of future registrations is recorded only in validation metadata, not exposed as a
model feature. Duplicate member rows for labeled users cause an error rather than
an arbitrary selection.

Members are an unversioned later snapshot. Masking future registration dates does
not establish that the remaining age/city/gender values were known at prediction time.
Document this assumption and consider a model comparison excluding demographics.

## Listening features

A recorded day is a day with an eligible log, even if measured activity is zero or
invalid. Multiple logs on a day are summed; dates are counted once. No log rows are
deduplicated because no event identifier establishes accidental duplicates.

| Column | Definition |
| --- | --- |
| `has_listening_logs` | 1 if any eligible log exists; otherwise 0 |
| `listening_record_count` | Number of eligible log rows |
| `invalid_listening_rows` | Rows with invalid seconds, song-bucket total, or daily unique count |
| `active_days` | Distinct recorded dates in historical logs |
| `total_songs` | Sum of all five play-count buckets, across eligible rows |
| `total_songs_100` | Sum of `num_100` (plays over 98.5% completion) |
| `total_unique_songs_daily` | Sum of daily `num_unq`; not distinct tracks across the whole history |
| `total_listening_secs` | Sum of valid nonnegative finite listening seconds |
| `avg_secs_per_active_day` | Valid observed seconds divided by all recorded active dates |
| `last_active_date` | Latest recorded eligible date |
| `days_since_last_active` | Days between that date and cutoff; 0 for cutoff-day records |
| `active_days_7d`, `active_days_30d` | Distinct recorded dates in the named windows |
| `listening_secs_7d`, `listening_secs_30d` | Valid seconds summed in the named windows |
| `listening_secs_previous_30d` | Valid seconds in the preceding, nonoverlapping 30 days |
| `songs_30d` | Five-bucket song total in the last 30 days |
| `listening_secs_change_30d` | Recent 30-day seconds minus previous 30-day seconds |
| `listening_secs_ratio_30d` | Recent divided by previous 30-day seconds; NULL when denominator is zero/NULL |

Negative, missing, or unparseable counts/seconds become NULL; nonfinite seconds also
become NULL. A row's five-bucket total is NULL if any bucket is invalid. Valid zero
measurements remain zero. SQL sums ignore NULL measurements: mixed-validity totals
are **partial observed totals**, and averages using them may understate actual activity.
`invalid_listening_rows` flags this limitation; no invalid measurement is silently imputed.

For users **without eligible logs**, every listening measurement/count/date is NULL,
with `has_listening_logs=0`. For users with history but **no rows in a recent window**,
the window's activity counts and sums are zero. If window rows exist but all values
of a measure are invalid, its sum stays NULL. A recorded zero-activity window has
positive recorded-day counts and zero measurements. Absence of logs means no
observations in this source; it does not prove real-world inactivity.

## Transaction features

Counts describe observed transaction **rows**, including plan-change/cancellation
events; they are not counts of distinct dates or successful purchases. Rows with
valid eligible dates are retained even when another field is invalid. Negative or
invalid amounts/plan lengths become NULL (zero is valid), nonfinite amounts become
NULL, and nonbinary auto-renew/cancel values become NULL. Dates with invalid expiry
stay eligible events but have NULL expiry. Field-level statistics use valid values.

| Column | Definition |
| --- | --- |
| `has_transactions` | 1 if any eligible transaction exists; otherwise 0 |
| `transaction_count` | All eligible transaction rows |
| `transaction_count_30d`, `transaction_count_90d` | Eligible rows in the respective windows |
| `invalid_transaction_rows` | Rows with NULL cleaned amount, duration, cancel, renew, or expiry |
| `last_transaction_date` | Latest eligible transaction date |
| `days_since_last_transaction` | Days from latest transaction to cutoff |
| `total_amount_paid`, `avg_amount_paid` | Sum/mean of valid amounts; NTD |
| `last_amount_paid` | Cleaned amount on the selected latest-state row |
| `avg_plan_days`, `last_plan_days` | Mean valid duration and duration on latest-state row |
| `last_membership_expire_date` | Expiry on latest-state row, not maximum expiry over history |
| `auto_renew_rate`, `last_is_auto_renew` | Mean valid 0/1 renew flag and latest-state flag |
| `ever_cancel` | Maximum valid cancellation flag; NULL if every flag is invalid |
| `cancel_count` | Rows with observed `is_cancel=1` |
| `cancel_ratio` | Mean valid cancellation flag |
| `last_is_cancel` | Cancellation flag on latest-state row |
| `days_since_last_cancel` | Days from most recent observed cancellation to cutoff; NULL if none |
| `cancel_count_30d`, `cancel_count_90d` | Observed cancellation rows in the respective windows |
| `transactions_since_last_cancel` | Rows strictly after the latest cancellation date; NULL if no cancellation |

For no eligible transaction history, all transaction measurements/counts are NULL,
with `has_transactions=0`. With history, empty recent windows have zero event counts.
Observed cancellation counts are zero when no valid positive flag is seen; consult
`invalid_transaction_rows`, `ever_cancel`, and `cancel_ratio` before interpreting
this as evidence of no cancellations. Invalid-field counts do not flag invalid dates,
which are excluded upstream. Partial payment/duration histories are not imputed.

### Same-day ordering and expiration assumptions

For latest state, follow the organizer generator's ordering: latest date first;
smallest **lexicographic concatenation** of raw `plan_list_price`, `payment_plan_days`,
`payment_method_id` wins; for the same signature, cancellation follows subscription;
among cancellations earliest expiry wins; among renewals latest expiry wins.
NULL ordering keys go last. Remaining ties use amount, renew flag, plan days, and
expiry descending to make exported latest-state fields deterministic. This final
tie-break is a reproducibility convention, not an observed intraday timestamp.

This reproduces the generator's last-row ordering, not its entire churn/expiration
reconstruction algorithm. `last_membership_expire_date` is the recorded state on
that row. Do not use maximum historical expiry as current state: cancellation may
shorten it. An expiry **after** cutoff is allowed when recorded in a transaction
**at/before** cutoff. Transactions after cutoff are never consulted. Transactions
on the last cancellation date are not counted as later events because intraday
sequence is unavailable. Cancellation is not the churn target.

## Validation and evaluation limits

Before export, require 970,960 unique labels, binary targets, unique feature-table
keys, 970,960 final rows and unique IDs, unchanged user/label pairs, and no eligible
transaction or contributing listening date after cutoff. Feature tables are joined
with LEFT JOINs; missing data never drops a labeled user. Checks raise exceptions
rather than Python assertions (so optimization cannot disable them).

The target remains the supplied competition label. Published month descriptions
are inconsistent; the exact supplied-label generation parameters have not been
independently reproduced. The intended target concerns a roughly 30-day renewal
gap following expiration, not simply cancellation in March. February 28 is this
project's explicit observation cutoff; feature timing control does not certify
label correctness or a fixed future outcome horizon for every supplied user.

An optional v1-to-v2 temporal experiment also needs a label-maturity caveat: earlier
cohort labels may require observations beyond the next cohort's prediction date.
Fit imputers, encoders, model selection, calibration, and thresholds only on training
folds. Do not use `msno` as a predictor. Raw timestamps should be deliberately encoded
or excluded by modeling code. This script performs no modeling or learned imputation.
