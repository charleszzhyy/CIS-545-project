#!/usr/bin/env python3
"""Build the KKBox v2 February-end snapshot. See docs/kkbox_v2_asof_features.md."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import duckdb

CUTOFF = "2017-02-28"
VERSION = "1.0.1"
EXPECTED_USERS = 970_960
LISTENING_BUCKETS = 64


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def prepare_train_and_members(con, expected_users):
    rows, users, invalid = con.execute("""
        SELECT COUNT(*), COUNT(DISTINCT msno),
               COUNT(*) FILTER (WHERE msno IS NULL OR TRIM(msno) = ''
                   OR is_churn IS NULL OR is_churn NOT IN ('0', '1'))
        FROM labels_raw
    """).fetchone()
    if (rows, users, invalid) != (expected_users, expected_users, 0):
        raise ValueError(f"Invalid labels: rows={rows}, users={users}, invalid={invalid}")
    con.execute("""
        CREATE TABLE train AS
        SELECT msno, CAST(is_churn AS INTEGER) AS is_churn FROM labels_raw
    """)
    con.execute(f"""
        CREATE TABLE member_features AS
        WITH parsed AS (
            SELECT m.*, TRY_STRPTIME(registration_init_time, '%Y%m%d')::DATE AS reg
            FROM members_raw m INNER JOIN train t USING (msno)
        ), usable AS (
            SELECT *, reg <= DATE '{CUTOFF}' AS available FROM parsed
        )
        SELECT msno,
               CAST(COALESCE(available, FALSE) AS INTEGER) AS has_demographics,
               CASE WHEN available THEN NULLIF(TRIM(city), '') END AS city,
               CASE WHEN available AND TRY_CAST(bd AS INTEGER) BETWEEN 1 AND 100
                    THEN TRY_CAST(bd AS INTEGER) END AS age,
               CASE WHEN available THEN NULLIF(TRIM(gender), '') END AS gender,
               CASE WHEN available THEN NULLIF(TRIM(registered_via), '') END AS registered_via,
               CASE WHEN available THEN reg END AS registration_date,
               reg > DATE '{CUTOFF}' AS registration_after_cutoff
        FROM usable
    """)
    duplicates = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT msno FROM member_features GROUP BY msno HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    if duplicates:
        raise ValueError(f"Conflicting/duplicate member records for {duplicates} training users")


def build_listening(con, work_dir=None):
    # Bound exact-distinct aggregation to one user's bucket, not the entire history.
    con.execute("SET partitioned_write_max_open_files = 8")
    with tempfile.TemporaryDirectory(prefix="listening_", dir=work_dir) as staging:
        con.execute(f"""
        COPY (
        WITH eligible AS NOT MATERIALIZED (
            SELECT l.msno, TRY_STRPTIME(date, '%Y%m%d')::DATE AS activity_date,
                   l.num_25, l.num_50, l.num_75, l.num_985,
                   l.num_100, l.num_unq, l.total_secs
            FROM logs_raw l INNER JOIN train t USING (msno)
            WHERE TRY_STRPTIME(date, '%Y%m%d')::DATE <= DATE '{CUTOFF}'
        ), parsed AS NOT MATERIALIZED (
            SELECT msno, activity_date,
                   TRY_CAST(num_25 AS BIGINT) AS n25,
                   TRY_CAST(num_50 AS BIGINT) AS n50,
                   TRY_CAST(num_75 AS BIGINT) AS n75,
                   TRY_CAST(num_985 AS BIGINT) AS n985,
                   TRY_CAST(num_100 AS BIGINT) AS n100,
                   TRY_CAST(num_unq AS BIGINT) AS unq,
                   TRY_CAST(total_secs AS DOUBLE) AS secs
            FROM eligible
        ), cleaned AS NOT MATERIALIZED (
            SELECT msno, activity_date,
                   CASE WHEN n25 >= 0 AND n50 >= 0 AND n75 >= 0
                                  AND n985 >= 0 AND n100 >= 0
                        THEN n25::HUGEINT + n50 + n75 + n985 + n100 END AS songs,
                   CASE WHEN n100 >= 0 THEN n100 END AS songs_100,
                   CASE WHEN unq >= 0 THEN unq END AS unique_songs,
                   CASE WHEN secs >= 0 AND ISFINITE(secs) THEN secs END AS seconds
            FROM parsed
        )
        SELECT *, CAST(HASH(msno) % {LISTENING_BUCKETS} AS INTEGER) AS bucket
        FROM cleaned
        ) TO {sql_string(staging)}
        (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (bucket), ROW_GROUP_SIZE 16384)
        """)
        con.execute("""
            CREATE TABLE listening_features (
                msno VARCHAR, listening_record_count BIGINT, invalid_listening_rows BIGINT,
                active_days BIGINT, total_songs HUGEINT, total_songs_100 HUGEINT,
                total_unique_songs_daily HUGEINT, total_listening_secs DOUBLE,
                avg_secs_per_active_day DOUBLE, last_active_date DATE,
                days_since_last_active BIGINT, active_days_7d BIGINT, active_days_30d BIGINT,
                listening_secs_7d DOUBLE, listening_secs_30d DOUBLE,
                listening_secs_previous_30d DOUBLE, songs_30d HUGEINT
            )
        """)
        for bucket in range(LISTENING_BUCKETS):
            folder = Path(staging) / f"bucket={bucket}"
            if not folder.exists():
                continue
            print(f"Aggregating listening bucket {bucket + 1}/{LISTENING_BUCKETS}...", flush=True)
            con.execute(f"""
        INSERT INTO listening_features
        SELECT msno, COUNT(*) AS listening_record_count,
               COUNT(*) FILTER (WHERE seconds IS NULL OR songs IS NULL
                                   OR unique_songs IS NULL) AS invalid_listening_rows,
               COUNT(DISTINCT activity_date) AS active_days,
               SUM(songs) AS total_songs,
               SUM(songs_100) AS total_songs_100,
               SUM(unique_songs) AS total_unique_songs_daily,
               SUM(seconds) AS total_listening_secs,
               SUM(seconds) / COUNT(DISTINCT activity_date) AS avg_secs_per_active_day,
               MAX(activity_date) AS last_active_date,
               DATE_DIFF('day', MAX(activity_date), DATE '{CUTOFF}') AS days_since_last_active,
               COUNT(DISTINCT activity_date) FILTER (
                   WHERE activity_date > DATE '{CUTOFF}' - INTERVAL 7 DAY) AS active_days_7d,
               COUNT(DISTINCT activity_date) FILTER (
                   WHERE activity_date > DATE '{CUTOFF}' - INTERVAL 30 DAY) AS active_days_30d,
               CASE WHEN COUNT(*) FILTER (
                   WHERE activity_date > DATE '{CUTOFF}' - INTERVAL 7 DAY) = 0 THEN 0
                   ELSE SUM(seconds) FILTER (
                       WHERE activity_date > DATE '{CUTOFF}' - INTERVAL 7 DAY)
               END AS listening_secs_7d,
               CASE WHEN COUNT(*) FILTER (
                   WHERE activity_date > DATE '{CUTOFF}' - INTERVAL 30 DAY) = 0 THEN 0
                   ELSE SUM(seconds) FILTER (
                       WHERE activity_date > DATE '{CUTOFF}' - INTERVAL 30 DAY)
               END AS listening_secs_30d,
               CASE WHEN COUNT(*) FILTER (
                   WHERE activity_date > DATE '{CUTOFF}' - INTERVAL 60 DAY
                     AND activity_date <= DATE '{CUTOFF}' - INTERVAL 30 DAY) = 0 THEN 0
                   ELSE SUM(seconds) FILTER (
                       WHERE activity_date > DATE '{CUTOFF}' - INTERVAL 60 DAY
                         AND activity_date <= DATE '{CUTOFF}' - INTERVAL 30 DAY)
               END AS listening_secs_previous_30d,
               CASE WHEN COUNT(*) FILTER (
                   WHERE activity_date > DATE '{CUTOFF}' - INTERVAL 30 DAY) = 0 THEN 0
                   ELSE SUM(songs) FILTER (
                       WHERE activity_date > DATE '{CUTOFF}' - INTERVAL 30 DAY)
               END AS songs_30d
        FROM read_parquet({sql_string(folder / '*.parquet')}, hive_partitioning=false)
        GROUP BY msno
            """)
            # This run owns these files; release disk space after their aggregation succeeds.
            for parquet in folder.glob("*.parquet"):
                parquet.unlink()
            folder.rmdir()


def build_transactions(con):
    # Store only eligible events for labeled users; ordering/aggregation follows filtering.
    con.execute(f"""
        CREATE TABLE transaction_events AS
        WITH eligible AS (
            SELECT x.*, TRY_STRPTIME(transaction_date, '%Y%m%d')::DATE AS txn_date
            FROM transactions_raw x INNER JOIN train t USING (msno)
            WHERE TRY_STRPTIME(transaction_date, '%Y%m%d')::DATE <= DATE '{CUTOFF}'
        ), parsed AS (
            SELECT msno, txn_date,
                   TRY_STRPTIME(membership_expire_date, '%Y%m%d')::DATE AS expire_date,
                   TRY_CAST(actual_amount_paid AS DOUBLE) AS paid,
                   TRY_CAST(payment_plan_days AS INTEGER) AS plan_days,
                   CASE WHEN is_cancel IN ('0', '1')
                        THEN CAST(is_cancel AS INTEGER) END AS cancel,
                   CASE WHEN is_auto_renew IN ('0', '1')
                        THEN CAST(is_auto_renew AS INTEGER) END AS renew,
                   CASE WHEN plan_list_price IS NOT NULL AND payment_plan_days IS NOT NULL
                                  AND payment_method_id IS NOT NULL
                        THEN CONCAT(plan_list_price, payment_plan_days, payment_method_id)
                   END AS plan_sort
            FROM eligible
        )
        SELECT msno, txn_date, expire_date, plan_sort,
               CASE WHEN paid >= 0 AND ISFINITE(paid) THEN paid END AS paid,
               CASE WHEN plan_days >= 0 THEN plan_days END AS plan_days,
               cancel, renew
        FROM parsed
    """)
    con.execute(f"""
        CREATE TABLE transaction_features AS
        WITH ordered AS (
            SELECT *,
                   MAX(txn_date) FILTER (WHERE cancel = 1)
                       OVER (PARTITION BY msno) AS last_cancel_date,
                   ROW_NUMBER() OVER (
                       PARTITION BY msno ORDER BY txn_date DESC,
                           plan_sort ASC NULLS LAST, cancel DESC NULLS LAST,
                           CASE WHEN cancel = 1 THEN expire_date END ASC NULLS LAST,
                           CASE WHEN cancel = 0 THEN expire_date END DESC NULLS LAST,
                           paid DESC NULLS LAST, renew DESC NULLS LAST,
                           plan_days DESC NULLS LAST, expire_date DESC NULLS LAST
                   ) AS rn
            FROM transaction_events
        )
        SELECT msno, COUNT(*) AS transaction_count,
               COUNT(*) FILTER (WHERE txn_date > DATE '{CUTOFF}' - INTERVAL 30 DAY)
                   AS transaction_count_30d,
               COUNT(*) FILTER (WHERE txn_date > DATE '{CUTOFF}' - INTERVAL 90 DAY)
                   AS transaction_count_90d,
               COUNT(*) FILTER (WHERE paid IS NULL OR plan_days IS NULL
                   OR cancel IS NULL OR renew IS NULL OR expire_date IS NULL)
                   AS invalid_transaction_rows,
               MAX(txn_date) AS last_transaction_date,
               DATE_DIFF('day', MAX(txn_date), DATE '{CUTOFF}') AS days_since_last_transaction,
               SUM(paid) AS total_amount_paid, AVG(paid) AS avg_amount_paid,
               MAX(paid) FILTER (WHERE rn = 1) AS last_amount_paid,
               AVG(plan_days) AS avg_plan_days,
               MAX(plan_days) FILTER (WHERE rn = 1) AS last_plan_days,
               MAX(expire_date) FILTER (WHERE rn = 1) AS last_membership_expire_date,
               AVG(renew) AS auto_renew_rate,
               MAX(renew) FILTER (WHERE rn = 1) AS last_is_auto_renew,
               MAX(cancel) AS ever_cancel,
               COUNT(*) FILTER (WHERE cancel = 1) AS cancel_count,
               AVG(cancel) AS cancel_ratio,
               MAX(cancel) FILTER (WHERE rn = 1) AS last_is_cancel,
               DATE_DIFF('day', MAX(last_cancel_date), DATE '{CUTOFF}') AS days_since_last_cancel,
               COUNT(*) FILTER (WHERE cancel = 1
                   AND txn_date > DATE '{CUTOFF}' - INTERVAL 30 DAY) AS cancel_count_30d,
               COUNT(*) FILTER (WHERE cancel = 1
                   AND txn_date > DATE '{CUTOFF}' - INTERVAL 90 DAY) AS cancel_count_90d,
               CASE WHEN MAX(last_cancel_date) IS NOT NULL
                    THEN COUNT(*) FILTER (WHERE txn_date > last_cancel_date)
               END AS transactions_since_last_cancel
        FROM ordered GROUP BY msno
    """)


def join_and_validate(con, expected_users):
    for table in ("member_features", "listening_features", "transaction_features"):
        rows, users = con.execute(f"SELECT COUNT(*), COUNT(DISTINCT msno) FROM {table}").fetchone()
        if rows != users:
            raise ValueError(f"Duplicate IDs in {table}")
    for table, column in (("transaction_events", "txn_date"),
                          ("listening_features", "last_active_date")):
        future = con.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} > DATE '{CUTOFF}'").fetchone()[0]
        if future:
            raise ValueError(f"Future events in {table}")
    con.execute("""
        CREATE TABLE final AS
        SELECT t.*, m.* EXCLUDE (msno, registration_after_cutoff, has_demographics),
               COALESCE(m.has_demographics, 0) AS has_demographics,
               CAST(m.age IS NULL AS INTEGER) AS age_missing,
               CAST(m.gender IS NULL AS INTEGER) AS gender_missing,
               CAST(l.msno IS NOT NULL AS INTEGER) AS has_listening_logs,
               l.* EXCLUDE (msno),
               l.listening_secs_30d - l.listening_secs_previous_30d AS listening_secs_change_30d,
               l.listening_secs_30d / NULLIF(l.listening_secs_previous_30d, 0)
                   AS listening_secs_ratio_30d,
               CAST(x.msno IS NOT NULL AS INTEGER) AS has_transactions,
               x.* EXCLUDE (msno)
        FROM train t
        LEFT JOIN member_features m USING (msno)
        LEFT JOIN listening_features l USING (msno)
        LEFT JOIN transaction_features x USING (msno)
    """)
    rows, users = con.execute("SELECT COUNT(*), COUNT(DISTINCT msno) FROM final").fetchone()
    changed = con.execute("""
        SELECT COUNT(*) FROM final f LEFT JOIN train t USING (msno)
        WHERE t.msno IS NULL OR f.is_churn IS DISTINCT FROM t.is_churn
    """).fetchone()[0]
    if (rows, users, changed) != (expected_users, expected_users, 0):
        raise ValueError(f"Final validation failed: rows={rows}, users={users}, changed={changed}")
    return dict(zip(("rows", "unique_users", "users_with_transactions", "users_with_logs"),
                    con.execute("""SELECT COUNT(*), COUNT(DISTINCT msno),
                        SUM(has_transactions), SUM(has_listening_logs) FROM final""").fetchone()))


def source_identity(paths):
    return {name: {"path": str(path), "bytes": path.stat().st_size,
                   "mtime_ns": path.stat().st_mtime_ns} for name, path in paths.items()}


def run(args):
    paths = {"labels_raw": args.labels.resolve(), "members_raw": args.members.resolve(),
             "transactions_raw": args.transactions.resolve(), "logs_raw": args.logs.resolve()}
    output = args.output.absolute()
    metadata = output.with_suffix(".metadata.json")
    for destination in (output, metadata):
        if os.path.lexists(destination):
            raise FileExistsError(f"Refusing to replace {destination}")
    before = source_identity(paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    # Every run gets fresh storage; no intermediate tables or caches are reused.
    with tempfile.TemporaryDirectory(prefix="kkbox_asof_", dir=args.work_dir) as work:
        con = duckdb.connect(str(Path(work) / "work.duckdb"))
        try:
            con.execute(f"SET memory_limit = {sql_string(args.memory_limit)}")
            con.execute(f"SET threads = {args.threads}")
            con.execute("SET preserve_insertion_order = false")
            con.execute(f"SET temp_directory = {sql_string(Path(work) / 'spill')}")
            con.execute(f"SET max_temp_directory_size = {sql_string(args.max_temp_directory_size)}")
            for name, path in paths.items():
                con.execute(f"""CREATE VIEW {name} AS SELECT * FROM read_csv(
                    {sql_string(path)}, header=true, all_varchar=true, ignore_errors=false)""")
            # Materialize labels once; preserve them as the validation reference.
            con.execute("CREATE TABLE label_snapshot AS SELECT * FROM labels_raw")
            con.execute("CREATE OR REPLACE VIEW labels_raw AS SELECT * FROM label_snapshot")
            print("Preparing labels and demographics...", flush=True)
            prepare_train_and_members(con, EXPECTED_USERS)
            print("Aggregating eligible transactions...", flush=True)
            build_transactions(con)
            print("Aggregating eligible listening logs (large CSV scan)...", flush=True)
            build_listening(con, work)
            summary = join_and_validate(con, EXPECTED_USERS)
            summary["masked_future_registrations"] = con.execute("""SELECT COUNT(*)
                FROM member_features WHERE registration_after_cutoff""").fetchone()[0]
            if source_identity(paths) != before:
                raise RuntimeError("A source file changed during processing; output was not published")
            provenance = {"preprocessing_version": VERSION, "duckdb_version": duckdb.__version__,
                          "cutoff_inclusive": CUTOFF, "sources": before, "validation": summary,
                          "resources": {"memory_limit": args.memory_limit, "threads": args.threads,
                              "max_temp_directory_size": args.max_temp_directory_size,
                              "listening_buckets": LISTENING_BUCKETS}}
            # Stage on the output filesystem, then publish with no-overwrite hard links.
            with tempfile.TemporaryDirectory(prefix=".kkbox_publish_", dir=output.parent) as staging:
                parquet = Path(staging) / "data.parquet"
                manifest = Path(staging) / "metadata.json"
                con.execute(f"COPY final TO {sql_string(parquet)} (FORMAT PARQUET, COMPRESSION ZSTD)")
                manifest.write_text(json.dumps(provenance, indent=2) + "\n")
                os.link(manifest, metadata)
                try:
                    os.link(parquet, output)
                except BaseException:
                    metadata.unlink()  # Only the sidecar created by this run.
                    raise
            print(json.dumps(summary, indent=2))
            print(f"Saved: {output}\nProvenance: {metadata}")
        finally:
            con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=Path("kkbox_work/data/churn_comp_refresh/train_v2.csv"))
    parser.add_argument("--members", type=Path, default=Path("kkbox_work/members_v3.csv"))
    parser.add_argument("--transactions", type=Path, default=Path("kkbox_work/v1/transactions.csv"))
    parser.add_argument("--logs", type=Path, default=Path("kkbox_work/user_logs.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/kkbox_v2_asof_20170228.parquet"))
    parser.add_argument("--work-dir", type=Path, default=Path("kkbox_work"))
    parser.add_argument("--memory-limit", default="3GB")
    parser.add_argument("--max-temp-directory-size", default="50GB",
                        help="DuckDB spill cap; excludes the work database and staged listening files")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    run(args)
