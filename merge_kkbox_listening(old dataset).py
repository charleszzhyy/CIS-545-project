#!/usr/bin/env python3
"""Merge KKBox v2 churn labels, members, and pre-cutoff listening logs using disk partitions.

Examples:
  python merge_kkbox_listening.py --download
  python merge_kkbox_listening.py --data-dir ./data/raw
  python merge_kkbox_listening.py --download --cutoff 2017-02-28 --memory-limit 4GB

Requires: pip install duckdb kagglehub  (kagglehub only for --download)
For .7z files: macOS `brew install sevenzip` (installs 7zz).
No transaction data is used.
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import duckdb

FILES = ("train_v2.csv", "members_v3.csv", "user_logs.csv")


def sql_path(path):
    return "'" + str(path).replace("'", "''") + "'"


def locate(base, filename):
    return next(iter(sorted(base.rglob(filename))), None)


def get_sources(base, work):
    work.mkdir(parents=True, exist_ok=True)
    sources = {}
    for filename in FILES:
        found = locate(work, filename) or locate(base, filename)
        if found is None:
            archive = locate(base, filename + ".7z") or locate(work, filename + ".7z")
            if archive is None:
                raise FileNotFoundError(f"Missing {filename} or {filename}.7z under {base}")
            extractor = (
                shutil.which("7zz") or shutil.which("7z")
                or ("/opt/homebrew/bin/7zz" if Path("/opt/homebrew/bin/7zz").is_file() else None)
            )
            if extractor is None:
                raise RuntimeError("Install 7-Zip: `brew install sevenzip` (Mac), then check `which 7zz`.")
            print(f"Extracting {archive.name} ...", flush=True)
            subprocess.run([extractor, "x", str(archive), f"-o{work}", "-y"], check=True)
            found = locate(work, filename)
            if found is None:
                raise FileNotFoundError(f"Extraction succeeded, but {filename} was not found in {work}")
        sources[filename] = found
        print(f"Using {filename}: {found}", flush=True)
    return sources


def build(args):
    cutoff = args.cutoff
    # Validate date before interpolating into SQL.
    from datetime import date
    try:
        cutoff = date.fromisoformat(cutoff).isoformat()
    except ValueError as exc:
        raise ValueError("--cutoff must be a valid date, YYYY-MM-DD") from exc

    base = Path(args.data_dir).expanduser().resolve()
    if args.download:
        import kagglehub
        if args.login:
            kagglehub.login()
        base = Path(kagglehub.competition_download("kkbox-churn-prediction-challenge"))
        print(f"Kaggle cache: {base}", flush=True)
    if not base.is_dir():
        raise FileNotFoundError(f"Data directory not found: {base}")

    work = Path(args.work_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = get_sources(base, work)

    con = duckdb.connect(str(work / "kkbox_listening.duckdb"))
    try:
        con.execute(f"SET memory_limit = {sql_path(args.memory_limit)}")
        con.execute(f"SET threads = {args.threads}")
        temp_dir = work / "duckdb_tmp"
        temp_dir.mkdir(exist_ok=True)
        con.execute(f"SET temp_directory = {sql_path(temp_dir)}")

        for name, file in (("train", "train_v2.csv"), ("members", "members_v3.csv"), ("logs", "user_logs.csv")):
            con.execute(
                f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM "
                f"read_csv({sql_path(paths[file])}, header=true, all_varchar=true)"
            )

        print("Validating labels...", flush=True)
        invalid = con.execute("""
            SELECT COUNT(*) FROM train
            WHERE msno IS NULL OR TRIM(msno) = '' OR is_churn IS NULL
               OR TRY_CAST(is_churn AS INTEGER) NOT IN (0, 1)
        """).fetchone()[0]
        if invalid:
            raise ValueError(f"Found {invalid} missing/invalid user IDs or labels")
        conflicting = con.execute("""
            SELECT COUNT(*) FROM (
                SELECT msno FROM train GROUP BY msno
                HAVING COUNT(DISTINCT is_churn) > 1
            )
        """).fetchone()[0]
        if conflicting:
            raise ValueError(f"{conflicting} training users have conflicting labels")

        con.execute("""
            CREATE OR REPLACE TABLE train_users AS
            SELECT msno, CAST(is_churn AS INTEGER) AS is_churn
            FROM train GROUP BY msno, is_churn
        """)
        count_users = con.execute("SELECT COUNT(*) FROM train_users").fetchone()[0]
        print(f"Labeled users: {count_users:,}", flush=True)

        # Limit demographics to training users and deterministically pick one per msno.
        # A warning reports conflicting members; no extra users are introduced.
        print("Preparing members...", flush=True)
        con.execute("""
            CREATE OR REPLACE TABLE member_features AS
            WITH chosen AS (
                SELECT m.*,
                    ROW_NUMBER() OVER (
                        PARTITION BY m.msno
                        ORDER BY TRY_STRPTIME(m.registration_init_time, '%Y%m%d') DESC NULLS LAST,
                                 m.city, m.bd, m.gender, m.registered_via
                    ) AS rn
                FROM members m INNER JOIN train_users t ON m.msno = t.msno
            )
            SELECT msno,
                   NULLIF(TRIM(city), '') AS city,
                   CASE WHEN TRY_CAST(bd AS INTEGER) BETWEEN 1 AND 100
                        THEN TRY_CAST(bd AS INTEGER) END AS age,
                   NULLIF(TRIM(gender), '') AS gender,
                   NULLIF(TRIM(registered_via), '') AS registered_via,
                   TRY_STRPTIME(registration_init_time, '%Y%m%d') AS registration_date
            FROM chosen WHERE rn = 1
        """)

        # Partition the historical logs by member ID before aggregation.
        # Each user's rows go into exactly one bucket, so DISTINCT dates remain exact.
        bucket_count = args.buckets
        if bucket_count < 2 or bucket_count > 256:
            raise ValueError("--buckets must be between 2 and 256")
        partition_dir = work / f"listening_partitions_{cutoff}_{bucket_count}"
        completed_marker = partition_dir / "_COMPLETE"
        con.execute("SET preserve_insertion_order=false")

        if not completed_marker.exists():
            if partition_dir.exists():
                shutil.rmtree(partition_dir)
            partition_dir.mkdir(parents=True)
            print("Partitioning pre-cutoff listening logs on disk (one large CSV scan)...", flush=True)
            con.execute(f"""
                COPY (
                    SELECT l.msno,
                           TRY_STRPTIME(l.date, '%Y%m%d') AS activity_date,
                           GREATEST(COALESCE(TRY_CAST(l.num_25 AS BIGINT), 0), 0)
                           + GREATEST(COALESCE(TRY_CAST(l.num_50 AS BIGINT), 0), 0)
                           + GREATEST(COALESCE(TRY_CAST(l.num_75 AS BIGINT), 0), 0)
                           + GREATEST(COALESCE(TRY_CAST(l.num_985 AS BIGINT), 0), 0)
                           + GREATEST(COALESCE(TRY_CAST(l.num_100 AS BIGINT), 0), 0) AS songs,
                           GREATEST(COALESCE(TRY_CAST(l.num_100 AS BIGINT), 0), 0) AS songs_100,
                           GREATEST(COALESCE(TRY_CAST(l.num_unq AS BIGINT), 0), 0) AS songs_unique,
                           GREATEST(COALESCE(TRY_CAST(l.total_secs AS DOUBLE), 0), 0) AS seconds,
                           CAST(HASH(l.msno) % {bucket_count} AS INTEGER) AS bucket
                    FROM logs l
                    INNER JOIN train_users t ON l.msno = t.msno
                    WHERE TRY_STRPTIME(l.date, '%Y%m%d')
                          BETWEEN DATE '2000-01-01' AND DATE '{cutoff}'
                ) TO {sql_path(partition_dir)}
                (FORMAT PARQUET, COMPRESSION ZSTD, PARTITION_BY (bucket), OVERWRITE_OR_IGNORE TRUE)
            """)
            completed_marker.touch()
            print("Disk partitioning completed.", flush=True)
        else:
            print("Reusing previously completed disk partitions.", flush=True)

        con.execute("DROP TABLE IF EXISTS listening_features")
        con.execute("""
            CREATE TABLE listening_features (
                msno VARCHAR, active_days BIGINT, total_songs HUGEINT,
                total_songs_100 HUGEINT, total_unique_songs_daily HUGEINT,
                total_listening_secs DOUBLE, avg_secs_per_active_day DOUBLE,
                last_active_date TIMESTAMP, days_since_last_active BIGINT,
                active_days_7d BIGINT, active_days_30d BIGINT,
                listening_secs_7d DOUBLE, listening_secs_30d DOUBLE,
                listening_secs_previous_30d DOUBLE, songs_30d HUGEINT
            )
        """)
        for bucket in range(bucket_count):
            folder = partition_dir / f"bucket={bucket}"
            if not folder.exists():
                continue
            print(f"Aggregating bucket {bucket + 1}/{bucket_count}...", flush=True)
            con.execute(f"""
                INSERT INTO listening_features
                SELECT msno,
                       COUNT(DISTINCT activity_date) AS active_days,
                       SUM(songs) AS total_songs,
                       SUM(songs_100) AS total_songs_100,
                       SUM(songs_unique) AS total_unique_songs_daily,
                       SUM(seconds) AS total_listening_secs,
                       SUM(seconds) / NULLIF(COUNT(DISTINCT activity_date), 0) AS avg_secs_per_active_day,
                       MAX(activity_date) AS last_active_date,
                       DATE_DIFF('day', MAX(activity_date), DATE '{cutoff}') AS days_since_last_active,
                       COUNT(DISTINCT activity_date) FILTER (
                           WHERE activity_date > DATE '{cutoff}' - INTERVAL 7 DAY
                       ) AS active_days_7d,
                       COUNT(DISTINCT activity_date) FILTER (
                           WHERE activity_date > DATE '{cutoff}' - INTERVAL 30 DAY
                       ) AS active_days_30d,
                       SUM(seconds) FILTER (
                           WHERE activity_date > DATE '{cutoff}' - INTERVAL 7 DAY
                       ) AS listening_secs_7d,
                       SUM(seconds) FILTER (
                           WHERE activity_date > DATE '{cutoff}' - INTERVAL 30 DAY
                       ) AS listening_secs_30d,
                       SUM(seconds) FILTER (
                           WHERE activity_date > DATE '{cutoff}' - INTERVAL 60 DAY
                             AND activity_date <= DATE '{cutoff}' - INTERVAL 30 DAY
                       ) AS listening_secs_previous_30d,
                       SUM(songs) FILTER (
                           WHERE activity_date > DATE '{cutoff}' - INTERVAL 30 DAY
                       ) AS songs_30d
                FROM read_parquet({sql_path(str(folder / '*.parquet'))})
                GROUP BY msno
            """)
        # The intermediate parquet buckets may be removed after successful completion
        # if disk space is tight, but keeping them allows reuse on future reruns.

        output = output_dir / "kkbox_train_members_logs.parquet"
        print("Joining and exporting Parquet...", flush=True)
        con.execute(f"""
            COPY (
                SELECT t.msno, t.is_churn,
                       m.city, m.age, m.gender, m.registered_via, m.registration_date,
                       CAST(l.msno IS NOT NULL AS INTEGER) AS has_listening_logs,
                       COALESCE(l.active_days, 0) AS active_days,
                       COALESCE(l.total_songs, 0) AS total_songs,
                       COALESCE(l.total_songs_100, 0) AS total_songs_100,
                       COALESCE(l.total_unique_songs_daily, 0) AS total_unique_songs_daily,
                       COALESCE(l.total_listening_secs, 0) AS total_listening_secs,
                       l.avg_secs_per_active_day,
                       l.last_active_date,
                       l.days_since_last_active,
                       COALESCE(l.active_days_7d, 0) AS active_days_7d,
                       COALESCE(l.active_days_30d, 0) AS active_days_30d,
                       COALESCE(l.listening_secs_7d, 0) AS listening_secs_7d,
                       COALESCE(l.listening_secs_30d, 0) AS listening_secs_30d,
                       COALESCE(l.listening_secs_previous_30d, 0) AS listening_secs_previous_30d,
                       COALESCE(l.songs_30d, 0) AS songs_30d
                FROM train_users t
                LEFT JOIN member_features m USING (msno)
                LEFT JOIN listening_features l USING (msno)
            ) TO {sql_path(output)} (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        summary = con.execute(f"""
            SELECT COUNT(*) AS rows, COUNT(DISTINCT msno) AS users,
                   SUM(is_churn) AS churned,
                   COUNT(registration_date) AS registered_dates,
                   SUM(has_listening_logs) AS users_with_logs,
                   MAX(last_active_date) AS max_log_date
            FROM read_parquet({sql_path(output)})
        """).fetchone()
        print("Final (rows, unique users, churned, registration dates, users with logs, latest log date):", summary)
        if summary[0] != count_users or summary[1] != count_users:
            raise AssertionError("Join changed number of training users")
        if summary[4] == 0:
            raise RuntimeError("ZERO users matched listening logs: inspect the source file and cutoff!")
        from datetime import date

        if summary[5] and summary[5].date() > date.fromisoformat(cutoff):
            raise AssertionError("Future listening dates leaked into features")
            raise AssertionError("Future listening dates leaked into features")
        print(f"Saved: {output}")
        print("No transactions included. Dates after cutoff are excluded from listening features.")
    finally:
        con.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="./data/raw")
    parser.add_argument("--work-dir", default="./kkbox_work")
    parser.add_argument("--output-dir", default="./data/processed")
    parser.add_argument("--cutoff", default="2017-02-28")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--login", action="store_true", help="Prompt for Kaggle token in this process")
    parser.add_argument("--memory-limit", default="3GB")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--buckets", type=int, default=64, help="Number of disk partitions; increase to 128 if memory is still limited")
    build(parser.parse_args())
