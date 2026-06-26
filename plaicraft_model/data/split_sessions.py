#!/usr/bin/env python3
"""Create a high-quality validation SQLite DB from the full training DB.

Selection rule:
- sample 100 players from the full DB whose player_id exists in the high-quality training DB
- for each selected player, sample exactly one session from the full DB
- the sampled session must not exist in the high-quality training DB

The output DB is a new SQLite file with the same `session_metadata` schema and
the same explicit indexes as the high-quality training DB.
"""
from __future__ import annotations

import argparse
import logging
import os
import random
import sqlite3
from typing import Dict, List, Sequence, Tuple


LOGGER = logging.getLogger("split_sessions")


def get_table_columns(conn: sqlite3.Connection, table_name: str) -> List[str]:
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info('{table_name}')")
    return [row[1] for row in cursor.fetchall()]


def get_sqlite_master_sql(conn: sqlite3.Connection, object_type: str, object_name: str) -> str:
    cursor = conn.cursor()
    cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type = ? AND name = ?",
        (object_type, object_name),
    )
    row = cursor.fetchone()
    if row is None or row[0] is None:
        raise RuntimeError(f"Missing {object_type} {object_name!r} in reference database")
    return row[0]


def create_output_db(reference_db: str, output_db: str, table_name: str) -> None:
    if os.path.exists(output_db):
        raise FileExistsError(f"Output DB already exists: {output_db}")

    with sqlite3.connect(reference_db) as reference_conn, sqlite3.connect(output_db) as output_conn:
        output_cursor = output_conn.cursor()
        output_cursor.execute(get_sqlite_master_sql(reference_conn, "table", table_name))

        reference_cursor = reference_conn.cursor()
        reference_cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'index' AND tbl_name = ? AND sql IS NOT NULL ORDER BY name",
            (table_name,),
        )
        for (index_sql,) in reference_cursor.fetchall():
            output_cursor.execute(index_sql)

        output_conn.commit()


def get_candidate_sessions(full_db: str, high_quality_db: str, table_name: str) -> List[Tuple[str, int]]:
    with sqlite3.connect(full_db) as full_conn:
        full_conn.execute("ATTACH DATABASE ? AS hq", (high_quality_db,))
        cursor = full_conn.cursor()
        cursor.execute(
            f"""
            SELECT f.session_id, f.player_id
            FROM {table_name} AS f
            WHERE f.session_id NOT IN (SELECT session_id FROM hq.{table_name})
              AND f.player_id IN (SELECT DISTINCT player_id FROM hq.{table_name})
            """
        )
        return [(row[0], row[1]) for row in cursor.fetchall()]


def group_sessions_by_player(candidate_sessions: Sequence[Tuple[str, int]]) -> Dict[int, List[str]]:
    grouped: Dict[int, List[str]] = {}
    for session_id, player_id in candidate_sessions:
        grouped.setdefault(player_id, []).append(session_id)
    return grouped


def insert_selected_sessions(
    output_db: str,
    full_db: str,
    table_name: str,
    columns: Sequence[str],
    session_ids: Sequence[str],
) -> None:
    if not session_ids:
        return

    column_list = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(session_ids))
    insert_sql = (
        f"INSERT INTO {table_name} ({column_list}) "
        f"SELECT {column_list} FROM full.{table_name} WHERE session_id IN ({placeholders})"
    )

    with sqlite3.connect(output_db) as output_conn:
        output_conn.execute("ATTACH DATABASE ? AS full", (full_db,))
        output_conn.execute("BEGIN")
        output_conn.execute(insert_sql, list(session_ids))
        output_conn.commit()


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a high-quality validation DB from a full training DB"
    )
    parser.add_argument(
        "--high-quality-db",
        default="data/high_quality_1.5k_hrs_1105_players_training.db",
        help="Path to the high-quality training DB used as the schema/index reference",
    )
    parser.add_argument(
        "--full-db",
        default="data/latest_9.5k_hrs_12619_players_training.db",
        help="Path to the full training DB containing candidate sessions",
    )
    parser.add_argument(
        "--output-db",
        default="data/high_quality_validation.db",
        help="Path for the new validation DB to create",
    )
    parser.add_argument("--validation-count", type=int, default=100, help="Number of sessions to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(message)s")

    if args.seed is not None:
        random.seed(args.seed)

    LOGGER.info("Reference DB: %s", args.high_quality_db)
    LOGGER.info("Full DB: %s", args.full_db)
    LOGGER.info("Output DB: %s", args.output_db)

    if not os.path.exists(args.high_quality_db):
        LOGGER.error("High-quality DB does not exist: %s", args.high_quality_db)
        return 2
    if not os.path.exists(args.full_db):
        LOGGER.error("Full DB does not exist: %s", args.full_db)
        return 2
    if os.path.exists(args.output_db):
        LOGGER.error("Output DB already exists, refusing to overwrite: %s", args.output_db)
        return 2

    candidate_sessions = get_candidate_sessions(args.full_db, args.high_quality_db, "session_metadata")
    LOGGER.info("Eligible candidate sessions: %d", len(candidate_sessions))

    if len(candidate_sessions) < args.validation_count:
        LOGGER.error(
            "Not enough eligible sessions: need %d, found %d",
            args.validation_count,
            len(candidate_sessions),
        )
        return 3

    candidate_sessions_by_player = group_sessions_by_player(candidate_sessions)
    eligible_players = sorted(candidate_sessions_by_player)

    if len(eligible_players) < args.validation_count:
        LOGGER.error(
            "Not enough eligible players: need %d, found %d",
            args.validation_count,
            len(eligible_players),
        )
        return 3

    selected_player_ids = random.sample(eligible_players, args.validation_count)
    selected_session_ids = [random.choice(candidate_sessions_by_player[player_id]) for player_id in selected_player_ids]
    selected_player_ids = sorted(selected_player_ids)

    LOGGER.info("Eligible players: %d", len(eligible_players))
    LOGGER.info("Selected validation sessions: %d", len(selected_session_ids))
    LOGGER.info("Selected validation players: %d", len(selected_player_ids))
    LOGGER.info("Sample selected session ids: %s", selected_session_ids[:10])

    # Use the full DB as the schema/index reference so the validation DB
    # matches the full training DB's schema (this ensures columns like
    # `player_age` present in the full DB are preserved in the output).
    schema_db = args.full_db
    create_output_db(schema_db, args.output_db, "session_metadata")
    with sqlite3.connect(schema_db) as reference_conn:
        reference_columns = get_table_columns(reference_conn, "session_metadata")

    insert_selected_sessions(
        output_db=args.output_db,
        full_db=args.full_db,
        table_name="session_metadata",
        columns=reference_columns,
        session_ids=selected_session_ids,
    )

    with sqlite3.connect(args.output_db) as output_conn:
        cursor = output_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM session_metadata")
        LOGGER.info("Validation DB rows inserted: %d", cursor.fetchone()[0])
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND tbl_name = 'session_metadata' AND sql IS NOT NULL")
        LOGGER.info("Explicit indexes copied: %d", cursor.fetchone()[0])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
