"""
One-off migration: assigns all pre-Stage-2 data (threads, pins,
renames, archives, RAG documents, long-term memories) to a single
specified user.

Run manually, once, after your account exists:
    python scripts/migrate_existing_data.py your-email@gmail.com
"""

from __future__ import annotations

import pickle
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.store import get_ui_connection, get_all_users
from agent.graph import CHECKPOINT_DB_PATH, create_checkpointer
from rag.ingest import METADATA_PATH
from database.db import get_connection as get_memories_connection


def find_user_by_email(email: str) -> dict | None:
    for user in get_all_users():
        if user["email"].lower() == email.lower():
            return user
    return None


def migrate_ui_tables(user_id: str) -> None:
    connection = get_ui_connection()
    try:
        for table in (
            "pinned_threads",
            "deleted_threads",
            "renamed_threads",
            "archived_threads",
        ):
            connection.execute(
                f"UPDATE {table} SET user_id = ? WHERE user_id IS NULL",
                (user_id,),
            )
        connection.commit()
        print("UI tables (pins/deletes/renames/archives) migrated.")
    finally:
        connection.close()


def migrate_checkpoint_threads(user_id: str) -> None:
    if not CHECKPOINT_DB_PATH.exists():
        print("No checkpoint database found — nothing to migrate.")
        return

    connection = sqlite3.connect(str(CHECKPOINT_DB_PATH))
    try:
        cursor = connection.cursor()
        tables = [
            row[0]
            for row in cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        ]

        for table in tables:
            columns = {
                row[1]
                for row in cursor.execute(f"PRAGMA table_info({table})").fetchall()
            }
            if "thread_id" not in columns:
                continue

            rows = cursor.execute(
                f"SELECT DISTINCT thread_id FROM {table}"
            ).fetchall()

            for (thread_id,) in rows:
                if thread_id.startswith(f"{user_id}:"):
                    continue
                if ":" in thread_id and thread_id.split(":", 1)[0]:
                    continue
                new_id = f"{user_id}:{thread_id}"
                cursor.execute(
                    f"UPDATE {table} SET thread_id = ? WHERE thread_id = ?",
                    (new_id, thread_id),
                )

        connection.commit()
        print("Checkpoint thread IDs migrated.")
    finally:
        connection.close()


def migrate_rag_metadata(user_id: str) -> None:
    if not METADATA_PATH.exists():
        print("No RAG metadata found — nothing to migrate.")
        return

    with open(METADATA_PATH, "rb") as f:
        registry = pickle.load(f)

    if not isinstance(registry, list):
        print("RAG metadata was not a list — skipping.")
        return

    changed = 0
    for entry in registry:
        if isinstance(entry, dict) and not entry.get("user_id"):
            entry["user_id"] = user_id
            changed += 1

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(registry, f)

    print(f"RAG metadata: {changed} chunk(s) assigned to user.")


def migrate_memories(user_id: str) -> None:
    connection = get_memories_connection()
    try:
        cursor = connection.cursor()
        cursor.execute(
            "UPDATE memories SET user_id = ? WHERE user_id IS NULL",
            (user_id,),
        )
        connection.commit()
        print(f"Memories: {cursor.rowcount} row(s) assigned to user.")
    finally:
        connection.close()


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/migrate_existing_data.py <your-email>")
        sys.exit(1)

    email = sys.argv[1]
    user = find_user_by_email(email)

    if not user:
        print(
            f"No user found with email '{email}'. "
            "Sign in with Google at least once first, then re-run this script."
        )
        sys.exit(1)

    user_id = user["user_id"]
    print(f"Migrating existing data to user_id={user_id} ({email})...")

    migrate_ui_tables(user_id)
    migrate_checkpoint_threads(user_id)
    migrate_rag_metadata(user_id)
    migrate_memories(user_id)

    print("Migration complete.")


if __name__ == "__main__":
    main()