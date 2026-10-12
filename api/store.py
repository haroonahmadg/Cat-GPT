"""
UI-level, per-user thread bookkeeping: pinning, soft-deletion,
renaming, and archiving. Also owns the users table.

Every function below now requires a user_id and filters all reads/
writes by it, so one person's pins/deletes/renames/archives can never
be seen or touched by another.
"""

from __future__ import annotations

import sqlite3
import uuid

from api.config import UI_DB_PATH
from agent.graph import CHECKPOINT_DB_PATH, create_checkpointer


def get_ui_connection() -> sqlite3.Connection:
    return sqlite3.connect(str(UI_DB_PATH), check_same_thread=False)


# ============================================================
# PER-USER TABLE DEFINITIONS
#
# user_id is deliberately nullable: rows that existed before
# multi-user auth keep user_id = NULL until
# scripts/migrate_existing_data.py assigns them to a real user.
# Every function below always writes a real user_id.
# ============================================================

_PER_USER_TABLES: dict[str, tuple[tuple[str, ...], str]] = {
    "pinned_threads": (
        ("thread_id", "user_id", "pinned_at"),
        """
        CREATE TABLE pinned_threads (
            thread_id TEXT NOT NULL,
            user_id TEXT,
            pinned_at TEXT,
            PRIMARY KEY (thread_id, user_id)
        )
        """,
    ),
    "deleted_threads": (
        ("thread_id", "user_id", "deleted_at"),
        """
        CREATE TABLE deleted_threads (
            thread_id TEXT NOT NULL,
            user_id TEXT,
            deleted_at TEXT,
            PRIMARY KEY (thread_id, user_id)
        )
        """,
    ),
    "renamed_threads": (
        ("thread_id", "user_id", "title", "renamed_at"),
        """
        CREATE TABLE renamed_threads (
            thread_id TEXT NOT NULL,
            user_id TEXT,
            title TEXT NOT NULL,
            renamed_at TEXT,
            PRIMARY KEY (thread_id, user_id)
        )
        """,
    ),
    "archived_threads": (
        ("thread_id", "user_id", "archived_at"),
        """
        CREATE TABLE archived_threads (
            thread_id TEXT NOT NULL,
            user_id TEXT,
            archived_at TEXT,
            PRIMARY KEY (thread_id, user_id)
        )
        """,
    ),
}


def _primary_key_columns(table_info: list) -> list[str]:
    """Primary-key column names, in key order, from PRAGMA table_info."""

    # PRAGMA row: (cid, name, type, notnull, default, pk_position)
    return [
        row[1]
        for row in sorted(
            (r for r in table_info if r[5] > 0),
            key=lambda r: r[5],
        )
    ]


def _ensure_per_user_table(
    connection: sqlite3.Connection,
    table: str,
    columns: tuple[str, ...],
    create_sql: str,
) -> None:
    """
    Create the table, or rebuild it when it still has the pre-auth
    schema (thread_id as the only primary key).

    Why a rebuild: CREATE TABLE IF NOT EXISTS is skipped when the old
    table exists, and ALTER TABLE ADD COLUMN cannot change a primary
    key. With the old key, INSERT OR REPLACE would let one user's pin,
    delete, archive or rename overwrite another user's row.

    Existing rows are preserved. The whole rebuild is one transaction,
    so a failure leaves the original table untouched. Safe to run on
    every startup: a table that is already correct is left alone.
    """

    # Make sure no transaction is already open before issuing BEGIN.
    connection.commit()

    info = connection.execute(f"PRAGMA table_info({table})").fetchall()

    if not info:
        connection.execute(create_sql)
        connection.commit()
        return

    existing = [row[1] for row in info]

    if (
        _primary_key_columns(info) == ["thread_id", "user_id"]
        and all(column in existing for column in columns)
    ):
        return

    if "thread_id" not in existing:
        raise RuntimeError(
            f"Table '{table}' has no thread_id column and cannot be migrated."
        )

    old_table = f"{table}_old"
    shared = ", ".join(column for column in columns if column in existing)

    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(f"DROP TABLE IF EXISTS {old_table}")
        connection.execute(f"ALTER TABLE {table} RENAME TO {old_table}")
        connection.execute(create_sql)
        connection.execute(
            f"INSERT OR IGNORE INTO {table} ({shared}) "
            f"SELECT {shared} FROM {old_table}"
        )
        connection.execute(f"DROP TABLE {old_table}")
        connection.commit()

    except Exception:
        connection.rollback()
        raise


class AccountExistsError(ValueError):
    """An account with this email address already exists."""


# ============================================================
# USERS TABLE
#
# google_id is nullable (accounts created with email + password have
# none) and password_hash is nullable (accounts created through Google
# have none). SQLite cannot drop a NOT NULL constraint with ALTER
# TABLE, so an older users table is rebuilt the same way the per-user
# thread tables are.
# ============================================================

_USERS_COLUMNS = (
    "user_id",
    "google_id",
    "email",
    "name",
    "password_hash",
    "created_at",
)

_USERS_CREATE_SQL = """
    CREATE TABLE users (
        user_id TEXT PRIMARY KEY,
        google_id TEXT UNIQUE,
        email TEXT NOT NULL,
        name TEXT,
        password_hash TEXT,
        created_at TEXT
    )
"""


def _ensure_users_table(connection: sqlite3.Connection) -> None:
    """
    Create the users table, or rebuild an older one (google_id NOT
    NULL, no password_hash). Rows are preserved, the rebuild is one
    transaction, and an already-correct table is left alone.
    """

    connection.commit()

    info = connection.execute("PRAGMA table_info(users)").fetchall()

    if not info:
        connection.execute(_USERS_CREATE_SQL)
        connection.commit()
        return

    by_name = {row[1]: row for row in info}

    # PRAGMA row: (cid, name, type, notnull, default, pk_position)
    if "password_hash" in by_name and by_name["google_id"][3] == 0:
        return

    old_table = "users_old"
    shared = ", ".join(column for column in _USERS_COLUMNS if column in by_name)

    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(f"DROP TABLE IF EXISTS {old_table}")
        connection.execute(f"ALTER TABLE users RENAME TO {old_table}")
        connection.execute(_USERS_CREATE_SQL)
        connection.execute(
            f"INSERT OR IGNORE INTO users ({shared}) "
            f"SELECT {shared} FROM {old_table}"
        )
        connection.execute(f"DROP TABLE {old_table}")
        connection.commit()

    except Exception:
        connection.rollback()
        raise


def initialize_ui_database() -> None:
    connection = get_ui_connection()

    try:
        _ensure_users_table(connection)

        # One account per email address, ignoring letter case. If old
        # data already contains duplicate emails the index cannot be
        # built; create_password_user() still checks for duplicates
        # itself, so new sign-ups stay protected either way.
        try:
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_lower "
                "ON users (lower(email))"
            )
            connection.commit()
        except sqlite3.IntegrityError:
            connection.rollback()

        for table, (columns, create_sql) in _PER_USER_TABLES.items():
            _ensure_per_user_table(connection, table, columns, create_sql)

        connection.execute(
            """
            UPDATE pinned_threads
            SET pinned_at = COALESCE(pinned_at, datetime('now'))
            WHERE pinned_at IS NULL
            """
        )

        connection.execute(
            """
            UPDATE deleted_threads
            SET deleted_at = COALESCE(deleted_at, datetime('now'))
            WHERE deleted_at IS NULL
            """
        )

        connection.commit()

    finally:
        connection.close()


# ============================================================
# USERS
# ============================================================

def get_or_create_user(google_id: str, email: str, name: str | None) -> dict:

    if not isinstance(google_id, str) or not google_id.strip():
        raise ValueError("A valid google_id is required.")

    if not isinstance(email, str) or not email.strip():
        raise ValueError("A valid email is required.")

    clean_google_id = google_id.strip()
    clean_email = email.strip()

    connection = get_ui_connection()

    try:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT user_id, email, name, password_hash
            FROM users WHERE google_id = ?
            """,
            (clean_google_id,),
        ).fetchone()
        if row:
            connection.commit()
            return {
                "user_id": row[0],
                "email": row[1],
                "name": row[2],
                "password_hash": row[3],
            }

        # Google has verified ownership of this address. Reuse an existing
        # password account with the same email so its chats and memories stay
        # attached to the same Cat GPT user.
        row = connection.execute(
            """
            SELECT user_id, email, name, password_hash, google_id
            FROM users WHERE lower(email) = lower(?)
            """,
            (clean_email,),
        ).fetchone()
        if row:
            if row[4] and row[4] != clean_google_id:
                raise ValueError(
                    "This Cat GPT email is already linked to a different Google account."
                )
            connection.execute(
                "UPDATE users SET google_id = ?, name = COALESCE(name, ?) WHERE user_id = ?",
                (clean_google_id, name, row[0]),
            )
            connection.commit()
            return {
                "user_id": row[0],
                "email": row[1],
                "name": row[2] or name,
                "password_hash": row[3],
            }

        new_user_id = uuid.uuid4().hex
        try:
            connection.execute(
                """
                INSERT INTO users (user_id, google_id, email, name, created_at)
                VALUES (?, ?, ?, ?, datetime('now'))
                """,
                (new_user_id, clean_google_id, clean_email, name),
            )
            connection.commit()

        except sqlite3.IntegrityError:
            connection.rollback()
            row = connection.execute(
                """
                SELECT user_id, email, name, password_hash
                FROM users WHERE google_id = ?
                """,
                (clean_google_id,),
            ).fetchone()
            if row:
                return {
                    "user_id": row[0],
                    "email": row[1],
                    "name": row[2],
                    "password_hash": row[3],
                }
            raise ValueError(
                "This Google account or email is already linked to another Cat GPT account."
            ) from None

        return {
            "user_id": new_user_id,
            "email": clean_email,
            "name": name,
            "password_hash": None,
        }
    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def get_all_users() -> list[dict]:
    """Used only by the one-off migration script."""

    connection = get_ui_connection()

    try:
        rows = connection.execute(
            "SELECT user_id, email, name FROM users ORDER BY created_at"
        ).fetchall()

        return [{"user_id": r[0], "email": r[1], "name": r[2]} for r in rows]

    finally:
        connection.close()


# ============================================================
# PASSWORD ACCOUNTS
#
# The dicts returned here include password_hash for auth.py's use
# only. They must never be sent to a client: the API response
# models expose email and name only.
# ============================================================

def _user_row_to_dict(row) -> dict:
    return {
        "user_id": row[0],
        "email": row[1],
        "name": row[2],
        "password_hash": row[3],
    }


def get_user_by_email(email: str) -> dict | None:

    if not isinstance(email, str) or not email.strip():
        return None

    connection = get_ui_connection()

    try:
        row = connection.execute(
            """
            SELECT user_id, email, name, password_hash FROM users
            WHERE lower(email) = lower(?)
            ORDER BY (password_hash IS NULL), created_at
            LIMIT 1
            """,
            (email.strip(),),
        ).fetchone()

        return _user_row_to_dict(row) if row else None

    finally:
        connection.close()


def get_user_by_id(user_id: str) -> dict | None:

    if not isinstance(user_id, str) or not user_id.strip():
        return None

    connection = get_ui_connection()

    try:
        row = connection.execute(
            """
            SELECT user_id, email, name, password_hash FROM users
            WHERE user_id = ?
            """,
            (user_id.strip(),),
        ).fetchone()

        return _user_row_to_dict(row) if row else None

    finally:
        connection.close()


def create_password_user(
    email: str,
    name: str | None,
    password_hash: str,
) -> dict:
    """
    Create an email + password account. Raises AccountExistsError when
    the email is already registered (by any method). The duplicate
    check and the insert run in one write transaction, so two
    simultaneous sign-ups for the same email cannot both succeed.
    """

    if not isinstance(email, str) or not email.strip():
        raise ValueError("A valid email is required.")

    if not isinstance(password_hash, str) or not password_hash:
        raise ValueError("A password hash is required.")

    clean_email = email.strip()
    new_user_id = uuid.uuid4().hex

    connection = get_ui_connection()

    try:
        connection.execute("BEGIN IMMEDIATE")

        exists = connection.execute(
            "SELECT 1 FROM users WHERE lower(email) = lower(?)",
            (clean_email,),
        ).fetchone()

        if exists:
            raise AccountExistsError(
                "An account with this email already exists."
            )

        connection.execute(
            """
            INSERT INTO users
            (user_id, google_id, email, name, password_hash, created_at)
            VALUES (?, NULL, ?, ?, ?, datetime('now'))
            """,
            (new_user_id, clean_email, name, password_hash),
        )
        connection.commit()

        return {"user_id": new_user_id, "email": clean_email, "name": name}

    except AccountExistsError:
        connection.rollback()
        raise

    except sqlite3.IntegrityError as exc:
        connection.rollback()
        raise AccountExistsError(
            "An account with this email already exists."
        ) from exc

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def set_password_hash(user_id: str, password_hash: str) -> None:

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user_id is required.")

    if not isinstance(password_hash, str) or not password_hash:
        raise ValueError("A password hash is required.")

    connection = get_ui_connection()

    try:
        cursor = connection.execute(
            "UPDATE users SET password_hash = ? WHERE user_id = ?",
            (password_hash, user_id.strip()),
        )

        if cursor.rowcount == 0:
            connection.rollback()
            raise ValueError("User not found.")

        connection.commit()

    finally:
        connection.close()


# ============================================================
# PIN / UNPIN
# ============================================================

def pin_thread(thread_id: str, user_id: str) -> None:

    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("A valid thread_id is required.")

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user_id is required.")

    connection = get_ui_connection()

    try:
        connection.execute(
            """
            INSERT OR REPLACE INTO pinned_threads (thread_id, user_id, pinned_at)
            VALUES (?, ?, datetime('now'))
            """,
            (thread_id.strip(), user_id.strip()),
        )
        connection.commit()

    finally:
        connection.close()


def unpin_thread(thread_id: str, user_id: str) -> None:

    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("A valid thread_id is required.")

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user_id is required.")

    connection = get_ui_connection()

    try:
        connection.execute(
            "DELETE FROM pinned_threads WHERE thread_id = ? AND user_id = ?",
            (thread_id.strip(), user_id.strip()),
        )
        connection.commit()

    finally:
        connection.close()


def get_pinned_threads(user_id: str) -> set[str]:

    if not isinstance(user_id, str) or not user_id.strip():
        return set()

    connection = get_ui_connection()

    try:
        rows = connection.execute(
            """
            SELECT thread_id FROM pinned_threads
            WHERE user_id = ?
            ORDER BY COALESCE(pinned_at, '') DESC, rowid DESC
            """,
            (user_id.strip(),),
        ).fetchall()

        return {row[0] for row in rows}

    finally:
        connection.close()


# ============================================================
# RENAME
# ============================================================

def rename_thread(thread_id: str, user_id: str, title: str) -> None:

    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("A valid thread_id is required.")

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user_id is required.")

    if not isinstance(title, str) or not title.strip():
        raise ValueError("Title cannot be empty.")

    clean_title = title.strip()

    if len(clean_title) > 200:
        raise ValueError("Title is too long (max 200 characters).")

    connection = get_ui_connection()

    try:
        connection.execute(
            """
            INSERT OR REPLACE INTO renamed_threads
            (thread_id, user_id, title, renamed_at)
            VALUES (?, ?, ?, datetime('now'))
            """,
            (thread_id.strip(), user_id.strip(), clean_title),
        )
        connection.commit()

    finally:
        connection.close()


def get_renamed_titles(user_id: str) -> dict[str, str]:

    if not isinstance(user_id, str) or not user_id.strip():
        return {}

    connection = get_ui_connection()

    try:
        rows = connection.execute(
            "SELECT thread_id, title FROM renamed_threads WHERE user_id = ?",
            (user_id.strip(),),
        ).fetchall()

        return {row[0]: row[1] for row in rows}

    finally:
        connection.close()


# ============================================================
# ARCHIVE
# ============================================================

def archive_thread(thread_id: str, user_id: str) -> None:

    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("A valid thread_id is required.")

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user_id is required.")

    connection = get_ui_connection()

    try:
        connection.execute(
            """
            INSERT OR REPLACE INTO archived_threads
            (thread_id, user_id, archived_at)
            VALUES (?, ?, datetime('now'))
            """,
            (thread_id.strip(), user_id.strip()),
        )
        connection.commit()

    finally:
        connection.close()


def unarchive_thread(thread_id: str, user_id: str) -> None:

    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("A valid thread_id is required.")

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user_id is required.")

    connection = get_ui_connection()

    try:
        connection.execute(
            "DELETE FROM archived_threads WHERE thread_id = ? AND user_id = ?",
            (thread_id.strip(), user_id.strip()),
        )
        connection.commit()

    finally:
        connection.close()


def get_archived_threads(user_id: str) -> set[str]:

    if not isinstance(user_id, str) or not user_id.strip():
        return set()

    connection = get_ui_connection()

    try:
        rows = connection.execute(
            "SELECT thread_id FROM archived_threads WHERE user_id = ?",
            (user_id.strip(),),
        ).fetchall()

        return {row[0] for row in rows}

    finally:
        connection.close()


# ============================================================
# DELETE
# ============================================================

def mark_thread_deleted(thread_id: str, user_id: str) -> None:

    if not isinstance(thread_id, str) or not thread_id.strip():
        raise ValueError("A valid thread_id is required.")

    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("A valid user_id is required.")

    connection = get_ui_connection()

    try:
        connection.execute(
            """
            INSERT OR REPLACE INTO deleted_threads (thread_id, user_id, deleted_at)
            VALUES (?, ?, datetime('now'))
            """,
            (thread_id.strip(), user_id.strip()),
        )

        connection.execute(
            "DELETE FROM pinned_threads WHERE thread_id = ? AND user_id = ?",
            (thread_id.strip(), user_id.strip()),
        )

        connection.execute(
            "DELETE FROM renamed_threads WHERE thread_id = ? AND user_id = ?",
            (thread_id.strip(), user_id.strip()),
        )

        connection.execute(
            "DELETE FROM archived_threads WHERE thread_id = ? AND user_id = ?",
            (thread_id.strip(), user_id.strip()),
        )

        connection.commit()

    finally:
        connection.close()


def get_deleted_threads(user_id: str) -> set[str]:

    if not isinstance(user_id, str) or not user_id.strip():
        return set()

    connection = get_ui_connection()

    try:
        rows = connection.execute(
            "SELECT thread_id FROM deleted_threads WHERE user_id = ?",
            (user_id.strip(),),
        ).fetchall()

        return {row[0] for row in rows}

    finally:
        connection.close()


def permanently_delete_checkpoint_thread(
    thread_id: str,
) -> tuple[bool, str | None]:

    if not thread_id or not CHECKPOINT_DB_PATH.exists():
        return True, None

    connection = None
    cleanup_error = None

    try:
        checkpointer, connection = create_checkpointer()

        delete_method = getattr(checkpointer, "delete_thread", None)

        if callable(delete_method):
            delete_method(thread_id)
            return True, None

    except Exception as exc:
        cleanup_error = str(exc)

    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    return False, cleanup_error or (
        "The installed checkpointer does not expose delete_thread()."
    )


def delete_thread(thread_id: str, user_id: str) -> tuple[bool, str | None]:

    if not isinstance(thread_id, str) or not thread_id.strip():
        return False, "Invalid thread ID."

    if not isinstance(user_id, str) or not user_id.strip():
        return False, "Invalid user ID."

    try:
        mark_thread_deleted(thread_id.strip(), user_id.strip())
    except Exception as exc:
        return False, f"Could not mark the chat as deleted: {exc}"

    cleanup_ok, cleanup_error = permanently_delete_checkpoint_thread(
        thread_id.strip()
    )

    return True, cleanup_error if not cleanup_ok else None