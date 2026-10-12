import os
import sqlite3
import pytest
from database.db import (
    get_connection,
    init_db,
    save_message,
    get_messages,
    save_memory,
    get_memories,
    delete_thread,
)


@pytest.fixture(autouse=True)
def setup_test_db(tmp_path, monkeypatch):
    test_db_path = str(tmp_path / "test_assistant.db")
    test_db_dir = str(tmp_path)

    monkeypatch.setattr("database.db.DB_PATH", test_db_path)
    monkeypatch.setattr("database.db.DB_DIR", test_db_dir)

    init_db()
    yield
    if os.path.exists(test_db_path):
        os.remove(test_db_path)


def test_wal_mode_enabled():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode;")
    mode = cursor.fetchone()[0]
    conn.close()
    assert mode.lower() == "wal"


def test_save_and_get_messages():
    thread_id = "test_thread_100"
    save_message(thread_id, "user", "Hello Assistant")
    save_message(thread_id, "assistant", "Hello! How can I help?")

    messages = get_messages(thread_id)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello Assistant"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Hello! How can I help?"


def test_save_and_get_memories():
    save_memory("user_name", "Alex")
    save_memory("favorite_language", "Python")

    memories = get_memories()
    memory_dict = dict(memories)

    assert memory_dict.get("user_name") == "Alex"
    assert memory_dict.get("favorite_language") == "Python"


def test_delete_thread_cascades():
    thread_id = "delete_me_thread"
    save_message(thread_id, "user", "Message to delete")
    delete_thread(thread_id)

    messages = get_messages(thread_id)
    assert len(messages) == 0