from types import SimpleNamespace

from langchain_core.messages import HumanMessage

from agent import graph


def test_codex_conversation_checkpoints_are_not_listed_as_chats(monkeypatch, tmp_path):
    monkeypatch.setattr(graph, "CHECKPOINT_DB_PATH", tmp_path / "checkpoints.sqlite")
    graph.CHECKPOINT_DB_PATH.touch()

    checkpoints = [
        SimpleNamespace(
            config={"configurable": {"thread_id": "user-1:codex-workspace"}},
            checkpoint={
                "channel_values": {
                    "messages": [HumanMessage(content="Explain this code")]
                }
            },
        ),
        SimpleNamespace(
            config={"configurable": {"thread_id": "user-1:codex-6f5b16d7"}},
            checkpoint={
                "channel_values": {
                    "messages": [HumanMessage(content="Refactor this function")]
                }
            },
        ),
        SimpleNamespace(
            config={"configurable": {"thread_id": "user-1:regular-chat"}},
            checkpoint={
                "channel_values": {
                    "messages": [HumanMessage(content="Help with this task")]
                }
            },
        ),
        SimpleNamespace(
            config={"configurable": {"thread_id": "user-2:regular-chat"}},
            checkpoint={
                "channel_values": {
                    "messages": [HumanMessage(content="Another account's chat")]
                }
            },
        ),
    ]

    class FakeCheckpointer:
        def list(self, _config):
            return checkpoints

    class FakeConnection:
        def close(self):
            pass

    monkeypatch.setattr(
        graph,
        "create_checkpointer",
        lambda: (FakeCheckpointer(), FakeConnection()),
    )

    assert graph.get_all_saved_threads("user-1") == {
        "regular-chat": "Help with this task"
    }
