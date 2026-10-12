from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from agent import graph


def test_model_candidates_use_configured_primary_for_legacy_deployments(monkeypatch):
    monkeypatch.setenv("GROQ_MODEL", "custom/primary")
    monkeypatch.delenv("GROQ_MODELS", raising=False)

    assert graph.get_text_model_candidates() == ["custom/primary"]


def test_model_candidates_require_configuration(monkeypatch):
    monkeypatch.delenv("GROQ_MODELS", raising=False)
    monkeypatch.delenv("GROQ_MODEL", raising=False)

    with pytest.raises(RuntimeError, match="GROQ_MODELS is not configured"):
        graph.get_text_model_candidates()


def test_model_candidates_validate_count_and_deduplicate(monkeypatch):
    monkeypatch.setenv("GROQ_MODELS", "model-a, model-b, model-a")
    assert graph.get_text_model_candidates() == ["model-a", "model-b"]

    monkeypatch.setenv("GROQ_MODELS", "a,b,c,d,e,f,g")
    with pytest.raises(RuntimeError, match="at most 6"):
        graph.get_text_model_candidates()


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("429 rate_limit_exceeded"),
        RuntimeError("model_not_found: model does not exist"),
    ],
)
def test_assistant_switches_to_next_model(failure, monkeypatch):
    monkeypatch.setattr(graph, "build_system_prompt", lambda: "system")
    monkeypatch.setattr(graph, "get_text_model_candidates", lambda: ["model-a", "model-b"])
    monkeypatch.setattr(graph, "_get_max_tokens", lambda _name, default: int(default))
    monkeypatch.setattr(graph, "emit_tool_event", lambda *_args, **_kwargs: None)

    selected_models = []

    class FakeRunnable:
        def __init__(self, model):
            self.model = model

        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            selected_models.append(self.model)
            if self.model == "model-a":
                raise failure
            return SimpleNamespace(content="fallback answer")

    monkeypatch.setattr(graph, "get_llm", FakeRunnable)
    monkeypatch.setattr(graph, "recover_tool_call", lambda *_args: None)

    result = graph.assistant_node({"messages": []})
    assert selected_models == ["model-a", "model-b"]
    assert result["messages"][0].content == "fallback answer"


def test_vision_model_does_not_use_text_fallbacks(monkeypatch):
    monkeypatch.setattr(graph, "build_system_prompt", lambda: "system")
    monkeypatch.setattr(graph, "get_vision_model_name", lambda: "vision-model")
    monkeypatch.setattr(graph, "get_text_model_candidates", lambda: ["text-a", "text-b"])
    monkeypatch.setattr(graph, "_get_max_tokens", lambda _name, default: int(default))
    monkeypatch.setattr(graph, "emit_tool_event", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(graph.time, "sleep", lambda _seconds: None)
    selected_models = []

    class FakeRunnable:
        def __init__(self, model):
            self.model = model

        def bind_tools(self, _tools):
            return self

        def invoke(self, _messages):
            selected_models.append(self.model)
            raise RuntimeError("429 rate_limit_exceeded")

    monkeypatch.setattr(graph, "get_llm", FakeRunnable)
    monkeypatch.setattr(graph, "recover_tool_call", lambda *_args: None)

    result = graph.assistant_node(
        {"messages": [HumanMessage(content=[{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}])]}
    )
    assert selected_models == ["vision-model"]
    assert "rate limit" in result["messages"][0].content.lower()
