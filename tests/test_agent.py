import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import END
from langgraph.prebuilt import tools_condition

from agent.graph import parse_input_messages


def test_parse_input_messages_valid():
    raw_messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    parsed = parse_input_messages(raw_messages)
    assert len(parsed) == 2
    assert isinstance(parsed[0], HumanMessage)
    assert isinstance(parsed[1], AIMessage)
    assert parsed[0].content == "Hello!"


def test_parse_input_messages_invalid():
    with pytest.raises(ValueError, match="Messages cannot be empty"):
        parse_input_messages([])

    with pytest.raises(ValueError, match="No valid messages were provided"):
        parse_input_messages([{"role": "system", "content": ""}])


def test_tools_condition_ends_without_tool_calls():
    state = {"messages": [AIMessage(content="Final answer.")]}
    next_node = tools_condition(state)
    assert next_node == END


def test_tools_condition_routes_tool_calls():
    msg = AIMessage(
        content="",
        tool_calls=[{"name": "calculator", "args": {"expression": "2+2"}, "id": "1"}],
    )
    state = {"messages": [msg]}
    next_node = tools_condition(state)
    assert next_node == "tools"