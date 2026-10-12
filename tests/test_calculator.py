from agent.tools import calculator


def test_basic_addition():
    result = calculator.invoke({"expression": "25 + 75"})
    assert result == "100"


def test_multiplication():
    result = calculator.invoke({"expression": "12 * 8"})
    assert result == "96"


def test_parentheses():
    result = calculator.invoke({"expression": "(10 + 5) * 2"})
    assert result == "30"


def test_square_root():
    result = calculator.invoke({"expression": "sqrt(144)"})
    assert result == "12"


def test_division_by_zero():
    result = calculator.invoke({"expression": "10 / 0"})
    assert "division by zero" in result.lower()


def test_empty_expression():
    result = calculator.invoke({"expression": ""})
    assert "expression" in result.lower()


def test_unsafe_expression():
    result = calculator.invoke(
        {"expression": "__import__('os').system('whoami')"}
    )
    assert "error" in result.lower()