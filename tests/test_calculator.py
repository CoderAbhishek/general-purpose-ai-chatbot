import pytest
from pydantic import ValidationError

from general_purpose_ai_chatbot.chatbot.tools.calculator import calculator


# --- Core operations -------------------------------------------------------

@pytest.mark.parametrize(
    ("a", "b", "operation", "expected"),
    [
        (6, 3, "add", 9.0),
        (6, 3, "subtract", 3.0),
        (6, 3, "multiply", 18.0),
        (6, 3, "divide", 2.0),
        (-2, 5, "add", 3.0),
        (-4, -2, "multiply", 8.0),
    ],
)
def test_operations(a, b, operation, expected):
    assert calculator.invoke({"a": a, "b": b, "operation": operation}) == expected


# --- Zero handling ---------------------------------------------------------

@pytest.mark.parametrize("zero", [0, 0.0, -0.0])
def test_divide_by_zero_returns_error(zero):
    result = calculator.invoke({"a": 5, "b": zero, "operation": "divide"})
    assert result.startswith("Error:")
    assert "divide by zero" in result.lower()


def test_multiply_by_zero_is_allowed():
    assert calculator.invoke({"a": 5, "b": 0, "operation": "multiply"}) == 0.0


# --- Input validation by the schema ----------------------------------------

def test_numeric_strings_are_converted():
    assert calculator.invoke({"a": "5", "b": "2", "operation": "add"}) == 7.0


def test_invalid_operation_rejected_by_schema():
    with pytest.raises(ValidationError):
        calculator.invoke({"a": 5, "b": 2, "operation": "modulo"})


def test_fallback_for_unsupported_operation_when_called_directly():
    # .func is the plain Python function underneath @tool; it skips schema validation.
    result = calculator.func(5, 2, "modulo")
    assert result.startswith("Error:")


# --- Rounding --------------------------------------------------------------

def test_rounding_removes_float_noise():
    assert calculator.invoke({"a": 0.1, "b": 0.2, "operation": "add"}) == 0.3


def test_rounding_keeps_small_values():
    assert calculator.invoke({"a": 0.00001, "b": 2, "operation": "multiply"}) == 2e-05


def test_rounding_uses_twelve_significant_digits():
    assert calculator.invoke({"a": 1, "b": 3, "operation": "divide"}) == 0.333333333333


# --- Infinity and NaN ------------------------------------------------------

def test_overflow_returns_error():
    result = calculator.invoke({"a": 1e308, "b": 10, "operation": "multiply"})
    assert result.startswith("Error:")


def test_nan_result_returns_error():
    result = calculator.invoke({"a": float("inf"), "b": float("inf"), "operation": "subtract"})
    assert result.startswith("Error:")


# --- What the model sees ---------------------------------------------------

def test_tool_schema():
    assert calculator.name == "calculator"
    assert set(calculator.args) == {"a", "b", "operation"}
    assert calculator.args["operation"]["enum"] == ["add", "subtract", "multiply", "divide"]
    assert "exact calculation" in calculator.description
