import math
from typing import Literal
from langchain_core.tools import tool

SIGNIFICANT_DIGITS = 12

@tool
def calculator(
    a: float, 
    b: float, 
    operation: Literal[
        "add", 
        "subtract", 
        "multiply", 
        "divide"
        ]
    ) -> float | str:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, subtract, multiply, divide.
    Use this for any exact calculation instead of doing the maths yourself.
    """
    if operation == "add":
        result = a + b
    elif operation == "subtract":
        result = a - b
    elif operation == "multiply":
        result = a * b
    elif operation == "divide":
        if b == 0:
            return "Error: Cannot divide by zero."
        result = a / b
    else:
        return f"Error: Unsupported operation: {operation}"

    if not math.isfinite(result):
        return "Error: Result is not a valid number."

    return float(f"{result:.{SIGNIFICANT_DIGITS}g}")