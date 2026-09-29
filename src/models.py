# ABOUTME: Pydantic models that represent every data structure in the project.
# ABOUTME: Covers function definitions, prompts, and output results.

from typing import Any, Dict

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Function definition schema (read from functions_definition.json)
# ---------------------------------------------------------------------------

class ParameterSchema(BaseModel):
    """Schema for a single function parameter.

    Attributes:
        type: JSON-schema-like type name, e.g. "number", "string", "boolean".
    """

    type: str


class FunctionDefinition(BaseModel):
    """Full definition of one callable function.

    Attributes:
        name: Identifier used to call the function (e.g. "fn_add_numbers").
        description: Human-readable description sent to the LLM as context.
        parameters: Mapping of parameter name → its schema.
        returns: Schema of the return value (used for documentation only).
    """

    name: str
    description: str
    parameters: Dict[str, ParameterSchema]
    returns: ParameterSchema


# ---------------------------------------------------------------------------
# Input prompt (read from function_calling_tests.json)
# ---------------------------------------------------------------------------

class InputPrompt(BaseModel):
    """A single natural-language prompt to be resolved into a function call.

    Attributes:
        prompt: The raw user request, e.g. "What is the sum of 2 and 3?".
    """

    prompt: str


# ---------------------------------------------------------------------------
# Output result (written to function_calling_results.json)
# ---------------------------------------------------------------------------

class FunctionCallResult(BaseModel):
    """The structured function call produced for one prompt.

    Attributes:
        prompt: The original natural-language request (echoed back).
        name: Name of the chosen function.
        parameters: Mapping of argument name → extracted value (typed correctly).
    """

    prompt: str
    name: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
