# ABOUTME: Unit tests for Pydantic models — validates schema parsing and edge cases.

import pytest
from pydantic import ValidationError

from src.models import (
    FunctionCallResult, FunctionDefinition, InputPrompt, ParameterSchema,
)


class TestParameterSchema:
    def test_valid(self) -> None:
        p = ParameterSchema(type="number")
        assert p.type == "number"

    def test_missing_type_raises(self) -> None:
        with pytest.raises(ValidationError):
            ParameterSchema.model_validate({})


class TestFunctionDefinition:
    def test_valid(self) -> None:
        fn = FunctionDefinition.model_validate({
            "name": "fn_add_numbers",
            "description": "Add two numbers.",
            "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
            "returns": {"type": "number"},
        })
        assert fn.name == "fn_add_numbers"
        assert fn.parameters["a"].type == "number"

    def test_missing_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            FunctionDefinition.model_validate({
                "description": "x",
                "parameters": {},
                "returns": {"type": "string"},
            })


class TestInputPrompt:
    def test_valid(self) -> None:
        p = InputPrompt(prompt="Greet shrek")
        assert p.prompt == "Greet shrek"

    def test_empty_prompt_allowed(self) -> None:
        p = InputPrompt(prompt="")
        assert p.prompt == ""


class TestFunctionCallResult:
    def test_valid(self) -> None:
        r = FunctionCallResult(
            prompt="What is 2+3?",
            name="fn_add_numbers",
            parameters={"a": 2.0, "b": 3.0},
        )
        assert r.name == "fn_add_numbers"
        assert r.parameters["a"] == 2.0

    def test_default_empty_parameters(self) -> None:
        r = FunctionCallResult(prompt="x", name="fn_greet")
        assert r.parameters == {}
