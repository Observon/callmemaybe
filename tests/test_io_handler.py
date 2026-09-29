# ABOUTME: Unit tests for io_handler — covers happy paths and all error branches.

import json
from pathlib import Path


from src.io_handler import load_function_definitions, load_input_prompts, write_results
from src.models import FunctionCallResult


class TestLoadFunctionDefinitions:
    def test_valid_file(self, tmp_path: Path) -> None:
        data = [
            {
                "name": "fn_add_numbers",
                "description": "Add two numbers.",
                "parameters": {"a": {"type": "number"}, "b": {"type": "number"}},
                "returns": {"type": "number"},
            }
        ]
        f = tmp_path / "fns.json"
        f.write_text(json.dumps(data))
        result = load_function_definitions(f)
        assert len(result) == 1
        assert result[0].name == "fn_add_numbers"

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = load_function_definitions(tmp_path / "nope.json")
        assert result == []

    def test_invalid_json_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("NOT JSON {{{{")
        result = load_function_definitions(f)
        assert result == []

    def test_skips_invalid_entries(self, tmp_path: Path) -> None:
        data = [
            {
                "name": "fn_ok", "description": "ok",
                "parameters": {}, "returns": {"type": "string"},
            },
            {"BROKEN": True},
        ]
        f = tmp_path / "mixed.json"
        f.write_text(json.dumps(data))
        result = load_function_definitions(f)
        assert len(result) == 1


class TestLoadInputPrompts:
    def test_valid_file(self, tmp_path: Path) -> None:
        data = [{"prompt": "Greet shrek"}, {"prompt": "Add 2 and 3"}]
        f = tmp_path / "prompts.json"
        f.write_text(json.dumps(data))
        result = load_input_prompts(f)
        assert len(result) == 2
        assert result[0].prompt == "Greet shrek"

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = load_input_prompts(Path("/nonexistent/file.json"))
        assert result == []

    def test_invalid_json_returns_empty(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("[{broken")
        result = load_input_prompts(f)
        assert result == []


class TestWriteResults:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        results = [
            FunctionCallResult(
                prompt="What is 2+3?",
                name="fn_add_numbers",
                parameters={"a": 2.0, "b": 3.0},
            )
        ]
        out = tmp_path / "out" / "results.json"
        write_results(results, out)
        assert out.exists()
        parsed = json.loads(out.read_text())
        assert parsed[0]["name"] == "fn_add_numbers"
        assert parsed[0]["parameters"]["a"] == 2.0

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        out = tmp_path / "deep" / "nested" / "file.json"
        write_results([], out)
        assert out.exists()
