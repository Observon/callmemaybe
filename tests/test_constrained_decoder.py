# ABOUTME: Tests for the constrained decoder's prefix parser and mask logic.
# ABOUTME: All tests are model-free — they test the JSON parsing and masking
# ABOUTME: logic directly, using a mock vocabulary.

from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from src.constrained_decoder import (
    ConstrainedDecoder,
    Vocabulary,
    _ParseState,
    _parse_prefix,
    _norm,
)
from src.models import FunctionDefinition, ParameterSchema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_fn(name: str, params: Dict[str, str]) -> FunctionDefinition:
    """Helper: build a FunctionDefinition with given param types."""
    return FunctionDefinition(
        name=name,
        description="test",
        parameters={k: ParameterSchema(type=v) for k, v in params.items()},
        returns=ParameterSchema(type="string"),
    )


# ---------------------------------------------------------------------------
# _norm
# ---------------------------------------------------------------------------

class TestNorm:
    def test_space_prefix(self) -> None:
        assert _norm("Ġhello") == " hello"

    def test_newline(self) -> None:
        assert _norm("Ċ") == "\n"

    def test_no_special(self) -> None:
        assert _norm("abc") == "abc"

    def test_mixed(self) -> None:
        assert _norm("ĠhelloĊ") == " hello\n"


# ---------------------------------------------------------------------------
# _parse_prefix — structural validity
# ---------------------------------------------------------------------------

class TestParsePrefix:
    NAMES = ["a", "b"]
    TYPES = {"a": "number", "b": "number"}

    def _p(self, buf: str) -> Optional[_ParseState]:
        return _parse_prefix(buf, self.NAMES, self.TYPES)

    # happy path
    def test_empty_is_open_brace(self) -> None:
        ps = self._p("")
        assert ps is not None
        assert ps.state == "OPEN_BRACE"

    def test_open_brace(self) -> None:
        ps = self._p("{")
        assert ps is not None
        assert ps.state == "EXPECT_KEY_QUOTE"

    def test_open_quote(self) -> None:
        ps = self._p("{\"")
        assert ps is not None
        assert ps.state == "IN_KEY"

    def test_partial_key(self) -> None:
        ps = self._p("{\"a")
        assert ps is not None
        assert ps.state == "IN_KEY"

    def test_closed_key(self) -> None:
        ps = self._p("{\"a\"")
        assert ps is not None
        assert ps.state == "EXPECT_COLON"

    def test_colon(self) -> None:
        ps = self._p("{\"a\":")
        assert ps is not None
        assert ps.state == "EXPECT_VALUE"

    def test_number_start(self) -> None:
        ps = self._p("{\"a\": 2")
        assert ps is not None
        assert ps.state == "IN_NUMBER"
        assert ps.value_buf == "2"

    def test_number_float(self) -> None:
        ps = self._p("{\"a\": 2.5")
        assert ps is not None
        assert ps.value_buf == "2.5"

    def test_double_dot_invalid(self) -> None:
        assert self._p("{\"a\": 2..5") is None

    def test_comma_advances_param(self) -> None:
        ps = self._p("{\"a\": 2.0,")
        assert ps is not None
        assert ps.param_idx == 1
        assert ps.state == "EXPECT_KEY_QUOTE"

    def test_full_two_params(self) -> None:
        ps = self._p("{\"a\": 2.0, \"b\": 3.0}")
        assert ps is not None
        assert ps.state == "DONE"

    # invalid cases
    def test_wrong_key_invalid(self) -> None:
        assert self._p("{\"x\"") is None

    def test_wrong_key_prefix_invalid(self) -> None:
        assert self._p("{\"z") is None

    def test_string_in_number_field_invalid(self) -> None:
        assert self._p("{\"a\": \"hello\"") is None

    def test_number_in_string_field_invalid(self) -> None:
        names = ["s"]
        types = {"s": "string"}
        assert _parse_prefix("{\"s\": 42", names, types) is None

    def test_early_close_invalid(self) -> None:
        # Closing before all params are written
        assert self._p("{\"a\": 1}") is None  # param_idx=1 != len=2

    def test_correct_close(self) -> None:
        # Single param closes correctly
        names = ["a"]
        types = {"a": "number"}
        ps = _parse_prefix("{\"a\": 1}", names, types)
        assert ps is not None
        assert ps.state == "DONE"

    def test_empty_parameters(self) -> None:
        ps = _parse_prefix("{}", [], {})
        assert ps is not None
        assert ps.state == "DONE"

    def test_incomplete_number_cannot_close(self) -> None:
        assert self._p('{"a": -}') is None
        assert self._p('{"a": 1.}') is None

    def test_boolean_value(self) -> None:
        ps = _parse_prefix('{"enabled": true}', ["enabled"], {"enabled": "boolean"})
        assert ps is not None
        assert ps.state == "DONE"

    def test_string_value(self) -> None:
        names = ["s"]
        types = {"s": "string"}
        ps = _parse_prefix("{\"s\": \"hello\"}", names, types)
        assert ps is not None
        assert ps.state == "DONE"

    def test_string_with_escape(self) -> None:
        names = ["s"]
        types = {"s": "string"}
        ps = _parse_prefix("{\"s\": \"hel\\\"lo\"}", names, types)
        assert ps is not None
        assert ps.state == "DONE"

    def test_raw_newline_in_string_invalid(self) -> None:
        names = ["s"]
        types = {"s": "string"}
        assert _parse_prefix("{\"s\": \"hel\nlo\"}", names, types) is None

    def test_compound_token_open_and_key(self) -> None:
        # A BPE token might be the entire '{\"a\"' as one token
        ps = self._p("{\"a\"")
        assert ps is not None
        assert ps.state == "EXPECT_COLON"

    def test_colon_with_space(self) -> None:
        ps = self._p("{\"a\": ")
        assert ps is not None
        assert ps.state == "EXPECT_VALUE"


# ---------------------------------------------------------------------------
# Vocabulary (mock-based)
# ---------------------------------------------------------------------------

class MockVocab:
    """Minimal vocabulary for testing the decoder without a real model."""

    def __init__(self, tokens: Dict[str, int]) -> None:
        self._d2i = tokens
        self._i2d = {v: k for k, v in tokens.items()}

    @property
    def size(self) -> int:
        return len(self._d2i)

    def display(self, tid: int) -> Optional[str]:
        return self._i2d.get(tid)

    def ids(self) -> range:
        return range(self.size)

    def encode(self, text: str) -> List[int]:
        return []

    def decode(self, ids: List[int]) -> str:
        return ""


# ---------------------------------------------------------------------------
# ConstrainedDecoder._build_mask  (unit test, no model needed)
# ---------------------------------------------------------------------------

class TestBuildMask:
    """Test the mask builder in isolation with a tiny synthetic vocabulary."""

    def _make_decoder(self, tokens: Dict[str, int]) -> ConstrainedDecoder:
        model_mock = MagicMock()
        vocab = MockVocab(tokens)
        # Patch the decoder to use our mock vocab
        decoder = object.__new__(ConstrainedDecoder)
        decoder._model = model_mock
        decoder._vocab = vocab  # type: ignore[assignment]
        decoder._max_new_tokens = 64
        decoder._verbose = False
        return decoder

    def test_open_brace_only_allows_brace(self) -> None:
        tokens = {"{": 0, "\"": 1, "a": 2, "}": 3}
        decoder = self._make_decoder(tokens)
        mask = decoder._build_mask("", ["x"], {"x": "number"})
        assert mask[0]   # '{' allowed
        assert not mask[1]  # '"' not allowed
        assert not mask[2]  # 'a' not allowed

    def test_after_brace_only_quote(self) -> None:
        tokens = {"{": 0, "\"": 1, "a": 2}
        decoder = self._make_decoder(tokens)
        mask = decoder._build_mask("{", ["x"], {"x": "number"})
        # Only '"' starts a valid next step
        assert mask[1]
        assert not mask[0]
        assert not mask[2]

    def test_in_number_allows_digits_and_terminator(self) -> None:
        tokens = {"0": 0, ".": 1, ",": 2, "}": 3, "\"": 4, "x": 5}
        decoder = self._make_decoder(tokens)
        # After the full first number, only '}' closes (single param)
        mask = decoder._build_mask("{\"x\": 2", ["x"], {"x": "number"})
        assert mask[0]   # '0' valid digit
        assert mask[1]   # '.' valid (no dot yet)
        assert mask[3]   # '}' closes (single param)
        assert not mask[4]  # '"' not valid in number

    def test_string_value_blocks_newline(self) -> None:
        tokens = {"a": 0, "\n": 1, "\"": 2}
        decoder = self._make_decoder(tokens)
        mask = decoder._build_mask("{\"s\": \"hel", ["s"], {"s": "string"})
        assert mask[0]    # 'a' valid inside string
        assert not mask[1]  # '\n' invalid inside JSON string
        assert mask[2]    # '"' closes string


# ---------------------------------------------------------------------------
# ConstrainedDecoder._extract_values
# ---------------------------------------------------------------------------

class TestExtractValues:
    def test_two_numbers(self) -> None:
        result = ConstrainedDecoder._extract_values(
            "{\"a\": 2.0, \"b\": 3.0}", ["a", "b"], {"a": "number", "b": "number"}
        )
        assert result["a"] == pytest.approx(2.0)
        assert result["b"] == pytest.approx(3.0)

    def test_string_value(self) -> None:
        result = ConstrainedDecoder._extract_values(
            "{\"name\": \"shrek\"}", ["name"], {"name": "string"}
        )
        assert result["name"] == "shrek"

    def test_invalid_json_returns_empty(self) -> None:
        result = ConstrainedDecoder._extract_values(
            "NOT JSON", ["a"], {"a": "number"}
        )
        assert result == {}

    def test_number_coercion(self) -> None:
        # Integer in JSON → float in result
        result = ConstrainedDecoder._extract_values(
            "{\"a\": 42}", ["a"], {"a": "number"}
        )
        assert result["a"] == pytest.approx(42.0)
        assert isinstance(result["a"], float)

    def test_boolean_value_preserved(self) -> None:
        result = ConstrainedDecoder._extract_values(
            '{"enabled": true}', ["enabled"], {"enabled": "boolean"}
        )
        assert result == {"enabled": True}


# ---------------------------------------------------------------------------
# Vocabulary.encode / decode  (bonus: own tokeniser)
# ---------------------------------------------------------------------------

class TestVocabularyEncodeDecode:
    def _make_vocab(self) -> Vocabulary:
        model_mock = MagicMock()
        model_mock.get_path_to_vocab_file.return_value = "/tmp/_test_vocab.json"
        import json as _json
        import pathlib
        tokens = {
            "{": 0, "}": 1, "\"": 2, ":": 3, ",": 4,
            " ": 5, "hello": 6, "world": 7, "Ġhello": 8,
        }
        pathlib.Path("/tmp/_test_vocab.json").write_text(_json.dumps(tokens))
        return Vocabulary(model_mock)

    def test_decode_basic(self) -> None:
        vocab = self._make_vocab()
        text = vocab.decode([0, 2, 6])
        assert text == "{\"hello"

    def test_decode_space_token(self) -> None:
        vocab = self._make_vocab()
        # token 8 is "Ġhello" → display " hello"
        text = vocab.decode([8])
        assert text == " hello"

    def test_encode_known_token(self) -> None:
        vocab = self._make_vocab()
        ids = vocab.encode("hello")
        assert 6 in ids

    def test_roundtrip(self) -> None:
        vocab = self._make_vocab()
        ids = vocab.encode("hello")
        text = vocab.decode(ids)
        assert "hello" in text
