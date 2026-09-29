# ABOUTME: Constrained decoder — generates JSON token-by-token, masking invalid tokens.
# ABOUTME: Uses a prefix-parser approach: each candidate token is validated by checking
# ABOUTME: whether buffer + token remains a valid JSON prefix for the target schema.
# ABOUTME: This correctly handles compound BPE tokens of arbitrary length.

import json
import math
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from llm_sdk import Small_LLM_Model
from src.models import FunctionDefinition


# ---------------------------------------------------------------------------
# BPE normalisation
# ---------------------------------------------------------------------------

def _norm(token_str: str) -> str:
    """Normalise a BPE token string to its display character sequence.

    Qwen (and most HuggingFace BPE models) use:
      Ġ  → leading space
      Ċ  → newline

    Args:
        token_str: Raw token string from the vocabulary file.

    Returns:
        Human-readable character sequence the token represents.
    """
    return token_str.replace("Ġ", " ").replace("Ċ", "\n")


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

class Vocabulary:
    """Token-id ↔ token-string mapping loaded from the model's vocab file.

    This is built from ``get_path_to_vocab_file()`` so we never access
    private tokeniser internals.

    Args:
        model: An initialised Small_LLM_Model.
    """

    def __init__(self, model: Small_LLM_Model) -> None:
        vocab_path = model.get_path_to_vocab_file()
        with open(vocab_path, encoding="utf-8") as fh:
            raw: Dict[str, int] = json.load(fh)

        self._id_to_raw: Dict[int, str] = {v: k for k, v in raw.items()}
        self._id_to_display: Dict[int, str] = {
            v: _norm(k) for k, v in raw.items()
        }
        self._display_to_id: Dict[str, int] = {
            _norm(k): v for k, v in raw.items()
        }
        self._size: int = len(raw)

    # -- public ---------------------------------------------------------------

    @property
    def size(self) -> int:
        """Total vocabulary size."""
        return self._size

    def display(self, token_id: int) -> Optional[str]:
        """Return the normalised (display) string for a token id."""
        return self._id_to_display.get(token_id)

    def ids(self) -> range:
        """Iterator over all valid token ids."""
        return range(self._size)

    # -- bonus: public encode/decode -----------------------------------------

    def encode(self, text: str) -> List[int]:
        """Encode a text string into token ids using the vocabulary.

        Implements a greedy longest-match BPE encoding over the display
        representations of the vocabulary.  This is the bonus implementation
        that avoids calling ``model.encode()`` directly.

        Args:
            text: Input string to tokenise.

        Returns:
            List of token ids.
        """
        ids: List[int] = []
        i = 0
        while i < len(text):
            # Greedy longest match
            best_id: Optional[int] = None
            best_len = 0
            for length in range(min(20, len(text) - i), 0, -1):
                candidate = text[i:i + length]
                tid = self._display_to_id.get(candidate)
                if tid is not None and length > best_len:
                    best_id = tid
                    best_len = length
                    break
            if best_id is not None:
                ids.append(best_id)
                i += best_len
            else:
                raise ValueError(
                    f"Vocabulary cannot encode character at position {i}: "
                    f"{text[i]!r}"
                )
        return ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode a list of token ids back into a string.

        Args:
            token_ids: Sequence of token ids to decode.

        Returns:
            Reconstructed text string.
        """
        return "".join(
            self._id_to_display.get(tid, "") for tid in token_ids
        )


# ---------------------------------------------------------------------------
# JSON prefix parser — the heart of the constraint logic
# ---------------------------------------------------------------------------

class _ParseState:
    """Result of parsing a partial JSON buffer.

    Attributes:
        state: Current position in the JSON generation state machine.
        param_idx: Index of the parameter currently being targeted.
        current_key: Name of the key currently being written (if any).
        value_buf: Characters accumulated for the current value.
        in_escape: Whether the last character was a backslash in a string.
    """

    __slots__ = ("state", "param_idx", "current_key", "value_buf", "in_escape")

    def __init__(
        self,
        state: str,
        param_idx: int,
        current_key: Optional[str],
        value_buf: str,
        in_escape: bool,
    ) -> None:
        self.state = state
        self.param_idx = param_idx
        self.current_key = current_key
        self.value_buf = value_buf
        self.in_escape = in_escape


def _parse_prefix(
    buf: str,
    param_names: List[str],
    param_types: Dict[str, str],
) -> Optional[_ParseState]:
    """Parse a partial JSON buffer character by character.

    Validates that ``buf`` is a valid prefix of the expected JSON object:
    ``{"param1": <value1>, "param2": <value2>, ...}``
    where each value type is constrained by ``param_types``.

    Args:
        buf: Accumulated JSON characters generated so far.
        param_names: Ordered list of expected parameter names.
        param_types: Mapping of parameter name → type string.

    Returns:
        A ``_ParseState`` describing where we are in the generation,
        or ``None`` if ``buf`` is not a valid prefix (token must be masked).
    """
    state = "OPEN_BRACE"
    current_key: Optional[str] = None
    value_buf = ""
    param_idx = 0
    in_escape = False
    unicode_escape_remaining = 0
    key_buf = ""

    for c in buf:
        if state == "OPEN_BRACE":
            if c == "{":
                state = "EXPECT_EMPTY_CLOSE" if not param_names else "EXPECT_KEY_QUOTE"
            elif c != " ":
                return None

        elif state == "EXPECT_EMPTY_CLOSE":
            if c == "}":
                state = "DONE"
            elif c not in (" ", "\n"):
                return None

        elif state == "EXPECT_KEY_QUOTE":
            if c == "\"":
                state = "IN_KEY"
                key_buf = ""
            elif c not in (" ", "\n"):
                return None

        elif state == "IN_KEY":
            if c == "\"":
                current_key = key_buf
                # Validate the key is the one we expect
                if param_idx >= len(param_names):
                    return None
                if current_key != param_names[param_idx]:
                    return None
                state = "EXPECT_COLON"
            else:
                key_buf += c
                # Validate the key is a valid prefix of the expected name
                expected = (
                    param_names[param_idx] if param_idx < len(param_names) else ""
                )
                if not expected.startswith(key_buf):
                    return None

        elif state == "EXPECT_COLON":
            if c == ":":
                state = "EXPECT_VALUE"
            elif c not in (" ", "\n"):
                return None

        elif state == "EXPECT_VALUE":
            if c in (" ", "\n"):
                pass  # skip whitespace before value
            elif c == "\"":
                ptype = param_types.get(current_key or "", "string")
                if ptype == "number":
                    return None
                state = "IN_STRING"
                value_buf = ""
            elif c in "-0123456789":
                ptype = param_types.get(current_key or "", "string")
                if ptype == "boolean":
                    return None
                if ptype != "number":
                    return None
                state = "IN_NUMBER"
                value_buf = c
            elif c in "tf":
                ptype = param_types.get(current_key or "", "string")
                if ptype != "boolean":
                    return None
                state = "IN_BOOLEAN"
                value_buf = c
            else:
                return None

        elif state == "IN_STRING":
            if unicode_escape_remaining:
                if c not in "0123456789abcdefABCDEF":
                    return None
                unicode_escape_remaining -= 1
            elif in_escape:
                if c not in '"\\/bfnrtu':
                    return None
                value_buf += c
                in_escape = False
                if c == "u":
                    unicode_escape_remaining = 4
            elif c == "\\":
                in_escape = True
            elif c == "\"":
                if unicode_escape_remaining:
                    return None
                param_idx += 1
                state = "AFTER_VALUE"
            elif ord(c) < 0x20:
                return None  # raw newlines invalid inside JSON strings
            else:
                value_buf += c

        elif state == "IN_NUMBER":
            if c in (",", "}"):
                if not re.fullmatch(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", value_buf):
                    return None
                param_idx += 1
                if c == "}":
                    if param_idx != len(param_names):
                        return None  # closed too early
                    state = "DONE"
                else:
                    state = "EXPECT_KEY_QUOTE"
            elif c == ".":
                if "." in value_buf:
                    return None  # second dot is invalid
                value_buf += c
            elif c in "0123456789":
                value_buf += c
            else:
                return None

        elif state == "IN_BOOLEAN":
            if c in "truefalse":
                value_buf += c
                if value_buf not in ("true", "false") and not (
                    "true".startswith(value_buf) or "false".startswith(value_buf)
                ):
                    return None
            elif c in (",", "}"):
                if value_buf not in ("true", "false"):
                    return None
                param_idx += 1
                if c == "}":
                    if param_idx != len(param_names):
                        return None
                    state = "DONE"
                else:
                    state = "EXPECT_KEY_QUOTE"
            else:
                return None

        elif state == "AFTER_VALUE":
            if c == "}":
                if param_idx != len(param_names):
                    return None
                state = "DONE"
            elif c == ",":
                state = "EXPECT_KEY_QUOTE"
            elif c not in (" ", "\n"):
                return None

        elif state == "DONE":
            return None  # nothing valid after the closing brace

    return _ParseState(state, param_idx, current_key, value_buf, in_escape)


# ---------------------------------------------------------------------------
# Constrained Decoder
# ---------------------------------------------------------------------------

class ConstrainedDecoder:
    """Generates a JSON parameters object via constrained decoding.

    At every generation step:
    1. Get logits for all vocabulary tokens from the LLM.
    2. For each candidate token, check whether ``current_buf + token``
       is still a valid prefix of the expected JSON (using ``_parse_prefix``).
    3. Set invalid-token logits to ``-inf``.
    4. Pick the token with the highest remaining logit (greedy argmax).
    5. Append to buffer and context, repeat until state == DONE.

    This approach correctly handles compound BPE tokens of any length.

    Args:
        model: Initialised Small_LLM_Model.
        vocab: Vocabulary built from the same model.
        max_new_tokens: Safety cap to prevent infinite loops.
        verbose: If True, print each generated token for debugging.
    """

    def __init__(
        self,
        model: Small_LLM_Model,
        vocab: Vocabulary,
        max_new_tokens: int = 256,
        verbose: bool = False,
    ) -> None:
        self._model = model
        self._vocab = vocab
        self._max_new_tokens = max_new_tokens
        self._verbose = verbose

    def generate(
        self,
        prompt_ids: List[int],
        fn_def: FunctionDefinition,
    ) -> Tuple[Dict[str, Any], str]:
        """Generate parameter values for the given function from a prompt.

        Args:
            prompt_ids: Token ids of the full prompt (already encoded).
            fn_def: The function whose parameters must be extracted.

        Returns:
            A tuple of:
            - dict mapping parameter names to their typed values
            - the raw JSON string that was generated
        """
        param_names = list(fn_def.parameters.keys())
        param_types = {k: v.type for k, v in fn_def.parameters.items()}

        context_ids: List[int] = list(prompt_ids)
        json_buf = ""   # accumulates the JSON characters generated so far

        for step in range(self._max_new_tokens):
            # Parse current buffer to check if we're already done
            ps = _parse_prefix(json_buf, param_names, param_types)
            if ps is not None and ps.state == "DONE":
                break

            logits = self._model.get_logits_from_input_ids(context_ids)
            logits_arr = np.array(logits, dtype=np.float64)

            # Build mask: for each token, check if buf + token is valid prefix
            valid_mask = self._build_mask(json_buf, param_names, param_types)
            logits_arr[~valid_mask] = -math.inf

            if not np.any(np.isfinite(logits_arr)):
                print(
                    f"[WARN] No valid token at step {step}, buf={json_buf!r}",
                    flush=True,
                )
                break

            chosen_id = int(np.argmax(logits_arr))
            chosen_display = self._vocab.display(chosen_id) or ""

            context_ids.append(chosen_id)
            json_buf += chosen_display

            if self._verbose:
                ps_now = _parse_prefix(json_buf, param_names, param_types)
                state_name = ps_now.state if ps_now else "INVALID"
                print(
                    f"  step {step:3d}: token={chosen_display!r:15} "
                    f"buf={json_buf!r:40} state={state_name}"
                )

        final_state = _parse_prefix(json_buf, param_names, param_types)
        if final_state is None or final_state.state != "DONE":
            raise ValueError(
                f"Constrained decoding did not produce complete JSON: {json_buf!r}"
            )

        values = self._extract_values(json_buf, param_names, param_types)
        if set(values) != set(param_names):
            raise ValueError(
                f"Generated JSON is missing required parameters: {json_buf!r}"
            )
        return values, json_buf

    # ------------------------------------------------------------------
    # Mask builder
    # ------------------------------------------------------------------

    def _build_mask(
        self,
        json_buf: str,
        param_names: List[str],
        param_types: Dict[str, str],
    ) -> np.ndarray:
        """Build a boolean mask of valid next tokens.

        A token is valid iff ``json_buf + token_display`` remains a valid
        JSON prefix for the target schema.

        Args:
            json_buf: Accumulated JSON string so far.
            param_names: Ordered parameter names.
            param_types: Parameter name → type string.

        Returns:
            Boolean numpy array of shape (vocab_size,).
        """
        mask = np.zeros(self._vocab.size, dtype=bool)

        for tid in self._vocab.ids():
            display = self._vocab.display(tid)
            if display is None or display == "":
                continue

            candidate_buf = json_buf + display
            result = _parse_prefix(candidate_buf, param_names, param_types)
            mask[tid] = result is not None

        return mask

    # ------------------------------------------------------------------
    # Value extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_values(
        json_buf: str,
        param_names: List[str],
        param_types: Dict[str, str],
    ) -> Dict[str, Any]:
        """Parse the completed JSON buffer and extract typed parameter values.

        Falls back to an empty dict if the buffer cannot be parsed.

        Args:
            json_buf: The full generated JSON string.
            param_names: Ordered parameter names.
            param_types: Parameter name → type string.

        Returns:
            Dict mapping parameter names to their Python-typed values.
        """
        try:
            raw: Dict[str, Any] = json.loads(json_buf)
        except json.JSONDecodeError:
            # Try to extract partial results via the prefix parser
            ps = _parse_prefix(json_buf + "}", param_names, param_types)
            if ps is None:
                return {}
            try:
                raw = json.loads(json_buf + "}")
            except json.JSONDecodeError:
                return {}

        result: Dict[str, Any] = {}
        for name in param_names:
            if name not in raw:
                continue
            ptype = param_types.get(name, "string")
            val = raw[name]
            if ptype == "number":
                try:
                    result[name] = float(val)
                except (TypeError, ValueError):
                    continue
            elif ptype == "boolean":
                if isinstance(val, bool):
                    result[name] = val
            else:
                result[name] = str(val)
        return result
