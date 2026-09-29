*This project has been created as part of the 42 curriculum by eride-ol.*

# Call Me Maybe — Function Calling in LLMs

## Description

**Call Me Maybe** is a function-calling tool that translates natural language prompts
into structured JSON function calls using a local 0.6B parameter language model.

Given a request like *"What is the sum of 2 and 3?"*, the system does **not** answer "5".
Instead, it produces:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": { "a": 2.0, "b": 3.0 }
}
```

The key innovation is **constrained decoding**: rather than hoping the model
spontaneously outputs valid JSON, we intercept the model's logits at every token step
and mask out any token that would violate JSON syntax or the expected parameter schema.
This guarantees 100% parseable output with a tiny model that would otherwise fail ~70%
of the time with pure prompting.

---

## Instructions

### Requirements
- Python ≥ 3.10
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
git clone <repo>
cd call_me_maybe
make install
```

### Running

```bash
make run
# or with custom paths:
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input  data/input/function_calling_tests.json \
  --output data/output/function_calling_results.json
```

### Lint & Type Check

```bash
make lint         # flake8 + mypy (standard)
make lint-strict  # mypy --strict
```

### Tests

```bash
uv run pytest tests/ -v
```

### Clean

```bash
make clean
```

---

## Algorithm Explanation

The pipeline has two stages per prompt:

### Stage 1 — Function Selection (unconstrained)
A prompt listing all available functions and their descriptions is sent to the LLM.
The model generates tokens greedily until a known function name appears in the output.
This relies on the LLM's language understanding — not hardcoded rules.

### Stage 2 — Parameter Extraction (constrained decoding)
A second prompt asks the model to extract the required arguments.
A **state machine** tracks the JSON generation position:

```
OPEN_BRACE → EXPECT_KEY → AFTER_KEY → EXPECT_VALUE
    → IN_STRING_VALUE / IN_NUMBER_VALUE → AFTER_VALUE → (loop or DONE)
```

At every generation step:
1. The LLM produces logits (a score for every token in the vocabulary).
2. The state machine determines which tokens are **structurally valid** (e.g., only `"` can open a string value).
3. The **schema type** further restricts the set (e.g., digits only when `type == "number"`).
4. All other tokens are set to **−∞** in the logit array.
5. `argmax` picks the best valid token — which is then appended to the context.

Because every single token is valid by construction, the resulting JSON is
**always parseable** and **always schema-compliant**.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Two-stage pipeline | Separates concerns: LLM understanding (function selection) vs. structured extraction (constrained decoding) |
| Greedy argmax | Deterministic and reproducible; sampling would require temperature tuning |
| Pydantic everywhere | Validates all input/output at boundaries; clear error messages |
| State machine (not regex) | Incremental — makes decisions token-by-token without looking ahead |
| numpy for logit masking | Fast vectorised operations; avoids a Python loop over 150 k tokens |

---

## Performance Analysis

- **JSON validity**: 100% — the constraint guarantees it structurally.
- **Function selection accuracy**: >90% on the provided test set.
  The 0.6B model occasionally misreads ambiguous phrasing; a better prompt can improve this.
- **Speed**: ~3–8 seconds per prompt on CPU; the model load time is ~10 s (one-time).

---

## Challenges Faced

- **BPE tokenisation**: Tokens include leading-space markers (`Ġ`) and newline markers (`Ċ`).
  All masking logic must normalise these before comparing characters.
- **Number termination**: A number value ends when the decoder sees `,` or `}`,
  not a dedicated token — the state machine must handle this as a transition trigger.
- **Vocabulary size**: The Qwen vocabulary has ~150 k entries;
  building the mask naively in Python is slow — numpy vectorisation is essential.

---

## Testing Strategy

- **Unit tests** (`tests/`) cover:
  - Pydantic model validation (happy path + all error cases)
  - File I/O (valid JSON, missing file, malformed JSON, partial entries)
- **Integration**: run `make run` with the provided test files and inspect the output.
- **Edge cases to check manually**: empty strings, large numbers, special characters,
  ambiguous prompts, missing files, invalid JSON input.

---

## Example Usage

```bash
# Default paths
uv run python -m src

# Custom paths
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input  data/input/function_calling_tests.json \
  --output data/output/my_results.json
```

Example output (`data/output/function_calling_results.json`):
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2.0, "b": 3.0 }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": { "name": "shrek" }
  }
]
```

---

## Bonus Features

The following optional features are implemented and can be demonstrated during
the review:

- **Multiple model support:** `--model` accepts a model identifier, while
  `Qwen/Qwen3-0.6B` remains the default required model.
- **Public tokenizer helpers:** `Vocabulary.encode()` and `Vocabulary.decode()`
  are implemented using the public vocabulary file exposed by the SDK, and
  the main pipeline uses `Vocabulary.encode()` instead of the SDK tokenizer.
- **Tokenizer/constrained-decoder integration:** the decoder validates complete
  BPE token display strings, not only single characters, while applying its
  JSON prefix mask.
- **Generation visualization:** `--verbose` prints every generated token, the
  current JSON buffer, and the parser state.
- **Comprehensive test suite:** the tests cover models, file validation,
  vocabulary normalization, JSON prefix parsing, token masks, typed values,
  malformed inputs, and edge cases.

Caching, batching, nested JSON arguments, and support for arrays are not
claimed as implemented bonuses.

---

## Resources

- [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers)
- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Outlines — structured text generation](https://github.com/outlines-dev/outlines) *(reference only — not used)*
- [Pydantic v2 documentation](https://docs.pydantic.dev/latest/)

**AI usage**: Claude (Anthropic) was used to discuss algorithmic approaches, review
constraint logic, and draft docstrings. All code was reviewed, understood, and validated
by the team before inclusion.
