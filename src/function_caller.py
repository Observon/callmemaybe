# ABOUTME: Orchestrates one full function-call resolution.
# ABOUTME: Stage 1: unconstrained LLM generation to select the function.
# ABOUTME: Stage 2: constrained decoding to extract typed parameter values.
# ABOUTME: Keeps selection and parameter extraction in separate stages.

from typing import List, Tuple

import numpy as np

from llm_sdk import Small_LLM_Model

from src.constrained_decoder import ConstrainedDecoder, Vocabulary
from src.models import FunctionCallResult, FunctionDefinition, InputPrompt


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_selection_prompt(
    user_prompt: str,
    functions: List[FunctionDefinition],
) -> str:
    """Build a prompt that asks the LLM to name the function to call.

    Lists all functions with descriptions so the model can make an informed
    choice.  The expected output is a single function name on one line.

    Args:
        user_prompt: The user's natural-language request.
        functions: All available function definitions.

    Returns:
        Formatted prompt string ready to be tokenised.
    """
    fn_list = "\n".join(
        f"- {fn.name}: {fn.description}" for fn in functions
    )
    return (
        f"You are a function dispatcher. Given a user request, output ONLY "
        f"the name of the function to call — nothing else.\n\n"
        f"Available functions:\n{fn_list}\n\n"
        f"User request: {user_prompt}\n\n"
        f"Function name:"
    )


def _build_parameter_prompt(
    user_prompt: str,
    fn_def: FunctionDefinition,
) -> str:
    """Build a prompt for constrained parameter extraction.

    The constrained decoder will generate a JSON object immediately after
    this prompt, so we only need to provide enough context for the model
    to choose correct argument values.

    Args:
        user_prompt: The user's natural-language request.
        fn_def: The function whose parameters must be filled in.

    Returns:
        Formatted prompt string.
    """
    param_desc = ", ".join(
        f"{name} ({schema.type})"
        for name, schema in fn_def.parameters.items()
    )
    return (
        f"Extract the arguments for the function call.\n"
        f"Function: {fn_def.name}\n"
        f"Description: {fn_def.description}\n"
        f"Parameters: {param_desc}\n"
        f"User request: {user_prompt}\n"
        f"JSON arguments:"
    )


# ---------------------------------------------------------------------------
# Function selector (stage 1 — unconstrained generation)
# ---------------------------------------------------------------------------

def _select_function(
    model: Small_LLM_Model,
    vocab: Vocabulary,
    user_prompt: str,
    functions: List[FunctionDefinition],
) -> Tuple[FunctionDefinition, str]:
    """Use the LLM to select the best-matching function for the request.

    Generates tokens greedily (no mask) until a known function name appears
    in the accumulated output or until a newline is produced.

    Falls back to the first function if no name is recognised.

    Args:
        model: Initialised language model.
        vocab: Vocabulary for decoding generated tokens.
        user_prompt: The user's natural-language request.
        functions: All available functions.

    Returns:
        A tuple of (selected FunctionDefinition, generated text).
    """
    prompt = _build_selection_prompt(user_prompt, functions)
    context_ids = vocab.encode(prompt)

    generated = ""
    for _ in range(48):
        logits = model.get_logits_from_input_ids(context_ids)
        logits_arr = np.array(logits, dtype=np.float64)
        chosen_id = int(np.argmax(logits_arr))
        display = vocab.display(chosen_id) or ""
        context_ids.append(chosen_id)
        generated += display

        # Match as soon as a function name appears
        for fn in functions:
            if fn.name in generated:
                return fn, generated

        # The model finished its one-line answer
        if "\n" in display or generated.count("\n") > 1:
            break

    # Fallback: substring scan of full generated text
    for fn in functions:
        if fn.name in generated:
            return fn, generated

    return functions[0], generated


# ---------------------------------------------------------------------------
# FunctionCaller — public API
# ---------------------------------------------------------------------------

class FunctionCaller:
    """Resolves natural-language prompts to structured FunctionCallResult objects.

    Uses a two-stage pipeline:
    1. Unconstrained generation to select the function.
    2. Constrained decoding to extract typed parameters.

    Args:
        model: An initialised Small_LLM_Model.
        functions: All available function definitions.
        verbose: If True, print per-token generation details.
    """

    def __init__(
        self,
        model: Small_LLM_Model,
        functions: List[FunctionDefinition],
        verbose: bool = False,
    ) -> None:
        self._model = model
        self._functions = functions
        self._vocab = Vocabulary(model)
        self._decoder = ConstrainedDecoder(model, self._vocab, verbose=verbose)

    def resolve(self, entry: InputPrompt) -> FunctionCallResult:
        """Resolve one natural-language prompt into a structured function call.

        Args:
            entry: The input prompt to process.

        Returns:
            A FunctionCallResult with prompt, function name, and parameters.
        """
        # Stage 1 — function selection
        fn_def, sel_text = _select_function(
            self._model, self._vocab, entry.prompt, self._functions
        )
        print(f"  → selected: {fn_def.name}  (from: {sel_text.strip()!r})")

        # Stage 2 — parameter extraction via constrained decoding
        param_prompt = _build_parameter_prompt(entry.prompt, fn_def)
        prompt_ids = self._vocab.encode(param_prompt)
        parameters, raw_json = self._decoder.generate(prompt_ids, fn_def)
        print(f"  → parameters JSON: {raw_json!r}")

        return FunctionCallResult(
            prompt=entry.prompt,
            name=fn_def.name,
            parameters=parameters,
        )
