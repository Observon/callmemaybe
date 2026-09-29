# ABOUTME: CLI entry point — parses arguments, loads model, runs pipeline.
# ABOUTME: Usage: uv run python -m src [--functions_definition F]
# ABOUTME:                              [--input I] [--output O] [--verbose]

import argparse
import sys
from pathlib import Path
from typing import List

from llm_sdk import Small_LLM_Model

from src.function_caller import FunctionCaller
from src.io_handler import load_function_definitions, load_input_prompts, write_results
from src.models import FunctionCallResult


DEFAULT_FUNCTIONS_DEF = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Namespace with functions_definition, input, output paths, and verbose flag.
    """
    parser = argparse.ArgumentParser(
        description="Call Me Maybe — function calling via constrained decoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  uv run python -m src\n"
            "  uv run python -m src --verbose\n"
            "  uv run python -m src \\\n"
            "    --functions_definition data/input/functions_definition.json \\\n"
            "    --input data/input/function_calling_tests.json \\\n"
            "    --output data/output/function_calling_results.json\n"
        ),
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS_DEF,
        metavar="FILE",
        help=f"Path to functions_definition.json (default: {DEFAULT_FUNCTIONS_DEF})",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        metavar="FILE",
        help=f"Path to function_calling_tests.json (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        metavar="FILE",
        help=f"Destination file for results (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        metavar="MODEL",
        help="HuggingFace model identifier (default: Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-token generation details (useful for debugging)",
    )
    return parser.parse_args()


def main() -> int:
    """Main pipeline: load inputs → load model → resolve prompts → write output.

    Returns:
        Exit code: 0 on success, 1 on fatal error.
    """
    args = parse_args()

    # --- Load inputs --------------------------------------------------------
    functions = load_function_definitions(args.functions_definition)
    if not functions:
        print("[ERROR] No valid function definitions. Aborting.", file=sys.stderr)
        return 1

    prompts = load_input_prompts(args.input)
    if not prompts:
        print("[ERROR] No valid prompts. Aborting.", file=sys.stderr)
        return 1

    print(f"Loaded {len(functions)} function(s), {len(prompts)} prompt(s).")

    # --- Load model ---------------------------------------------------------
    print(f"Loading model '{args.model}' (may take a moment)…")
    try:
        model = Small_LLM_Model(model_name=args.model)
    except Exception as exc:
        print(f"[ERROR] Failed to load model: {exc}", file=sys.stderr)
        return 1
    print("Model ready.\n")

    # --- Run pipeline -------------------------------------------------------
    caller = FunctionCaller(model, functions, verbose=args.verbose)
    results: List[FunctionCallResult] = []

    for i, entry in enumerate(prompts, 1):
        print(f"[{i}/{len(prompts)}] {entry.prompt!r}")
        try:
            result = caller.resolve(entry)
            results.append(result)
            print()
        except Exception as exc:
            print(f"  [WARN] Failed: {exc}", file=sys.stderr)

    # --- Write output -------------------------------------------------------
    write_results(results, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
