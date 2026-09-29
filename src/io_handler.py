# ABOUTME: Handles all file I/O: reading JSON inputs and writing JSON outputs.
# ABOUTME: Validates files with Pydantic and reports clear errors without crashing.

import json
import sys
from pathlib import Path
from typing import List

from pydantic import ValidationError

from src.models import FunctionCallResult, FunctionDefinition, InputPrompt


def load_function_definitions(path: Path) -> List[FunctionDefinition]:
    """Load and validate the list of available functions from a JSON file.

    Args:
        path: Path to functions_definition.json.

    Returns:
        A list of validated FunctionDefinition objects.
        Returns an empty list and prints an error if the file is missing,
        contains invalid JSON, or fails schema validation.
    """
    if not path.exists():
        print(f"[ERROR] Functions definition file not found: {path}", file=sys.stderr)
        return []

    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON in {path}: {exc}", file=sys.stderr)
        return []

    if not isinstance(raw, list):
        print(f"[ERROR] Expected a JSON array in {path}", file=sys.stderr)
        return []

    definitions: List[FunctionDefinition] = []
    for item in raw:
        try:
            definitions.append(FunctionDefinition.model_validate(item))
        except ValidationError as exc:
            print(f"[WARN] Skipping invalid function entry: {exc}", file=sys.stderr)

    return definitions


def load_input_prompts(path: Path) -> List[InputPrompt]:
    """Load and validate the list of natural-language prompts.

    Args:
        path: Path to function_calling_tests.json.

    Returns:
        A list of validated InputPrompt objects.
        Returns an empty list on any error.
    """
    if not path.exists():
        print(f"[ERROR] Input prompts file not found: {path}", file=sys.stderr)
        return []

    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except json.JSONDecodeError as exc:
        print(f"[ERROR] Invalid JSON in {path}: {exc}", file=sys.stderr)
        return []

    if not isinstance(raw, list):
        print(f"[ERROR] Expected a JSON array in {path}", file=sys.stderr)
        return []

    prompts: List[InputPrompt] = []
    for item in raw:
        try:
            prompts.append(InputPrompt.model_validate(item))
        except ValidationError as exc:
            print(f"[WARN] Skipping invalid prompt entry: {exc}", file=sys.stderr)

    return prompts


def write_results(results: List[FunctionCallResult], path: Path) -> None:
    """Serialize and write results to a JSON file.

    Creates parent directories automatically if they do not exist.

    Args:
        results: List of FunctionCallResult objects to serialise.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = [r.model_dump() for r in results]

    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"[OK] Results written to {path}")
    except OSError as exc:
        print(f"[ERROR] Could not write output file {path}: {exc}", file=sys.stderr)
