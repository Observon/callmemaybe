# ABOUTME: pytest configuration — mocks llm_sdk before test collection
# ABOUTME: so tests run without torch/transformers installed.

import sys
import types
from unittest.mock import MagicMock


def _mock_llm_sdk() -> None:
    """Install a lightweight mock of llm_sdk into sys.modules.

    This prevents torch from being imported during test collection,
    allowing all non-model tests to run without GPU dependencies.
    """
    mock_sdk = types.ModuleType("llm_sdk")

    class Small_LLM_Model:  # noqa: N801
        """Stub — replaced by MagicMock in individual tests."""
        pass

    mock_sdk.Small_LLM_Model = Small_LLM_Model  # type: ignore[attr-defined]
    sys.modules.setdefault("llm_sdk", mock_sdk)
    # Also mock torch so llm_sdk's own import doesn't fail if it's imported elsewhere
    for mod in ("torch", "transformers", "huggingface_hub"):
        sys.modules.setdefault(mod, MagicMock())


_mock_llm_sdk()
