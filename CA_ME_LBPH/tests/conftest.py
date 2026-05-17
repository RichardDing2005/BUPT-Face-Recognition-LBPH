from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolate_ca_me_imports() -> None:
    original_path = list(sys.path)
    _drop_src_modules()
    sys.path.insert(0, str(ROOT))
    try:
        yield
    finally:
        _drop_src_modules()
        sys.path[:] = original_path


def _drop_src_modules() -> None:
    for module_name in list(sys.modules):
        if module_name == "src" or module_name.startswith("src."):
            del sys.modules[module_name]
