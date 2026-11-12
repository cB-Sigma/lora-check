"""The README shows real output: regenerate the blocks and compare."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"


def _load_updater() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "update_readme", ROOT / "scripts" / "update_readme.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.skipif(not README.exists(), reason="README.md not shipped with this install")
def test_readme_generated_blocks_are_current() -> None:
    updater = _load_updater()
    text = README.read_text(encoding="utf-8")
    for name, content in updater.blocks().items():
        match = re.search(rf"<!-- BEGIN {name} -->\n(.*?)\n<!-- END {name} -->", text, re.DOTALL)
        assert match is not None, f"README.md lacks the generated block {name!r}"
        assert match.group(1) == content, (
            f"README block {name!r} is stale; run `python scripts/update_readme.py`"
        )


def test_demo_output_is_deterministic() -> None:
    from lora_check import demo
    from lora_check.report import render_verify

    first = render_verify(demo.demo_verify("mixed"), "text", "x")
    second = render_verify(demo.demo_verify("mixed"), "text", "x")
    assert first == second


def test_readme_makes_no_unpublished_claims() -> None:
    if not README.exists():
        pytest.skip("README.md not shipped with this install")
    text = README.read_text(encoding="utf-8")
    assert "pip install git+https://github.com/cb-Sigma/lora-check" in text
    assert "img.shields.io/pypi" not in text
    assert "SIMULATED" in text
