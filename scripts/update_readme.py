"""Regenerate the generated blocks in README.md (demo output, rule list, evaluation table).

Blocks are delimited by ``<!-- BEGIN name -->`` / ``<!-- END name -->``.
``tests/test_readme_sync.py`` fails when the README is out of date.

Usage: python scripts/update_readme.py [--check]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from lora_check import demo
from lora_check.lint import RULES
from lora_check.report import render_lint, render_verify

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
EVALUATION = ROOT / "docs" / "evaluation-table.md"


def _rules_table() -> str:
    lines = ["| Rule | Applies to | What it flags | Source |", "|---|---|---|---|"]
    for rule in RULES.values():
        if rule.source:
            source = f"[{rule.source_kind}]({rule.source})"
        else:
            source = "structural"
        fixed = f" (fixed in {rule.fixed_in})" if rule.fixed_in else ""
        lines.append(f"| `{rule.id}` | {rule.engine or 'any'} | {rule.title}{fixed} | {source} |")
    return "\n".join(lines)


def blocks() -> dict[str, str]:
    """Name -> current content of every generated block."""
    verify = render_verify(demo.demo_verify("mixed"), "text", demo.demo_label("mixed"))
    lint = render_lint(demo.demo_lint(), "text")
    return {
        "demo-verify": f"```text\n$ lora-check verify --demo\n{verify}\n```",
        "demo-lint": f"```text\n$ lora-check lint --demo\n{lint}\n```",
        "rules": _rules_table(),
        "evaluation": EVALUATION.read_text(encoding="utf-8").strip(),
    }


def render(text: str) -> str:
    for name, content in blocks().items():
        begin, end = f"<!-- BEGIN {name} -->\n", f"\n<!-- END {name} -->"
        pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end.lstrip("\n")), re.DOTALL)
        if not pattern.search(text):
            raise SystemExit(f"README.md has no block named {name!r}")
        match = pattern.search(text)
        assert match is not None
        text = text[: match.start()] + begin + content + end + text[match.end() :]
    return text


def main() -> int:
    current = README.read_text(encoding="utf-8")
    updated = render(current)
    if "--check" in sys.argv[1:]:
        if current != updated:
            print("README.md is out of date; run python scripts/update_readme.py", file=sys.stderr)
            return 1
        return 0
    README.write_text(updated, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
