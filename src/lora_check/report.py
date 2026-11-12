"""Rendering of verify and lint results as text, Markdown or JSON."""

from __future__ import annotations

import json
import textwrap
from typing import Any

from lora_check import __version__
from lora_check.lint.core import CHECKED, LintReport
from lora_check.verify import CHECKS, SKIP, ModelReport, VerifyReport


def _cell(report: ModelReport, check: str) -> str:
    result = report.checks.get(check)
    if result is None or result.status == SKIP:
        return "-"
    return result.status


def _table(rows: list[list[str]]) -> list[str]:
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows]


def verify_text(report: VerifyReport, label: str | None = None, verbose: bool = False) -> str:
    """Human-readable verify report.

    Only the checks that did not pass are detailed; ``verbose`` prints every
    check of every model.
    """
    lines = [
        f"lora-check {__version__} - verify {label or report.endpoint}",
        f"base model: {report.base.name}   adapters: {len(report.adapters)}   "
        f"probes: {report.probes} x {report.repeats} repeats   requests: {report.requests}",
        f"run-to-run noise (mean |dlogprob|, p90): base {report.base.noise:.4f}"
        + "".join(f", {a.name} {a.noise:.4f}" for a in report.adapters),
        "",
    ]
    header = ["MODEL", *[c.upper() for c in CHECKS], "VERDICT"]
    rows = [header]
    for rep in [*report.adapters, report.base]:
        name = f"{rep.name} (base)" if rep.is_base else rep.name
        rows.append([name, *[_cell(rep, c) for c in CHECKS], rep.verdict])
    lines.extend(_table(rows))
    lines.append("")
    for rep in [*report.adapters, report.base]:
        shown = [c for c in rep.checks.values() if c.status != SKIP]
        if not shown or (rep.verdict == "PASS" and not verbose):
            continue
        if not verbose:
            shown = [c for c in shown if c.status != "PASS"]
        lines.append(f"{rep.name}{' (base)' if rep.is_base else ''}: {rep.verdict}")
        for check in shown:
            lines.extend(
                textwrap.wrap(
                    f"{check.name:<9} {check.status:<5} {check.summary}",
                    98,
                    initial_indent="  ",
                    subsequent_indent=" " * 18,
                )
            )
    if report.notes:
        lines.append("")
        lines.extend(f"note: {note}" for note in report.notes)
    lines.append("")
    lines.append(f"overall: {report.verdict}")
    return "\n".join(lines)


def verify_markdown(report: VerifyReport, label: str | None = None) -> str:
    """Markdown verify report, suitable for a CI job summary."""
    lines = [
        f"## lora-check verify: {report.verdict}",
        "",
        f"Endpoint `{label or report.endpoint}`, base model `{report.base.name}`, "
        f"{report.probes} probes x {report.repeats} repeats, {report.requests} requests.",
        "",
        "| Model | " + " | ".join(c.capitalize() for c in CHECKS) + " | Verdict |",
        "|---|" + "---|" * (len(CHECKS) + 1),
    ]
    for rep in [*report.adapters, report.base]:
        name = f"`{rep.name}` (base)" if rep.is_base else f"`{rep.name}`"
        cells = " | ".join(_cell(rep, c) for c in CHECKS)
        lines.append(f"| {name} | {cells} | **{rep.verdict}** |")
    lines.append("")
    for rep in [*report.adapters, report.base]:
        shown = [c for c in rep.checks.values() if c.status != SKIP]
        if not shown:
            continue
        lines.append(f"### `{rep.name}`: {rep.verdict}")
        lines.append("")
        lines.extend(f"- **{check.name}** {check.status}: {check.summary}" for check in shown)
        lines.append("")
    lines.extend(f"> {note}" for note in report.notes)
    return "\n".join(lines).rstrip() + "\n"


def to_json(payload: dict[str, Any]) -> str:
    """Stable, indented JSON."""
    return json.dumps(payload, indent=2, ensure_ascii=False)


def render_verify(
    report: VerifyReport, fmt: str, label: str | None = None, verbose: bool = False
) -> str:
    if fmt == "json":
        payload = report.to_json()
        if label:
            payload["endpoint"] = label
        return to_json(payload)
    if fmt == "markdown":
        return verify_markdown(report, label)
    return verify_text(report, label, verbose)


# ---------------------------------------------------------------------------
# lint
# ---------------------------------------------------------------------------


_PAD = " " * 8


def lint_text(report: LintReport, label: str | None = None) -> str:
    """Human-readable lint report."""
    engines = ", ".join(report.engines)
    if report.engine_version:
        engines += f" {report.engine_version}"
    summary = report.summary
    lines = [
        f"lora-check {__version__} - lint {label or report.adapter}",
        f"base: {report.base or '(not given)'}   engine rules: {engines}   sources checked: {CHECKED}",
    ]
    if summary:
        lines.append(
            f"r={summary.get('r')} lora_alpha={summary.get('lora_alpha')} "
            f"use_rslora={summary.get('use_rslora')} use_dora={summary.get('use_dora')} "
            f"lora modules={summary.get('lora_modules_in_weights')} "
            f"ranks={summary.get('ranks_in_weights')} scalings={summary.get('distinct_scalings')}"
        )
    lines.append("")
    findings = report.sorted_findings()
    if not findings:
        lines.append("no findings")
    for finding in findings:
        rule = finding.info
        lines.append(f"{finding.severity.upper():<7} {finding.rule}  {rule.title}")
        lines.extend(
            textwrap.wrap(finding.message, 92, initial_indent=_PAD, subsequent_indent=_PAD)
        )
        if rule.source:
            lines.append(f"        source ({rule.source_kind}): {rule.source}")
    lines.append("")
    lines.append(
        f"{report.count('error')} error(s), {report.count('warning')} warning(s), {report.count('info')} note(s)"
    )
    return "\n".join(lines)


def lint_markdown(report: LintReport, label: str | None = None) -> str:
    """Markdown lint report."""
    lines = [
        f"## lora-check lint: `{label or report.adapter}`",
        "",
        f"{report.count('error')} error(s), {report.count('warning')} warning(s), "
        f"{report.count('info')} note(s). Engine rules: {', '.join(report.engines)}.",
        "",
        "| Severity | Rule | Finding | Source |",
        "|---|---|---|---|",
    ]
    for finding in report.sorted_findings():
        rule = finding.info
        source = f"[{rule.source_kind}]({rule.source})" if rule.source else ""
        message = finding.message.replace("|", "\\|")
        lines.append(
            f"| {finding.severity} | {finding.rule} | **{rule.title}.** {message} | {source} |"
        )
    return "\n".join(lines) + "\n"


def render_lint(report: LintReport, fmt: str, label: str | None = None) -> str:
    if fmt == "json":
        payload = report.to_json()
        if label:
            payload["adapter"] = label
        return to_json(payload)
    if fmt == "markdown":
        return lint_markdown(report, label)
    return lint_text(report, label)
