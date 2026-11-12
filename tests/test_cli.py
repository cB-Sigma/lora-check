from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from lora_check import __version__
from lora_check.cli import main
from lora_check.fingerprint import Fingerprint
from lora_check.probes import builtin_probes, dump_probes, load_probes
from lora_check.sim import Faults, SimConfig, SimServer, running_server


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "verify" in capsys.readouterr().out


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_verify_demo_text_and_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verify", "--demo"]) == 1
    out = capsys.readouterr().out
    assert "demo://mixed" in out
    assert "the adapter is loaded but ignored" in out
    assert "overall: FAIL" in out


def test_verify_demo_healthy_json(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verify", "--demo", "healthy", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["verdict"] == "PASS"
    assert payload["endpoint"].startswith("demo://healthy")
    assert {a["name"] for a in payload["adapters"]} >= {"sql-gen", "legal-v2"}


def test_verify_demo_other_scenarios(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verify", "--demo", "leak", "--format", "markdown"]) == 1
    assert "| `sql-gen` |" in capsys.readouterr().out
    assert main(["verify", "--demo", "duplicate", "--no-fingerprint"]) == 1
    assert "behaves identically" in capsys.readouterr().out
    assert main(["verify", "--demo", "cache", "--no-fingerprint"]) == 0
    capsys.readouterr()
    assert main(["verify", "--demo", "cache", "--no-fingerprint", "--cache-salt"]) == 1
    assert "prefix cache is bypassed" in capsys.readouterr().out
    assert main(["verify", "--demo", "nope"]) == 2


def test_verify_requires_base_url(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verify"]) == 2
    assert "--base-url" in capsys.readouterr().err


def test_verify_unreachable_endpoint_is_exit_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["verify", "--base-url", "http://127.0.0.1:9", "--quiet", "--timeout", "2"]) == 2
    assert "failed" in capsys.readouterr().err


def test_full_workflow_against_http_server(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """fingerprint (toy) -> mine -> verify with the mined fingerprint, over HTTP."""
    sim = SimServer(SimConfig(adapters=("a1", "a2"), noise=0.001, faults=Faults(scale={"a2": 2.0})))
    full = tmp_path / "full.json"
    sim.reference_fingerprint(builtin_probes()).save(full)
    mined_probes = tmp_path / "mined.jsonl"
    mined_fp = tmp_path / "mined.json"
    assert (
        main(
            [
                "mine",
                "--fingerprint",
                str(full),
                "--top",
                "10",
                "-o",
                str(mined_probes),
                "--fingerprint-out",
                str(mined_fp),
            ]
        )
        == 0
    )
    assert "10 most sensitive probes" in capsys.readouterr().out
    assert len(load_probes(mined_probes)) == 10
    assert len(Fingerprint.load(mined_fp).probes) == 10
    report_file = tmp_path / "report.md"
    with running_server(sim) as url:
        code = main(
            [
                "verify", "--base-url", url, "--fingerprint", str(mined_fp), "--format", "markdown",
                "--output", str(report_file), "--quiet", "--concurrency", "4",
            ]
        )  # fmt: skip
        assert code == 1
        text = report_file.read_text()
        assert "## lora-check verify: FAIL" in text
        assert "scale=2" in text
        code = main(
            ["verify", "--base-url", url, "--adapter", "a1", "--base-model", "toy-base", "--format", "json",
             "--quiet", "--max-probes", "8", "--api", "chat", "--top-logprobs", "10", "--verbose"]
        )  # fmt: skip
        assert code == 0
        assert json.loads(capsys.readouterr().out)["adapters"][0]["name"] == "a1"


def test_verify_argument_errors(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fp = tmp_path / "fp.json"
    SimServer().reference_fingerprint(builtin_probes(4)).save(fp)
    other = tmp_path / "other.jsonl"
    dump_probes(builtin_probes(8)[4:], other)
    base = ["verify", "--base-url", "http://127.0.0.1:9", "--quiet"]
    assert main([*base, "--fingerprint", str(fp), "--probes", str(other)]) == 2
    assert "subset" in capsys.readouterr().err
    assert main([*base, "--map", "broken"]) == 2
    assert main([*base, "--extra-body", "{nope"]) == 2
    assert main([*base, "--extra-body", "[1]"]) == 2
    assert main([*base, "--fingerprint", str(tmp_path / "missing.json")]) == 2
    assert main([*base, "--probes", str(tmp_path / "missing.txt")]) == 2


def test_mine_json_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    fp = tmp_path / "fp.json"
    assert main(["fingerprint", "--demo", "-o", str(fp), "--quiet"]) == 0
    assert (
        main(
            [
                "mine",
                "--fingerprint",
                str(fp),
                "--adapter",
                "sql-gen",
                "--top",
                "3",
                "--format",
                "json",
            ]
        )
        == 0
    )
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 3
    assert rows[0]["chosen_for"] == "sql-gen"
    assert main(["mine", "--fingerprint", str(fp), "--adapter", "unknown"]) == 2


def test_fingerprint_argument_errors(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["fingerprint"]) == 2
    assert main(["fingerprint", "--base", "b", "--adapter", "a=x", "--adapter", "a=y"]) == 2
    assert main(["fingerprint", "--base", "b", "--adapter", "a=x", "--scales", "one,two"]) == 2
    assert "--scales" in capsys.readouterr().err


def test_fingerprint_without_ml_stack_explains_itself(capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("httpx")
    try:
        import torch  # noqa: F401
    except ImportError:
        assert main(["fingerprint", "--base", "b", "--adapter", "a=/nowhere", "--quiet"]) == 2
        assert "lora-check[fingerprint]" in capsys.readouterr().err
    else:
        pytest.skip("torch is installed")


def test_lint_demo_and_formats(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["lint", "--demo"]) == 1
    out = capsys.readouterr().out
    assert "V107" in out
    assert "https://github.com/vllm-project/vllm/issues/59799" in out
    assert (
        main(["lint", "--demo", "--engine-version", "0.32.0", "--strict"]) == 1
    )  # warnings remain
    capsys.readouterr()
    assert main(["lint", "--demo", "--format", "json"]) == 1
    assert json.loads(capsys.readouterr().out)["errors"] == 1
    assert main(["lint", "--demo", "--format", "markdown"]) == 1
    assert "| error | V107 |" in capsys.readouterr().out


def test_lint_list_rules_and_errors(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    assert main(["lint", "--list-rules"]) == 0
    out = capsys.readouterr().out
    assert "S203" in out
    assert "fixed in 0.32.0" in out
    assert main(["lint"]) == 2
    assert main(["lint", str(tmp_path / "nope nope")]) == 2
    assert main(["lint", str(tmp_path), "--engine-version", "1.0"]) == 2


def test_lint_local_directory(
    make_adapter: Callable[..., Path], make_base: Callable[..., Path], tmp_path: Path
) -> None:
    out = tmp_path / "lint.json"
    code = main(
        [
            "lint",
            str(make_adapter()),
            "--base",
            str(make_base()),
            "--engine",
            "vllm",
            "--format",
            "json",
            "-o",
            str(out),
        ]
    )
    assert code == 0
    assert json.loads(out.read_text())["errors"] == 0
