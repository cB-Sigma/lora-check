from __future__ import annotations

import json
import struct
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from lora_check.lint import (
    RULES,
    HubSource,
    LintReport,
    LocalSource,
    SourceError,
    lint_adapter,
    open_source,
)
from lora_check.lint.core import match_pattern, parse_version
from lora_check.lint.io import parse_safetensors_header, write_safetensors

MakeAdapter = Callable[..., Path]


def _lint(
    path: Path, base: Path | None = None, engine: str | None = None, version: str | None = None
) -> LintReport:
    return lint_adapter(LocalSource(path), LocalSource(base) if base else None, engine, version)


def _rules(report: LintReport, severity: str | None = None) -> set[str]:
    return {f.rule for f in report.findings if severity is None or f.severity == severity}


def test_every_engine_rule_cites_a_source() -> None:
    for rule in RULES.values():
        if rule.engine is not None:
            assert rule.source is not None, rule.id
            assert rule.source.startswith("https://"), rule.id
            assert rule.source_kind in ("docs", "source", "issue"), rule.id
            assert rule.engine in ("vllm", "sglang")


def test_clean_adapter_has_no_errors_or_warnings(
    make_adapter: MakeAdapter, make_base: MakeAdapter
) -> None:
    report = _lint(make_adapter(), make_base())
    assert report.count("error") == 0
    assert report.count("warning") == 0
    assert report.exit_code(strict=True) == 0
    assert report.summary["lora_modules_in_weights"] == 4
    assert report.summary["base_modules_matched"] == 4
    assert report.summary["distinct_scalings"] == [2.0]
    assert _rules(report) <= {"V108", "S205"}  # informational engine notes only


def test_missing_config_and_weights(tmp_path: Path, make_adapter: MakeAdapter) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert _rules(_lint(empty)) == {"L001"}
    path = make_adapter()
    (path / "adapter_model.safetensors").unlink()
    assert "L002" in _rules(_lint(path), "error")
    (path / "adapter_model.bin").write_bytes(b"\x80\x04")
    assert "L003" in _rules(_lint(path), "warning")


def test_bad_required_fields(make_adapter: MakeAdapter) -> None:
    report = _lint(make_adapter(config={"r": 0, "lora_alpha": None, "peft_type": "IA3"}))
    assert {"L004", "L010"} <= _rules(report, "error")
    assert report.exit_code() == 1


def test_rank_and_alpha_patterns_version_gated_for_vllm(make_adapter: MakeAdapter) -> None:
    modules = {
        "model.layers.0.self_attn.q_proj": (8, 32, 32),
        "model.layers.0.self_attn.v_proj": (16, 32, 32),
    }
    path = make_adapter(
        config={"rank_pattern": {"v_proj": 16}, "alpha_pattern": {"v_proj": 64}}, modules=modules
    )
    old = _lint(path, engine="vllm", version="0.30.0")
    assert "V107" in _rules(old, "error")
    assert old.summary["distinct_scalings"] == [2.0, 4.0]
    assert "V107" not in _rules(_lint(path, engine="vllm", version="0.32.0"))
    assert "V107" in _rules(_lint(path, engine="vllm"), "warning")
    assert "S203" in _rules(_lint(path, engine="sglang"), "warning")
    assert "V107" not in _rules(_lint(path, engine="sglang"))
    both = _lint(path)
    assert {"V107", "S203", "L020"} <= _rules(both)


def test_patterns_that_do_not_change_the_scale_are_only_noted(make_adapter: MakeAdapter) -> None:
    modules = {
        "model.layers.0.self_attn.q_proj": (8, 32, 32),
        "model.layers.0.self_attn.v_proj": (16, 32, 32),
    }
    path = make_adapter(
        config={"rank_pattern": {"v_proj": 16}, "alpha_pattern": {"v_proj": 32}}, modules=modules
    )
    report = _lint(path, engine="vllm", version="0.30.0")
    assert "L020" in _rules(report, "info")
    assert "V107" not in _rules(report)


def test_rank_mismatch_between_config_and_weights(make_adapter: MakeAdapter) -> None:
    modules = {
        "model.layers.0.self_attn.q_proj": (4, 32, 32),
        "model.layers.0.self_attn.v_proj": (8, 32, 32),
    }
    report = _lint(make_adapter(modules=modules))
    assert any(
        f.rule == "L030" and "q_proj (has 4, config says 8)" in f.message for f in report.findings
    )


def test_target_modules_vs_weights(make_adapter: MakeAdapter) -> None:
    report = _lint(make_adapter(config={"target_modules": ["q_proj", "k_proj"]}))
    messages = " ".join(f.message for f in report.findings if f.rule == "L030")
    assert "k_proj" in messages
    assert "v_proj" in messages


def test_unsupported_features_per_engine(make_adapter: MakeAdapter) -> None:
    path = make_adapter(
        config={
            "use_dora": True,
            "bias": "all",
            "modules_to_save": ["lm_head"],
            "init_lora_weights": "pissa_niter_4",
            "lora_bias": True,
            "use_rslora": True,
            "r": 64,
            "target_parameters": ["experts.down_proj"],
        },
        modules={"model.layers.0.self_attn.q_proj": (64, 32, 32)},
    )
    vllm = _lint(path, engine="vllm", version="0.6.4")
    assert {"V101", "V102", "V103", "V104", "V105"} <= _rules(vllm, "error")
    assert {"V106", "V108"} <= _rules(vllm, "info")
    assert {"V109", "V110", "L023", "L024", "L025", "L050"} <= _rules(vllm, "warning")
    sglang = _lint(path, engine="sglang")
    assert "S201" in _rules(sglang, "error")
    assert {"S203", "S204"} <= _rules(sglang, "warning")
    assert not any(rule.startswith("V") for rule in _rules(sglang))
    assert "V110" not in _rules(_lint(path, engine="vllm", version="0.30.0"))


def test_base_prefix_mismatch_is_the_silent_failure_pattern(
    make_adapter: MakeAdapter, make_base: MakeAdapter
) -> None:
    report = _lint(make_adapter(), make_base(prefix="language_model.model."))
    finding = next(f for f in report.findings if f.rule == "L043")
    assert finding.severity == "warning"
    assert "'language_model.' (4 modules)" in finding.message
    assert "34186" in finding.message


def test_adapter_side_prefix_is_only_informational(
    make_adapter: MakeAdapter, make_base: MakeAdapter
) -> None:
    modules = {
        f"transformer.model.layers.0.self_attn.{p}": (8, 32, 32) for p in ("q_proj", "v_proj")
    }
    report = _lint(make_adapter(modules=modules), make_base())
    finding = next(f for f in report.findings if f.rule == "L043")
    assert finding.severity == "info"
    assert report.count("error") == 0


def test_missing_modules_and_shape_mismatch(
    make_adapter: MakeAdapter, make_base: MakeAdapter
) -> None:
    modules = {
        "model.layers.7.self_attn.q_proj": (8, 32, 32),
        "model.layers.0.self_attn.v_proj": (8, 48, 32),
    }
    report = _lint(make_adapter(modules=modules), make_base())
    assert {"L040", "L041"} <= _rules(report, "error")
    assert any("1/2 adapter modules" in f.message for f in report.findings)


def test_base_without_safetensors(make_adapter: MakeAdapter, tmp_path: Path) -> None:
    base = tmp_path / "bin-only"
    base.mkdir()
    (base / "pytorch_model.bin").write_bytes(b"x")
    report = _lint(make_adapter(), base)
    assert "L040" in _rules(report, "warning")


def test_base_name_mismatch_note(make_adapter: MakeAdapter, make_base: MakeAdapter) -> None:
    report = _lint(make_adapter(config={"base_model_name_or_path": "org/other"}), make_base())
    assert "L042" in _rules(report, "info")


def test_tokenizer_changes(make_adapter: MakeAdapter, make_base: MakeAdapter) -> None:
    path = make_adapter(
        config={"modules_to_save": ["embed_tokens"]},
        extra_tensors={"base_model.model.model.embed_tokens.weight": ("F32", (103, 32))},
        files={
            "added_tokens.json": {"<a>": 100, "<b>": 101, "<c>": 102, "the": 5},
            "tokenizer_config.json": {},
        },
    )
    report = _lint(path, make_base(vocab=100))
    finding = next(f for f in report.findings if f.rule == "L050")
    assert "3 beyond the base vocabulary of 100" in finding.message
    assert "103 rows, resized from 100" in finding.message
    assert "tokenizer_config.json" in finding.message
    assert "S202" in _rules(report, "error")
    assert "S204" in _rules(report, "warning")
    without_base = _lint(path, engine="sglang")
    assert "S202" in _rules(without_base, "warning")


def test_added_tokens_copied_from_base_are_not_new(
    make_adapter: MakeAdapter, make_base: MakeAdapter
) -> None:
    path = make_adapter(files={"added_tokens.json": {"<eos>": 2}})
    report = _lint(path, make_base(vocab=100), engine="sglang")
    assert "S202" not in _rules(report)


def test_unprefixed_keys_and_incomplete_pairs(make_adapter: MakeAdapter, tmp_path: Path) -> None:
    path = make_adapter(prefix="")
    assert "L031" in _rules(_lint(path), "warning")
    lonely = tmp_path / "lonely"
    lonely.mkdir()
    (lonely / "adapter_config.json").write_text(
        json.dumps({"r": 8, "lora_alpha": 8, "target_modules": ["q"]})
    )
    write_safetensors(
        lonely / "adapter_model.safetensors",
        {"base_model.model.x.q.lora_A.weight": ("F32", (8, 4))},
    )
    assert any("only one of" in f.message for f in _lint(lonely).findings)
    nolora = tmp_path / "nolora"
    nolora.mkdir()
    (nolora / "adapter_config.json").write_text(
        json.dumps({"r": 8, "lora_alpha": 8, "target_modules": ["q"]})
    )
    write_safetensors(nolora / "adapter_model.safetensors", {"something.weight": ("F32", (2, 2))})
    assert any("no lora_A/lora_B" in f.message for f in _lint(nolora).findings)


def test_report_json_and_sorting(make_adapter: MakeAdapter) -> None:
    report = _lint(make_adapter(config={"use_dora": True}), engine="vllm")
    payload = report.to_json()
    assert payload["errors"] == 1
    severities = [f["severity"] for f in payload["findings"]]
    assert severities == sorted(severities, key=["error", "warning", "info"].index)
    assert payload["findings"][0]["source"].startswith("https://github.com/vllm-project/vllm")


def test_helpers() -> None:
    assert parse_version("v0.30.0rc1") == (0, 30, 0)
    assert parse_version("0.6.4.post1") == (0, 6, 4)
    assert parse_version("nightly") == ()
    assert match_pattern({"q_proj": 4}, "model.layers.0.q_proj") == 4
    assert match_pattern({"q_proj": 4}, "model.layers.0.xq_proj") is None
    assert match_pattern({r"layers\.0\..*": 2, "q_proj": 4}, "model.layers.0.q_proj") == 2
    assert match_pattern({"(": 1}, "a.(") == 1


def test_safetensors_io_roundtrip_and_errors(tmp_path: Path) -> None:
    path = tmp_path / "x.safetensors"
    write_safetensors(path, {"a.weight": ("F16", (3, 5)), "b": ("F32", (2,))})
    source = LocalSource(tmp_path)
    tensors = source.tensors("x.safetensors")
    assert tensors["a.weight"].shape == (3, 5)
    assert tensors["a.weight"].dtype == "F16"
    assert path.stat().st_size >= 8 + 3 * 5 * 2 + 2 * 4
    (tmp_path / "short.safetensors").write_bytes(b"abc")
    with pytest.raises(SourceError):
        source.tensors("short.safetensors")
    (tmp_path / "huge.safetensors").write_bytes(struct.pack("<Q", 1 << 40) + b"{}")
    with pytest.raises(SourceError, match="implausible"):
        source.tensors("huge.safetensors")
    with pytest.raises(SourceError, match="corrupt"):
        parse_safetensors_header(struct.pack("<Q", 3), b"{{{")
    with pytest.raises(SourceError):
        LocalSource(tmp_path / "missing")
    with pytest.raises(SourceError):
        open_source("not a repo id")
    assert isinstance(open_source(str(tmp_path)), LocalSource)
    assert isinstance(open_source("org/name"), HubSource)


def test_hub_source_reads_only_listing_json_and_headers(make_adapter: MakeAdapter) -> None:
    local = make_adapter(config={"use_dora": True})
    requests: list[tuple[str, str | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        requests.append((path, request.headers.get("range")))
        if path == "/api/models/org/adapter/revision/main":
            names = [p.name for p in local.iterdir()] + ["sub/ignored.txt"]
            return httpx.Response(200, json={"siblings": [{"rfilename": n} for n in names]})
        prefix = "/org/adapter/resolve/main/"
        if path.startswith(prefix):
            data = (local / path[len(prefix) :]).read_bytes()
            rng = request.headers.get("range")
            if rng:
                start, _, end = rng.removeprefix("bytes=").partition("-")
                return httpx.Response(206, content=data[int(start) : int(end) + 1])
            return httpx.Response(200, content=data)
        return httpx.Response(404)

    source = HubSource("org/adapter", transport=httpx.MockTransport(handler))
    report = lint_adapter(source, None, "vllm", None)
    assert "V101" in {f.rule for f in report.findings}
    assert report.summary["lora_modules_in_weights"] == 4
    weight_requests = [r for p, r in requests if p.endswith(".safetensors")]
    assert weight_requests
    assert all(r is not None for r in weight_requests)  # never the whole file

    missing = HubSource("org/none", transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    assert {f.rule for f in lint_adapter(missing).findings} == {"L001"}


def test_hub_source_handles_servers_without_range_support(make_adapter: MakeAdapter) -> None:
    local = make_adapter()

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.startswith("/api/"):
            return httpx.Response(
                200, json={"siblings": [{"rfilename": p.name} for p in local.iterdir()]}
            )
        return httpx.Response(
            200, content=(local / request.url.path.rsplit("/", 1)[-1]).read_bytes()
        )

    source: Any = HubSource("org/adapter", revision="v1", transport=httpx.MockTransport(handler))
    assert source.label == "org/adapter@v1"
    assert len(source.tensors("adapter_model.safetensors")) == 8
