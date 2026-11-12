"""Validation against real model maths: transformers + PEFT on CPU.

Skipped unless torch, transformers and peft are installed and the tiny test
model can be loaded.  A random-weight LoRA is created locally, fingerprinted
with the real ``lora-check fingerprint`` code path, and served by the small
reference server in ``tests/peft_server.py`` so that ``verify`` runs end to
end over HTTP against genuine adapter arithmetic.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
peft = pytest.importorskip("peft")

from lora_check.cli import main  # noqa: E402
from lora_check.client import Client  # noqa: E402
from lora_check.fingerprint import Fingerprint  # noqa: E402
from lora_check.lint import LocalSource, lint_adapter  # noqa: E402
from lora_check.probes import builtin_probes  # noqa: E402
from lora_check.reference import compute_fingerprint  # noqa: E402
from lora_check.sim import running_server  # noqa: E402
from lora_check.verify import FAIL, PASS, VerifyConfig, VerifyReport, run_verify  # noqa: E402
from tests.peft_server import PeftApp  # noqa: E402

pytestmark = pytest.mark.realpeft

BASE = "hf-internal-testing/tiny-random-gpt2"
PROBES = builtin_probes(12)


def _make_adapter(path: Path, seed: int, **config: Any) -> None:
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(BASE)
    cfg = peft.LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["c_attn", "c_proj"],
        fan_in_fan_out=True,
        init_lora_weights=False,
        task_type="CAUSAL_LM",
        **config,
    )
    wrapped = peft.get_peft_model(model, cfg)
    with torch.no_grad():
        for name, param in wrapped.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn_like(param) * 0.5)
            elif "lora_A" in name:
                param.copy_(torch.randn_like(param) * 0.3)
    wrapped.save_pretrained(str(path))


@pytest.fixture(scope="module")
def adapters(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    torch.set_num_threads(1)
    root = tmp_path_factory.mktemp("adapters")
    try:
        _make_adapter(root / "plain", 1)
    except OSError as exc:  # model hub unreachable
        pytest.skip(f"cannot load {BASE}: {exc}")
    _make_adapter(root / "other", 2)
    _make_adapter(root / "patterned", 3, rank_pattern={"c_proj": 4}, alpha_pattern={"c_proj": 32})
    _make_adapter(root / "rslora", 4, use_rslora=True)
    return {name: str(root / name) for name in ("plain", "other", "patterned", "rslora")}


@pytest.fixture(scope="module")
def fingerprint(adapters: dict[str, str]) -> Fingerprint:
    return compute_fingerprint(BASE, adapters, PROBES)


def _verify(app: PeftApp, fingerprint: Fingerprint | None) -> VerifyReport:
    with running_server(app) as url, Client(url) as client:
        return run_verify(
            client, VerifyConfig(probes=PROBES, fingerprint=fingerprint, concurrency=2)
        )


def _check(report: VerifyReport, adapter: str, check: str) -> Any:
    return next(a for a in report.adapters if a.name == adapter).checks[check]


def _with_config(src: str, dst: Path, **changes: Any) -> str:
    """Copy an adapter and edit its adapter_config.json (a mis-read config)."""
    shutil.copytree(src, dst)
    config_path = dst / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config.update(changes)
    config_path.write_text(json.dumps(config))
    return str(dst)


def test_fingerprint_contents(fingerprint: Fingerprint, adapters: dict[str, str]) -> None:
    assert fingerprint.meta["reference"] == "transformers+peft"
    assert set(fingerprint.adapters) == set(adapters)
    for name in adapters:
        assert (
            min(fingerprint.sensitivity(name).values()) > 1e-3
        )  # random LoRA really moves the output
    names = {name: [v.name for v in ref.variants] for name, ref in fingerprint.adapters.items()}
    assert "patterns-ignored" in names["patterned"]
    assert "patterns-ignored" not in names["plain"]
    assert "rslora-ignored" in names["rslora"]
    assert fingerprint.adapters["patterned"].config["distinct_scalings"] == [2.0, 8.0]
    assert fingerprint.adapters["rslora"].config["distinct_scalings"] == [
        pytest.approx(16 / 8**0.5, abs=1e-5)
    ]
    entry = fingerprint.base[PROBES[0].id].entries[0]
    assert isinstance(entry[0], int)
    assert isinstance(entry[1], str)
    assert entry[2] < 0


def test_fingerprint_matches_merged_weights(
    fingerprint: Fingerprint, adapters: dict[str, str]
) -> None:
    """Independent check of the reference: merge the LoRA into the weights and compare."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    merged = peft.PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(BASE), adapters["patterned"]
    ).merge_and_unload()
    merged.eval()
    for probe in PROBES[:4]:
        with torch.no_grad():
            logits = merged(**tokenizer(probe.prompt, return_tensors="pt")).logits[0, -1]
        logprobs = torch.log_softmax(logits, dim=-1)
        for token_id, _text, lp in fingerprint.adapters["patterned"].reference[probe.id].entries:
            assert lp == pytest.approx(float(logprobs[token_id]), abs=2e-4)


def test_verify_healthy_real_server(fingerprint: Fingerprint, adapters: dict[str, str]) -> None:
    report = _verify(PeftApp(BASE, adapters), fingerprint)
    assert report.verdict == PASS, [a.to_json() for a in report.adapters]
    for name in adapters:
        assert _check(report, name, "applied").status == PASS
        assert _check(report, name, "distinct").status == PASS
        assert _check(report, name, "stable").status == PASS
        assert _check(report, name, "reference").status == PASS


def test_verify_ignored_adapter_real_server(
    fingerprint: Fingerprint, adapters: dict[str, str]
) -> None:
    report = _verify(PeftApp(BASE, adapters, ignored=frozenset({"other"})), fingerprint)
    assert _check(report, "other", "applied").status == FAIL
    assert _check(report, "other", "reference").evidence["cause"] == "adapter-ignored"
    assert _check(report, "plain", "reference").status == PASS


def test_verify_wrong_alpha_real_server(
    fingerprint: Fingerprint, adapters: dict[str, str], tmp_path: Path
) -> None:
    served = dict(adapters)
    served["plain"] = _with_config(adapters["plain"], tmp_path / "half-alpha", lora_alpha=8)
    report = _verify(PeftApp(BASE, served), fingerprint)
    reference = _check(report, "plain", "reference")
    assert reference.status == FAIL
    assert reference.evidence["cause"] == "wrong-scale"
    assert "scale=0.5" in reference.summary
    assert _check(report, "plain", "applied").status == PASS  # invisible without the fingerprint
    assert _check(report, "other", "reference").status == PASS


def test_verify_patterns_ignored_real_server(
    fingerprint: Fingerprint, adapters: dict[str, str]
) -> None:
    """Serve the adapter the way vllm#59799 describes: one lora_alpha/r for every module."""
    report = _verify(PeftApp(BASE, adapters, uniform_scale=frozenset({"patterned"})), fingerprint)
    reference = _check(report, "patterned", "reference")
    assert reference.status == FAIL
    assert reference.evidence["cause"] == "wrong-application"
    assert "patterns-ignored" in reference.summary
    assert _check(report, "plain", "reference").status == PASS


def test_verify_dropped_alpha_pattern_real_server(
    fingerprint: Fingerprint, adapters: dict[str, str], tmp_path: Path
) -> None:
    """A per-module scale that matches no stored variant is still a mismatch."""
    served = dict(adapters)
    served["patterned"] = _with_config(
        adapters["patterned"], tmp_path / "no-alpha-pattern", alpha_pattern={}
    )
    report = _verify(PeftApp(BASE, served), fingerprint)
    assert _check(report, "patterned", "reference").status == FAIL


def test_verify_rslora_ignored_real_server(
    fingerprint: Fingerprint, adapters: dict[str, str], tmp_path: Path
) -> None:
    served = dict(adapters)
    served["rslora"] = _with_config(adapters["rslora"], tmp_path / "no-rslora", use_rslora=False)
    report = _verify(PeftApp(BASE, served), fingerprint)
    reference = _check(report, "rslora", "reference")
    assert reference.status == FAIL
    assert "rslora-ignored" in reference.summary


def test_verify_swapped_adapters_real_server(
    fingerprint: Fingerprint, adapters: dict[str, str]
) -> None:
    served = dict(adapters)
    served["plain"], served["other"] = adapters["other"], adapters["plain"]
    report = _verify(PeftApp(BASE, served), fingerprint)
    assert _check(report, "plain", "reference").evidence["cause"] == "wrong-adapter"
    assert _check(report, "plain", "applied").status == PASS


def test_verify_float64_server_within_tolerance(
    fingerprint: Fingerprint, adapters: dict[str, str]
) -> None:
    """A different dtype on the serving side must not trip the reference check."""
    report = _verify(PeftApp(BASE, adapters, dtype="float64"), fingerprint)
    assert report.verdict == PASS
    assert report.base.checks["reference"].evidence["base_offset"] < 1e-3


def test_lint_on_real_peft_adapter(adapters: dict[str, str]) -> None:
    report = lint_adapter(LocalSource(adapters["patterned"]), None, "vllm", "0.30.0")
    rules = {f.rule: f.severity for f in report.findings}
    assert rules["V107"] == "error"
    assert report.summary["distinct_scalings"] == [2.0, 8.0]
    assert report.summary["ranks_in_weights"] == [4, 8]
    assert not any(
        f.rule in ("L030", "L031") for f in report.findings
    )  # real PEFT layout parses cleanly
    assert lint_adapter(LocalSource(adapters["plain"]), None, "vllm", "0.32.0").count("error") == 0


def test_fingerprint_cli(adapters: dict[str, str], tmp_path: Path) -> None:
    out = tmp_path / "fp.json"
    code = main(
        [
            "fingerprint", "--base", BASE, "--adapter", f"plain={adapters['plain']}", "--max-probes", "3",
            "--scales", "0.5", "-o", str(out), "--quiet",
        ]
    )  # fmt: skip
    assert code == 0
    loaded = Fingerprint.load(out)
    assert [v.name for v in loaded.adapters["plain"].variants] == ["scale=0.5"]
    assert len(loaded.probes) == 3
