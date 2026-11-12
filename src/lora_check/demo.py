"""Self-contained demos backed by the bundled simulator.

``lora-check verify --demo`` starts the simulated multi-LoRA server on a
local port, injects the faults of the chosen scenario and runs the real
verification code against it over HTTP.  The output shown in the README is
produced by these functions and kept in sync by a test.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from lora_check.client import Client
from lora_check.fingerprint import Fingerprint
from lora_check.lint import LintReport, LocalSource, lint_adapter, write_safetensors
from lora_check.probes import Probe, builtin_probes
from lora_check.sim import Faults, SimConfig, SimServer, running_server
from lora_check.verify import VerifyConfig, VerifyReport, run_verify

DEMO_ADAPTERS = ("sql-gen", "support-bot", "summarizer", "legal-v1", "legal-v2")

SCENARIOS: dict[str, str] = {
    "mixed": "one adapter ignored, one served at 0.25x scale, two swapped (default)",
    "healthy": "every adapter served correctly",
    "leak": "the first request after switching model runs with the previous adapter",
    "duplicate": "two adapter names served with the same weights",
    "cache": "prefix cache shared between adapters (needs --cache-salt or a fingerprint)",
}


def demo_sim(scenario: str = "mixed", noise: float = 0.002) -> SimServer:
    """The simulated server for a named scenario."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown demo scenario {scenario!r}; choose from {', '.join(SCENARIOS)}")
    faults = Faults()
    if scenario == "mixed":
        faults = Faults(
            ignored=frozenset({"support-bot"}),
            scale={"summarizer": 0.25},
            alias={"legal-v1": "legal-v2", "legal-v2": "legal-v1"},
        )
    elif scenario == "leak":
        faults = Faults(leak_after_switch=1.0)
    elif scenario == "duplicate":
        faults = Faults(alias={"legal-v2": "legal-v1"})
    elif scenario == "cache":
        faults = Faults(cache_ignores_adapter=True)
    return SimServer(SimConfig(adapters=DEMO_ADAPTERS, noise=noise, faults=faults))


def demo_probes() -> list[Probe]:
    return builtin_probes(24)


def demo_fingerprint(scenario: str = "mixed") -> Fingerprint:
    """Ground-truth fingerprint for the demo adapters (from the toy model)."""
    return demo_sim(scenario).reference_fingerprint(demo_probes())


def demo_verify(
    scenario: str = "mixed",
    with_fingerprint: bool = True,
    cache_salt: bool = False,
    concurrency: int = 8,
) -> VerifyReport:
    """Run a real verification over HTTP against the simulated server."""
    sim = demo_sim(scenario)
    probes = demo_probes()
    with running_server(sim) as url, Client(url) as client:
        config = VerifyConfig(
            probes=probes,
            fingerprint=sim.reference_fingerprint(probes) if with_fingerprint else None,
            cache_salt=cache_salt,
            # The leak scenario depends on request order; keep it reproducible.
            concurrency=1 if scenario == "leak" else concurrency,
        )
        return run_verify(client, config)


def demo_label(scenario: str) -> str:
    return f"demo://{scenario} (simulated server)"


def write_demo_adapter(root: Path) -> tuple[Path, Path]:
    """Create a small adapter and base checkpoint that trip several lint rules.

    The layout mimics a multimodal checkpoint whose language model sits under
    ``language_model.`` while the adapter was trained on the text-only class.
    Tensors are zero-filled; only names and shapes matter to the linter.
    """
    base = root / "demo-base"
    adapter = root / "demo-adapter"
    base.mkdir(parents=True, exist_ok=True)
    adapter.mkdir(parents=True, exist_ok=True)
    hidden, vocab = 64, 512
    base_tensors: dict[str, tuple[str, tuple[int, ...]]] = {
        "language_model.model.embed_tokens.weight": ("F16", (vocab, hidden)),
    }
    adapter_tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
    for layer in range(2):
        for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
            base_tensors[f"language_model.model.layers.{layer}.self_attn.{proj}.weight"] = (
                "F16",
                (hidden, hidden),
            )
        for proj, rank in (("q_proj", 8), ("v_proj", 16)):
            stem = f"base_model.model.model.layers.{layer}.self_attn.{proj}"
            adapter_tensors[f"{stem}.lora_A.weight"] = ("F32", (rank, hidden))
            adapter_tensors[f"{stem}.lora_B.weight"] = ("F32", (hidden, rank))
    write_safetensors(base / "model.safetensors", base_tensors)
    (base / "config.json").write_text(json.dumps({"model_type": "demo", "vocab_size": vocab}))
    write_safetensors(adapter / "adapter_model.safetensors", adapter_tensors)
    (adapter / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "demo-base",
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj"],
                "rank_pattern": {"v_proj": 16},
                "alpha_pattern": {"v_proj": 64},
                "use_rslora": False,
                "use_dora": False,
                "bias": "none",
            },
            indent=2,
        )
    )
    return adapter, base


def demo_lint(engine: str | None = "vllm", engine_version: str | None = "0.30.0") -> LintReport:
    """Lint the bundled demo adapter against its demo base."""
    with tempfile.TemporaryDirectory(prefix="lora-check-demo-") as tmp:
        adapter, base = write_demo_adapter(Path(tmp))
        return lint_adapter(
            LocalSource(adapter, label="demo-adapter"),
            LocalSource(base, label="demo-base"),
            engine,
            engine_version,
        )
