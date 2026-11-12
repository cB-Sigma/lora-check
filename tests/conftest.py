"""Shared fixtures."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from lora_check.client import Client
from lora_check.lint import write_safetensors
from lora_check.probes import Probe, builtin_probes
from lora_check.sim import SimServer, mock_transport


@pytest.fixture
def probes() -> list[Probe]:
    return builtin_probes(16)


@pytest.fixture
def client_for() -> Callable[..., Client]:
    """Build a client wired straight into a simulator (no sockets)."""

    def make(sim: SimServer, **kwargs: Any) -> Client:
        return Client("http://sim", transport=mock_transport(sim), **kwargs)

    return make


AdapterFactory = Callable[..., Path]


@pytest.fixture
def make_adapter(tmp_path: Path) -> AdapterFactory:
    """Write a zero-filled PEFT-style adapter directory."""

    def make(
        name: str = "adapter",
        config: dict[str, Any] | None = None,
        modules: dict[str, tuple[int, int, int]] | None = None,
        prefix: str = "base_model.model.",
        extra_tensors: dict[str, tuple[str, tuple[int, ...]]] | None = None,
        files: dict[str, Any] | None = None,
    ) -> Path:
        path = tmp_path / name
        path.mkdir()
        cfg: dict[str, Any] = {
            "peft_type": "LORA",
            "r": 8,
            "lora_alpha": 16,
            "target_modules": ["q_proj", "v_proj"],
            "bias": "none",
        }
        cfg.update(config or {})
        (path / "adapter_config.json").write_text(json.dumps(cfg))
        mods = modules
        if mods is None:
            mods = {
                f"model.layers.{i}.self_attn.{proj}": (8, 32, 32)
                for i in range(2)
                for proj in ("q_proj", "v_proj")
            }
        tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
        for module, (rank, fan_in, fan_out) in mods.items():
            tensors[f"{prefix}{module}.lora_A.weight"] = ("F32", (rank, fan_in))
            tensors[f"{prefix}{module}.lora_B.weight"] = ("F32", (fan_out, rank))
        tensors.update(extra_tensors or {})
        write_safetensors(path / "adapter_model.safetensors", tensors)
        for filename, content in (files or {}).items():
            (path / filename).write_text(json.dumps(content))
        return path

    return make


@pytest.fixture
def make_base(tmp_path: Path) -> Callable[..., Path]:
    """Write a zero-filled base checkpoint directory."""

    def make(
        name: str = "base", prefix: str = "model.", hidden: int = 32, vocab: int = 100
    ) -> Path:
        path = tmp_path / name
        path.mkdir()
        tensors: dict[str, tuple[str, tuple[int, ...]]] = {
            f"{prefix}embed_tokens.weight": ("F16", (vocab, hidden))
        }
        for i in range(2):
            for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
                tensors[f"{prefix}layers.{i}.self_attn.{proj}.weight"] = ("F16", (hidden, hidden))
        write_safetensors(path / "model.safetensors", tensors)
        (path / "config.json").write_text(json.dumps({"vocab_size": vocab}))
        return path

    return make
