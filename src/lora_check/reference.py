"""Compute a reference fingerprint with transformers + PEFT.

This is the only module that needs the heavy optional dependencies
(``pip install "lora-check[fingerprint]"`` after installing a suitable torch
build).  Run it where the adapter was trained, or on any machine that can
load the base model; CPU is fine for small models.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from lora_check import __version__
from lora_check.fingerprint import AdapterRef, DistTable, Fingerprint, RefDist, Variant
from lora_check.probes import Probe

DEFAULT_SCALES: tuple[float, ...] = (0.25, 0.5, 0.75, 1.5, 2.0, 4.0)
_ADAPTER = "lora_check"
_CONFIG_KEYS = (
    "r",
    "lora_alpha",
    "use_rslora",
    "use_dora",
    "rank_pattern",
    "alpha_pattern",
    "target_modules",
    "bias",
    "modules_to_save",
    "base_model_name_or_path",
)


class ReferenceError(RuntimeError):
    """The reference could not be computed."""


def _import_stack() -> tuple[Any, Any, Any]:
    try:
        import peft
        import torch
        import transformers
    except ImportError as exc:
        raise ReferenceError(
            "computing a fingerprint needs torch, transformers and peft: install a torch build for "
            'your machine, then `pip install "lora-check[fingerprint]"` '
            f"(import failed: {exc})"
        ) from exc
    return torch, transformers, peft


def _jsonable(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        return sorted(str(v) for v in value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class _Scorer:
    """Next-token top-K for a list of probes under whatever the model currently is."""

    def __init__(self, torch: Any, tokenizer: Any, device: str, top_k: int, chat: bool) -> None:
        self.torch = torch
        self.tokenizer = tokenizer
        self.device = device
        self.top_k = top_k
        self.chat = chat
        self._text: dict[int, str] = {}

    def encode(self, prompt: str) -> Any:
        if self.chat:
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
            return self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        return self.tokenizer(prompt, return_tensors="pt")

    def one(self, model: Any, prompt: str) -> RefDist:
        torch = self.torch
        encoded = {k: v.to(self.device) for k, v in self.encode(prompt).items()}
        with torch.no_grad():
            logits = model(**encoded).logits[0, -1].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        k = min(self.top_k, int(logprobs.shape[-1]))
        values, indices = torch.topk(logprobs, k)
        entries = []
        for lp, token_id in zip(values.tolist(), indices.tolist()):
            if token_id not in self._text:
                self._text[token_id] = self.tokenizer.decode([token_id])
            entries.append((int(token_id), self._text[token_id], float(lp)))
        return RefDist(tuple(entries))

    def table(self, model: Any, probes: Sequence[Probe]) -> DistTable:
        return {p.id: self.one(model, p.prompt) for p in probes}


def compute_fingerprint(
    base_model: str,
    adapters: dict[str, str],
    probes: Sequence[Probe],
    top_k: int = 20,
    dtype: str = "float32",
    device: str = "cpu",
    chat: bool = False,
    scales: Sequence[float] = DEFAULT_SCALES,
    variants: bool = True,
    revision: str | None = None,
    trust_remote_code: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Fingerprint:
    """Reference distributions of ``base_model`` and each adapter on ``probes``.

    For every adapter the fingerprint also stores *variants*: the adapter at
    other global scales, with ``rank_pattern``/``alpha_pattern`` ignored and
    with rsLoRA ignored (each only when it would change anything).  They are
    produced by editing ``LoraLayer.scaling`` in PEFT, i.e. with the real
    model maths.
    """
    torch, transformers, peft = _import_stack()
    from peft.tuners.lora import LoraLayer

    def say(message: str) -> None:
        if progress is not None:
            progress(message)

    if not hasattr(torch, dtype):
        raise ReferenceError(f"unknown torch dtype {dtype!r}")
    say(f"loading base model {base_model}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model, revision=revision, trust_remote_code=trust_remote_code
    )
    # transformers renamed `torch_dtype` to `dtype`; support both spellings.
    major_minor = tuple(int(x) for x in str(transformers.__version__).split(".")[:2] if x.isdigit())
    dtype_kwarg = "dtype" if major_minor >= (4, 56) else "torch_dtype"
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        revision=revision,
        trust_remote_code=trust_remote_code,
        **{dtype_kwarg: getattr(torch, dtype)},
    )
    model.to(device)
    model.eval()
    scorer = _Scorer(torch, tokenizer, device, top_k, chat)
    probe_list = list(probes)
    base_table = scorer.table(model, probe_list)

    refs: dict[str, AdapterRef] = {}
    for name, path in adapters.items():
        say(f"adapter {name}: loading {path}")
        peft_model = peft.PeftModel.from_pretrained(model, path, adapter_name=_ADAPTER)
        peft_model.eval()
        config = peft_model.peft_config[_ADAPTER]
        layers = [
            m for m in peft_model.modules() if isinstance(m, LoraLayer) and _ADAPTER in m.scaling
        ]
        if not layers:
            raise ReferenceError(
                f"adapter {name!r} ({path}) attached no LoRA layers to {base_model}"
            )
        original = [float(layer.scaling[_ADAPTER]) for layer in layers]

        def apply(values: Sequence[float], layers: list[Any] = layers) -> None:
            for layer, value in zip(layers, values):
                layer.scaling[_ADAPTER] = value

        reference = scorer.table(peft_model, probe_list)
        found: list[Variant] = []
        if variants:
            for s in scales:
                apply([value * s for value in original])
                found.append(
                    Variant(
                        name=f"scale={s:g}",
                        kind="scale",
                        description=f"the adapter applied at {s:g}x its intended scale",
                        dists=scorer.table(peft_model, probe_list),
                        scale=float(s),
                    )
                )
            r = float(config.r)
            alpha = float(config.lora_alpha)
            rslora = bool(getattr(config, "use_rslora", False))
            uniform = alpha / math.sqrt(r) if rslora else alpha / r
            if any(abs(value - uniform) > 1e-9 * max(1.0, abs(uniform)) for value in original):
                apply([uniform] * len(layers))
                found.append(
                    Variant(
                        name="patterns-ignored",
                        kind="patterns-ignored",
                        description="one global lora_alpha/r scale applied to every module, ignoring "
                        "rank_pattern/alpha_pattern (the behaviour reported in vllm#59799)",
                        dists=scorer.table(peft_model, probe_list),
                    )
                )
            if rslora:
                apply(
                    [
                        float(layer.lora_alpha[_ADAPTER]) / float(layer.r[_ADAPTER])
                        for layer in layers
                    ]
                )
                found.append(
                    Variant(
                        name="rslora-ignored",
                        kind="rslora-ignored",
                        description="lora_alpha/r used instead of lora_alpha/sqrt(r), i.e. use_rslora ignored",
                        dists=scorer.table(peft_model, probe_list),
                    )
                )
            apply(original)
        config_dict = config.to_dict()
        refs[name] = AdapterRef(
            name=name,
            reference=reference,
            variants=found,
            config={
                **{k: _jsonable(config_dict.get(k)) for k in _CONFIG_KEYS if k in config_dict},
                "path": path,
                "lora_layers": len(layers),
                "distinct_scalings": sorted({round(v, 6) for v in original}),
            },
        )
        model = peft_model.unload()
        model.eval()
        check = scorer.one(model, probe_list[0].prompt)
        drift = max(
            abs(a[2] - b[2]) for a, b in zip(check.entries, base_table[probe_list[0].id].entries)
        )
        if drift > 1e-4:
            raise ReferenceError(
                f"base model output changed by {drift:.2g} after unloading adapter {name!r}; "
                "the adapter modifies the base weights (bias, modules_to_save or a base-modifying init)"
            )

    return Fingerprint(
        base_model=base_model,
        probes=probe_list,
        base=base_table,
        adapters=refs,
        api="chat" if chat else "completions",
        top_k=top_k,
        meta={
            "reference": "transformers+peft",
            "lora_check": __version__,
            "torch": str(torch.__version__),
            "transformers": str(transformers.__version__),
            "peft": str(peft.__version__),
            "dtype": dtype,
            "device": device,
            "revision": revision,
        },
    )
