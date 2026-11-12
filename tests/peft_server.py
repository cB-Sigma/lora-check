"""A small reference server built on transformers + PEFT (test-only).

It answers the same HTTP surface lora-check probes on real engines, with the
real model maths behind it, so ``verify`` can be exercised end to end on CPU.
It is not an inference engine: one request at a time, one token per request.
"""

from __future__ import annotations

import threading
from typing import Any


class PeftApp:
    """Serve a base model and named PEFT adapters with next-token log-probs."""

    def __init__(
        self,
        base_model: str,
        adapters: dict[str, str],
        dtype: str = "float32",
        ignored: frozenset[str] = frozenset(),
        uniform_scale: frozenset[str] = frozenset(),
        max_logprobs: int = 20,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.base_model = base_model
        self.names = list(adapters)
        self.ignored = ignored
        self.max_logprobs = max_logprobs
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=getattr(torch, dtype))
        self.model: Any = None
        for name, path in adapters.items():
            if self.model is None:
                self.model = PeftModel.from_pretrained(model, path, adapter_name=name)
            else:
                self.model.load_adapter(path, adapter_name=name)
        assert self.model is not None
        self.model.eval()
        # Fault injection: apply one global lora_alpha/r to every module of
        # these adapters, ignoring rank_pattern/alpha_pattern (vllm#59799).
        for name in uniform_scale:
            config = self.model.peft_config[name]
            for module in self.model.modules():
                scaling = getattr(module, "scaling", None)
                if isinstance(scaling, dict) and name in scaling:
                    scaling[name] = config.lora_alpha / config.r
        self._lock = threading.Lock()

    def _logprobs(self, adapter: str | None, prompt: str, k: int) -> list[tuple[str, float]]:
        torch = self.torch
        encoded = self.tokenizer(prompt, return_tensors="pt")
        with self._lock, torch.no_grad():
            if adapter is None or adapter in self.ignored:
                with self.model.disable_adapter():
                    logits = self.model(**encoded).logits[0, -1]
            else:
                self.model.set_adapter(adapter)
                logits = self.model(**encoded).logits[0, -1]
        values, indices = torch.topk(torch.log_softmax(logits.float(), dim=-1), k)
        return [
            (f"token_id:{token_id}", float(lp))
            for lp, token_id in zip(values.tolist(), indices.tolist())
        ]

    def handle(
        self, method: str, path: str, body: dict[str, Any] | None
    ) -> tuple[int, dict[str, Any]]:
        route = path.rstrip("/")
        if method == "GET" and route == "/v1/models":
            data: list[dict[str, Any]] = [
                {"id": self.base_model, "object": "model", "root": self.base_model, "parent": None}
            ]
            data.extend(
                {
                    "id": name,
                    "object": "model",
                    "root": f"local://{name}",
                    "parent": self.base_model,
                }
                for name in self.names
            )
            return 200, {"object": "list", "data": data}
        if method == "POST" and route == "/v1/completions":
            payload = body or {}
            model = str(payload.get("model", ""))
            if model != self.base_model and model not in self.names:
                return 404, {"error": {"message": f"The model `{model}` does not exist."}}
            k = int(payload.get("logprobs") or 0)
            if k < 1 or k > self.max_logprobs:
                return 400, {"error": {"message": "logprobs out of range"}}
            entries = self._logprobs(
                None if model == self.base_model else model, str(payload.get("prompt", "")), k
            )
            return 200, {
                "object": "text_completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "text": "",
                        "logprobs": {
                            "tokens": [entries[0][0]],
                            "token_logprobs": [entries[0][1]],
                            "top_logprobs": [dict(entries)],
                        },
                        "finish_reason": "length",
                    }
                ],
            }
        return 404, {"error": {"message": "not found"}}
