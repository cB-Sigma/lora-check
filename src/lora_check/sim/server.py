"""A fake OpenAI-compatible multi-LoRA server with injectable faults.

The simulator serves the toy model from :mod:`lora_check.sim.model` behind
the same HTTP surface lora-check probes on real engines (``/v1/models`` with
``parent``/``root``, ``/v1/completions`` and ``/v1/chat/completions`` with
log-probs, vLLM-style ``/v1/load_lora_adapter``).  Each fault reproduces the
*observable shape* of a failure class reported against real engines; none of
them claims to reproduce an engine's internals.

It is used by ``--demo``, by the test-suite and by the evaluation script.
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Protocol

import httpx

from lora_check.fingerprint import AdapterRef, Fingerprint, RefDist, Variant
from lora_check.probes import Probe
from lora_check.sim.model import ToyAdapter, ToyLM, Vector, log_softmax, tokenize, top_k

DEFAULT_SCALES: tuple[float, ...] = (0.25, 0.5, 0.75, 1.5, 2.0, 4.0)


@dataclass
class Faults:
    """Faults to inject.  All default to "healthy"."""

    ignored: frozenset[str] = frozenset()
    """Adapters that load fine but are never applied (base output)."""
    scale: dict[str, float] = field(default_factory=dict)
    """Adapter -> multiplier on its nominal scale (a wrong alpha/r)."""
    alias: dict[str, str] = field(default_factory=dict)
    """Requests for the key are served with the value's weights (swap, duplicate)."""
    leak_after_switch: float = 0.0
    """Probability that the first request after a model switch runs with the
    previous request's adapter state."""
    batch_mix: float = 0.0
    """Probability that a request overlapping other models' requests is
    computed with one of *their* adapters."""
    cache_ignores_adapter: bool = False
    """Prefix-cache keys omit the adapter, so adapters reuse each other's state."""
    stale_cache_on_reload: bool = False
    """Prefix-cache keys use only the adapter *name*, so an in-place reload
    keeps serving state computed with the old weights."""
    no_logprobs: bool = False
    """Answer without log-probs, like an engine that does not support them."""


@dataclass
class SimConfig:
    """Shape of the simulated deployment."""

    base_model: str = "toy-base"
    adapters: tuple[str, ...] = ("alpha", "beta")
    seed: int = 0
    noise: float = 0.0
    """Std-dev of per-request Gaussian noise on the logits (run-to-run jitter)."""
    batch_noise: float = 0.0
    """Extra noise when other requests are in flight (batch-composition jitter)."""
    systematic: float = 0.0
    """Std-dev of a fixed per-prompt logit offset (engine vs reference numerics)."""
    strength: dict[str, float] = field(default_factory=dict)
    """Adapter -> effect strength (1.0 default; small = narrow adapter)."""
    max_logprobs: int = 20
    prefix_cache: bool = True
    latency: float = 0.0
    """Seconds each request holds its slot; makes concurrent requests overlap."""
    faults: Faults = field(default_factory=Faults)


@dataclass
class _Loaded:
    adapter: ToyAdapter
    generation: int


class SimError(Exception):
    """An error response: ``(status, message)``."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


class SimServer:
    """The simulated engine.  ``handle`` is the whole HTTP surface."""

    def __init__(self, config: SimConfig | None = None) -> None:
        self.config = config or SimConfig()
        self.lm = ToyLM(seed=self.config.seed)
        self._lock = threading.Lock()
        self._loaded: dict[str, _Loaded] = {}
        self._cache: dict[tuple[str, int, str, tuple[int, ...]], Vector] = {}
        self._counters: dict[tuple[str, str], int] = {}
        self._inflight: list[str] = []
        self._previous: str | None = None
        self.requests = 0
        for name in self.config.adapters:
            self._loaded[name] = _Loaded(self.make_adapter(name), 0)

    # -- adapters ----------------------------------------------------------

    def make_adapter(self, name: str, seed: str | None = None) -> ToyAdapter:
        """The adapter weights the server (and its reference) use for ``name``."""
        return self.lm.make_adapter(name, seed=seed, strength=self.config.strength.get(name, 1.0))

    def adapter(self, name: str) -> ToyAdapter:
        return self._loaded[name].adapter

    def load_adapter(self, name: str, path: str, inplace: bool = False) -> None:
        """vLLM-style dynamic load; ``path`` seeds the weights."""
        with self._lock:
            current = self._loaded.get(name)
            if current is not None and not inplace:
                raise SimError(400, f"The lora adapter '{name}' has already been loaded.")
            generation = current.generation + 1 if current is not None else 0
            adapter = self.lm.make_adapter(
                name, seed=path, strength=self.config.strength.get(name, 1.0)
            )
            self._loaded[name] = _Loaded(adapter, generation)

    def unload_adapter(self, name: str) -> None:
        with self._lock:
            if name not in self._loaded:
                raise SimError(404, f"The lora adapter '{name}' cannot be found.")
            del self._loaded[name]

    # -- request handling --------------------------------------------------

    def handle(
        self, method: str, path: str, body: dict[str, Any] | None
    ) -> tuple[int, dict[str, Any]]:
        """Dispatch one request; returns ``(status, json)``."""
        route = path.split("?", 1)[0].rstrip("/")
        try:
            if method == "GET" and route == "/v1/models":
                return 200, self._models()
            if method == "POST" and route == "/v1/completions":
                return 200, self._completion(body or {}, chat=False)
            if method == "POST" and route == "/v1/chat/completions":
                return 200, self._completion(body or {}, chat=True)
            if method == "POST" and route == "/v1/load_lora_adapter":
                payload = body or {}
                self.load_adapter(
                    str(payload.get("lora_name", "")),
                    str(payload.get("lora_path", "")),
                    bool(payload.get("load_inplace", False)),
                )
                return 200, {
                    "message": f"Success: LoRA adapter '{payload.get('lora_name')}' added successfully"
                }
            if method == "POST" and route == "/v1/unload_lora_adapter":
                self.unload_adapter(str((body or {}).get("lora_name", "")))
                return 200, {"message": "Success"}
        except SimError as exc:
            return exc.status, {"error": {"message": exc.message, "type": "invalid_request_error"}}
        return 404, {"error": {"message": f"no route {method} {path}", "type": "not_found"}}

    def _models(self) -> dict[str, Any]:
        base = self.config.base_model
        data: list[dict[str, Any]] = [
            {
                "id": base,
                "object": "model",
                "owned_by": "lora-check-sim",
                "root": base,
                "parent": None,
            }
        ]
        with self._lock:
            names = list(self._loaded)
        data.extend(
            {
                "id": name,
                "object": "model",
                "owned_by": "lora-check-sim",
                "root": f"sim://{name}",
                "parent": base,
            }
            for name in names
        )
        return {"object": "list", "data": data}

    def _uniform(self, tag: str, model: str, prompt: str, n: int) -> random.Random:
        return random.Random(f"{self.config.seed}|{tag}|{model}|{n}|{prompt}")

    def _resolve(self, model: str) -> str | None:
        """Requested model id -> adapter name (None for the base)."""
        base = self.config.base_model
        if model == base:
            return None
        name = model[len(base) + 1 :] if model.startswith(base + ":") else model
        if name not in self._loaded:
            raise SimError(404, f"The model `{model}` does not exist.")
        return name

    def _completion(self, body: dict[str, Any], chat: bool) -> dict[str, Any]:
        cfg = self.config
        faults = cfg.faults
        model = str(body.get("model", ""))
        if chat:
            messages = body.get("messages") or []
            prompt = (
                "".join(f"{m.get('role')}: {m.get('content')}\n" for m in messages) + "assistant:"
            )
            want = int(body.get("top_logprobs") or 0) if body.get("logprobs") else 0
        else:
            raw_prompt = body.get("prompt", "")
            prompt = raw_prompt if isinstance(raw_prompt, str) else str(raw_prompt)
            want = int(body.get("logprobs") or 0)
        if want > cfg.max_logprobs:
            raise SimError(400, f"logprobs must be at most {cfg.max_logprobs}")
        salt = str(body.get("cache_salt") or "")

        with self._lock:
            requested = self._resolve(model)
            self.requests += 1
            count = self._counters.get((model, prompt), 0)
            self._counters[(model, prompt)] = count + 1
            effective = faults.alias.get(requested, requested) if requested else None
            if effective in faults.ignored or requested in faults.ignored:
                effective = None
            previous = self._previous
            if (
                faults.leak_after_switch > 0
                and previous is not None
                and previous != model
                and self._uniform("leak", model, prompt, count).random() < faults.leak_after_switch
            ):
                effective = self._effective_of(previous)
            others = [m for m in self._inflight if m != model]
            if (
                faults.batch_mix > 0
                and others
                and self._uniform("mix", model, prompt, count).random() < faults.batch_mix
            ):
                effective = self._effective_of(others[0])
            self._previous = model
            busy = bool(self._inflight)
            self._inflight.append(model)
            loaded = self._loaded.get(effective) if effective else None
        try:
            if cfg.latency > 0:
                time.sleep(cfg.latency)
            adapter = loaded.adapter if loaded else None
            scale = 0.0
            if adapter is not None:
                scale = adapter.scaling * faults.scale.get(adapter.name, 1.0)
            tokens = tokenize(prompt)
            state = self._hidden(tokens, adapter, scale, loaded.generation if loaded else 0, salt)
            logits = self.lm.head(state, adapter, scale)
            sigma = cfg.noise + (cfg.batch_noise if busy else 0.0)
            if sigma > 0:
                rng = self._uniform("noise", model, prompt, count)
                logits = [x + rng.gauss(0.0, sigma) for x in logits]
            if cfg.systematic > 0:
                rng = random.Random(f"{cfg.seed}|sys|{adapter.name if adapter else ''}|{prompt}")
                logits = [x + rng.gauss(0.0, cfg.systematic) for x in logits]
            entries = top_k(log_softmax(logits), max(want, 1))
        finally:
            with self._lock:
                self._inflight.remove(model)

        as_ids = bool(body.get("return_tokens_as_token_ids"))

        def key(token_id: int, text: str) -> str:
            return f"token_id:{token_id}" if as_ids else text

        best_id, best_text, best_lp = entries[0]
        show = want > 0 and not faults.no_logprobs
        usage = {
            "prompt_tokens": len(tokens),
            "completion_tokens": 1,
            "total_tokens": len(tokens) + 1,
        }
        if chat:
            chat_logprobs = None
            if show:
                chat_logprobs = {
                    "content": [
                        {
                            "token": key(best_id, best_text),
                            "logprob": best_lp,
                            "top_logprobs": [
                                {"token": key(i, t), "logprob": lp} for i, t, lp in entries[:want]
                            ],
                        }
                    ]
                }
            return {
                "id": f"chatcmpl-sim-{self.requests}",
                "object": "chat.completion",
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": best_text},
                        "logprobs": chat_logprobs,
                        "finish_reason": "length",
                    }
                ],
                "usage": usage,
            }
        text_logprobs = None
        if show:
            text_logprobs = {
                "tokens": [key(best_id, best_text)],
                "token_logprobs": [best_lp],
                "top_logprobs": [{key(i, t): lp for i, t, lp in entries[:want]}],
                "text_offset": [0],
            }
        return {
            "id": f"cmpl-sim-{self.requests}",
            "object": "text_completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "text": best_text,
                    "logprobs": text_logprobs,
                    "finish_reason": "length",
                }
            ],
            "usage": usage,
        }

    def _effective_of(self, model: str) -> str | None:
        """Adapter that a *healthy* request for ``model`` would run with."""
        base = self.config.base_model
        if model == base:
            return None
        name = model[len(base) + 1 :] if model.startswith(base + ":") else model
        return name if name in self._loaded else None

    def _hidden(
        self,
        tokens: list[int],
        adapter: ToyAdapter | None,
        scale: float,
        generation: int,
        salt: str,
    ) -> Vector:
        """Final hidden state, going through the (possibly faulty) prefix cache."""
        cfg = self.config
        if not cfg.prefix_cache:
            return self.lm.run(tokens, adapter, scale)
        name = adapter.name if adapter is not None else ""
        if cfg.faults.cache_ignores_adapter:
            name, generation = "", 0
        elif cfg.faults.stale_cache_on_reload:
            generation = 0
        start = 0
        state = self.lm.initial_state()
        with self._lock:
            for n in range(len(tokens), 0, -1):
                hit = self._cache.get((name, generation, salt, tuple(tokens[:n])))
                if hit is not None:
                    state, start = hit, n
                    break
        fresh: list[tuple[int, Vector]] = []
        for n in range(start, len(tokens)):
            state = self.lm.step(state, tokens[n], adapter, scale)
            fresh.append((n + 1, state))
        if fresh:
            with self._lock:
                if len(self._cache) > 200_000:
                    self._cache.clear()
                for n, value in fresh:
                    self._cache.setdefault((name, generation, salt, tuple(tokens[:n])), value)
        return state

    # -- ground truth ------------------------------------------------------

    def reference_fingerprint(
        self,
        probes: list[Probe],
        top: int = 20,
        scales: tuple[float, ...] = DEFAULT_SCALES,
        adapters: dict[str, ToyAdapter] | None = None,
    ) -> Fingerprint:
        """What a correct server would answer: the simulator's ground truth.

        Plays the role `lora-check fingerprint` plays for real models.
        """
        chosen = adapters or {name: self.make_adapter(name) for name in self.config.adapters}

        def table(adapter: ToyAdapter | None, scale: float | None) -> dict[str, RefDist]:
            return {
                p.id: RefDist(tuple(self.lm.top_logprobs(p.prompt, adapter, scale, top)))
                for p in probes
            }

        refs: dict[str, AdapterRef] = {}
        for name, adapter in chosen.items():
            variants = [
                Variant(
                    name=f"scale={s:g}",
                    kind="scale",
                    description=f"the adapter applied at {s:g}x its intended scale",
                    dists=table(adapter, adapter.scaling * s),
                    scale=s,
                )
                for s in scales
            ]
            refs[name] = AdapterRef(
                name=name,
                reference=table(adapter, None),
                variants=variants,
                config={"r": adapter.rank, "lora_alpha": adapter.alpha, "use_rslora": False},
            )
        return Fingerprint(
            base_model=self.config.base_model,
            probes=list(probes),
            base=table(None, None),
            adapters=refs,
            top_k=top,
            meta={"reference": "lora-check simulator (toy model)", "seed": self.config.seed},
        )


# ---------------------------------------------------------------------------
# Transports
# ---------------------------------------------------------------------------


class App(Protocol):
    """Anything that can answer ``(method, path, json body)``."""

    def handle(
        self, method: str, path: str, body: dict[str, Any] | None
    ) -> tuple[int, dict[str, Any]]: ...


def mock_transport(sim: App) -> httpx.MockTransport:
    """An in-process httpx transport: no sockets, same code path as HTTP."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        status, payload = sim.handle(request.method, request.url.path, body)
        return httpx.Response(status, json=payload)

    return httpx.MockTransport(handler)


def _make_handler(sim: App) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True

        def _reply(self, body: dict[str, Any] | None) -> None:
            status, payload = sim.handle(self.command, self.path, body)
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            self._reply(None)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                body = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                body = {}
            self._reply(body if isinstance(body, dict) else {})

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


@contextmanager
def running_server(sim: App, host: str = "127.0.0.1", port: int = 0) -> Iterator[str]:
    """Serve ``sim`` over real HTTP; yields the base URL (without ``/v1``)."""
    httpd = ThreadingHTTPServer((host, port), _make_handler(sim))
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://{host}:{httpd.server_address[1]}"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)
