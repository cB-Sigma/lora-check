"""Minimal OpenAI-compatible client: model discovery and next-token log-probs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from lora_check.dist import Dist, clean

TOKEN_ID_PREFIX = "token_id:"


class EndpointError(RuntimeError):
    """The endpoint could not be reached or did not answer usefully."""


@dataclass(frozen=True)
class ModelInfo:
    """One entry of ``/v1/models``.

    vLLM and SGLang list every loaded LoRA adapter as a model whose
    ``parent`` is the base model and whose ``root`` is the adapter path.
    """

    id: str
    parent: str | None = None
    root: str | None = None

    @property
    def is_adapter(self) -> bool:
        return bool(self.parent)


def normalise_base_url(url: str) -> str:
    """Accept ``http://host:8000`` or ``http://host:8000/v1``; return the latter."""
    stripped = url.strip().rstrip("/")
    if not stripped:
        raise EndpointError("empty base URL")
    if not stripped.startswith(("http://", "https://")):
        stripped = "http://" + stripped
    return stripped if stripped.endswith("/v1") else stripped + "/v1"


@dataclass
class Client:
    """Thread-safe client for the two endpoints lora-check needs."""

    base_url: str
    api_key: str | None = None
    api: str = "completions"
    top_logprobs: int = 20
    timeout: float = 60.0
    retries: int = 2
    token_ids: bool = False
    extra_body: dict[str, Any] = field(default_factory=dict)
    transport: httpx.BaseTransport | None = None
    request_count: int = field(default=0, init=False)
    _http: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.api not in ("completions", "chat"):
            raise EndpointError(f"unknown api {self.api!r}; use 'completions' or 'chat'")
        self.base_url = normalise_base_url(self.base_url)
        headers = {"User-Agent": "lora-check"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        self._http = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=self.timeout,
            transport=self.transport,
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- low level ---------------------------------------------------------

    def _request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                response = self._http.request(method, path, json=body)
            except httpx.HTTPError as exc:
                last = EndpointError(f"{method} {self.base_url}{path} failed: {exc}")
            else:
                if response.status_code < 400:
                    try:
                        return response.json()
                    except ValueError as exc:
                        raise EndpointError(
                            f"{method} {path}: response is not JSON: {response.text[:200]!r}"
                        ) from exc
                detail = response.text[:300].replace("\n", " ")
                last = EndpointError(
                    f"{method} {path} returned HTTP {response.status_code}: {detail}"
                )
                if response.status_code < 500 and response.status_code != 429:
                    break
            if attempt < self.retries:
                time.sleep(0.2 * (attempt + 1))
        assert last is not None
        raise last

    # -- public API --------------------------------------------------------

    def list_models(self) -> list[ModelInfo]:
        """Return the entries of ``/v1/models``."""
        payload = self._request("GET", "/models")
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list):
            raise EndpointError("GET /models: response has no 'data' list")
        models: list[ModelInfo] = []
        for entry in data:
            if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
                continue
            parent = entry.get("parent")
            root = entry.get("root")
            models.append(
                ModelInfo(
                    id=entry["id"],
                    parent=parent if isinstance(parent, str) and parent else None,
                    root=root if isinstance(root, str) and root else None,
                )
            )
        return models

    def next_token(self, model: str, prompt: str, cache_salt: str | None = None) -> Dist:
        """Top-K next-token log-probabilities for ``prompt`` under ``model``.

        Uses greedy settings (temperature 0, one token).  The returned keys
        are token strings, or ``token_id:<n>`` when ``token_ids`` is set and
        the server supports vLLM's ``return_tokens_as_token_ids``.
        """
        body: dict[str, Any] = {"model": model, "max_tokens": 1, "temperature": 0}
        if self.api == "completions":
            body["prompt"] = prompt
            body["logprobs"] = self.top_logprobs
        else:
            body["messages"] = [{"role": "user", "content": prompt}]
            body["logprobs"] = True
            body["top_logprobs"] = self.top_logprobs
        if self.token_ids:
            body["return_tokens_as_token_ids"] = True
        if cache_salt is not None:
            body["cache_salt"] = cache_salt
        body.update(self.extra_body)
        path = "/completions" if self.api == "completions" else "/chat/completions"
        payload = self._request("POST", path, body)
        self.request_count += 1
        return self._parse(payload, model)

    def _parse(self, payload: Any, model: str) -> Dist:
        hint = (
            "the server returned no next-token logprobs; lora-check needs an endpoint that "
            "supports the 'logprobs' request parameter"
        )
        try:
            logprobs = payload["choices"][0]["logprobs"]
        except (KeyError, IndexError, TypeError) as exc:
            raise EndpointError(f"model {model!r}: malformed response ({hint})") from exc
        if not logprobs:
            raise EndpointError(f"model {model!r}: {hint}")
        raw: dict[str, float] = {}
        if self.api == "completions":
            top = logprobs.get("top_logprobs") if isinstance(logprobs, dict) else None
            if not top or not isinstance(top[0], dict):
                raise EndpointError(f"model {model!r}: {hint}")
            raw = {str(k): float(v) for k, v in top[0].items() if v is not None}
        else:
            content = logprobs.get("content") if isinstance(logprobs, dict) else None
            if not content or not isinstance(content[0], dict):
                raise EndpointError(f"model {model!r}: {hint}")
            for item in content[0].get("top_logprobs") or []:
                if isinstance(item, dict) and "token" in item and item.get("logprob") is not None:
                    token = str(item["token"])
                    lp = float(item["logprob"])
                    # Distinct token ids can decode to the same text; keep the likelier.
                    raw[token] = max(lp, raw.get(token, float("-inf")))
        dist = clean(raw)
        if not dist:
            raise EndpointError(f"model {model!r}: {hint}")
        return dist
