from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from lora_check.client import Client, EndpointError, ModelInfo, normalise_base_url
from lora_check.sim import Faults, SimConfig, SimServer


def test_normalise_base_url() -> None:
    assert normalise_base_url("http://h:8000") == "http://h:8000/v1"
    assert normalise_base_url("http://h:8000/v1/") == "http://h:8000/v1"
    assert normalise_base_url("h:8000") == "http://h:8000/v1"
    with pytest.raises(EndpointError):
        normalise_base_url("  ")


def test_list_models_reports_parent_and_root(client_for: Callable[..., Client]) -> None:
    models = client_for(SimServer()).list_models()
    assert models[0] == ModelInfo(id="toy-base", parent=None, root="toy-base")
    assert [m.id for m in models if m.is_adapter] == ["alpha", "beta"]
    assert all(m.parent == "toy-base" for m in models[1:])


@pytest.mark.parametrize("api", ["completions", "chat"])
def test_next_token_parses_both_apis(client_for: Callable[..., Client], api: str) -> None:
    client = client_for(SimServer(), api=api, top_logprobs=7)
    dist = client.next_token("alpha", "The capital of France is")
    assert len(dist) == 7
    assert all(v <= 0 for v in dist.values())
    assert client.request_count == 1


def test_token_ids_and_request_body(client_for: Callable[..., Client]) -> None:
    seen: list[dict[str, Any]] = []
    sim = SimServer()

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        status, payload = sim.handle(request.method, request.url.path, body)
        return httpx.Response(status, json=payload)

    client = Client(
        "http://sim",
        transport=httpx.MockTransport(handler),
        token_ids=True,
        api_key="secret",
        extra_body={"seed": 7},
    )
    dist = client.next_token("alpha", "hello", cache_salt="s1")
    assert all(k.startswith("token_id:") for k in dist)
    body = seen[0]
    assert body["temperature"] == 0
    assert body["max_tokens"] == 1
    assert body["logprobs"] == 20
    assert body["cache_salt"] == "s1"
    assert body["seed"] == 7
    assert body["return_tokens_as_token_ids"] is True


def test_missing_logprobs_is_a_clear_error(client_for: Callable[..., Client]) -> None:
    client = client_for(SimServer(SimConfig(faults=Faults(no_logprobs=True))))
    with pytest.raises(EndpointError, match="logprobs"):
        client.next_token("alpha", "hello")


def test_unknown_model_is_reported_without_retry(client_for: Callable[..., Client]) -> None:
    sim = SimServer()
    client = client_for(sim)
    with pytest.raises(EndpointError, match="404"):
        client.next_token("nope", "hello")
    assert sim.requests == 0


def test_too_many_logprobs_is_rejected(client_for: Callable[..., Client]) -> None:
    client = client_for(SimServer(SimConfig(max_logprobs=5)), top_logprobs=20)
    with pytest.raises(EndpointError, match="400"):
        client.next_token("alpha", "hello")


def test_transport_errors_are_retried_then_raised() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("refused")

    client = Client("http://sim", transport=httpx.MockTransport(handler), retries=1)
    with pytest.raises(EndpointError, match="refused"):
        client.list_models()
    assert calls == 2


def test_server_errors_are_retried() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"data": [{"id": "m"}]})

    client = Client("http://sim", transport=httpx.MockTransport(handler))
    assert client.list_models() == [ModelInfo("m")]


def test_bad_api_name() -> None:
    with pytest.raises(EndpointError):
        Client("http://sim", api="embeddings")
