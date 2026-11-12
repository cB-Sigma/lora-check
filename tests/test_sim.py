from __future__ import annotations

import math

import httpx
import pytest

from lora_check.client import Client
from lora_check.dist import distance
from lora_check.sim import Faults, SimConfig, SimServer, ToyLM, running_server
from lora_check.sim.model import VOCAB, log_softmax, tokenize


def _dist(sim: SimServer, model: str, prompt: str, **extra: object) -> dict[str, float]:
    status, payload = sim.handle(
        "POST", "/v1/completions", {"model": model, "prompt": prompt, "logprobs": 20, **extra}
    )
    assert status == 200, payload
    top: dict[str, float] = payload["choices"][0]["logprobs"]["top_logprobs"][0]
    return top


def test_vocab_is_unique_and_tokenizer_is_total() -> None:
    assert len(set(VOCAB)) == len(VOCAB)
    assert tokenize("The cat, a zyzzyva!")  # unknown words still map somewhere
    assert tokenize("") == [VOCAB.index("<unk>")]


def test_model_is_deterministic_and_normalised() -> None:
    lm = ToyLM(seed=1)
    a = lm.logits("the cat is")
    assert a == ToyLM(seed=1).logits("the cat is")
    assert a != ToyLM(seed=2).logits("the cat is")
    assert math.isclose(sum(math.exp(x) for x in log_softmax(a)), 1.0, rel_tol=1e-9)


def test_adapter_changes_output_and_scale_zero_is_base() -> None:
    lm = ToyLM()
    adapter = lm.make_adapter("x")
    assert lm.logits("the cat is", adapter, 0.0) == lm.logits("the cat is")
    assert lm.logits("the cat is", adapter) != lm.logits("the cat is")
    assert lm.make_adapter("x").b_o == adapter.b_o
    assert lm.make_adapter("y").b_o != adapter.b_o


def test_healthy_server_matches_ground_truth_and_cache_is_transparent() -> None:
    sim = SimServer()
    prompt = "select the first order where"
    truth = {t: lp for _i, t, lp in sim.lm.top_logprobs(prompt, sim.adapter("alpha"))}
    first = _dist(sim, "alpha", prompt)
    assert first == pytest.approx(truth)
    _dist(sim, "beta", prompt)
    _dist(sim, "toy-base", prompt)
    assert _dist(sim, "alpha", prompt) == first  # served from the prefix cache
    assert _dist(sim, "alpha", prompt, cache_salt="x") == pytest.approx(first)
    no_cache = SimServer(SimConfig(prefix_cache=False))
    assert _dist(no_cache, "alpha", prompt) == pytest.approx(first)


def test_sglang_style_model_field_is_accepted() -> None:
    sim = SimServer()
    assert _dist(sim, "toy-base:alpha", "the cat") == _dist(sim, "alpha", "the cat")


def test_fault_ignored_and_alias_and_scale() -> None:
    prompt = "the patient is"
    healthy = SimServer()
    base = _dist(healthy, "toy-base", prompt)
    alpha = _dist(healthy, "alpha", prompt)
    beta = _dist(healthy, "beta", prompt)
    assert (
        _dist(SimServer(SimConfig(faults=Faults(ignored=frozenset({"alpha"})))), "alpha", prompt)
        == base
    )
    assert (
        _dist(SimServer(SimConfig(faults=Faults(alias={"alpha": "beta"}))), "alpha", prompt) == beta
    )
    half = _dist(SimServer(SimConfig(faults=Faults(scale={"alpha": 0.5}))), "alpha", prompt)
    assert 0 < distance(half, base) < distance(alpha, base)


def test_fault_leak_after_switch() -> None:
    sim = SimServer(SimConfig(faults=Faults(leak_after_switch=1.0), prefix_cache=False))
    prompt = "the model is"
    clean_alpha = _dist(SimServer(), "alpha", prompt)
    clean_base = _dist(SimServer(), "toy-base", prompt)
    assert _dist(sim, "toy-base", prompt) == clean_base  # nothing before it
    assert _dist(sim, "alpha", prompt) == clean_base  # leaked base state
    assert _dist(sim, "alpha", prompt) == clean_alpha  # no switch this time


def test_fault_shared_prefix_cache_depends_on_first_writer() -> None:
    prompt = "import the data and return"
    clean = _dist(SimServer(), "alpha", prompt)
    sim = SimServer(SimConfig(faults=Faults(cache_ignores_adapter=True)))
    _dist(sim, "toy-base", prompt)
    contaminated = _dist(sim, "alpha", prompt)
    assert distance(contaminated, clean) > 0.01
    assert _dist(sim, "alpha", prompt, cache_salt="fresh") == pytest.approx(clean)


def test_dynamic_load_and_stale_cache_fault() -> None:
    prompt = "error : the order was"
    sim = SimServer(SimConfig(faults=Faults(stale_cache_on_reload=True)))
    before = _dist(sim, "alpha", prompt)
    status, _ = sim.handle(
        "POST", "/v1/load_lora_adapter", {"lora_name": "alpha", "lora_path": "v2"}
    )
    assert status == 400  # already loaded, not in-place
    status, _ = sim.handle(
        "POST",
        "/v1/load_lora_adapter",
        {"lora_name": "alpha", "lora_path": "v2", "load_inplace": True},
    )
    assert status == 200
    stale = _dist(sim, "alpha", prompt)
    fresh = _dist(sim, "alpha", prompt, cache_salt="s")
    assert distance(stale, fresh) > 0.01
    assert distance(fresh, before) > 0.01
    healthy = SimServer()
    _dist(healthy, "alpha", prompt)
    healthy.load_adapter("alpha", "v2", inplace=True)
    assert _dist(healthy, "alpha", prompt) == pytest.approx(fresh)
    assert sim.handle("POST", "/v1/unload_lora_adapter", {"lora_name": "alpha"})[0] == 200
    assert sim.handle("POST", "/v1/unload_lora_adapter", {"lora_name": "alpha"})[0] == 404
    assert sim.handle("GET", "/nothing", None)[0] == 404


def test_noise_is_reproducible_per_request_index() -> None:
    a = SimServer(SimConfig(noise=0.01))
    b = SimServer(SimConfig(noise=0.01))
    first = _dist(a, "alpha", "the cat")
    second = _dist(a, "alpha", "the cat")
    assert first != second
    assert _dist(b, "alpha", "the cat") == first


def test_real_http_roundtrip() -> None:
    sim = SimServer()
    with running_server(sim) as url, Client(url) as client:
        assert [m.id for m in client.list_models()] == ["toy-base", "alpha", "beta"]
        assert len(client.next_token("beta", "the sea is")) == 20
        assert httpx.post(f"{url}/v1/completions", content=b"{not json").status_code == 404
