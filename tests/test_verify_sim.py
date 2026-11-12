"""End-to-end checks of `verify` against the simulator."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from lora_check.client import Client, EndpointError, ModelInfo
from lora_check.probes import Probe
from lora_check.sim import Faults, SimConfig, SimServer, running_server
from lora_check.verify import (
    ERROR,
    FAIL,
    PASS,
    SKIP,
    WARN,
    VerifyConfig,
    VerifyReport,
    _schedule,
    discover,
    run_verify,
)

ADAPTERS = ("a1", "a2", "a3")
MakeClient = Callable[..., Client]


def _run(
    client_for: MakeClient,
    probes: list[Probe],
    faults: Faults | None = None,
    fingerprint: bool = False,
    noise: float = 0.002,
    sim_kwargs: dict[str, object] | None = None,
    **config: object,
) -> VerifyReport:
    sim = SimServer(
        SimConfig(adapters=ADAPTERS, noise=noise, faults=faults or Faults(), **(sim_kwargs or {}))  # type: ignore[arg-type]
    )
    config.setdefault("concurrency", 4)
    cfg = VerifyConfig(
        probes=probes,
        fingerprint=sim.reference_fingerprint(probes) if fingerprint else None,
        **config,  # type: ignore[arg-type]
    )
    return run_verify(client_for(sim), cfg)


def _status(report: VerifyReport, adapter: str, check: str) -> str:
    return next(a for a in report.adapters if a.name == adapter).checks[check].status


def test_healthy_server_passes_everything(client_for: MakeClient, probes: list[Probe]) -> None:
    report = _run(client_for, probes, fingerprint=True, cache_salt=True)
    assert report.verdict == PASS
    assert report.exit_code() == 0
    for adapter in ADAPTERS:
        for check in ("applied", "distinct", "stable", "cache", "reference"):
            assert _status(report, adapter, check) == PASS
    assert report.base.checks["stable"].status == PASS
    assert report.base.checks["reference"].status == PASS
    assert report.requests > 0
    payload = report.to_json()
    assert payload["verdict"] == PASS
    assert len(payload["adapters"]) == 3


def test_without_fingerprint_reference_is_skipped_and_noted(
    client_for: MakeClient, probes: list[Probe]
) -> None:
    report = _run(client_for, probes)
    assert report.verdict == PASS
    assert _status(report, "a1", "reference") == SKIP
    assert _status(report, "a1", "cache") == SKIP
    assert any("No fingerprint" in note for note in report.notes)


def test_ignored_adapter_fails_applied(client_for: MakeClient, probes: list[Probe]) -> None:
    report = _run(client_for, probes, Faults(ignored=frozenset({"a2"})), fingerprint=True)
    assert _status(report, "a2", "applied") == FAIL
    assert _status(report, "a1", "applied") == PASS
    reference = next(a for a in report.adapters if a.name == "a2").checks["reference"]
    assert reference.status == FAIL
    assert reference.evidence["cause"] == "adapter-ignored"
    assert report.exit_code() == 1


def test_all_adapters_ignored_fail_applied_and_distinct(
    client_for: MakeClient, probes: list[Probe]
) -> None:
    report = _run(client_for, probes, Faults(ignored=frozenset(ADAPTERS)))
    for adapter in ADAPTERS:
        assert _status(report, adapter, "applied") == FAIL
        assert _status(report, adapter, "distinct") == FAIL


def test_duplicate_weights_fail_distinct(client_for: MakeClient, probes: list[Probe]) -> None:
    report = _run(client_for, probes, Faults(alias={"a3": "a2"}))
    assert _status(report, "a2", "distinct") == FAIL
    assert _status(report, "a3", "distinct") == FAIL
    assert _status(report, "a1", "distinct") == PASS
    assert _status(report, "a3", "applied") == PASS


def test_wrong_scale_needs_fingerprint_and_names_the_scale(
    client_for: MakeClient, probes: list[Probe]
) -> None:
    faults = Faults(scale={"a2": 0.5})
    assert _run(client_for, probes, faults).verdict == PASS  # invisible without a reference
    report = _run(client_for, probes, faults, fingerprint=True)
    reference = next(a for a in report.adapters if a.name == "a2").checks["reference"]
    assert reference.status == FAIL
    assert reference.evidence["cause"] == "wrong-scale"
    assert "scale=0.5" in reference.summary
    assert reference.evidence["fitted_scale"] == pytest.approx(0.5, abs=0.15)
    assert _status(report, "a1", "reference") == PASS


def test_unusual_scale_is_estimated_by_fit(client_for: MakeClient, probes: list[Probe]) -> None:
    report = _run(client_for, probes, Faults(scale={"a2": 1.5}), fingerprint=True)
    reference = next(a for a in report.adapters if a.name == "a2").checks["reference"]
    assert reference.status == FAIL
    assert reference.evidence["cause"] == "wrong-scale"
    assert 1.2 < reference.evidence["fitted_scale"] < 1.8


def test_swapped_adapters_are_named(client_for: MakeClient, probes: list[Probe]) -> None:
    report = _run(client_for, probes, Faults(alias={"a1": "a3", "a3": "a1"}), fingerprint=True)
    for adapter, other in (("a1", "a3"), ("a3", "a1")):
        reference = next(a for a in report.adapters if a.name == adapter).checks["reference"]
        assert reference.status == FAIL
        assert reference.evidence["cause"] == "wrong-adapter"
        assert repr(other) in reference.summary
    assert _status(report, "a2", "reference") == PASS


def test_leak_after_switch_fails_stable_with_attribution(
    client_for: MakeClient, probes: list[Probe]
) -> None:
    report = _run(client_for, probes, Faults(leak_after_switch=1.0), concurrency=1)
    stable = next(a for a in report.adapters if a.name == "a1").checks["stable"]
    assert stable.status == FAIL
    assert "leakage" in stable.summary
    assert stable.evidence["answers_moved"] >= len(probes)
    assert stable.evidence["examples"][0]["after"] != "a1"
    assert report.base.checks["stable"].status == FAIL
    assert _status(report, "a1", "applied") == PASS  # isolated answers stay clean


def test_batch_mixing_fails_stable_only_under_concurrency(
    client_for: MakeClient, probes: list[Probe]
) -> None:
    report = _run(client_for, probes, Faults(batch_mix=0.5), sim_kwargs={"latency": 0.002})
    moved = [
        example
        for rep in [report.base, *report.adapters]
        for example in rep.checks["stable"].evidence.get("examples", [])
    ]
    assert moved
    assert {example["mode"] for example in moved} == {"concurrent"}
    assert any(rep.checks["stable"].status == FAIL for rep in report.adapters)


def test_benign_batch_noise_does_not_fail_stable(
    client_for: MakeClient, probes: list[Probe]
) -> None:
    report = _run(
        client_for, probes, noise=0.001, sim_kwargs={"batch_noise": 0.02, "latency": 0.001}
    )
    assert report.verdict in (PASS, WARN)
    assert all(rep.checks["stable"].status != FAIL for rep in report.adapters)


def test_shared_prefix_cache_is_caught_by_salt_or_fingerprint(
    client_for: MakeClient, probes: list[Probe]
) -> None:
    faults = Faults(cache_ignores_adapter=True)
    # Honest limitation: consistent first-writer-wins contamination is invisible by itself.
    assert _run(client_for, probes, faults).verdict == PASS
    salted = _run(client_for, probes, faults, cache_salt=True)
    assert _status(salted, "a2", "cache") == FAIL
    assert salted.base.checks["cache"].status == PASS  # the base populated the cache
    referenced = _run(client_for, probes, faults, fingerprint=True)
    assert _status(referenced, "a2", "reference") == FAIL


def test_stale_cache_after_inplace_reload(client_for: MakeClient, probes: list[Probe]) -> None:
    sim = SimServer(SimConfig(adapters=ADAPTERS, faults=Faults(stale_cache_on_reload=True)))
    client = client_for(sim)
    config = VerifyConfig(probes=probes, cache_salt=True, concurrency=4)
    assert run_verify(client, config).verdict == PASS
    sim.load_adapter("a2", "retrained", inplace=True)
    report = run_verify(client, config)
    assert _status(report, "a2", "cache") == FAIL
    assert _status(report, "a1", "cache") == PASS


def test_cache_salt_rejected_by_server_is_skipped(probes: list[Probe]) -> None:
    import json

    import httpx

    sim = SimServer(SimConfig(adapters=ADAPTERS))

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        if body and "cache_salt" in body:
            return httpx.Response(400, json={"error": {"message": "unknown field cache_salt"}})
        status, payload = sim.handle(request.method, request.url.path, body)
        return httpx.Response(status, json=payload)

    client = Client("http://sim", transport=httpx.MockTransport(handler))
    report = run_verify(client, VerifyConfig(probes=probes, cache_salt=True, concurrency=2))
    cache = report.adapters[0].checks["cache"]
    assert cache.status == SKIP
    assert "rejected" in cache.summary
    assert report.verdict == PASS


def test_narrow_adapter_is_applied_not_failed(client_for: MakeClient, probes: list[Probe]) -> None:
    report = _run(
        client_for, probes, noise=0.002, sim_kwargs={"strength": dict.fromkeys(ADAPTERS, 0.05)}
    )
    assert all(_status(report, a, "applied") in (PASS, WARN) for a in ADAPTERS)


def test_unknown_adapter_is_an_error_not_a_crash(
    client_for: MakeClient, probes: list[Probe]
) -> None:
    report = _run(client_for, probes, adapters=["a1", "missing"], base_model="toy-base")
    assert _status(report, "missing", "applied") == ERROR
    assert _status(report, "a1", "applied") == PASS
    assert report.exit_code() == 2


def test_chat_api_and_token_ids(probes: list[Probe]) -> None:
    from lora_check.sim import mock_transport

    sim = SimServer(SimConfig(adapters=ADAPTERS, faults=Faults(ignored=frozenset({"a3"}))))
    fingerprint = sim.reference_fingerprint(probes)
    client = Client("http://sim", transport=mock_transport(sim), token_ids=True)
    report = run_verify(client, VerifyConfig(probes=probes, fingerprint=fingerprint, concurrency=2))
    assert _status(report, "a1", "reference") == PASS  # matched by token id
    assert _status(report, "a3", "applied") == FAIL
    chat = Client("http://sim", transport=mock_transport(sim), api="chat")
    report = run_verify(chat, VerifyConfig(probes=probes, concurrency=2))
    assert _status(report, "a3", "applied") == FAIL
    assert _status(report, "a1", "applied") == PASS


def test_sglang_style_addressing(client_for: MakeClient, probes: list[Probe]) -> None:
    report = _run(client_for, probes, adapter_model_format="{base}:{adapter}")
    assert report.verdict == PASS
    assert report.adapters[0].model_id == "toy-base:a1"


def test_name_map_and_missing_reference(client_for: MakeClient, probes: list[Probe]) -> None:
    sim = SimServer(SimConfig(adapters=ADAPTERS))
    fingerprint = sim.reference_fingerprint(probes)
    fingerprint.adapters["trained-a1"] = fingerprint.adapters.pop("a1")
    report = run_verify(
        client_for(sim), VerifyConfig(probes=probes, fingerprint=fingerprint, concurrency=2)
    )
    skipped = next(a for a in report.adapters if a.name == "a1").checks["reference"]
    assert skipped.status == SKIP
    assert "--map" in skipped.summary
    report = run_verify(
        client_for(sim),
        VerifyConfig(
            probes=probes, fingerprint=fingerprint, name_map={"a1": "trained-a1"}, concurrency=2
        ),
    )
    assert _status(report, "a1", "reference") == PASS


def test_different_base_model_is_reported_not_blamed_on_adapter(
    client_for: MakeClient, probes: list[Probe]
) -> None:
    other = SimServer(SimConfig(adapters=ADAPTERS, seed=99))
    sim = SimServer(SimConfig(adapters=ADAPTERS))
    report = run_verify(
        client_for(sim),
        VerifyConfig(probes=probes, fingerprint=other.reference_fingerprint(probes), concurrency=2),
    )
    assert report.base.checks["reference"].status == WARN
    assert _status(report, "a1", "reference") == WARN
    assert "base" in report.adapters[0].checks["reference"].summary


def test_systematic_numeric_offset_widens_tolerance(
    client_for: MakeClient, probes: list[Probe]
) -> None:
    report = _run(client_for, probes, fingerprint=True, sim_kwargs={"systematic": 0.02})
    assert report.verdict == PASS
    assert report.base.checks["reference"].evidence["base_offset"] > 0.005


def test_no_logprobs_endpoint_is_a_clear_error(client_for: MakeClient, probes: list[Probe]) -> None:
    with pytest.raises(EndpointError, match="logprobs"):
        _run(client_for, probes, Faults(no_logprobs=True))


def test_no_interleave_skips_stable(client_for: MakeClient, probes: list[Probe]) -> None:
    report = _run(client_for, probes, Faults(leak_after_switch=1.0), interleave=False)
    assert _status(report, "a1", "stable") == SKIP


def test_config_validation(client_for: MakeClient, probes: list[Probe]) -> None:
    with pytest.raises(EndpointError, match="repeats"):
        _run(client_for, probes, repeats=1)
    with pytest.raises(EndpointError, match="no probes"):
        _run(client_for, [])


def test_over_real_http(probes: list[Probe]) -> None:
    sim = SimServer(
        SimConfig(adapters=ADAPTERS, noise=0.002, faults=Faults(ignored=frozenset({"a1"})))
    )
    with running_server(sim) as url, Client(url) as client:
        report = run_verify(client, VerifyConfig(probes=probes[:8], concurrency=4))
    assert _status(report, "a1", "applied") == FAIL
    assert _status(report, "a2", "applied") == PASS
    assert report.endpoint == f"{url}/v1"


def test_discover() -> None:
    models = [
        ModelInfo("base", None, "base"),
        ModelInfo("x", "base", "/p/x"),
        ModelInfo("y", "base", "/p/y"),
    ]
    assert discover(models, None, None) == ("base", ["x", "y"])
    assert discover(models, None, ["y"]) == ("base", ["y"])
    assert discover([], "b", ["x"]) == ("b", ["x"])
    with pytest.raises(EndpointError, match="no LoRA adapters"):
        discover([ModelInfo("base")], None, None)
    with pytest.raises(EndpointError, match="pass --base-model"):
        discover([ModelInfo("m1"), ModelInfo("m2"), ModelInfo("m3")], None, ["m1"])
    assert discover([ModelInfo("m1"), ModelInfo("m2")], None, ["m1"]) == ("m2", ["m1"])
    with pytest.raises(EndpointError, match="both"):
        discover(models, "x", ["x"])


def test_schedule_switches_model_every_request(probes: list[Probe]) -> None:
    names = ["base", "a", "b"]
    order = _schedule(names, probes)
    assert len(order) == len(names) * len(probes)
    assert all(order[i][0] != order[i + 1][0] for i in range(len(order) - 1))
    followers = {(order[i][0], order[i + 1][0]) for i in range(len(order) - 1)}
    assert len(followers) == 6  # every ordered pair of distinct models occurs
