from __future__ import annotations

import json
from pathlib import Path

import pytest

from lora_check.fingerprint import (
    Fingerprint,
    FingerprintError,
    RefDist,
    compare_to_reference,
    detect_key_style,
    scale_hypotheses,
)
from lora_check.probes import Probe
from lora_check.sim import SimConfig, SimServer


@pytest.fixture
def sim() -> SimServer:
    return SimServer(SimConfig(adapters=("a1", "a2")))


@pytest.fixture
def fingerprint(sim: SimServer, probes: list[Probe]) -> Fingerprint:
    return sim.reference_fingerprint(probes)


def _observe(
    sim: SimServer, probes: list[Probe], adapter: str | None, scale: float = 1.0
) -> dict[str, list[dict[str, float]]]:
    weights = sim.adapter(adapter) if adapter else None
    s = weights.scaling * scale if weights else None
    return {
        p.id: [{t: lp for _i, t, lp in sim.lm.top_logprobs(p.prompt, weights, s)}] for p in probes
    }


def test_roundtrip_preserves_everything(fingerprint: Fingerprint, tmp_path: Path) -> None:
    path = tmp_path / "fp.json"
    fingerprint.save(path)
    loaded = Fingerprint.load(path)
    assert loaded.probes == fingerprint.probes
    assert set(loaded.adapters) == {"a1", "a2"}
    assert [v.name for v in loaded.adapters["a1"].variants] == [
        "scale=0.25",
        "scale=0.5",
        "scale=0.75",
        "scale=1.5",
        "scale=2",
        "scale=4",
    ]
    a = loaded.base["g01"].as_dist()
    b = fingerprint.base["g01"].as_dist()
    assert a == pytest.approx(b, abs=1e-6)
    assert loaded.sensitivity("a1")["g01"] > 0.01


def test_refdist_key_styles_and_collisions() -> None:
    ref = RefDist(((5, "the", -0.5), (9, "the", -2.0), (7, " a", -1.0)))
    assert ref.as_dist("id") == {"token_id:5": -0.5, "token_id:9": -2.0, "token_id:7": -1.0}
    assert ref.as_dist("text") == {"the": -0.5, " a": -1.0}
    assert ref.as_dist("text", k=1) == {"the": -0.5}
    assert detect_key_style({"p": [{"token_id:1": -1.0}]}) == "id"
    assert detect_key_style({"p": [{"hello": -1.0}]}) == "text"
    assert RefDist.from_dist({"b": -2.0, "a": -1.0}).entries[0] == (None, "a", -1.0)


def test_subset(fingerprint: Fingerprint) -> None:
    small = fingerprint.subset(["g03", "g01"])
    assert [p.id for p in small.probes] == ["g03", "g01"]
    small.validate()
    assert set(small.adapters["a1"].variants[0].dists) == {"g01", "g03"}
    with pytest.raises(FingerprintError):
        fingerprint.subset(["nope"])


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"format": "other"},
        {"format": "lora-check/fingerprint", "version": 99, "probes": [], "base": {}},
        {"format": "lora-check/fingerprint", "version": 1, "probes": [{"id": "a"}], "base": {}},
        {
            "format": "lora-check/fingerprint",
            "version": 1,
            "probes": [{"id": "a", "prompt": "x"}],
            "base": {},
        },
        {
            "format": "lora-check/fingerprint",
            "version": 1,
            "probes": [{"id": "a", "prompt": "x"}],
            "base": {"a": [[1, "t"]]},
        },
    ],
)
def test_rejects_malformed_files(tmp_path: Path, payload: object) -> None:
    path = tmp_path / "fp.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(FingerprintError):
        Fingerprint.load(path)


def test_load_errors(tmp_path: Path) -> None:
    with pytest.raises(FingerprintError):
        Fingerprint.load(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("{")
    with pytest.raises(FingerprintError):
        Fingerprint.load(bad)


def test_compare_match(sim: SimServer, fingerprint: Fingerprint, probes: list[Probe]) -> None:
    outcome = compare_to_reference(
        fingerprint, "a1", _observe(sim, probes, "a1"), _observe(sim, probes, None), noise=0.0
    )
    assert outcome.status == "PASS"
    assert outcome.cause == "match"
    assert outcome.candidates[0].label == "reference"
    assert outcome.fit is not None
    assert outcome.fit.scale == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize(
    ("served", "scale", "cause"),
    [
        (None, 1.0, "adapter-ignored"),
        ("a2", 1.0, "wrong-adapter"),
        ("a1", 0.5, "wrong-scale"),
        ("a1", 4.0, "wrong-scale"),
        ("a1", 0.8, "wrong-scale"),
        ("a1", 1.3, "wrong-scale"),
    ],
)
def test_compare_diagnoses(
    sim: SimServer,
    fingerprint: Fingerprint,
    probes: list[Probe],
    served: str | None,
    scale: float,
    cause: str,
) -> None:
    outcome = compare_to_reference(
        fingerprint,
        "a1",
        _observe(sim, probes, served, scale),
        _observe(sim, probes, None),
        noise=0.0,
    )
    assert outcome.status == "FAIL"
    assert outcome.cause == cause
    assert "tolerance" in outcome.evidence()


def test_compare_unknown_weights(
    sim: SimServer, fingerprint: Fingerprint, probes: list[Probe]
) -> None:
    stranger = sim.lm.make_adapter("someone-else")
    observed = {
        p.id: [{t: lp for _i, t, lp in sim.lm.top_logprobs(p.prompt, stranger)}] for p in probes
    }
    outcome = compare_to_reference(
        fingerprint, "a1", observed, _observe(sim, probes, None), noise=0.0
    )
    assert outcome.status == "FAIL"
    assert outcome.cause == "unknown-mismatch"


def test_compare_truncated_endpoint_topk(
    sim: SimServer, fingerprint: Fingerprint, probes: list[Probe]
) -> None:
    observed = {
        p.id: [{t: lp for _i, t, lp in sim.lm.top_logprobs(p.prompt, sim.adapter("a1"), k=5)}]
        for p in probes
    }
    outcome = compare_to_reference(fingerprint, "a1", observed, None, noise=0.0)
    assert outcome.status == "PASS"
    assert outcome.base_offset is None


def test_scale_hypotheses() -> None:
    rs = scale_hypotheses({"r": 16, "lora_alpha": 32, "use_rslora": True})
    assert any(f == pytest.approx(0.25) and "use_rslora ignored" in text for f, text in rs)
    plain = scale_hypotheses({"r": 16, "lora_alpha": 32})
    assert any(f == pytest.approx(4.0) for f, _ in plain)
    assert any(f == pytest.approx(0.5) and "lora_alpha ignored" in text for f, text in plain)
    assert scale_hypotheses({"r": 0, "lora_alpha": 1}) == []
    assert scale_hypotheses({"r": "x"}) == []
