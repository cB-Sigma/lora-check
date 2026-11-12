"""Detection and false-alarm rates on the simulator.

Every cell runs the real ``verify`` code against a freshly seeded simulated
server (through an in-process transport) and records whether the expected
check failed.  The numbers describe lora-check on a *toy model with synthetic
faults*; they say nothing about any particular engine.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from lora_check.client import Client
from lora_check.probes import Probe, builtin_probes
from lora_check.sim.server import Faults, SimConfig, SimServer, mock_transport
from lora_check.verify import FAIL, VerifyConfig, VerifyReport, run_verify

ADAPTERS = ("a1", "a2", "a3")
NOISE_LEVELS: tuple[float, ...] = (0.0, 0.002, 0.02, 0.1)


@dataclass(frozen=True)
class Scenario:
    """One row of the evaluation table."""

    key: str
    label: str
    mode: str  # which optional inputs verify gets: plain | fingerprint | salt
    faults: Callable[[], Faults]
    detected: Callable[[VerifyReport], bool]
    healthy: bool = False
    strength: float = 1.0
    latency: float = 0.0
    reload: bool = False


def _check_failed(
    adapter: str, check: str, cause: str | None = None
) -> Callable[[VerifyReport], bool]:
    def test(report: VerifyReport) -> bool:
        for rep in report.adapters:
            if rep.name == adapter and check in rep.checks:
                result = rep.checks[check]
                if result.status != FAIL:
                    return False
                return cause is None or result.evidence.get("cause") == cause
        return False

    return test


def _any_fail(report: VerifyReport) -> bool:
    return report.verdict == FAIL


def _any_stable_fail(report: VerifyReport) -> bool:
    return any(
        rep.checks.get("stable") is not None and rep.checks["stable"].status == FAIL
        for rep in [report.base, *report.adapters]
    )


SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        "healthy",
        "healthy server: any FAIL (false alarm)",
        "plain",
        Faults,
        _any_fail,
        healthy=True,
    ),
    Scenario(
        "healthy-fp",
        "healthy server + fingerprint: any FAIL (false alarm)",
        "fingerprint",
        Faults,
        _any_fail,
        healthy=True,
    ),
    Scenario(
        "healthy-salt",
        "healthy server + cache-salt control: any FAIL (false alarm)",
        "salt",
        Faults,
        _any_fail,
        healthy=True,
    ),
    Scenario(
        "healthy-narrow",
        "healthy server, narrow adapters (0.1x effect): any FAIL (false alarm)",
        "plain",
        Faults,
        _any_fail,
        healthy=True,
        strength=0.1,
    ),
    Scenario(
        "ignored",
        "adapter ignored -> APPLIED fails",
        "plain",
        lambda: Faults(ignored=frozenset({"a2"})),
        _check_failed("a2", "applied"),
    ),
    Scenario(
        "duplicate",
        "two names, same weights -> DISTINCT fails",
        "plain",
        lambda: Faults(alias={"a3": "a2"}),
        _check_failed("a3", "distinct"),
    ),
    Scenario(
        "leak",
        "previous adapter leaks into 50% of requests after a switch -> STABLE fails",
        "plain",
        lambda: Faults(leak_after_switch=0.5),
        _any_stable_fail,
    ),
    Scenario(
        "batch-mix",
        "20% of overlapping requests computed with a neighbour's adapter -> STABLE fails",
        "plain",
        lambda: Faults(batch_mix=0.2),
        _any_stable_fail,
        latency=0.001,
    ),
    Scenario(
        "scale-plain",
        "adapter at 0.5x scale, no fingerprint -> any FAIL",
        "plain",
        lambda: Faults(scale={"a2": 0.5}),
        _any_fail,
    ),
    Scenario(
        "scale",
        "adapter at 0.5x scale + fingerprint -> REFERENCE fails, cause wrong-scale",
        "fingerprint",
        lambda: Faults(scale={"a2": 0.5}),
        _check_failed("a2", "reference", "wrong-scale"),
    ),
    Scenario(
        "scale-odd",
        "adapter at 0.7x scale (no stored variant) + fingerprint -> REFERENCE fails",
        "fingerprint",
        lambda: Faults(scale={"a2": 0.7}),
        _check_failed("a2", "reference"),
    ),
    Scenario(
        "swap-plain",
        "two adapters swapped, no fingerprint -> any FAIL",
        "plain",
        lambda: Faults(alias={"a2": "a3", "a3": "a2"}),
        _any_fail,
    ),
    Scenario(
        "swap",
        "two adapters swapped + fingerprint -> REFERENCE fails, cause wrong-adapter",
        "fingerprint",
        lambda: Faults(alias={"a2": "a3", "a3": "a2"}),
        _check_failed("a2", "reference", "wrong-adapter"),
    ),
    Scenario(
        "cache-plain",
        "prefix cache shared between adapters, no fingerprint, no salt -> any FAIL",
        "plain",
        lambda: Faults(cache_ignores_adapter=True),
        _any_fail,
    ),
    Scenario(
        "cache",
        "prefix cache shared between adapters + cache-salt control -> CACHE fails",
        "salt",
        lambda: Faults(cache_ignores_adapter=True),
        _check_failed("a2", "cache"),
    ),
    Scenario(
        "cache-fp",
        "prefix cache shared between adapters + fingerprint -> REFERENCE fails",
        "fingerprint",
        lambda: Faults(cache_ignores_adapter=True),
        _check_failed("a2", "reference"),
    ),
    Scenario(
        "stale",
        "adapter reloaded under the same name, stale cache + cache-salt control -> CACHE fails",
        "salt",
        lambda: Faults(stale_cache_on_reload=True),
        _check_failed("a2", "cache"),
        reload=True,
    ),
)


def run_scenario(
    scenario: Scenario,
    noise: float,
    seed: int,
    probes: Sequence[Probe] | None = None,
    concurrency: int = 4,
) -> VerifyReport:
    """One verification run of ``scenario`` at the given noise and seed."""
    probe_list = list(probes) if probes is not None else builtin_probes(16)
    sim = SimServer(
        SimConfig(
            adapters=ADAPTERS,
            seed=seed,
            noise=noise,
            strength=dict.fromkeys(ADAPTERS, scenario.strength),
            latency=scenario.latency,
            faults=scenario.faults(),
        )
    )
    config = VerifyConfig(
        probes=probe_list,
        concurrency=concurrency,
        cache_salt=scenario.mode == "salt",
        fingerprint=sim.reference_fingerprint(probe_list)
        if scenario.mode == "fingerprint"
        else None,
    )
    with Client("http://sim", transport=mock_transport(sim)) as client:
        if scenario.reload:
            # Warm the cache with the old weights, then swap them in place.
            run_verify(client, config)
            sim.load_adapter("a2", "retrained-weights", inplace=True)
        return run_verify(client, config)


@dataclass(frozen=True)
class Cell:
    """Outcome of one (scenario, noise) cell."""

    hits: int
    trials: int

    @property
    def rate(self) -> float:
        return self.hits / self.trials if self.trials else 0.0


def evaluate(
    trials: int = 20,
    noise_levels: Sequence[float] = NOISE_LEVELS,
    scenarios: Sequence[Scenario] = SCENARIOS,
    progress: Callable[[str], None] | None = None,
) -> dict[str, dict[float, Cell]]:
    """Run every scenario at every noise level ``trials`` times."""
    table: dict[str, dict[float, Cell]] = {}
    for scenario in scenarios:
        row: dict[float, Cell] = {}
        for noise in noise_levels:
            hits = sum(
                1
                for trial in range(trials)
                if scenario.detected(run_scenario(scenario, noise, seed=1000 + trial))
            )
            row[noise] = Cell(hits, trials)
            if progress is not None:
                progress(f"{scenario.key} noise={noise:g}: {hits}/{trials}")
        table[scenario.key] = row
    return table


def markdown_table(
    table: dict[str, dict[float, Cell]], scenarios: Sequence[Scenario] = SCENARIOS
) -> str:
    """The evaluation as a Markdown table (rates as hits/trials)."""
    by_key = {s.key: s for s in scenarios}
    noise_levels = list(next(iter(table.values())))
    head = "| Scenario (SIMULATED) | " + " | ".join(f"noise {n:g}" for n in noise_levels) + " |"
    lines = [head, "|---|" + "---:|" * len(noise_levels)]
    for key, row in table.items():
        cells = " | ".join(f"{row[n].hits}/{row[n].trials}" for n in noise_levels)
        lines.append(f"| {by_key[key].label} | {cells} |")
    return "\n".join(lines)
