from __future__ import annotations

import pytest

from lora_check.fingerprint import FingerprintError
from lora_check.mine import mine_probes, mined_table
from lora_check.probes import builtin_probes
from lora_check.sim import SimConfig, SimServer


def test_mine_picks_each_adapters_most_sensitive_probes() -> None:
    sim = SimServer(SimConfig(adapters=("a1", "a2")))
    fingerprint = sim.reference_fingerprint(builtin_probes())
    mined = mine_probes(fingerprint, top=6)
    assert len(mined) == 6
    assert len({m.probe.id for m in mined}) == 6
    for name in ("a1", "a2"):
        table = fingerprint.sensitivity(name)
        best = max(table, key=lambda pid: table[pid])
        assert best in {m.probe.id for m in mined}
    assert [m.chosen_for for m in mined[:2]] == ["a1", "a2"]
    assert mined[0].score == max(fingerprint.sensitivity("a1").values())
    mean_all = sum(fingerprint.sensitivity("a1").values()) / len(fingerprint.probes)
    assert sum(m.sensitivity["a1"] for m in mined if m.chosen_for == "a1") / 3 > mean_all
    assert "PROBE" in mined_table(mined)
    assert mined_table([]) == "no probes selected"


def test_mine_limits_and_errors() -> None:
    sim = SimServer(SimConfig(adapters=("a1",)))
    fingerprint = sim.reference_fingerprint(builtin_probes(5))
    assert len(mine_probes(fingerprint, top=50)) == 5
    assert len(mine_probes(fingerprint, ["a1"], top=2)) == 2
    with pytest.raises(FingerprintError):
        mine_probes(fingerprint, ["nope"])
    fingerprint.adapters.clear()
    with pytest.raises(FingerprintError):
        mine_probes(fingerprint)
