"""Select the probes on which adapters move the distribution most."""

from __future__ import annotations

from dataclasses import dataclass

from lora_check.fingerprint import Fingerprint, FingerprintError
from lora_check.probes import Probe


@dataclass(frozen=True)
class MinedProbe:
    """A selected probe and how strongly each adapter reacts to it."""

    probe: Probe
    sensitivity: dict[str, float]
    chosen_for: str

    @property
    def score(self) -> float:
        return self.sensitivity[self.chosen_for]


def _rank(table: dict[str, float]) -> list[str]:
    return sorted(table, key=lambda pid: (-table[pid], pid))


def mine_probes(
    fingerprint: Fingerprint, adapters: list[str] | None = None, top: int = 16
) -> list[MinedProbe]:
    """Pick up to ``top`` probes, taking each adapter's most sensitive in turn.

    Sensitivity is the distance between the adapter's and the base model's
    reference distribution on a probe.  Round-robin selection guarantees
    every adapter gets its own most sensitive probes, which matters when
    adapters are narrow in different directions.
    """
    names = adapters or sorted(fingerprint.adapters)
    unknown = [n for n in names if n not in fingerprint.adapters]
    if unknown:
        raise FingerprintError(
            f"fingerprint has no adapter {unknown[0]!r} (it has: {', '.join(sorted(fingerprint.adapters))})"
        )
    if not names:
        raise FingerprintError("fingerprint contains no adapters to mine probes for")
    tables = {name: fingerprint.sensitivity(name) for name in names}
    by_id = {p.id: p for p in fingerprint.probes}
    ranked = {name: _rank(table) for name, table in tables.items()}
    chosen: list[MinedProbe] = []
    taken: set[str] = set()
    cursor = dict.fromkeys(names, 0)
    limit = min(max(1, top), len(by_id))
    while len(chosen) < limit:
        progressed = False
        for name in names:
            order = ranked[name]
            while cursor[name] < len(order) and order[cursor[name]] in taken:
                cursor[name] += 1
            if cursor[name] >= len(order) or len(chosen) >= limit:
                continue
            pid = order[cursor[name]]
            taken.add(pid)
            chosen.append(MinedProbe(by_id[pid], {n: tables[n][pid] for n in names}, name))
            progressed = True
        if not progressed:
            break
    return chosen


def mined_table(mined: list[MinedProbe]) -> str:
    """Plain-text table of the selection."""
    if not mined:
        return "no probes selected"
    names = list(mined[0].sensitivity)
    rows = [["PROBE", *names, "PROMPT"]]
    for item in mined:
        prompt = item.probe.prompt.replace("\n", "\\n")
        prompt = prompt if len(prompt) <= 48 else prompt[:45] + "..."
        rows.append([item.probe.id, *[f"{item.sensitivity[n]:.4f}" for n in names], prompt])
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(c.ljust(widths[i]) for i, c in enumerate(row)).rstrip() for row in rows
    )
