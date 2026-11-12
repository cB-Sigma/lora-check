"""Use lora-check as a library: verify an endpoint and inspect the evidence.

Run without arguments to try it against the bundled simulator, or pass the
base URL of a real OpenAI-compatible server.
"""

from __future__ import annotations

import sys

from lora_check.client import Client
from lora_check.probes import builtin_probes
from lora_check.sim import Faults, SimConfig, SimServer, running_server
from lora_check.verify import VerifyConfig, run_verify


def check(base_url: str) -> int:
    with Client(base_url) as client:
        report = run_verify(client, VerifyConfig(probes=builtin_probes(16)))
    for adapter in report.adapters:
        print(f"{adapter.name}: {adapter.verdict}")
        for check_result in adapter.checks.values():
            if check_result.status in ("FAIL", "WARN"):
                print(f"  {check_result.name}: {check_result.summary}")
    return report.exit_code()


def main() -> int:
    if len(sys.argv) > 1:
        return check(sys.argv[1])
    sim = SimServer(
        SimConfig(adapters=("good", "ignored"), faults=Faults(ignored=frozenset({"ignored"})))
    )
    with running_server(sim) as url:
        return check(url)


if __name__ == "__main__":
    raise SystemExit(main())
