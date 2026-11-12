| Scenario (SIMULATED) | noise 0 | noise 0.002 | noise 0.02 | noise 0.1 |
|---|---:|---:|---:|---:|
| healthy server: any FAIL (false alarm) | 0/25 | 0/25 | 0/25 | 0/25 |
| healthy server + fingerprint: any FAIL (false alarm) | 0/25 | 0/25 | 0/25 | 0/25 |
| healthy server + cache-salt control: any FAIL (false alarm) | 0/25 | 0/25 | 0/25 | 0/25 |
| healthy server, narrow adapters (0.1x effect): any FAIL (false alarm) | 0/25 | 0/25 | 0/25 | 7/25 |
| adapter ignored -> APPLIED fails | 25/25 | 25/25 | 25/25 | 25/25 |
| two names, same weights -> DISTINCT fails | 25/25 | 25/25 | 25/25 | 25/25 |
| previous adapter leaks into 50% of requests after a switch -> STABLE fails | 25/25 | 25/25 | 25/25 | 25/25 |
| 20% of overlapping requests computed with a neighbour's adapter -> STABLE fails | 25/25 | 25/25 | 25/25 | 25/25 |
| adapter at 0.5x scale, no fingerprint -> any FAIL | 0/25 | 0/25 | 0/25 | 0/25 |
| adapter at 0.5x scale + fingerprint -> REFERENCE fails, cause wrong-scale | 25/25 | 25/25 | 25/25 | 25/25 |
| adapter at 0.7x scale (no stored variant) + fingerprint -> REFERENCE fails | 25/25 | 25/25 | 25/25 | 19/25 |
| two adapters swapped, no fingerprint -> any FAIL | 0/25 | 0/25 | 0/25 | 0/25 |
| two adapters swapped + fingerprint -> REFERENCE fails, cause wrong-adapter | 25/25 | 25/25 | 25/25 | 25/25 |
| prefix cache shared between adapters, no fingerprint, no salt -> any FAIL | 0/25 | 0/25 | 0/25 | 0/25 |
| prefix cache shared between adapters + cache-salt control -> CACHE fails | 25/25 | 25/25 | 25/25 | 25/25 |
| prefix cache shared between adapters + fingerprint -> REFERENCE fails | 25/25 | 25/25 | 25/25 | 17/25 |
| adapter reloaded under the same name, stale cache + cache-salt control -> CACHE fails | 25/25 | 25/25 | 25/25 | 25/25 |
