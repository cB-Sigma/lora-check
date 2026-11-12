# Examples

| File | What it shows |
|---|---|
| [`github-actions.yml`](github-actions.yml) | Lint an adapter and verify a staging endpoint on a schedule |
| [`verify_from_python.py`](verify_from_python.py) | Calling `run_verify` from Python and reading the evidence |
| [`make_random_lora.py`](make_random_lora.py) | Writing a LoRA with non-zero weights to experiment with (needs torch + peft) |
| [`probes.txt`](probes.txt) | The probe file format, with domain prompts |

Try everything without a server:

```bash
lora-check verify --demo                 # mixed faults
lora-check verify --demo leak            # cross-adapter leakage
lora-check verify --demo cache --no-fingerprint --cache-salt
lora-check lint --demo
lora-check fingerprint --demo -o demo.fp.json && lora-check mine --fingerprint demo.fp.json --top 8
lora-check sim --port 8000               # then point `verify --base-url` at it
```
