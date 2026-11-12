# Contributing

Thanks for looking. The most valuable contributions right now are reports
from real engines: this project has so far only been exercised against its
own simulator and against transformers + PEFT on CPU.

## Reporting results from a real server

Run

```bash
lora-check verify --base-url ... --format json --output report.json
```

and attach `report.json` to an issue together with the engine name and
version, the base model, how the adapters were loaded and, if you can, whether
the verdict was right. False alarms are as interesting as missed faults.

## Development setup

```bash
git clone https://github.com/cb-Sigma/lora-check
cd lora-check
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

ruff check . && ruff format --check .
mypy
pytest
```

The default test run needs no GPU, no model downloads and about a minute.
`tests/test_real_peft.py` is skipped unless torch, transformers and peft are
installed; to run it, install a CPU build of torch first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers peft
pytest tests/test_real_peft.py
```

It downloads `hf-internal-testing/tiny-random-gpt2` (a few hundred kilobytes).

## Ground rules

- **Lint rules about an engine must cite a source**: a documentation page, a
  file in the engine's repository or an issue, plus the date you read it
  (`CHECKED` in `src/lora_check/lint/core.py`). A test enforces the citation.
  Do not add rules from memory or from blog posts.
- **Do not overstate.** If something was verified on the simulator only, say
  so in the docs. The README separates "read in the docs", "tested on the
  simulator" and "tested on real PEFT arithmetic"; keep it that way.
- **New faults go into the simulator first**, with a test that shows `verify`
  catching them and, where relevant, a row in `lora_check/sim/evaluate.py`.
- The README demo blocks are generated. After changing output, run
  `python scripts/update_readme.py`; `tests/test_readme_sync.py` checks it.
- Keep `verify` free of heavy dependencies: `httpx` only. torch, transformers
  and peft may be imported only inside `lora_check/reference.py`.

## Pull requests

Small and focused is best. Add a line to `CHANGELOG.md` under *Unreleased*.
CI runs ruff, mypy, pytest on Python 3.10 to 3.14 and a package build.

## Releases

Maintainers tag `vX.Y.Z` after updating `__version__` and the changelog; the
release workflow builds and publishes through PyPI trusted publishing.
