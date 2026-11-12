# lora-check

**Find out whether your inference server is really applying the LoRA adapter you asked for.**

[![CI](https://github.com/cb-Sigma/lora-check/actions/workflows/ci.yml/badge.svg)](https://github.com/cb-Sigma/lora-check/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](pyproject.toml)

Multi-LoRA servers such as vLLM and SGLang can load an adapter "successfully"
and then serve something else: plain base-model output, the adapter at the
wrong strength, another adapter's weights, or answers that depend on what the
previous request was. Nothing crashes and nothing is logged. `lora-check` is a
small CLI that catches this from the outside:

- **`verify`** probes a live OpenAI-compatible endpoint with next-token
  log-probabilities and gives each adapter a verdict you can gate CI on.
- **`fingerprint`** records what base and base+adapter *should* answer
  (transformers + PEFT), so `verify` can say not only "applied" but "applied
  correctly", and name the likely cause when it is not.
- **`lint`** statically checks an adapter directory or Hub repository for
  features and layouts that specific engines reject or mishandle, each rule
  with its source.
- **`mine`** picks the probes on which your adapters are most sensitive.

`verify` needs only an HTTP client. No GPU, no model weights.

```bash
pip install git+https://github.com/cb-Sigma/lora-check
lora-check verify --demo          # try it now against the bundled simulated server
```

## What it looks like

This is real output, produced by the command shown against the bundled
simulator (a toy model with injected faults; one adapter ignored, one served
at a quarter of its scale, two swapped). A test keeps this block in sync with
the code.

<!-- BEGIN demo-verify -->
```text
$ lora-check verify --demo
lora-check 0.1.0 - verify demo://mixed (simulated server)
base model: toy-base   adapters: 5   probes: 24 x 3 repeats   requests: 876
run-to-run noise (mean |dlogprob|, p90): base 0.0030, sql-gen 0.0030, support-bot 0.0030, summarizer 0.0031, legal-v1 0.0031, legal-v2 0.0032

MODEL            APPLIED  DISTINCT  STABLE  CACHE  REFERENCE  VERDICT
sql-gen          PASS     PASS      PASS    -      PASS       PASS
support-bot      FAIL     PASS      PASS    -      FAIL       FAIL
summarizer       PASS     PASS      PASS    -      FAIL       FAIL
legal-v1         PASS     PASS      PASS    -      FAIL       FAIL
legal-v2         PASS     PASS      PASS    -      FAIL       FAIL
toy-base (base)  -        -         PASS    -      PASS       PASS

support-bot: FAIL
  applied   FAIL  indistinguishable from the base model on all 24 probes (max distance 0.0027 <=
                  threshold 0.0090, no consistent difference across repeats): the adapter has no
                  effect
  reference FAIL  endpoint matches the BASE model (d=0.0016), not the adapter (d=0.5333 to the
                  reference): the adapter is loaded but ignored
summarizer: FAIL
  reference FAIL  endpoint matches variant 'scale=0.25' (d=0.0016), not the reference (d=0.5584 to
                  the reference): the adapter applied at 0.25x its intended scale; estimated scale
                  0.25x (R2=1.00)
legal-v1: FAIL
  reference FAIL  endpoint matches adapter 'legal-v2' (d=0.0016), not this adapter (d=0.7212 to
                  the reference): adapters are swapped or mis-routed
legal-v2: FAIL
  reference FAIL  endpoint matches adapter 'legal-v1' (d=0.0016), not this adapter (d=0.7214 to
                  the reference): adapters are swapped or mis-routed

overall: FAIL
```
<!-- END demo-verify -->

Exit code is `1` here, so a CI job fails. With `--format markdown` the same
report drops into a job summary; `--format json` carries the evidence for
every check.

## The failures this is built for

Each of these is an upstream report we read (on 2026-11-12) before designing a
check. Statuses are as of that date; follow the links for the current state.

| Failure | Upstream report | What `lora-check` does about it |
|---|---|---|
| Adapter loads, output is byte-identical to base, because module-name prefixes do not match the served model class | [vllm#34186](https://github.com/vllm-project/vllm/issues/34186) (open): "silently produce output identical to the base model" | `verify` APPLIED fails; `lint --base` flags the prefix mismatch (`L043`) |
| Adapter loads and is ignored on a specific model family | [vllm#41754](https://github.com/vllm-project/vllm/issues/41754) (open; Gemma 4 with Unsloth adapters, several root causes in the thread) | `verify` APPLIED fails |
| `rank_pattern` / `alpha_pattern` ignored: one `lora_alpha/r` for every module | [vllm#59799](https://github.com/vllm-project/vllm/issues/59799) (reported on 0.30.0; per-module scaling is present in the source from tag v0.32.0) | `lint` rule `V107`; `verify --fingerprint` matches the stored `patterns-ignored` variant |
| rsLoRA scale not applied | [vllm#10798](https://github.com/vllm-project/vllm/issues/10798) (closed; 0.6.4.post1) | `lint` rules `L021`/`V110`/`S203`; `verify --fingerprint` matches the `rslora-ignored` variant |
| One data-parallel replica's LoRA state applied to another replica's tokens (MoE expert LoRA, DP > 1) | [vllm#61453](https://github.com/vllm-project/vllm/issues/61453) (open) | `verify` STABLE: answers must not change under interleaved and concurrent traffic; `--fingerprint` for the idle-peer case |
| Prefix cache reused across LoRA weights that share a name | [vllm#30931](https://github.com/vllm-project/vllm/issues/30931) (open; the maintainer states it affects offline use, not the online server) | `verify --cache-salt` and `--fingerprint` |
| Umbrella tracker for LoRA correctness in RL training loops | [vllm#48297](https://github.com/vllm-project/vllm/issues/48297) (RFC, open) | Its acceptance criteria ("accepted adapter tensors take effect", "adapter identity stays isolated") are what `verify` tests from outside |

Separately, a NeurIPS 2026 workshop paper,
[*Same Command, Different Answer*](https://neurips.cc/virtual/2026/171257),
reports that requests naming a LoRA adapter are not byte-reproducible inside a
single vLLM process while base-model requests are. That is why `lora-check`
never compares for equality: every threshold is a multiple of the run-to-run
noise it measures on your endpoint.

## How each check works

All requests are greedy single-token completions with `logprobs` (top-20 by
default). The comparison unit is the **mean absolute log-probability
difference** over the returned tokens, in nats, the same quantity the upstream
reports quote.

1. **Calibration.** Every model (base and each adapter) answers every probe
   `--repeats` times (default 3). The spread among identical requests is that
   model's noise; thresholds are `--noise-multiplier` (default 3) times it,
   with a floor of `--min-delta` (0.001 nats).
2. **APPLIED.** The adapter's answers must differ from the base model's by
   more than the threshold. If no single probe does, a second, more sensitive
   test looks for a difference that is *consistent across repeats* even though
   it is below the per-request noise (a cross-validated inner product, see
   [docs/how-it-works.md](docs/how-it-works.md)). Still nothing: **FAIL**, the
   headline failure.
3. **DISTINCT.** The same test between every pair of adapters. Two names that
   behave identically usually mean the same weights are being served twice.
4. **STABLE.** A strictly sequential pass that switches model on every
   request, then a concurrent burst that mixes all models on the same prompts.
   Each answer must stay where calibration put it. A separate same-model burst
   measures how much answers move merely because of batching, so benign
   batch-composition jitter is not blamed on adapters. Moved answers are
   compared with the other models' answers; a match is reported as leakage.
5. **CACHE** (`--cache-salt`, vLLM). Every request is repeated with a unique
   `cache_salt`, which bypasses the prefix cache; the answer must not change.
6. **REFERENCE** (`--fingerprint`). The endpoint is compared with the
   fingerprint's reference, and also with every *wrong* candidate the
   fingerprint holds: the base model, the other adapters, the adapter at other
   scales, with patterns ignored, with rsLoRA ignored. The endpoint's own base
   model is compared with the reference base first, which measures the
   engine's numeric offset (dtype, kernels) and widens the tolerance
   accordingly. When the match fails, the closest candidate names the cause,
   and the effective scale is estimated by fitting between the stored scales.

## Quick start

### 1. Check a running server (no reference needed)

```bash
# vLLM: adapters are discovered from /v1/models (entries with a "parent")
lora-check verify --base-url http://localhost:8000

# SGLang: adapters are addressed as base-model:adapter-name
lora-check verify --base-url http://localhost:30000 --engine sglang

# Any OpenAI-compatible server: name things yourself
lora-check verify --base-url http://host:8000 --base-model my-base \
    --adapter sql-lora --adapter support-lora --api chat --top-logprobs 5
```

Without a fingerprint a PASS means *applied, distinct and stable*. It does not
mean *correct*: an adapter served at half strength still passes.

### 2. Add a reference fingerprint

Run this where the adapter was trained, or on any machine that can load the
base model (CPU is fine for small models):

```bash
pip install torch            # pick the build for your machine first
pip install "lora-check[fingerprint] @ git+https://github.com/cb-Sigma/lora-check"

lora-check fingerprint --base meta-llama/Llama-3.2-3B-Instruct \
    --adapter sql-lora=./out/sql-lora -o sql-lora.fp.json

lora-check verify --base-url http://localhost:8000 --fingerprint sql-lora.fp.json
```

The name before `=` should be the name the adapter is served under (or use
`--map SERVED=NAME`). Use `--chat` with `fingerprint` if you verify with
`--api chat`. On vLLM, add `--token-ids` to `verify` so tokens are matched by
id rather than by decoded text.

### 3. Lint before you deploy

```bash
lora-check lint ./out/sql-lora --base meta-llama/Llama-3.2-3B-Instruct \
    --engine vllm --engine-version 0.30.0
lora-check lint some-org/some-adapter          # straight from the Hub
```

<!-- BEGIN demo-lint -->
```text
$ lora-check lint --demo
lora-check 0.1.0 - lint demo-adapter
base: demo-base   engine rules: vllm 0.30.0   sources checked: 2026-11-12
r=8 lora_alpha=16 use_rslora=False use_dora=False lora modules=4 ranks=[8, 16] scalings=[2.0, 4.0]

ERROR   V107  vLLM applied one lora_alpha/r scale to every module
        vLLM 0.30.0 is affected: one lora_alpha/r scale was applied to every module, so
        those 2 module(s) are served at the wrong strength with no warning. Upgrade, or fold
        the per-module scale into lora_B and set lora_alpha = r.
        source (issue): https://github.com/vllm-project/vllm/issues/59799
WARNING L043  adapter and base module paths differ by a prefix
        the base checkpoint nests the adapted modules under an extra prefix:
        'language_model.' (4 modules). The adapter was probably trained against a different
        model class than the checkpoint layout (for example a text-only class of a
        multimodal model). This is the situation in which vLLM was reported to load an
        adapter without error and never apply it (vllm#34186): check it with `lora-check
        verify`.
INFO    L020  per-module scaling differs (rank_pattern / alpha_pattern)
        rank_pattern/alpha_pattern present: 2 module(s) have a scale different from
        lora_alpha/r = 2 (by factors 2x), e.g. model.layers.0.self_attn.v_proj,
        model.layers.1.self_attn.v_proj
        source (docs): https://huggingface.co/docs/peft/main/en/package_reference/lora
INFO    V108  vLLM validates only the last component of module names
        vLLM checks only the last component of module names when loading; an adapter whose
        module paths do not line up with the served model class can load cleanly and never
        be applied. Run `lora-check verify` after loading
        source (issue): https://github.com/vllm-project/vllm/issues/34186

1 error(s), 1 warning(s), 2 note(s)
```
<!-- END demo-lint -->

`lint` reads `adapter_config.json` and the safetensors **header** only (tensor
names and shapes, fetched with HTTP range requests for Hub repositories). It
never downloads weights and never opens pickle files.

### 4. Tune the probes to your adapter

The built-in probes are generic. A narrow adapter may barely react to them.

```bash
lora-check fingerprint --base BASE --adapter a=./a --probes candidates.txt -o big.fp.json
lora-check mine --fingerprint big.fp.json --top 24 -o probes.jsonl --fingerprint-out a.fp.json
lora-check verify --base-url ... --fingerprint a.fp.json
```

## Use in CI

```yaml
- name: Verify LoRA serving
  run: |
    pip install git+https://github.com/cb-Sigma/lora-check
    lora-check verify --base-url "$ENDPOINT" --fingerprint adapters.fp.json \
        --format markdown --output "$GITHUB_STEP_SUMMARY" --strict
```

Exit codes: `0` healthy, `1` at least one check failed (with `--strict`, also
on warnings), `2` the run could not be completed (endpoint unreachable, no
log-probs, bad arguments). A complete example is in
[examples/github-actions.yml](examples/github-actions.yml).

## Lint rules

Engine rules state how a specific engine treats a feature. Every one cites
where that was read; "source" means the engine's code on its main branch on
2026-11-12, which can change. `lora-check lint --list-rules` prints this table.

<!-- BEGIN rules -->
| Rule | Applies to | What it flags | Source |
|---|---|---|---|
| `L001` | any | adapter_config.json is missing or unreadable | [docs](https://huggingface.co/docs/peft/main/en/package_reference/lora) |
| `L002` | any | no adapter weights file | [docs](https://huggingface.co/docs/peft/main/en/package_reference/lora) |
| `L003` | any | weights are a pickle (.bin) and cannot be inspected safely | structural |
| `L004` | any | adapter is not a LoRA (peft_type) | [docs](https://huggingface.co/docs/peft/main/en/package_reference/lora) |
| `L010` | any | required config field missing or invalid | [docs](https://huggingface.co/docs/peft/main/en/package_reference/lora) |
| `L020` | any | per-module scaling differs (rank_pattern / alpha_pattern) | [docs](https://huggingface.co/docs/peft/main/en/package_reference/lora) |
| `L021` | any | rank-stabilised scaling (use_rslora) | [docs](https://huggingface.co/docs/peft/main/en/package_reference/lora) |
| `L022` | any | DoRA adapter (use_dora) | [docs](https://huggingface.co/docs/peft/main/en/package_reference/lora) |
| `L023` | any | adapter trains base-model biases (bias) | [docs](https://huggingface.co/docs/peft/main/en/package_reference/lora) |
| `L024` | any | full modules saved next to the LoRA (modules_to_save) | [docs](https://huggingface.co/docs/peft/main/en/package_reference/lora) |
| `L025` | any | initialisation that modifies the base weights (init_lora_weights) | [docs](https://huggingface.co/docs/peft/main/en/package_reference/lora) |
| `L026` | any | LoRA on nn.Parameter targets (target_parameters) | [docs](https://huggingface.co/docs/peft/main/en/package_reference/lora) |
| `L030` | any | weights do not match the config | [docs](https://huggingface.co/docs/peft/main/en/package_reference/lora) |
| `L031` | any | unexpected weight key layout | structural |
| `L040` | any | adapter modules not found in the base checkpoint | structural |
| `L041` | any | adapter and base module shapes disagree | structural |
| `L042` | any | base_model_name_or_path differs from --base | structural |
| `L043` | any | adapter and base module paths differ by a prefix | structural |
| `L050` | any | adapter changes the tokenizer or embeddings | [docs](https://huggingface.co/docs/peft/main/en/package_reference/lora) |
| `V101` | vllm | vLLM does not support DoRA | [source](https://github.com/vllm-project/vllm/blob/main/vllm/lora/peft_helper.py) |
| `V102` | vllm | vLLM does not support adapter bias | [source](https://github.com/vllm-project/vllm/blob/main/vllm/lora/peft_helper.py) |
| `V103` | vllm | vLLM only supports modules_to_save for classifier heads | [source](https://github.com/vllm-project/vllm/blob/main/vllm/lora/peft_helper.py) |
| `V104` | vllm | vLLM rejects inits that modify the base weights | [source](https://github.com/vllm-project/vllm/blob/main/vllm/lora/peft_helper.py) |
| `V105` | vllm | vLLM does not implement this PEFT LoRA feature | [source](https://github.com/vllm-project/vllm/blob/main/vllm/lora/peft_helper.py) |
| `V106` | vllm | rank exceeds vLLM's default --max-lora-rank | [docs](https://docs.vllm.ai/en/latest/features/lora.html) |
| `V107` | vllm | vLLM applied one lora_alpha/r scale to every module (fixed in 0.32.0) | [issue](https://github.com/vllm-project/vllm/issues/59799) |
| `V108` | vllm | vLLM validates only the last component of module names | [issue](https://github.com/vllm-project/vllm/issues/34186) |
| `V109` | vllm | MoE expert LoRA needs extra vLLM settings and is unsafe with DP > 1 | [issue](https://github.com/vllm-project/vllm/issues/61453) |
| `V110` | vllm | rsLoRA was not supported by old vLLM releases | [issue](https://github.com/vllm-project/vllm/issues/10798) |
| `S201` | sglang | SGLang does not support DoRA adapters | [source](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py) |
| `S202` | sglang | SGLang does not support adapters that add tokens | [source](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py) |
| `S203` | sglang | SGLang uses a single lora_alpha/r scale per adapter | [source](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora.py) |
| `S204` | sglang | SGLang LoRA on the embedding layer is under development | [docs](https://docs.sglang.io/advanced_features/lora.html) |
| `S205` | sglang | SGLang must be started with matching rank and target modules | [docs](https://docs.sglang.io/advanced_features/lora.html) |
<!-- END rules -->

## Engine notes

**Nothing in this project has been run against a real vLLM or SGLang server.**
No GPU was available during development. Engine behaviour below was read from
documentation and source, not observed.

| | vLLM | SGLang |
|---|---|---|
| Adapters in `/v1/models` with `parent` and `root` | from [docs](https://docs.vllm.ai/en/latest/features/lora.html) | from [source](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py) (the docs do not say) |
| Selecting an adapter | `model` = adapter name ([docs](https://docs.vllm.ai/en/latest/features/lora.html)) | `model` = `base-model:adapter-name`, or `lora_path` in the body ([docs](https://docs.sglang.io/advanced_features/lora.html)) |
| Dynamic loading | `POST /v1/load_lora_adapter` with `lora_name`, `lora_path`, optional `load_inplace`; needs `VLLM_ALLOW_RUNTIME_LORA_UPDATING` ([docs](https://docs.vllm.ai/en/latest/features/lora.html)) | `POST /load_lora_adapter` with `lora_name`, `lora_path`, optional `pinned` ([docs](https://docs.sglang.io/advanced_features/lora.html)) |
| Log-probs | `logprobs` on completions and chat, capped by `--max-logprobs` (default 20, [source](https://github.com/vllm-project/vllm/blob/main/vllm/config/model.py)) | documented for chat completions (`logprobs`, `top_logprobs`); try `--api chat` |
| `cache_salt`, `return_tokens_as_token_ids` | request fields in the completions protocol ([source](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/completion/protocol.py)) | not known |
| Tested with `lora-check` | **no** | **no** |

What *has* been tested is listed under [Evaluation](#evaluation): the bundled
simulator, and real transformers + PEFT model arithmetic on CPU behind a small
test server.

## Evaluation

### Simulated server (SIMULATED)

Detection and false-alarm counts over 25 independently seeded runs per cell,
on the bundled toy model with synthetic faults. "noise" is the standard
deviation of per-request Gaussian noise added to the logits; the adapters move
the output by about 0.5 nats. The rightmost column is far noisier than the
0.04-0.05 nats run-to-run noise quoted in vllm#61453. Regenerate with
`python scripts/evaluate_sim.py`.

<!-- BEGIN evaluation -->
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
<!-- END evaluation -->

Two things to read carefully. First, the one false-alarm cell: with adapters
ten times weaker than usual *and* the heaviest noise, a healthy server got a
FAIL in 7 of 25 runs (an adapter judged indistinguishable from the base or
from another adapter), because the effect is far below the noise. Second, the
zero rows: a wrong scale, swapped adapters and a consistently contaminated
prefix cache are **invisible without a fingerprint** (or, for the cache,
`--cache-salt`). These numbers describe the tool on a toy model. They are not
a claim about any engine.

### Real PEFT arithmetic on CPU

`tests/test_real_peft.py` ran here with torch 2.15.0 (CPU build), transformers
5.22.2 and peft 0.21.2 on `hf-internal-testing/tiny-random-gpt2`. Four LoRA
adapters with random non-zero weights are created locally (plain, a second
one, one with `rank_pattern`/`alpha_pattern`, one with `use_rslora`),
fingerprinted with the real `lora-check fingerprint` code, and served over
HTTP by a small test-only server built on transformers + PEFT
(`tests/peft_server.py`). All 12 tests pass:

| Case (real model maths, 12 probes) | Result |
|---|---|
| Fingerprint reference equals an independent `merge_and_unload()` forward pass | within 2e-4 nats |
| Healthy server, 4 adapters | all checks PASS |
| Server in float64 vs float32 fingerprint | PASS, base offset < 0.001 nats |
| Adapter ignored | APPLIED FAIL, cause `adapter-ignored` |
| `lora_alpha` halved in the served config | REFERENCE FAIL, matches variant `scale=0.5`; APPLIED still passes |
| One `lora_alpha/r` for all modules of the patterned adapter | REFERENCE FAIL, matches variant `patterns-ignored` |
| `alpha_pattern` dropped from the served config | REFERENCE FAIL |
| `use_rslora` dropped from the served config | REFERENCE FAIL, matches variant `rslora-ignored` |
| Two adapters swapped | REFERENCE FAIL, cause `wrong-adapter` |
| `lint` on the PEFT-written adapter directory | parses cleanly; `V107` for vLLM 0.30.0 |

The tiny model has random weights and near-uniform outputs, so adapter effects
are small (about 0.04 nats); the test server is deterministic and serves one
request at a time. It validates the arithmetic and the file formats, not
behaviour under a real engine's batching, quantisation or caching.

## Limitations

- **Without a fingerprint, `lora-check` cannot tell you an adapter is
  correct.** It can only tell you it is applied, distinct from the others and
  stable. Wrong scale and swapped adapters pass.
- **Engines or deployments without log-probs are not supported.** `verify`
  stops with exit code 2. Comparing generated text instead is on the roadmap.
- **Probes may miss a narrow adapter.** An adapter that only changes behaviour
  on inputs unlike the probes can look weak or absent. Use `mine`.
- **A consistently contaminated prefix cache is invisible to black-box
  repetition**, because every repeat hits the same cache entry. It takes
  `--cache-salt` (vLLM only, and not yet tried on a real server) or a
  fingerprint.
- **Top-K only.** Differences confined to tokens outside the top-K are not
  seen. Hosted APIs that cap `top_logprobs` at 5 give less signal.
- **Token text can be ambiguous.** Fingerprints store token ids and decoded
  text; engines that render tokens differently from `tokenizer.decode` will
  show a mismatch on the base model, which `verify` reports as a base
  mismatch rather than blaming the adapter. `--token-ids` avoids this on vLLM.
- **The scale estimate is an estimate.** It interpolates between stored
  references and assumes one global factor. Per-module errors show up as a
  mismatch with a poor fit, not as a number.
- **Static lint sees checkpoint key names, not the engine's runtime module
  tree.** A prefix difference is a warning to go and verify, not proof.
- **Engine rules age.** They cite the page or file they came from and the
  date it was read. Check `--list-rules` against your engine version.
- **The STABLE check sends concurrent traffic** (about 75 extra short
  requests per model with defaults). Point it at a staging replica if that
  matters.

## Related work

We searched GitHub, PyPI and the web (November 2026) for tools that verify
LoRA application on a served endpoint and found none with that purpose. The
closest things we know of:

- **Manual comparison against transformers + PEFT**, which is what the issue
  reporters above did by hand. `lora-check` packages that method.
- [**loraspectra**](https://github.com/SalyyS1/loraspectra): a Rust tool for
  spectral analysis, rank compression and merging of `.safetensors` LoRA
  files, suggested in vllm#59799 for folding per-module scales into the
  weights. It rewrites adapters; it does not check what a server does.
- **`/v1/models` listings** (vLLM, SGLang, NVIDIA NIM) confirm an adapter is
  *registered*. Every failure above happens after registration.
- **Engine test-suites** check LoRA correctness for the configurations the
  maintainers have hardware for. `lora-check` checks yours.
- **Evaluation harnesses** would notice a broken adapter as a quality drop,
  slowly and without telling you why.

If you know of a tool we missed, please open an issue.

## FAQ

**How many requests does `verify` send?** With defaults, roughly
`(adapters + 1) x probes x 6`: 876 one-token requests for five adapters and 24
probes in the demo. Reduce with `--max-probes`, `--repeats 2`,
`--no-interleave`.

**My endpoint is deterministic. Is calibration wasted?** It costs two extra
passes and gives you the `--min-delta` floor. It also tells you the endpoint
*is* deterministic, which is worth knowing.

**APPLIED is WARN, not PASS.** The adapter does something, but less than the
endpoint's noise on most probes. Either it is narrow (mine better probes) or
it is only partly applied (make a fingerprint).

**REFERENCE says "base mismatch".** The endpoint's *base model* is far from
the reference base: different weights or revision, quantisation, a chat
template applied on one side only, or token strings that do not line up. Fix
that first; the adapter cannot be judged against this fingerprint.

**Does it work with merged adapters?** There is nothing to distinguish: a
merged model is just a model. Fingerprint it as an adapter and compare if you
want to confirm the merge.

**Can I use it on hosted APIs?** If they expose log-probs and let you select
the adapter by model name, yes: `--engine openai --adapter ... --base-model
...`.

**Is the fingerprint sensitive data?** It contains your probe prompts and the
top-20 next-token log-probabilities for them, nothing else. It does not
contain weights.

## Roadmap

- Validation against real vLLM and SGLang servers, with recorded results per
  engine version.
- A text-generation fallback for endpoints without log-probs.
- `prompt_logprobs`-based probing to cover more token positions per request.
- Reload workflows: verify, hot-swap through the engine's load endpoint,
  verify again.
- More engines' rules (TGI, llama.cpp server, LoRAX, NIM) as they can be
  sourced.
- Multimodal probes.

## Contributing

Bug reports with a `--format json` report attached are the most useful thing
you can send, especially from real engines. See
[CONTRIBUTING.md](CONTRIBUTING.md). New lint rules must cite a source.

## Disclaimer and licence

`lora-check` is an independent project. It is not affiliated with, endorsed by
or sponsored by the vLLM project, the SGLang project or Hugging Face; their
names are used only to describe compatibility.

Licensed under the [Apache License 2.0](LICENSE).
