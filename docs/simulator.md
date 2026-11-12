# The simulator

`lora_check.sim` is a fake multi-LoRA server used by `--demo`, the tests and
the evaluation. Run it yourself with `lora-check sim --scenario mixed`.

## The toy model

A recurrent network in pure Python: `h_t = tanh(W_h h_{t-1} + E[x_t])`,
`logits = W_o h_T`, hidden size 16, 97 tokens. An adapter is a rank-2 update
`scale * B A` on both `W_h` and `W_o`, with the nominal scale `alpha / r`.
Because the recurrence is non-linear, a wrong scale does not act linearly on
the output. Weights come from seeded generators, so everything is
reproducible.

It is not a language model and nothing about its numbers transfers to real
models. It exists so that the *statistics* of the checks can be exercised
with known ground truth.

## The HTTP surface

`GET /v1/models` (with `parent` and `root`), `POST /v1/completions` and
`/v1/chat/completions` with log-probs, `return_tokens_as_token_ids`,
`cache_salt`, a `max_logprobs` cap, the `base:adapter` model form, and
`POST /v1/load_lora_adapter` / `/v1/unload_lora_adapter`.

## Benign noise

| Setting | Meaning |
|---|---|
| `noise` | Gaussian noise on the logits, fresh per request |
| `batch_noise` | extra noise when other requests are in flight |
| `systematic` | a fixed per-prompt offset, like a different dtype or kernel |

## Faults

| Fault | What the server does | Upstream shape it imitates |
|---|---|---|
| `ignored` | serves base output for an adapter | vllm#34186, #41754 |
| `scale` | applies the adapter at a multiple of its scale | vllm#59799, #10798 |
| `alias` | serves one adapter's weights under another name | routing mix-ups |
| `leak_after_switch` | first request after a model switch uses the previous adapter | per-request state leakage |
| `batch_mix` | an overlapping request is computed with a neighbour's adapter | vllm#61453 (cross-applied state) |
| `cache_ignores_adapter` | prefix-cache keys omit the adapter | cross-adapter cache reuse |
| `stale_cache_on_reload` | prefix-cache keys use only the adapter name, so an in-place reload keeps old state | vllm#30931 |
| `no_logprobs` | answers without log-probs | engines without the feature |

Each fault reproduces the *observable shape* of a failure class. None claims
to reproduce an engine's internals.
