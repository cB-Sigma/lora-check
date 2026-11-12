# How the checks work

This page gives the definitions behind the verdicts so that you can judge a
report yourself. The implementation is in `src/lora_check/dist.py`,
`verify.py` and `fingerprint.py`.

## What is measured

Every request is `max_tokens=1`, `temperature=0`, with `logprobs` set to K
(20 by default). The answer is a *truncated distribution*: the K most likely
next tokens with their log-probabilities.

The **distance** between two such answers is the mean absolute difference of
log-probabilities over the union of their tokens. A token that one side does
not list is known to be at most as likely as that side's least likely listed
token, so that minimum is substituted. The substitution can only shrink the
true difference, so the distance never invents a difference. The unit is
nats.

## Noise

For model `m` and probe `i`, the *within* distance is the largest distance
among the repeats of the identical request. A model's **noise** is the 90th
percentile of its within distances over probes; the percentile keeps a single
freak answer from inflating every threshold. A difference counts only if it
exceeds

```
threshold = max(min_delta, noise_multiplier * noise)
```

with defaults 0.001 nats and 3.

## APPLIED and DISTINCT

For two models, the per-probe *cross* distance is the mean distance over all
pairs of repeats. A probe "differs" when its cross distance exceeds the
threshold of the noisier model.

- no probe differs: go to the sub-noise test below;
- fewer than a quarter differ: WARN (narrow, or partly applied);
- otherwise PASS.

### The sub-noise test

A plain distance cannot see an effect smaller than the noise: the noise itself
contributes a positive amount to every distance. So for repeats `r = 1..R` let
`d_r = a_r - b_r` be the vector of log-prob differences between the two models
on the tokens every answer lists. If the models are the same, the `d_r` are
independent noise and the inner product `<d_r, d_s>` of two *different*
repeats has mean zero. If they differ by `delta`, its mean is `|delta|^2`.
Averaging these cross products per probe gives an unbiased estimate of the
squared effect; a t-statistic over probes says whether it is distinguishable
from zero. The check requires `t >= 4` and an estimated effect of at least
`min_delta`.

When this test fires and the per-probe test did not, APPLIED is WARN: the
adapter does something, but less than the endpoint's noise. Only when both are
silent is the verdict FAIL.

## STABLE

Three more sets of answers are collected per model:

1. a concurrent burst of all probes for that model *alone*;
2. a sequential pass in which every request goes to a different model than
   the one before, rotating so that every model follows every other;
3. a concurrent burst mixing all models on the same prompts.

Each answer's **deviation** is its mean distance to the isolated repeats of
the same model and probe. In (2) it must stay within the model's threshold; in
(3) it must also stay within three times the deviation seen in (1), so that
ordinary batch-composition jitter is calibrated out. A moved answer is then
compared with the *other* models' isolated answers; if it lands on one, that
is reported as leakage with the model it matches. One unexplained moved answer
is a WARN, two or more (or any that matches another model) a FAIL.

The base model gets this check too: adapter state leaking into base requests
is the same bug.

## CACHE

With `--cache-salt` each request is repeated once with a fresh random
`cache_salt` field, which vLLM uses to partition its prefix cache. The salted
answer is computed without any cached state, so it must agree with the
unsalted ones. Disagreement on two or more probes is a FAIL. If the server
rejects the field the check is skipped and says so.

Why this needs help from the server: a cache that hands adapter B the state
adapter A computed does so *every time* for the same prompt, so repetition
alone sees perfectly consistent, wrong answers.

## REFERENCE

A fingerprint holds reference answers for the base model, for each adapter,
and for *variants* of each adapter applied wrongly on purpose. For the served
adapter the mean distance to every candidate is computed:

| Candidate | If it is the best match |
|---|---|
| the adapter's reference | PASS, if within tolerance |
| the base model | adapter ignored |
| another adapter in the fingerprint | swapped or mis-routed |
| variant `scale=s` | applied at `s` times the intended scale |
| variant `patterns-ignored` | one `lora_alpha/r` used for all modules |
| variant `rslora-ignored` | `lora_alpha/r` used instead of `lora_alpha/sqrt(r)` |

The **tolerance** is

```
max( min(--tolerance, 0.1 * effect),  3 * noise,  3 * base_offset )
```

where `effect` is the adapter's own mean distance from the base in the
fingerprint and `base_offset` is the distance between the endpoint's base
model and the reference base. The first term stops a fixed tolerance from
swallowing a weak adapter; the last adapts to the engine's numerics. If the
base offset is itself more than a quarter of the adapter's effect, the verdict
is WARN "base mismatch": the comparison is not trustworthy and the adapter is
not blamed.

### Scale estimate

The references at known scales (0 is the base, 1 the adapter, plus the stored
`scale=` variants) form a grid. Between each pair of neighbours the observed
answers are fitted as an interpolation, using log-prob differences centred
per prompt so that the softmax normaliser cancels. The bracket that explains
the observations best gives the estimate and an R². The estimate is reported
only when R² is at least 0.6 and the adapter's effect clearly exceeds the
tolerance; a deviation of more than 15% from 1.0 is then a FAIL on its own.

The fit assumes one global factor. A per-module error (the usual result of
ignored patterns) fits poorly, which is why the `patterns-ignored` variant is
stored explicitly.

## What a PASS does not mean

Without a fingerprint: not that the adapter is the right one, nor that it is
at the right strength. With a fingerprint: not that the adapter is good, only
that the server reproduces what transformers + PEFT compute on these probes.
