# Fingerprint file format

A fingerprint is one JSON object. Version 1:

```jsonc
{
  "format": "lora-check/fingerprint",
  "version": 1,
  "base_model": "org/base",
  "api": "completions",            // or "chat": probes were wrapped in the chat template
  "top_k": 20,
  "meta": { "reference": "transformers+peft", "torch": "...", "peft": "...", "dtype": "float32" },
  "probes": [ { "id": "g01", "prompt": "The capital of France is" } ],
  "base": { "g01": [ [301, " is", -6.52], [551, "ib", -6.55] ] },
  "adapters": {
    "sql-lora": {
      "config": { "r": 8, "lora_alpha": 16, "use_rslora": false, "distinct_scalings": [2.0] },
      "reference": { "g01": [ [301, " is", -6.31] ] },
      "variants": [
        { "name": "scale=0.5", "kind": "scale", "scale": 0.5,
          "description": "the adapter applied at 0.5x its intended scale",
          "dists": { "g01": [ [301, " is", -6.41] ] } }
      ]
    }
  }
}
```

Each distribution is a list of `[token_id, decoded_text, logprob]`, most
likely first, natural log. `token_id` may be `null` for references that have
no ids. When the endpoint reports `token_id:<n>` keys (vLLM with
`--token-ids`) matching is by id, otherwise by text; if several ids decode to
the same text the likelier one is kept, as an endpoint reporting text would.

Variant kinds written by `lora-check fingerprint`:

| kind | when | meaning |
|---|---|---|
| `scale` | always (`--scales`) | every LoRA layer's scaling multiplied by `scale` |
| `patterns-ignored` | per-module scalings differ | every layer set to the global `lora_alpha/r` |
| `rslora-ignored` | `use_rslora` | every layer set to `lora_alpha/r` instead of `lora_alpha/sqrt(r)` |

Variants are produced by editing `LoraLayer.scaling` in PEFT and running the
real forward pass. Size is roughly 0.5 kB per probe per stored distribution;
`--no-variants` or a shorter `--scales` list shrinks it.

Files with a higher `version` than the tool knows are rejected.
