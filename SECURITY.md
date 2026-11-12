# Security policy

## Supported versions

Only the latest release receives fixes.

## Reporting a vulnerability

Please do not open a public issue for a security problem. Use GitHub's
private vulnerability reporting for this repository
(<https://github.com/cb-Sigma/lora-check/security/advisories/new>) and include
steps to reproduce. You should get an acknowledgement within a week.

## What the tool touches

- `verify` sends prompts from the probe set to the endpoint you name and reads
  the answers. An API key given with `--api-key`, `LORA_CHECK_API_KEY` or
  `OPENAI_API_KEY` is sent only to that endpoint as a bearer token.
- `lint` reads JSON files and safetensors headers. It never deserialises
  pickle (`.bin`) weights. For Hub repositories it talks to `huggingface.co`
  only, and sends `HF_TOKEN` there if that variable is set.
- `fingerprint` loads models with transformers and PEFT. Loading a model you
  do not trust carries the usual risks; `--trust-remote-code` is off unless
  you pass it.
- Fingerprint files contain your probe prompts and top-K log-probabilities.
  Treat them like any other artefact derived from your model.
- The bundled simulator binds to `127.0.0.1` by default and is meant for
  demos and tests, not for exposure to a network.
