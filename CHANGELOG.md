# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-11-12

First public version.

### Added

- `lora-check verify`: black-box probing of an OpenAI-compatible endpoint with
  the checks APPLIED, DISTINCT, STABLE, CACHE (`--cache-salt`) and REFERENCE
  (`--fingerprint`); adapter discovery from `/v1/models`; noise calibration
  from repeated requests; text, JSON and Markdown reports; CI exit codes.
- `lora-check fingerprint`: reference next-token distributions computed with
  transformers + PEFT, including mis-application variants (other scales,
  `rank_pattern`/`alpha_pattern` ignored, rsLoRA ignored).
- `lora-check lint`: static checks of `adapter_config.json` and the
  safetensors header for local directories and Hub repositories, against a
  base checkpoint and against sourced vLLM and SGLang rules.
- `lora-check mine`: selection of the most sensitive probes from a fingerprint.
- `lora-check sim` and `--demo`: a simulated multi-LoRA server with injectable
  faults, used for the demo, the tests and the evaluation table.

### Known limitations

- Not yet run against a real vLLM or SGLang server.
- Endpoints without log-probs are not supported.
