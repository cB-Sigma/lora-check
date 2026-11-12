# Engine facts and where they come from

Everything `lora-check` assumes about an engine is listed here with its
source. All pages and files were read on **2026-11-12**; "source" means the
main branch on that day. **None of it was observed on a running engine.**

## vLLM

| Fact | Source |
|---|---|
| Adapters are served with `--enable-lora --lora-modules name=path` (or a JSON form with `base_model_name`) | [LoRA docs](https://docs.vllm.ai/en/latest/features/lora.html) |
| `/v1/models` lists each adapter with `parent` = base model and `root` = adapter path | LoRA docs |
| An adapter is selected by putting its name in `model` | LoRA docs |
| `POST /v1/load_lora_adapter` (`lora_name`, `lora_path`, `load_inplace`) and `/v1/unload_lora_adapter`, enabled by `VLLM_ALLOW_RUNTIME_LORA_UPDATING` | LoRA docs |
| `max_lora_rank` defaults to 16, `max_loras` to 1 | [`vllm/config/lora.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/lora.py) |
| `max_logprobs` defaults to 20; `logprobs_mode` defaults to `raw_logprobs` | [`vllm/config/model.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/model.py) |
| Completion requests accept `cache_salt` and `return_tokens_as_token_ids` | [`completion/protocol.py`](https://github.com/vllm-project/vllm/blob/main/vllm/entrypoints/openai/completion/protocol.py) |
| Load-time rejections: DoRA, adapter `bias`, `lora_bias`, `modules_to_save` other than `classifier`/`score`, base-modifying `init_lora_weights`, aLoRA, `layer_replication`, BD-LoRA, QALoRA, rank above `max_lora_rank` | [`vllm/lora/peft_helper.py`](https://github.com/vllm-project/vllm/blob/main/vllm/lora/peft_helper.py) |
| Per-module `rank_pattern`/`alpha_pattern` scaling: absent from `peft_helper.py` at tags v0.30.0 and v0.31.0, present at v0.32.0 and later | the same file at those tags; [#59799](https://github.com/vllm-project/vllm/issues/59799) |
| rsLoRA is read (`use_rslora`) on main; it was not supported in 0.6.4.post1 | `peft_helper.py`; [#10798](https://github.com/vllm-project/vllm/issues/10798) |

Issues read in full: [#34186](https://github.com/vllm-project/vllm/issues/34186),
[#41754](https://github.com/vllm-project/vllm/issues/41754),
[#59799](https://github.com/vllm-project/vllm/issues/59799),
[#61453](https://github.com/vllm-project/vllm/issues/61453),
[#30931](https://github.com/vllm-project/vllm/issues/30931),
[#48297](https://github.com/vllm-project/vllm/issues/48297),
[#10798](https://github.com/vllm-project/vllm/issues/10798).

Notes on reading them fairly:

- #30931 is about two LoRA requests that share a name but carry different
  ids. The maintainer's position in the thread is that this arises in offline
  use and that the online server is not affected. `lora-check` therefore does
  not claim the online server has this bug; the `--cache-salt` control exists
  to make the question checkable.
- #41754 collects several distinct causes (a dense-model bug tracked
  elsewhere, an MoE expert-name mapping problem). The common symptom, output
  identical to base, is what `verify` tests.
- For #34186 and #61453 no merged fix was visible on the issue pages on the
  date above. That is not the same as "still broken in the latest release".

## SGLang

| Fact | Source |
|---|---|
| `--enable-lora`, `--lora-paths` (`path`, `name=path` or JSON), `--max-loras-per-batch`, `--max-lora-rank`, `--lora-target-modules` | [LoRA docs](https://docs.sglang.io/advanced_features/lora.html) |
| An adapter is selected with `model` = `base-model:adapter-name`; `lora_path` in the body also works and loses if both are given | [OpenAI API docs](https://docs.sglang.io/basic_usage/openai_api_completions.html) |
| `/v1/models` lists loaded adapters with `parent` and `root` | [`http_server.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/entrypoints/http_server.py) (not stated in the docs) |
| `POST /load_lora_adapter` (`lora_name`, `lora_path`, `pinned`), `/unload_lora_adapter` | LoRA docs |
| DoRA adapters and adapters that add tokens beyond the base vocabulary are refused | [`lora_manager.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora_manager.py) |
| The adapter scale is `lora_alpha / r`; `lora.py`, `lora_manager.py`, `mem_pool.py` and `lora_config.py` contain no reference to `use_rslora`, `rank_pattern` or `alpha_pattern` | [`lora.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/lora/lora.py) |
| Embedding-layer LoRA is listed as under development | LoRA docs |
| `logprobs`/`top_logprobs` are documented for chat completions | OpenAI API docs |

The scaling row is an inference from reading four files, not a documented
statement and not a reported bug. Rule `S203` is therefore a warning that
tells you to confirm with a fingerprint.

Whether SGLang accepts a bare adapter name in `model` was not established;
`--engine sglang` uses the documented `base:adapter` form.

## PEFT

Field meanings (`r`, `lora_alpha`, `target_modules`, `rank_pattern`,
`alpha_pattern`, `use_rslora`, `use_dora`, `bias`, `modules_to_save`,
`init_lora_weights`, `target_parameters`, ...) are taken from the
[LoRA reference](https://huggingface.co/docs/peft/main/en/package_reference/lora):
the scale is `lora_alpha/r`, or `lora_alpha/sqrt(r)` with `use_rslora`;
pattern keys match as suffixes at a dot boundary. The fingerprint and lint
code were run against peft 0.21.2.
