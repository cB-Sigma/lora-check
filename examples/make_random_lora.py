"""Create a LoRA adapter with random non-zero weights, for experiments.

A freshly initialised LoRA has B = 0 and changes nothing, which makes it
useless for testing whether a server applies adapters. This writes one that
visibly moves the output. Needs torch, transformers and peft.

    python examples/make_random_lora.py hf-internal-testing/tiny-random-gpt2 ./rand-lora \
        --target c_attn --fan-in-fan-out
    lora-check fingerprint --base hf-internal-testing/tiny-random-gpt2 \
        --adapter rand=./rand-lora -o rand.fp.json
"""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("base")
    parser.add_argument("output")
    parser.add_argument(
        "--target", action="append", required=True, help="target module name (repeatable)"
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--std", type=float, default=0.05, help="std-dev of the random lora_B weights"
    )
    parser.add_argument(
        "--fan-in-fan-out", action="store_true", help="needed for GPT-2 style Conv1D layers"
    )
    parser.add_argument("--rslora", action="store_true")
    args = parser.parse_args()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.base)
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        target_modules=args.target,
        fan_in_fan_out=args.fan_in_fan_out,
        use_rslora=args.rslora,
        task_type="CAUSAL_LM",
    )
    wrapped = get_peft_model(model, config)
    with torch.no_grad():
        for name, param in wrapped.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn_like(param) * args.std)
    wrapped.save_pretrained(args.output)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
