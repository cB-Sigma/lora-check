"""Command-line interface: ``lora-check verify | fingerprint | lint | mine | sim``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from lora_check import __version__
from lora_check.client import Client, EndpointError
from lora_check.fingerprint import Fingerprint, FingerprintError
from lora_check.probes import Probe, ProbeError, builtin_probes, dump_probes, load_probes

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2

ENGINE_PRESETS: dict[str, dict[str, Any]] = {
    # vLLM lists each adapter in /v1/models and addresses it by its name.
    "vllm": {"adapter_model_format": "{adapter}"},
    # SGLang documents `base-model:adapter-name` in the model field.
    "sglang": {"adapter_model_format": "{base}:{adapter}"},
    "openai": {"adapter_model_format": "{adapter}"},
}


def _err(message: str) -> None:
    print(f"lora-check: {message}", file=sys.stderr)


def _emit(text: str, output: str | None) -> None:
    if output:
        Path(output).write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    else:
        print(text)


def _probes(args: argparse.Namespace, default: list[Probe] | None = None) -> list[Probe]:
    probes = load_probes(args.probes) if args.probes else (default or builtin_probes())
    if args.max_probes:
        probes = probes[: args.max_probes]
    return probes


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def _cmd_verify(args: argparse.Namespace) -> int:
    from lora_check.report import render_verify
    from lora_check.verify import VerifyConfig, run_verify

    if args.demo:
        from lora_check import demo

        try:
            report = demo.demo_verify(
                args.demo,
                with_fingerprint=not args.no_fingerprint,
                cache_salt=args.cache_salt,
                concurrency=args.concurrency,
            )
        except ValueError as exc:
            _err(str(exc))
            return EXIT_USAGE
        _emit(
            render_verify(report, args.format, demo.demo_label(args.demo), args.verbose),
            args.output,
        )
        return report.exit_code(args.strict)

    if not args.base_url:
        _err("verify needs --base-url (or --demo to try it against the bundled simulator)")
        return EXIT_USAGE
    fingerprint = Fingerprint.load(args.fingerprint) if args.fingerprint else None
    probes = _probes(args, fingerprint.probes if fingerprint else None)
    if fingerprint is not None and args.probes:
        known = {p.id: p.prompt for p in fingerprint.probes}
        if any(known.get(p.id) != p.prompt for p in probes):
            raise ProbeError(
                "--probes must be a subset of the fingerprint's probes (same ids and prompts)"
            )
    name_map: dict[str, str] = {}
    for item in args.map or []:
        served, sep, ref = item.partition("=")
        if not sep or not served or not ref:
            _err(f"--map expects SERVED=FINGERPRINT_NAME, got {item!r}")
            return EXIT_USAGE
        name_map[served] = ref
    extra: dict[str, Any] = {}
    if args.extra_body:
        try:
            parsed: Any = json.loads(args.extra_body)
        except json.JSONDecodeError as exc:
            _err(f"--extra-body is not valid JSON: {exc}")
            return EXIT_USAGE
        if not isinstance(parsed, dict):
            _err("--extra-body must be a JSON object")
            return EXIT_USAGE
        extra = parsed
    preset = ENGINE_PRESETS[args.engine]
    api = args.api or (fingerprint.api if fingerprint else "completions")
    top = args.top_logprobs or (min(20, fingerprint.top_k) if fingerprint else 20)
    api_key = (
        args.api_key or os.environ.get("LORA_CHECK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    )

    def progress(message: str) -> None:
        if not args.quiet:
            print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr)

    with Client(
        args.base_url,
        api_key=api_key,
        api=api,
        top_logprobs=top,
        timeout=args.timeout,
        token_ids=args.token_ids,
        extra_body=extra,
    ) as client:
        report = run_verify(
            client,
            VerifyConfig(
                probes=probes,
                adapters=args.adapter or None,
                base_model=args.base_model,
                adapter_model_format=args.adapter_model_format or preset["adapter_model_format"],
                repeats=args.repeats,
                concurrency=args.concurrency,
                noise_multiplier=args.noise_multiplier,
                min_delta=args.min_delta,
                interleave=not args.no_interleave,
                cache_salt=args.cache_salt,
                fingerprint=fingerprint,
                name_map=name_map,
                reference_tolerance=args.tolerance,
                progress=progress,
            ),
        )
    _emit(render_verify(report, args.format, verbose=args.verbose), args.output)
    return report.exit_code(args.strict)


# ---------------------------------------------------------------------------
# fingerprint
# ---------------------------------------------------------------------------


def _cmd_fingerprint(args: argparse.Namespace) -> int:
    if args.demo:
        from lora_check import demo

        fingerprint = demo.demo_fingerprint(args.demo)
    else:
        if not args.base or not args.adapter:
            _err("fingerprint needs --base MODEL and at least one --adapter NAME=PATH (or --demo)")
            return EXIT_USAGE
        adapters: dict[str, str] = {}
        for item in args.adapter:
            name, sep, path = item.partition("=")
            if not sep:
                name, path = Path(item.rstrip("/")).name, item
            if not name or not path or name in adapters:
                _err(f"--adapter expects unique NAME=PATH entries, got {item!r}")
                return EXIT_USAGE
            adapters[name] = path
        try:
            scales = tuple(float(x) for x in args.scales.split(",") if x.strip())
        except ValueError:
            _err(f"--scales must be comma-separated numbers, got {args.scales!r}")
            return EXIT_USAGE
        from lora_check.reference import ReferenceError, compute_fingerprint

        try:
            fingerprint = compute_fingerprint(
                args.base,
                adapters,
                _probes(args),
                top_k=args.top_k,
                dtype=args.dtype,
                device=args.device,
                chat=args.chat,
                scales=scales,
                variants=not args.no_variants,
                revision=args.revision,
                trust_remote_code=args.trust_remote_code,
                progress=None if args.quiet else lambda m: print(m, file=sys.stderr),
            )
        except ReferenceError as exc:
            _err(str(exc))
            return EXIT_USAGE
    fingerprint.save(args.output)
    if not args.quiet:
        print(
            f"wrote {args.output}: base {fingerprint.base_model}, {len(fingerprint.probes)} probes, "
            f"top-{fingerprint.top_k}, adapters:",
            file=sys.stderr,
        )
        for name, ref in fingerprint.adapters.items():
            values = sorted(fingerprint.sensitivity(name).values())
            mean_effect = sum(values) / len(values)
            variants = ", ".join(v.name for v in ref.variants) or "none"
            print(
                f"  {name}: mean effect {mean_effect:.4f} nats (min {values[0]:.4f}, max {values[-1]:.4f}); "
                f"variants: {variants}",
                file=sys.stderr,
            )
    return EXIT_OK


# ---------------------------------------------------------------------------
# lint
# ---------------------------------------------------------------------------


def _cmd_lint(args: argparse.Namespace) -> int:
    from lora_check.lint import RULES, SourceError, lint_adapter, open_source
    from lora_check.report import render_lint

    if args.list_rules:
        for rule in RULES.values():
            engine = rule.engine or "generic"
            fixed = f" (fixed in {rule.fixed_in})" if rule.fixed_in else ""
            print(f"{rule.id}  [{engine}] {rule.title}{fixed}")
            if rule.source:
                print(f"      {rule.source_kind}: {rule.source}")
        return EXIT_OK
    if args.demo:
        from lora_check import demo

        report = demo.demo_lint(args.engine or "vllm", args.engine_version or "0.30.0")
    else:
        if not args.adapter:
            _err("lint needs an adapter directory or Hub id (or --demo)")
            return EXIT_USAGE
        if args.engine_version and not args.engine:
            _err("--engine-version needs --engine")
            return EXIT_USAGE
        try:
            adapter = open_source(args.adapter, revision=args.revision)
            base = open_source(args.base, revision=args.base_revision) if args.base else None
        except SourceError as exc:
            _err(str(exc))
            return EXIT_USAGE
        report = lint_adapter(adapter, base, args.engine, args.engine_version)
    _emit(render_lint(report, args.format), args.output)
    return report.exit_code(args.strict)


# ---------------------------------------------------------------------------
# mine
# ---------------------------------------------------------------------------


def _cmd_mine(args: argparse.Namespace) -> int:
    from lora_check.mine import mine_probes, mined_table

    fingerprint = Fingerprint.load(args.fingerprint)
    mined = mine_probes(fingerprint, args.adapter or None, args.top)
    probes = [m.probe for m in mined]
    if args.output:
        dump_probes(probes, args.output)
    if args.fingerprint_out:
        fingerprint.subset([p.id for p in probes]).save(args.fingerprint_out)
    if args.format == "json":
        print(
            json.dumps(
                [
                    {
                        "id": m.probe.id,
                        "prompt": m.probe.prompt,
                        "chosen_for": m.chosen_for,
                        "sensitivity": {k: round(v, 6) for k, v in m.sensitivity.items()},
                    }
                    for m in mined
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
    else:
        print(
            f"{len(mined)} most sensitive probes of {len(fingerprint.probes)} (distance adapter vs base, nats)"
        )
        print(mined_table(mined))
    return EXIT_OK


# ---------------------------------------------------------------------------
# sim
# ---------------------------------------------------------------------------


def _cmd_sim(args: argparse.Namespace) -> int:
    from lora_check import demo
    from lora_check.sim import running_server

    try:
        sim = demo.demo_sim(args.scenario, noise=args.noise)
    except ValueError as exc:
        _err(str(exc))
        return EXIT_USAGE
    with running_server(sim, host=args.host, port=args.port) as url:
        print(
            f"simulated multi-LoRA server ({args.scenario}) on {url}/v1 - Ctrl-C to stop",
            file=sys.stderr,
        )
        print(f"try: lora-check verify --base-url {url}", file=sys.stderr)
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            return EXIT_OK


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lora-check",
        description="Verify that an inference server really applies the LoRA adapters you asked for.",
    )
    parser.add_argument("--version", action="version", version=f"lora-check {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    v = sub.add_parser(
        "verify",
        help="probe a live OpenAI-compatible endpoint",
        description="Black-box check that each adapter is applied, distinct, stable under interleaving "
        "and (with --fingerprint) matches its reference. Exit code 0 healthy, 1 a check failed, "
        "2 the run could not be completed.",
    )
    v.add_argument(
        "--base-url", help="endpoint, e.g. http://localhost:8000 (the /v1 suffix is optional)"
    )
    v.add_argument(
        "--api-key", help="bearer token (default: $LORA_CHECK_API_KEY or $OPENAI_API_KEY)"
    )
    v.add_argument(
        "--adapter",
        action="append",
        metavar="NAME",
        help="adapter id to check (repeatable; default: discover from /v1/models)",
    )
    v.add_argument(
        "--base-model", help="base model id (default: the adapters' parent in /v1/models)"
    )
    v.add_argument(
        "--engine",
        choices=sorted(ENGINE_PRESETS),
        default="vllm",
        help="how adapters are addressed (default: vllm)",
    )
    v.add_argument(
        "--adapter-model-format",
        metavar="FMT",
        help="model field for an adapter, e.g. '{base}:{adapter}' (overrides --engine)",
    )
    v.add_argument(
        "--api",
        choices=("completions", "chat"),
        help="endpoint family to probe (default: completions)",
    )
    v.add_argument(
        "--probes",
        metavar="FILE",
        help="probe file (.txt, .jsonl or .json); default: built-in set or the fingerprint's",
    )
    v.add_argument("--max-probes", type=int, metavar="N", help="use only the first N probes")
    v.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="identical requests per probe for noise calibration (default: 3)",
    )
    v.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="parallel requests in the burst phases (default: 8)",
    )
    v.add_argument(
        "--top-logprobs", type=int, metavar="K", help="top-K log-probs to request (default: 20)"
    )
    v.add_argument(
        "--token-ids",
        action="store_true",
        help="ask for token ids instead of token text (vLLM return_tokens_as_token_ids)",
    )
    v.add_argument(
        "--no-interleave", action="store_true", help="skip the interleaving phases (STABLE check)"
    )
    v.add_argument(
        "--cache-salt",
        action="store_true",
        help="also repeat every request with a unique cache_salt (vLLM) and require identical answers",
    )
    v.add_argument(
        "--fingerprint", metavar="FILE", help="reference fingerprint from `lora-check fingerprint`"
    )
    v.add_argument(
        "--no-fingerprint",
        action="store_true",
        help="with --demo: run without the reference fingerprint",
    )
    v.add_argument(
        "--map",
        action="append",
        metavar="SERVED=NAME",
        help="served adapter id -> adapter name in the fingerprint (repeatable)",
    )
    v.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help="absolute tolerance (nats) when matching the fingerprint (default: 0.02)",
    )
    v.add_argument(
        "--noise-multiplier",
        type=float,
        default=3.0,
        help="a difference must exceed this many times the measured noise (default: 3)",
    )
    v.add_argument(
        "--min-delta",
        type=float,
        default=1e-3,
        help="smallest distance (nats) treated as a difference (default: 0.001)",
    )
    v.add_argument(
        "--extra-body", metavar="JSON", help="extra fields merged into every request body"
    )
    v.add_argument(
        "--timeout", type=float, default=60.0, help="per-request timeout in seconds (default: 60)"
    )
    v.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    v.add_argument(
        "--output", "-o", metavar="FILE", help="write the report to FILE instead of stdout"
    )
    v.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    v.add_argument("--verbose", action="store_true", help="show details for passing models too")
    v.add_argument("--quiet", action="store_true", help="no progress messages on stderr")
    v.add_argument(
        "--demo",
        nargs="?",
        const="mixed",
        metavar="SCENARIO",
        help="run against the bundled simulated server (mixed, healthy, leak, duplicate, cache)",
    )
    v.set_defaults(func=_cmd_verify)

    f = sub.add_parser(
        "fingerprint",
        help="compute reference distributions with transformers + PEFT",
        description="Compute the reference next-token distributions of the base model and each adapter "
        "on the probe set and write them to a small JSON file for `verify --fingerprint`. "
        'Needs torch, transformers and peft (pip install "lora-check[fingerprint]").',
    )
    f.add_argument("--base", metavar="MODEL", help="base model directory or Hub id")
    f.add_argument(
        "--adapter",
        action="append",
        metavar="NAME=PATH",
        help="adapter to fingerprint; NAME should be the name it is served under (repeatable)",
    )
    f.add_argument("--probes", metavar="FILE", help="probe file (default: built-in set)")
    f.add_argument("--max-probes", type=int, metavar="N")
    f.add_argument("--top-k", type=int, default=20, help="log-probs stored per probe (default: 20)")
    f.add_argument(
        "--dtype", default="float32", help="torch dtype for the reference (default: float32)"
    )
    f.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    f.add_argument(
        "--chat",
        action="store_true",
        help="wrap each probe in the tokenizer's chat template (for verify --api chat)",
    )
    f.add_argument(
        "--scales",
        default="0.25,0.5,0.75,1.5,2,4",
        help="global scale variants to store (default: 0.25,0.5,0.75,1.5,2,4)",
    )
    f.add_argument("--no-variants", action="store_true", help="store only the correct reference")
    f.add_argument("--revision", help="base model revision")
    f.add_argument("--trust-remote-code", action="store_true")
    f.add_argument("--output", "-o", metavar="FILE", default="fingerprint.json")
    f.add_argument("--quiet", action="store_true")
    f.add_argument(
        "--demo",
        nargs="?",
        const="mixed",
        metavar="SCENARIO",
        help="write the simulator's ground-truth fingerprint instead",
    )
    f.set_defaults(func=_cmd_fingerprint)

    li = sub.add_parser(
        "lint",
        help="static checks of an adapter directory or Hub repository",
        description="Check adapter_config.json and the weights header (names and shapes only) against "
        "PEFT's rules, a base model and engine-specific behaviour. Exit code 1 if there are errors.",
    )
    li.add_argument("adapter", nargs="?", help="adapter directory or Hugging Face Hub id")
    li.add_argument(
        "--base",
        metavar="MODEL",
        help="base model directory or Hub id to compare module paths and shapes with",
    )
    li.add_argument("--revision", default="main", help="adapter revision on the Hub")
    li.add_argument("--base-revision", default="main", help="base model revision on the Hub")
    li.add_argument(
        "--engine",
        choices=("vllm", "sglang"),
        help="apply only this engine's rules (default: all engines)",
    )
    li.add_argument(
        "--engine-version",
        metavar="X.Y.Z",
        help="engine release, to settle version-dependent rules",
    )
    li.add_argument("--format", choices=("text", "json", "markdown"), default="text")
    li.add_argument("--output", "-o", metavar="FILE")
    li.add_argument("--strict", action="store_true", help="exit 1 on warnings too")
    li.add_argument(
        "--list-rules", action="store_true", help="print every rule with its source and exit"
    )
    li.add_argument("--demo", action="store_true", help="lint a bundled example adapter")
    li.set_defaults(func=_cmd_lint)

    m = sub.add_parser(
        "mine",
        help="select the most sensitive probes from a fingerprint",
        description="Given a fingerprint computed on a large candidate probe file, pick the probes on "
        "which each adapter changes the distribution most.",
    )
    m.add_argument("--fingerprint", required=True, metavar="FILE")
    m.add_argument(
        "--adapter", action="append", metavar="NAME", help="adapters to mine for (default: all)"
    )
    m.add_argument("--top", type=int, default=16, help="number of probes to keep (default: 16)")
    m.add_argument("--output", "-o", metavar="FILE", help="write the selected probes as JSON Lines")
    m.add_argument(
        "--fingerprint-out",
        metavar="FILE",
        help="write a fingerprint restricted to the selected probes",
    )
    m.add_argument("--format", choices=("text", "json"), default="text")
    m.set_defaults(func=_cmd_mine)

    s = sub.add_parser("sim", help="run the bundled simulated multi-LoRA server")
    s.add_argument(
        "--scenario",
        default="mixed",
        help="mixed, healthy, leak, duplicate or cache (default: mixed)",
    )
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.add_argument(
        "--noise", type=float, default=0.002, help="run-to-run logit noise (default: 0.002)"
    )
    s.set_defaults(func=_cmd_sim)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_USAGE
    try:
        return int(args.func(args))
    except (EndpointError, FingerprintError, ProbeError) as exc:
        _err(str(exc))
        return EXIT_USAGE
    except BrokenPipeError:
        return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
