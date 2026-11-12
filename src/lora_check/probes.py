"""Probe sets: the prompts whose next-token distributions are compared."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Probe:
    """One prompt.  ``id`` is stable so fingerprints can be matched to it."""

    id: str
    prompt: str


class ProbeError(ValueError):
    """Raised for unreadable or inconsistent probe files."""


# Short, deliberately varied prompts.  They are generic on purpose: the goal
# is to exercise many directions of the residual stream, not to be good
# benchmark questions.  Use `lora-check mine` to build a set that is tuned to
# a particular adapter.
_BUILTIN: tuple[str, ...] = (
    "The capital of France is",
    "Once upon a time, in a small village by the sea, there lived a",
    "SELECT name, email FROM users WHERE",
    "def fibonacci(n):\n    if n < 2:\n        return",
    "Q: What is 17 plus 25?\nA:",
    "Dear customer, thank you for contacting support. Unfortunately, your order",
    'Translate to German: "Good morning, how are you?"\nGerman:',
    "The patient presented with fever, cough and shortness of breath. The most likely diagnosis is",
    '{"name": "Alice", "age": 30, "city":',
    "In conclusion, the results of the experiment show that",
    "Step 1: Preheat the oven to 180 degrees. Step 2:",
    "The quick brown fox jumps over the lazy",
    "import numpy as np\n\narr = np.",
    "Roses are red, violets are",
    "Breaking news: scientists have discovered a new species of",
    "User: Can you summarise this article for me?\nAssistant:",
    "The three primary colours are red, blue and",
    "WHEREAS the parties wish to enter into this agreement, NOW, THEREFORE, the parties",
    "git commit -m",
    "Yes or no: is the Earth flat? Answer:",
    "La capitale de l'Espagne est",
    "My favourite thing about weekends is",
    "Error: connection refused. To fix this problem you should first",
    "1, 1, 2, 3, 5, 8,",
    "The stock market closed lower today as investors",
    "<html>\n<head>\n<title>",
    "Write a haiku about autumn:\n",
    "Review: This restaurant was absolutely",
    "To be, or not to be, that is the",
    "The function returns None when the input list is",
    "Subject: Meeting tomorrow\n\nHi team,\n\nJust a reminder that",
    "Water boils at 100 degrees Celsius, which is equal to",
)


def builtin_probes(limit: int | None = None) -> list[Probe]:
    """The bundled generic probe set (optionally only the first ``limit``)."""
    probes = [Probe(id=f"g{i + 1:02d}", prompt=text) for i, text in enumerate(_BUILTIN)]
    return probes if limit is None else probes[: max(1, limit)]


def _from_obj(obj: object, index: int) -> Probe:
    if isinstance(obj, str):
        return Probe(id=f"p{index + 1:03d}", prompt=obj)
    if isinstance(obj, dict) and isinstance(obj.get("prompt"), str):
        ident = obj.get("id")
        return Probe(
            id=str(ident) if ident is not None else f"p{index + 1:03d}", prompt=obj["prompt"]
        )
    raise ProbeError(f"probe #{index + 1} must be a string or an object with a 'prompt' field")


def load_probes(path: str | Path) -> list[Probe]:
    """Load probes from ``.txt`` (one per line), ``.jsonl`` or ``.json``.

    In text files ``\\n`` escapes are expanded and lines starting with ``#``
    are ignored.  JSON entries are either strings or ``{"id", "prompt"}``.
    """
    file = Path(path)
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProbeError(f"cannot read probe file {file}: {exc}") from exc
    objs: list[object] = []
    try:
        if file.suffix == ".json":
            data = json.loads(text)
            if isinstance(data, dict) and isinstance(data.get("probes"), list):
                data = data["probes"]
            if not isinstance(data, list):
                raise ProbeError(f"{file}: expected a JSON list of probes")
            objs = list(data)
        elif file.suffix == ".jsonl":
            objs = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            objs = [
                line.replace("\\n", "\n")
                for line in text.splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]
    except json.JSONDecodeError as exc:
        raise ProbeError(f"{file}: invalid JSON: {exc}") from exc
    probes = [_from_obj(obj, i) for i, obj in enumerate(objs)]
    if not probes:
        raise ProbeError(f"{file}: no probes found")
    ids = [p.id for p in probes]
    if len(set(ids)) != len(ids):
        raise ProbeError(f"{file}: probe ids are not unique")
    return probes


def dump_probes(probes: list[Probe], path: str | Path) -> None:
    """Write probes as JSON Lines."""
    lines = [json.dumps({"id": p.id, "prompt": p.prompt}, ensure_ascii=False) for p in probes]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
