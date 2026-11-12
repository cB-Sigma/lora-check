from __future__ import annotations

import json
from pathlib import Path

import pytest

from lora_check.probes import ProbeError, builtin_probes, dump_probes, load_probes


def test_builtin_probes_are_unique_and_stable() -> None:
    probes = builtin_probes()
    assert len(probes) >= 24
    assert len({p.id for p in probes}) == len(probes)
    assert len({p.prompt for p in probes}) == len(probes)
    assert builtin_probes(5) == probes[:5]


def test_load_txt_expands_newlines_and_skips_comments(tmp_path: Path) -> None:
    file = tmp_path / "p.txt"
    file.write_text("# comment\nfirst line\\nsecond\n\nother\n")
    probes = load_probes(file)
    assert [p.prompt for p in probes] == ["first line\nsecond", "other"]
    assert [p.id for p in probes] == ["p001", "p002"]


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    file = tmp_path / "p.jsonl"
    dump_probes(builtin_probes(4), file)
    assert load_probes(file) == builtin_probes(4)


def test_load_json_variants(tmp_path: Path) -> None:
    file = tmp_path / "p.json"
    file.write_text(json.dumps({"probes": ["a", {"id": "x", "prompt": "b"}]}))
    assert [(p.id, p.prompt) for p in load_probes(file)] == [("p001", "a"), ("x", "b")]


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("p.json", "{broken"),
        ("p.json", '{"probes": 3}'),
        ("p.jsonl", '{"id": "a"}\n'),
        ("p.jsonl", '{"id": "a", "prompt": "x"}\n{"id": "a", "prompt": "y"}\n'),
        ("p.txt", "# only a comment\n"),
    ],
)
def test_bad_probe_files(tmp_path: Path, name: str, content: str) -> None:
    file = tmp_path / name
    file.write_text(content)
    with pytest.raises(ProbeError):
        load_probes(file)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ProbeError):
        load_probes(tmp_path / "nope.txt")
