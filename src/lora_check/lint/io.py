"""Reading adapter and model repositories without heavy dependencies.

Only JSON files and *safetensors headers* are read: tensor names, dtypes and
shapes.  No weights are downloaded or deserialised, and pickle files are
never opened.
"""

from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

MAX_HEADER_BYTES = 100 * 1024 * 1024
HUB = "https://huggingface.co"
_REPO_RE = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")


class SourceError(RuntimeError):
    """A repository or file could not be read."""


@dataclass(frozen=True)
class TensorInfo:
    """Name-level facts about one stored tensor."""

    dtype: str
    shape: tuple[int, ...]


def parse_safetensors_header(prefix: bytes, rest: bytes) -> dict[str, TensorInfo]:
    """Decode a safetensors header given its 8-byte length prefix and JSON body."""
    (length,) = struct.unpack("<Q", prefix)
    if length != len(rest):
        raise SourceError("truncated safetensors header")
    try:
        raw = json.loads(rest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceError(f"corrupt safetensors header: {exc}") from exc
    out: dict[str, TensorInfo] = {}
    for name, info in raw.items():
        if name == "__metadata__" or not isinstance(info, dict):
            continue
        out[name] = TensorInfo(
            str(info.get("dtype", "")), tuple(int(x) for x in info.get("shape", []))
        )
    return out


def _header_length(prefix: bytes, name: str) -> int:
    if len(prefix) != 8:
        raise SourceError(f"{name}: too short to be a safetensors file")
    (length,) = struct.unpack("<Q", prefix)
    if length <= 0 or length > MAX_HEADER_BYTES:
        raise SourceError(f"{name}: implausible safetensors header length {length}")
    return int(length)


def write_safetensors(path: str | Path, tensors: dict[str, tuple[str, tuple[int, ...]]]) -> None:
    """Write a valid safetensors file whose tensors are all zeros.

    Used to build fixtures and the lint demo; ``tensors`` maps a name to
    ``(dtype, shape)`` with dtype ``F32`` or ``F16``.
    """
    sizes = {"F32": 4, "F16": 2, "BF16": 2, "F64": 8}
    header: dict[str, Any] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        count = 1
        for dim in shape:
            count *= dim
        nbytes = count * sizes[dtype]
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<Q", len(blob)))
        handle.write(blob)
        handle.write(b"\x00" * offset)


class Source:
    """A directory-like place holding an adapter or a model."""

    label: str = ""

    def files(self) -> list[str]:
        raise NotImplementedError

    def read_bytes(self, name: str, start: int = 0, length: int | None = None) -> bytes:
        raise NotImplementedError

    def read_json(self, name: str) -> Any:
        """Parsed JSON of ``name``, or ``None`` if the file is absent."""
        if name not in self.files():
            return None
        try:
            return json.loads(self.read_bytes(name).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SourceError(f"{self.label}/{name}: invalid JSON: {exc}") from exc

    def tensors(self, name: str) -> dict[str, TensorInfo]:
        """Tensor names and shapes from the header of a safetensors file."""
        prefix = self.read_bytes(name, 0, 8)
        length = _header_length(prefix, name)
        return parse_safetensors_header(prefix, self.read_bytes(name, 8, length))


class LocalSource(Source):
    """A directory on disk."""

    def __init__(self, path: str | Path, label: str | None = None) -> None:
        self.path = Path(path)
        self.label = label or str(path)
        if not self.path.is_dir():
            raise SourceError(f"{path} is not a directory")

    def files(self) -> list[str]:
        return sorted(p.name for p in self.path.iterdir() if p.is_file())

    def read_bytes(self, name: str, start: int = 0, length: int | None = None) -> bytes:
        try:
            with open(self.path / name, "rb") as handle:
                handle.seek(start)
                return handle.read() if length is None else handle.read(length)
        except OSError as exc:
            raise SourceError(f"cannot read {self.path / name}: {exc}") from exc


class HubSource(Source):
    """A Hugging Face Hub model repository, read over HTTPS.

    Only the file listing, small JSON files and safetensors headers (via HTTP
    range requests) are fetched.  ``HF_TOKEN`` is sent if set.
    """

    def __init__(
        self, repo_id: str, revision: str = "main", transport: httpx.BaseTransport | None = None
    ) -> None:
        self.repo_id = repo_id
        self.revision = revision
        self.label = repo_id if revision == "main" else f"{repo_id}@{revision}"
        headers = {"User-Agent": "lora-check"}
        token = os.environ.get("HF_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._http = httpx.Client(
            headers=headers, timeout=60.0, follow_redirects=True, transport=transport
        )
        self._files: list[str] | None = None

    def files(self) -> list[str]:
        if self._files is None:
            url = f"{HUB}/api/models/{self.repo_id}/revision/{self.revision}"
            try:
                response = self._http.get(url)
            except httpx.HTTPError as exc:
                raise SourceError(
                    f"cannot reach the Hugging Face Hub for {self.repo_id}: {exc}"
                ) from exc
            if response.status_code != 200:
                raise SourceError(
                    f"Hugging Face Hub returned HTTP {response.status_code} for {self.repo_id} "
                    "(does the repository exist, and is HF_TOKEN set if it is private or gated?)"
                )
            siblings = response.json().get("siblings", [])
            self._files = sorted(
                s["rfilename"]
                for s in siblings
                if isinstance(s, dict) and "/" not in s.get("rfilename", "/")
            )
        return self._files

    def read_bytes(self, name: str, start: int = 0, length: int | None = None) -> bytes:
        url = f"{HUB}/{self.repo_id}/resolve/{self.revision}/{name}"
        headers = {}
        if start or length is not None:
            end = "" if length is None else str(start + length - 1)
            headers["Range"] = f"bytes={start}-{end}"
        try:
            response = self._http.get(url, headers=headers)
        except httpx.HTTPError as exc:
            raise SourceError(f"cannot download {name} from {self.repo_id}: {exc}") from exc
        if response.status_code not in (200, 206):
            raise SourceError(f"{self.repo_id}/{name}: HTTP {response.status_code}")
        data = response.content
        if response.status_code == 200 and (start or length is not None):
            data = data[start : None if length is None else start + length]
        return data


def open_source(
    spec: str, revision: str = "main", transport: httpx.BaseTransport | None = None
) -> Source:
    """Open ``spec`` as a local directory if it exists, else as a Hub repo id."""
    if Path(spec).is_dir():
        return LocalSource(spec)
    if _REPO_RE.match(spec):
        return HubSource(spec, revision=revision, transport=transport)
    raise SourceError(
        f"{spec!r} is neither an existing directory nor a Hub repository id (org/name)"
    )
