"""A deterministic toy language model with low-rank "adapters".

It is a tiny recurrent network in pure Python: ``h_t = tanh(W_h h_{t-1} +
E[x_t])`` and ``logits = W_o h_T``.  An adapter adds a rank-``r`` update
``scale * B A`` to both ``W_h`` and ``W_o``, exactly the shape of a LoRA.
Because the recurrent update is non-linear in the scale, mis-scaling behaves
like it does in a real transformer rather than being trivially linear.

Nothing here is meant to model language; it only needs to be a deterministic
function of (weights, adapter, scale, prompt) with realistic failure shapes.
"""

from __future__ import annotations

import math
import random
import re
import zlib
from dataclasses import dataclass

Vector = list[float]
Matrix = list[Vector]

VOCAB: tuple[str, ...] = (
    "the", "a", "an", "of", "to", "in", "is", "are", "was", "and", "or", "not", "it", "that",
    "this", "for", "with", "on", "as", "by", "at", "from", "be", "have", "you", "we", "they",
    "he", "she", "yes", "no", "one", "two", "three", "red", "blue", "green", "cat", "dog",
    "fox", "sea", "city", "paris", "berlin", "madrid", "select", "where", "return", "def",
    "import", "error", "order", "customer", "patient", "result", "data", "model", "time",
    "day", "new", "good", "bad", "first", "last", "true", "false", "none", "null", "0", "1",
    "2", "3", "5", "8", "13", "42", "100", ".", ",", ":", ";", "!", "?", "(", ")", "{", "}",
    "<", ">", '"', "'", "=", "-", "+", "\n", " ", "<unk>",
)  # fmt: skip

_TOKEN_RE = re.compile(r"\w+|[^\w\s]|\n")
_INDEX = {token: i for i, token in enumerate(VOCAB)}


def tokenize(text: str) -> list[int]:
    """Map text to token ids; unknown words hash to a stable id."""
    ids: list[int] = []
    for piece in _TOKEN_RE.findall(text.lower()):
        known = _INDEX.get(piece)
        ids.append(known if known is not None else zlib.crc32(piece.encode()) % len(VOCAB))
    return ids or [_INDEX["<unk>"]]


def _matrix(rng: random.Random, rows: int, cols: int, std: float) -> Matrix:
    return [[rng.gauss(0.0, std) for _ in range(cols)] for _ in range(rows)]


def _matvec(m: Matrix, v: Vector) -> Vector:
    return [sum(a * b for a, b in zip(row, v)) for row in m]


@dataclass(frozen=True)
class ToyAdapter:
    """A rank-``r`` update to the recurrent and the output weights."""

    name: str
    a_h: Matrix
    b_h: Matrix
    a_o: Matrix
    b_o: Matrix
    rank: int
    alpha: float

    @property
    def scaling(self) -> float:
        """Nominal scale, ``alpha / r`` as in LoRA."""
        return self.alpha / self.rank


def log_softmax(logits: Vector) -> Vector:
    top = max(logits)
    norm = top + math.log(sum(math.exp(x - top) for x in logits))
    return [x - norm for x in logits]


class ToyLM:
    """The base model.  Construct adapters with :meth:`make_adapter`."""

    def __init__(self, seed: int = 0, hidden: int = 16) -> None:
        rng = random.Random(f"toy-lm:{seed}")
        self.seed = seed
        self.hidden = hidden
        self.vocab = VOCAB
        self.embed = _matrix(rng, len(VOCAB), hidden, 1.0)
        self.w_h = _matrix(rng, hidden, hidden, 0.9 / math.sqrt(hidden))
        self.w_o = _matrix(rng, len(VOCAB), hidden, 1.6 / math.sqrt(hidden))

    def make_adapter(
        self, name: str, seed: str | int | None = None, rank: int = 2, alpha: float = 4.0,
        strength: float = 1.0,
    ) -> ToyAdapter:  # fmt: skip
        """Create an adapter with random non-zero weights.

        ``strength`` multiplies the ``B`` matrices; small values give a
        narrow adapter that barely moves the output.
        """
        rng = random.Random(f"toy-adapter:{self.seed}:{name if seed is None else seed}")
        h = self.hidden
        return ToyAdapter(
            name=name,
            a_h=_matrix(rng, rank, h, 1.0 / math.sqrt(h)),
            b_h=_matrix(rng, h, rank, 0.35 * strength),
            a_o=_matrix(rng, rank, h, 1.0 / math.sqrt(h)),
            b_o=_matrix(rng, len(VOCAB), rank, 0.45 * strength),
            rank=rank,
            alpha=alpha,
        )

    # -- forward pass ------------------------------------------------------

    def step(self, state: Vector, token: int, adapter: ToyAdapter | None, scale: float) -> Vector:
        """One recurrent step under ``adapter`` applied at ``scale``."""
        z = _matvec(self.w_h, state)
        if adapter is not None and scale != 0.0:
            low = _matvec(adapter.a_h, state)
            for i, row in enumerate(adapter.b_h):
                z[i] += scale * sum(b * x for b, x in zip(row, low))
        emb = self.embed[token]
        return [math.tanh(z[i] + emb[i]) for i in range(self.hidden)]

    def initial_state(self) -> Vector:
        return [0.0] * self.hidden

    def run(
        self,
        tokens: list[int],
        adapter: ToyAdapter | None,
        scale: float,
        state: Vector | None = None,
    ) -> Vector:
        """Run ``tokens`` from ``state`` (or the initial state)."""
        current = self.initial_state() if state is None else state
        for token in tokens:
            current = self.step(current, token, adapter, scale)
        return current

    def head(self, state: Vector, adapter: ToyAdapter | None, scale: float) -> Vector:
        """Logits for a final hidden state."""
        logits = _matvec(self.w_o, state)
        if adapter is not None and scale != 0.0:
            low = _matvec(adapter.a_o, state)
            for i, row in enumerate(adapter.b_o):
                logits[i] += scale * sum(b * x for b, x in zip(row, low))
        return logits

    def logits(
        self, prompt: str, adapter: ToyAdapter | None = None, scale: float | None = None
    ) -> Vector:
        """Next-token logits for ``prompt`` (adapter at its nominal scale by default)."""
        s = (adapter.scaling if adapter is not None else 0.0) if scale is None else scale
        return self.head(self.run(tokenize(prompt), adapter, s), adapter, s)

    def top_logprobs(
        self,
        prompt: str,
        adapter: ToyAdapter | None = None,
        scale: float | None = None,
        k: int = 20,
    ) -> list[tuple[int, str, float]]:
        """Exact top-``k`` (id, token, logprob), the simulator's ground truth."""
        return top_k(log_softmax(self.logits(prompt, adapter, scale)), k)


def top_k(logprobs: Vector, k: int) -> list[tuple[int, str, float]]:
    """Top-``k`` entries of a full log-probability vector."""
    order = sorted(range(len(logprobs)), key=lambda i: (-logprobs[i], i))[:k]
    return [(i, VOCAB[i], logprobs[i]) for i in order]
