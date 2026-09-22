"""Word / token embeddings from scratch (skip-gram + negative sampling).

Educational NumPy implementation:
  - Embedding table E in R^{V x D} (input / center vectors)
  - Context table C in R^{V x D} (output / context vectors)
  - Skip-gram objective with negative sampling (Mikolov et al. style)
  - Cosine similarity nearest-neighbor retrieval
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Toy corpus: short themed sentences so related words co-occur
# ---------------------------------------------------------------------------

TOY_CORPUS: List[str] = [
    "the cat sat on the mat",
    "the dog sat on the rug",
    "the cat chased the mouse",
    "the dog chased the cat",
    "birds fly in the sky",
    "planes fly in the sky",
    "fish swim in the water",
    "ducks swim in the pond",
    "the king and queen rule the kingdom",
    "the prince and princess visit the castle",
    "coffee and tea are hot drinks",
    "milk and juice are cold drinks",
    "apples and oranges are sweet fruit",
    "carrots and potatoes are root vegetables",
    "python and java are programming languages",
    "html and css are web languages",
    "neural networks learn from data",
    "deep learning uses neural networks",
    "embeddings map words to vectors",
    "vectors capture semantic similarity",
    "the sun rises in the east",
    "the moon shines at night",
    "students study math and science",
    "teachers teach math and science",
    "cars drive on the road",
    "trains travel on the rails",
]


def tokenize(text: str) -> List[str]:
    return text.lower().strip().split()


def build_vocab(sentences: Sequence[str], min_count: int = 1) -> Tuple[Dict[str, int], List[str]]:
    """Return word->id and id->word from tokenized sentences."""
    counts: Dict[str, int] = {}
    for s in sentences:
        for tok in tokenize(s):
            counts[tok] = counts.get(tok, 0) + 1
    words = sorted(w for w, c in counts.items() if c >= min_count)
    word2id = {w: i for i, w in enumerate(words)}
    return word2id, words


def corpus_token_ids(sentences: Sequence[str], word2id: Dict[str, int]) -> List[List[int]]:
    out: List[List[int]] = []
    for s in sentences:
        ids = [word2id[t] for t in tokenize(s) if t in word2id]
        if ids:
            out.append(ids)
    return out


def skipgram_pairs(
    docs: Sequence[Sequence[int]],
    window: int = 2,
) -> List[Tuple[int, int]]:
    """(center, context) pairs within +/-window (excluding self)."""
    pairs: List[Tuple[int, int]] = []
    for doc in docs:
        n = len(doc)
        for i, center in enumerate(doc):
            lo = max(0, i - window)
            hi = min(n, i + window + 1)
            for j in range(lo, hi):
                if j == i:
                    continue
                pairs.append((center, doc[j]))
    return pairs


def unigram_noise_probs(docs: Sequence[Sequence[int]], vocab_size: int, power: float = 0.75) -> np.ndarray:
    """P_n(w) proportional to freq(w)^power -- standard negative-sampling distribution."""
    freq = np.zeros(vocab_size, dtype=np.float64)
    for doc in docs:
        for t in doc:
            freq[t] += 1.0
    freq = np.maximum(freq, 1.0)
    powered = freq**power
    return powered / powered.sum()


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    # stable sigmoid
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out if out.shape else float(out)


class SkipGramNS:
    """Skip-gram with negative sampling (pure NumPy)."""

    def __init__(
        self,
        vocab_size: int,
        dim: int = 32,
        n_neg: int = 5,
        lr: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.V = vocab_size
        self.D = dim
        self.n_neg = n_neg
        self.lr = lr
        self.rng = np.random.default_rng(seed)
        # Xavier-ish init
        scale = 1.0 / np.sqrt(dim)
        self.E = self.rng.normal(0, scale, size=(vocab_size, dim)).astype(np.float64)
        self.C = self.rng.normal(0, scale, size=(vocab_size, dim)).astype(np.float64)

    def _neg_sample(self, noise_probs: np.ndarray, exclude: int) -> np.ndarray:
        negs = []
        while len(negs) < self.n_neg:
            cand = int(self.rng.choice(self.V, p=noise_probs))
            if cand != exclude:
                negs.append(cand)
        return np.array(negs, dtype=np.int64)

    def train_epoch(
        self,
        pairs: Sequence[Tuple[int, int]],
        noise_probs: np.ndarray,
        shuffle: bool = True,
    ) -> float:
        """One pass over (center, context) pairs. Returns mean NCE-style loss."""
        order = np.arange(len(pairs))
        if shuffle:
            self.rng.shuffle(order)
        total_loss = 0.0
        for idx in order:
            center, context = pairs[int(idx)]
            v = self.E[center]  # (D,)
            # positive
            u_pos = self.C[context]
            score_pos = float(np.dot(v, u_pos))
            sig_pos = float(sigmoid(score_pos))
            loss = -np.log(sig_pos + 1e-12)
            # grad for positive: (sig - 1) * other
            g_pos = sig_pos - 1.0
            # negatives
            negs = self._neg_sample(noise_probs, exclude=center)
            g_v = g_pos * u_pos
            g_u_pos = g_pos * v
            for neg in negs:
                u_neg = self.C[neg]
                score_neg = float(np.dot(v, u_neg))
                sig_neg = float(sigmoid(score_neg))
                loss += -np.log(1.0 - sig_neg + 1e-12)
                g_neg = sig_neg  # sig(score) - 0 for label 0
                g_v = g_v + g_neg * u_neg
                self.C[neg] = u_neg - self.lr * (g_neg * v)
            self.E[center] = v - self.lr * g_v
            self.C[context] = u_pos - self.lr * g_u_pos
            total_loss += loss
        return total_loss / max(len(pairs), 1)

    def fit(
        self,
        pairs: Sequence[Tuple[int, int]],
        noise_probs: np.ndarray,
        epochs: int = 40,
    ) -> List[float]:
        history: List[float] = []
        for _ in range(epochs):
            loss = self.train_epoch(pairs, noise_probs)
            history.append(loss)
        return history

    def embeddings(self) -> np.ndarray:
        """Return input embeddings (center vectors)."""
        return self.E.copy()


def cosine_similarity_matrix(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    Xn = X / norms
    return Xn @ Xn.T


def nearest_neighbors(
    embeddings: np.ndarray,
    word2id: Dict[str, int],
    id2word: Sequence[str],
    query: str,
    k: int = 5,
) -> List[Tuple[str, float]]:
    """Top-k nearest words by cosine similarity (excluding self)."""
    if query not in word2id:
        raise KeyError(f"Unknown word: {query}")
    qid = word2id[query]
    sims = cosine_similarity_matrix(embeddings)[qid]
    sims[qid] = -np.inf
    top = np.argsort(-sims)[:k]
    return [(id2word[int(i)], float(sims[int(i)])) for i in top]


def pca_2d(X: np.ndarray) -> np.ndarray:
    """Project rows of X to 2D via PCA (NumPy SVD)."""
    Xc = X - X.mean(axis=0, keepdims=True)
    # economy SVD
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T
