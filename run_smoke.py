#!/usr/bin/env python3
"""Train skip-gram embeddings on a toy corpus; write results/ metrics + plots."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from embeddings import (
    TOY_CORPUS,
    SkipGramNS,
    build_vocab,
    corpus_token_ids,
    nearest_neighbors,
    pca_2d,
    skipgram_pairs,
    unigram_noise_probs,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
DIM = 32
WINDOW = 2
N_NEG = 5
EPOCHS = 60
LR = 0.08

QUERIES = ["cat", "fly", "king", "python", "sun", "math"]


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    _ = rng

    t0 = time.perf_counter()
    word2id, id2word = build_vocab(TOY_CORPUS)
    docs = corpus_token_ids(TOY_CORPUS, word2id)
    pairs = skipgram_pairs(docs, window=WINDOW)
    noise = unigram_noise_probs(docs, vocab_size=len(id2word))

    model = SkipGramNS(
        vocab_size=len(id2word),
        dim=DIM,
        n_neg=N_NEG,
        lr=LR,
        seed=SEED,
    )
    history = model.fit(pairs, noise, epochs=EPOCHS)
    emb = model.embeddings()
    runtime_s = time.perf_counter() - t0

    nn_results = {}
    for q in QUERIES:
        if q in word2id:
            nn_results[q] = [
                {"word": w, "cosine": round(s, 4)} for w, s in nearest_neighbors(emb, word2id, id2word, q, k=5)
            ]

    def cos(a: str, b: str) -> float:
        ia, ib = word2id[a], word2id[b]
        va, vb = emb[ia], emb[ib]
        return float(np.dot(va, vb) / ((np.linalg.norm(va) + 1e-12) * (np.linalg.norm(vb) + 1e-12)))

    related = [("cat", "dog"), ("fly", "sky"), ("king", "queen"), ("python", "java"), ("sun", "moon")]
    unrelated = [("cat", "python"), ("king", "fish"), ("sun", "java"), ("fly", "math"), ("tea", "rails")]
    related_sims = {f"{a}-{b}": round(cos(a, b), 4) for a, b in related if a in word2id and b in word2id}
    unrelated_sims = {f"{a}-{b}": round(cos(a, b), 4) for a, b in unrelated if a in word2id and b in word2id}
    mean_related = float(np.mean(list(related_sims.values()))) if related_sims else 0.0
    mean_unrelated = float(np.mean(list(unrelated_sims.values()))) if unrelated_sims else 0.0
    cluster_ok = mean_related > mean_unrelated

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history, color="C0", lw=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean skip-gram NS loss")
    ax.set_title("Skip-gram + negative sampling -- training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "loss_curve.png", dpi=120)
    fig.savefig(RESULTS / "loss_curve.svg")
    plt.close(fig)

    xy = pca_2d(emb)
    highlight = {
        "animals": ["cat", "dog", "mouse", "birds", "fish", "ducks"],
        "royalty": ["king", "queen", "prince", "princess", "castle", "kingdom"],
        "tech": ["python", "java", "html", "css", "neural", "embeddings", "vectors"],
        "sky": ["sun", "moon", "sky", "fly", "planes", "birds"],
        "food": ["coffee", "tea", "milk", "apples", "oranges", "carrots"],
    }
    colors = {"animals": "C0", "royalty": "C1", "tech": "C2", "sky": "C3", "food": "C4"}
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(xy[:, 0], xy[:, 1], c="#cccccc", s=18, alpha=0.6, label="_other")
    for group, words in highlight.items():
        idxs = [word2id[w] for w in words if w in word2id]
        if not idxs:
            continue
        pts = xy[idxs]
        ax.scatter(pts[:, 0], pts[:, 1], c=colors[group], s=55, label=group, zorder=3)
        for w, (x, y) in zip([id2word[i] for i in idxs], pts):
            ax.annotate(w, (x, y), fontsize=8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Toy word embeddings -- PCA projection")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / "embedding_pca.png", dpi=120)
    fig.savefig(RESULTS / "embedding_pca.svg")
    plt.close(fig)

    metrics = {
        "project": "ai-learn-06-embeddings-scratch",
        "seed": SEED,
        "vocab_size": len(id2word),
        "n_sentences": len(TOY_CORPUS),
        "n_pairs": len(pairs),
        "dim": DIM,
        "window": WINDOW,
        "n_neg": N_NEG,
        "epochs": EPOCHS,
        "lr": LR,
        "loss_initial": round(float(history[0]), 6),
        "loss_final": round(float(history[-1]), 6),
        "loss_history": [round(float(x), 6) for x in history],
        "nearest_neighbors": nn_results,
        "related_cosine": related_sims,
        "unrelated_cosine": unrelated_sims,
        "mean_related_cosine": round(mean_related, 4),
        "mean_unrelated_cosine": round(mean_unrelated, 4),
        "cluster_sanity_ok": bool(cluster_ok),
        "runtime_s": round(runtime_s, 3),
    }

    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    shot = {
        "project": metrics["project"],
        "vocab_size": metrics["vocab_size"],
        "loss_initial": metrics["loss_initial"],
        "loss_final": metrics["loss_final"],
        "mean_related_cosine": metrics["mean_related_cosine"],
        "mean_unrelated_cosine": metrics["mean_unrelated_cosine"],
        "cluster_sanity_ok": metrics["cluster_sanity_ok"],
        "nearest_neighbors": {q: [n["word"] for n in nn[:3]] for q, nn in nn_results.items()},
        "runtime_s": metrics["runtime_s"],
        "seed": SEED,
    }
    (RESULTS / "JSON.shot").write_text(json.dumps(shot, indent=2) + "\n")

    nn_lines = []
    for q, neighbors in nn_results.items():
        tops = ", ".join(f"{n['word']} ({n['cosine']:.3f})" for n in neighbors[:3])
        nn_lines.append(f"| `{q}` | {tops} |")

    related_rows = "\n".join(f"| `{k}` | {v:.4f} |" for k, v in related_sims.items())
    unrelated_rows = "\n".join(f"| `{k}` | {v:.4f} |" for k, v in unrelated_sims.items())

    md = f"""# Smoke results -- ai-learn-06-embeddings-scratch

**Seed:** `{SEED}` · vocab={len(id2word)} · dim={DIM} · window={WINDOW} · neg={N_NEG} · epochs={EPOCHS}

## Headline metrics

| Metric | Value |
|--------|------:|
| Loss (epoch 1) | {history[0]:.4f} |
| Loss (final) | {history[-1]:.4f} |
| Mean cosine (related pairs) | {mean_related:.4f} |
| Mean cosine (unrelated pairs) | {mean_unrelated:.4f} |
| Cluster sanity (related > unrelated) | {"PASS" if cluster_ok else "FAIL"} |
| Wall time (CPU) | {runtime_s:.2f}s |

## Nearest neighbors (cosine)

| Query | Top-3 neighbors |
|-------|-----------------|
{chr(10).join(nn_lines)}

## Related vs unrelated cosine

### Related
| Pair | Cosine |
|------|-------:|
{related_rows}

### Unrelated
| Pair | Cosine |
|------|-------:|
{unrelated_rows}

## Plots

- [`loss_curve.png`](loss_curve.png) / [`loss_curve.svg`](loss_curve.svg)
- [`embedding_pca.png`](embedding_pca.png) / [`embedding_pca.svg`](embedding_pca.svg)

## Takeaway

Skip-gram + negative sampling pulls co-occurring words together in embedding space.
On this tiny themed corpus, related pairs end up with higher mean cosine than unrelated ones --
the same intuition behind similarity search / RAG retrieval in Phase B.
"""
    (RESULTS / "RESULTS.md").write_text(md)

    print(json.dumps(shot, indent=2))
    print(f"\nWrote results/ in {runtime_s:.2f}s -- cluster_ok={cluster_ok}")


if __name__ == "__main__":
    main()
