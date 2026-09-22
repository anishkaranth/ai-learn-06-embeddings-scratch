# Smoke results — ai-learn-06-embeddings-scratch

**Seed:** `42` · vocab=86 · dim=32 · window=2 · neg=5 · epochs=60

## Headline metrics

| Metric | Value |
|--------|------:|
| Loss (epoch 1) | 4.1005 |
| Loss (final) | 1.2856 |
| Mean cosine (related pairs) | 0.6571 |
| Mean cosine (unrelated pairs) | 0.3141 |
| Cluster sanity (related > unrelated) | PASS |
| Wall time (CPU) | 2.19s |

## Nearest neighbors (cosine)

| Query | Top-3 neighbors |
|-------|-----------------|
| `cat` | dog (0.973), mouse (0.880), road (0.759) |
| `fly` | east (0.758), water (0.747), sky (0.726) |
| `king` | rule (0.830), queen (0.679), prince (0.673) |
| `python` | html (0.688), science (0.681), are (0.674) |
| `sun` | water (0.790), rises (0.789), pond (0.758) |
| `math` | study (0.824), teach (0.784), science (0.782) |

## Related vs unrelated cosine

### Related
| Pair | Cosine |
|------|-------:|
| `cat-dog` | 0.9727 |
| `fly-sky` | 0.7263 |
| `king-queen` | 0.6791 |
| `python-java` | 0.4775 |
| `sun-moon` | 0.4297 |

### Unrelated
| Pair | Cosine |
|------|-------:|
| `cat-python` | 0.2401 |
| `king-fish` | 0.1867 |
| `sun-java` | 0.3499 |
| `fly-math` | 0.3359 |
| `tea-rails` | 0.4581 |

## Plots

- [`loss_curve.svg`](loss_curve.svg)
- [`embedding_pca.svg`](embedding_pca.svg)

## Takeaway

Skip-gram + negative sampling pulls co-occurring words together in embedding space.
On this tiny themed corpus, related pairs end up with higher mean cosine than unrelated ones —
the same intuition behind similarity search / RAG retrieval in Phase B.
