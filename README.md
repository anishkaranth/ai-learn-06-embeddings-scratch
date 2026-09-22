# AI Learn 06 — Word Embeddings from Scratch

Train a tiny **skip-gram + negative sampling** word embedding model in pure NumPy on a hand-written toy corpus. Retrieve nearest neighbors by cosine similarity and visualize clusters with a 2D PCA projection.

Phase A wrap / bridge to Phase B (embeddings intuition → similarity search / RAG).

## Learning goals

- An **embedding table** maps each token id → a dense vector in $\mathbb{R}^D$
- **Skip-gram**: predict context words given a center word (within a window)
- **Negative sampling**: train with a few noise words instead of a full softmax over $V$
- **Cosine similarity** retrieves nearest neighbors; related words should cluster
- PCA projects high-D embeddings to 2D for inspection

## Brief math

For center $w$, positive context $c$, and negatives $n_1,\ldots,n_k$:

$$
\mathcal{L} = -\log \sigma(v_w^\top u_c) - \sum_{i=1}^{k} \log \sigma(-v_w^\top u_{n_i})
$$

where $v$ are input (center) vectors, $u$ are output (context) vectors, and $\sigma$ is the sigmoid.
After training, use $v_w$ (or $v_w + u_w$) as the embedding for retrieval.

## Layout

```
embeddings.py          # vocab, skip-gram NS trainer, cosine NN, PCA
run_smoke.py           # end-to-end smoke → results/
notebooks/embeddings_scratch.ipynb
results/               # committed metrics + plots
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_smoke.py
```

Runs on CPU in a few seconds. See `results/RESULTS.md` for the latest smoke metrics.

## What you'll learn next (Phase B)

Similarity search over embeddings, toy RAG retrieval, tools, evals, and light fine-tuning (e.g. LoRA).
