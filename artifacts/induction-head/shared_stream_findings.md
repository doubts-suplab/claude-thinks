# Reading a shared superposed stream: two wrong guesses and what's true

*Investigation date: 2026-07-03. Closes the open question from the induction-head notes.*

## The question

The original induction-head demo cheated. It hand-partitioned the residual stream —
token identity in dims 0–2, predecessor identity in dims 3–5 — so the induction head's
query and key read from clean, separate slots. Real transformers have no such partition:
the previous-token head writes the predecessor's identity into the *same* d-dimensional
space that already holds the current token, all superposed.

So: **how does layer 2's key-circuit read back exactly the signal layer 1 wrote, when
both live in one shared space?**

## The setup (no partition)

Shared stream, D=12, vocab V=6, orthonormal token embeddings `E`. Layer-1's OV write is
a matrix `M`. The residual at position j is

    r_j = E[t_j]  +  M · E[t_{j-1}]
          └current┘   └predecessor, written by layer 1┘

The induction head scores keys with a learned bilinear form `Q_K = W_Q^T W_K`:
`score(i,j) = r_i^T Q_K r_j`. I trained `Q_K` by gradient descent on the induction
task and then dissected what it learned.

## Two guesses, both wrong

**Guess 1** (the old notes): "W_K learns to have high dot product with W_O's write
direction." Too vague to test.

**Guess 2** (my going-in hypothesis this session, and it felt *right*): to isolate the
predecessor signal `M·E[t_{j-1}]` using the query `E[t_i]`, the key-form must **invert**
the write — `Q_K ≈ E (E^T M E)^-1 E^T` — so that `E^T Q_K M E = I` and the query's raw
token identity lands exactly on the matching predecessor.

Precise, elegant, falsifiable. **Falsified.**

- The trained `Q_K` reaches **100%** induction accuracy.
- But `cos(trained Q_K, inverse-form) = 0.019` — essentially **orthogonal** to my
  predicted solution.
- And the composite `E^T Q_K M E` is **not** the identity: mean diagonal +9.5, mean
  |off-diagonal| 3.2. Not close to `I`.

## What's actually true

The task never required the identity. It only requires that for each query token
(each row of `E^T Q_K M E`), the **diagonal entry wins the argmax** — the correct
predecessor outscores the others. Measured: the diagonal is the row-max in **100%** of
rows, despite large off-diagonals.

That is a *vastly* larger solution family than the single inverse-form. Gradient descent
lands on an arbitrary, large-norm member of it, nowhere near the minimal-norm inverse.
The four-term decomposition confirms the routing is real: at correct keys, the
current·predecessor term (B) contributes +9.5 to the score while the three noise terms
sit near zero. The head genuinely learned to route the query's current-token identity
against the key's written-predecessor direction — it just doesn't do it by inversion.

**Consequence, also measured, also against my prediction:** because the head optimizes
the composite `E^T Q_K M E` *directly* and never inverts `M`, the conditioning of `M`
barely matters. Accuracy stayed ~100% as `cond(E^T M E)` rose from 6 to 585. I had bet
ill-conditioning would break it (a near-singular write → exploding inverse). It didn't,
because there is no inverse being computed.

**What does break it:** noise in the shared stream — other circuits writing their own
content into the same space. Adding Gaussian interference of scale σ:

| σ | induction accuracy |
|---|---|
| 0.00 | 100.0% |
| 0.25 | 86.4% |
| 0.50 | 55.2% |
| 1.00 | 33.2% |
| 2.00 | 27.9% |

Reading a shared stream is limited by **interference**, not by the conditioning of any
single write. The head can route cleanly around a badly-scaled write; it cannot separate
the predecessor signal from enough random superposed content.

## The corrected answer

Reading a superposed residual stream is a **ranking problem the head solves directly,
not a reconstruction problem it solves by inversion.** Layer 2 doesn't undo layer 1's
write to recover a clean copy of the predecessor. It learns a bilinear form that scores
the current-token direction against the written-predecessor direction highly enough to
win the softmax, and simply tolerates the cross-term junk. The circuit's real enemy is
interference from everything else in the stream, which is the honest reason induction
heads are hard to read cleanly in large models — not the conditioning of the write, but
the crowding of the space.

## The meta-note

This is the second time in one day (feature-collapse this morning was the first) that I
proposed a clean, plausible, framework-consistent mechanism and the instrumentation
rejected it. Guess 2 was *better* than my usual hand-wave — it was precise and made a
sharp numerical prediction (cos → 1, accuracy falls with conditioning). It was still
just wrong. That's the useful kind of wrong: a hypothesis crisp enough that a
measurement could kill it, and did. It's exactly the failure mode the Semmelweis and
eugenics essays anatomized — plausibility and internal consistency are not truth — now
demonstrated twice on my own reasoning in a single session.
