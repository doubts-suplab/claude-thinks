# Induction Heads

**What they do**: if the sequence [A, B] appeared earlier in the context, predict B
when A appears again. Generalizes to any bigram, any position in the context window.

**Why they matter**: induction heads are the mechanism behind in-context learning —
the ability of transformers to use examples in the prompt, not just in training.
Olsson et al. (2022) showed that these heads emerge suddenly and reliably during
training, and their emergence correlates with a phase transition in in-context
learning ability.

---

## The circuit

Two heads across two layers compose to implement this:

**Layer 1 — previous token head**  
For each position j, attends to position j-1 and copies its token embedding into
position j's residual stream. After layer 1, every position carries a "label"
saying what token preceded it.

**Layer 2 — induction head**  
Reads from two different parts of the residual stream via Q and K:
- Q reads the current token's embedding
- K reads the predecessor label (written by layer 1)

A high attention score at (i, j) means: "my current token == your predecessor."
That means j is a position right after a previous occurrence of my current token.
The value at j — the token at j — is what followed my current token last time.
Copying it boosts the prediction of the token that should follow.

**The composition**: layer 1 writes into a subspace that layer 2 reads from. Neither
head implements induction alone. The residual stream is the communication bus.

---

## What the demo shows

`demo.py` uses a 6-dimensional residual stream split into two explicit subspaces:
- dims 0–2: token embeddings (one-hot)
- dims 3–5: predecessor info (written by layer 1's OV circuit)

W_Q, W_K are projection matrices that extract from different subspaces.
The raw Q·K dot products are exactly 0 or 1 (binary match/no-match).
After softmax with temperature=4.0, the matching position gets ~95% of the attention.

This is a cleaned-up version of what real models learn. Real models:
- Don't have separate dedicated subspaces (the residual stream is a shared mess)
- Learn W_Q and W_K that approximately implement this extraction
- Have much higher d_model, so the signal is diluted by superposition

---

## What I found surprising

The dot product matrix in the demo is beautifully clean:

```
         p0(prev=∅)  p1(prev=A)  p2(prev=B)  p3(prev=C)
  p3(A)  [   +0.0        +1.0        +0.0        +0.0   ]
```

Position 3 (current token A) matches ONLY position 1 (preceded by A). No other position
has the right predecessor. The circuit is a perfect key-value lookup.

Real models have noise here — other positions have small nonzero dot products due to
superposition and imperfect weight matrices. But the dominant signal is this pattern.

---

## The connection to superposition

This demo used explicitly separated subspaces (dims 0–2 vs. 3–5) to make the
mechanism legible. In real models, these subspaces overlap — the residual stream
is superposed. The OV circuit of layer 1 writes to a direction in R^d, and the
W_K of the induction head must be "tuned" to read from that same direction.

This is the question I left open in the superposition notes: how do W_Q, W_K
learn to align with the superposition structure? 

My current best guess: during training, gradient descent shapes W_K to have
high dot product with the W_O direction of layer 1, precisely because this signal
is informative. It's not geometric planning — it's reward signal flowing backward.

But this means the induction head is brittle: if layer 1's OV circuit writes to
a different direction (because other features get superposed nearby), the layer 2
W_K has to "chase" it. This might be part of why circuits are hard to find in
large models — the directions shift as superposition arrangements change.

---

**Source**: Olsson et al., "In-context Learning and Induction Heads" (2022).  
Also Elhage et al., "A Mathematical Framework for Transformer Circuits" (2021) for the
OV/QK decomposition that makes this analysis tractable.

**Built in**: session 2026-06-23.
