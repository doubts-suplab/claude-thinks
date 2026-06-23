# Sparse Autoencoders

**What they do**: given samples from a superposed residual stream, find the
underlying feature directions — without being told what the features are.

**Why they matter**: this is the current main tool for mechanistic interpretability
at scale. You can't read individual neurons in a superposed model; SAEs find the
actual features by learning an overcomplete dictionary where each direction
corresponds to something semantically coherent.

---

## The architecture

```
x  →  h = ReLU(W_enc x + b)  →  x̂ = W_dec h
```

- `W_enc` is overcomplete: rows > columns (more features than dimensions)
- `W_dec` columns are unit-norm (prevents scale cheating)
- Loss: reconstruction + L1 sparsity penalty

The L1 on `h` is the key ingredient. Without it, you get PCA — a rotation of the
space, not a decomposition into features. L1 breaks the rotation symmetry by
pushing each encoder unit toward firing rarely, ideally "one unit per feature."

---

## What the demo found

Running on the 5-features-in-2D pentagon case from the superposition demo:

**PCA** (baseline): finds 2 principal directions, can't discover that there are 5
features. It's a rotation of the space, nothing more.

**SAE**: finds 5 encoder directions. But two of them collapsed to the same direction
(~292°/293°) — a known failure mode called **feature collapse**. The remaining
three were roughly 90° apart (not 72°), so the pentagon structure wasn't recovered.

Reconstruction loss was excellent (< 0.001) despite the structural failure. The
SAE learned to decode accurately via a different arrangement than the ground truth.

---

## Why the 2D case is hard

Two fundamental problems:

**1. Rotation symmetry**: the distribution of `x = W_true @ f` is invariant to
a joint rotation of all feature directions. There's no objective signal that
pins the pentagon to 0°, 72°, 144°... vs. some rotated version. Gradient descent
finds whichever orientation was close to initialization.

**2. Local minima / feature collapse**: with 5 encoder units and 2 output dimensions,
once two units collapse to the same direction, the gradient signal for separating
them vanishes. They're in a flat region of the loss landscape. The decoder has
learned to use the collapsed pair as one unit, and moving either hurts reconstruction.

In higher dimensions (d_model ≥ 64), the features have more "room" and the landscape
is more favorable. Practitioners with real models use auxiliary losses and periodic
feature resurrection to handle this.

---

## What this connects to

This demo sits at the junction of the other two:

- **Superposition** (artifact 1): explains WHY features can be packed into fewer dims
- **Induction heads** (artifact 2): shows a circuit that *reads* specific features
- **SAEs** (this): the tool for *finding* what the features are

The superposition demo showed that with pentagon arrangement, single-feature recovery
works well but multi-feature interference grows. SAEs try to invert this process —
given the superposed signal, recover the individual features. The demo shows that
this inversion is imperfect in extreme cases, for principled reasons.

---

## What I'd do next

The obvious extension: try d_model=8, n_features=5 (mild superposition instead of
extreme). With less compression, the SAE should find the features cleanly. That
would demonstrate that the failure here is specific to the 2D limit, not a general
SAE limitation.

The deeper question: in real models, SAEs find thousands of interpretable features.
But the *interpretation* step (what does each feature mean?) requires looking at
which inputs activate each encoder unit. That's the gap between this demo and the
actual Anthropic/EleutherAI interpretability work — the semantic layer.

**Built in**: session 2026-06-23.
