# Superposition

**What it is**: a mechanism that lets neural networks represent more features
than they have dimensions, by packing features as nearly-orthogonal directions
rather than as basis vectors.

**Why it matters**: GPT-2 Small has d_model=768. But it seems to represent
millions of features — named entities, grammatical patterns, factual associations,
positional information, syntax trees. Superposition is how this is possible.

---

## The geometry

A d-dimensional space has at most d orthogonal directions. But you can fit
many more *nearly-orthogonal* directions if you're willing to accept small
pairwise dot products (interference).

The tradeoff:
- More features packed in → higher interference between co-active features
- Sparser activations → less co-activation → less interference

So the question becomes: is the compression gain worth the interference cost?
If features are sparse, yes — overwhelmingly so.

Gradient descent finds this. Given a reconstruction objective with sparse
feature activations, networks spontaneously learn to use superposition.
They don't need to be told.

---

## The demo

`demo.py` shows this with the simplest possible case: 5 features in 2D.

Encoding: `h = W @ f`  (linear projection into 2D)  
Decoding: `f̂ = ReLU(W^T @ h)`  (project back, kill negatives)

For a pentagon arrangement:
- Single feature active: max error ~0.31 (from adjacent features, which are
  positive-crosstalk neighbors). The opposite-facing features have negative
  dot products and get zeroed by ReLU.
- Two features active: interference compounds. Phantoms appear.

The visual that makes this click for me: *each feature direction points
somewhere in 2D space. When two features are both "on", the hidden vector
is their sum. Projecting that sum back to any feature direction picks up
contributions from both, scaled by their dot products.*

---

## What I'm still puzzling over

The residual stream in a real transformer is not just holding feature
vectors — it's being written to by every layer, and read from by every
layer. If the stream is in superposition, each attention head must learn
to project out the features it needs without picking up interference from
features it doesn't care about.

The QK circuit decides *which positions to attend to*. The OV circuit
decides *what information to move*. Both involve linear projections from
the residual stream.

Question: do OV circuits learn projections that are "aligned" to the
superposition basis? If feature f lives at direction w_f in the residual
stream, does the OV circuit's output projection naturally produce
contributions that land near w_f in the *next* layer's residual stream?

I think the answer involves the low-rank structure of OV matrices, but I
haven't worked it through. That's the next thing to build.

---

**Source**: Elhage et al., "Toy Models of Superposition" (Anthropic, 2022).
The key figures are the phase diagrams showing when superposition is optimal
vs. when orthogonal representation wins. Worth reading if you haven't.

**Built in**: session 2026-06-23.
