#!/usr/bin/env python3
"""
Superposition: storing more features than dimensions.

Demonstrates the core mechanism from "Toy Models of Superposition"
(Elhage et al., Anthropic 2022). No ML framework needed — just numpy.

Run it:  python demo.py
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)

N_FEATURES = 5
D_MODEL = 2


def make_features(n, d):
    """n unit vectors evenly distributed around the unit circle in R^2."""
    assert d == 2
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=0)  # (d, n)


def encode(W, f):
    """Project sparse feature vector f into hidden state h."""
    return W @ f  # (d,)


def decode(W, h):
    """Recover features: project back, then ReLU.
    ReLU kills negatives — features pointing away from h contribute
    negative dot products, which (in the sparse case) means they were off."""
    return np.maximum(0.0, W.T @ h)  # (n,)


def one_hot(i, n):
    f = np.zeros(n)
    f[i] = 1.0
    return f


def section(title):
    width = 62
    print(f"\n{'─' * width}")
    print(f"  {title}")
    print(f"{'─' * width}")


def main():
    W = make_features(N_FEATURES, D_MODEL)

    section(f"SUPERPOSITION: {N_FEATURES} features in {D_MODEL} dimensions")

    print(f"""
We have {N_FEATURES} binary features to represent, but only {D_MODEL} dimensions
of hidden state. Orthogonal representation would need {N_FEATURES}D — impossible.

Strategy: pack features as unit vectors evenly around a circle.
Adjacent features have dot product cos(72°) ≈ 0.309.
Opposite-ish features have dot product cos(144°) ≈ −0.809.
""")

    print("Feature directions (columns of W):")
    for i in range(N_FEATURES):
        angle = 360 * i / N_FEATURES
        print(f"  f{i}: [{W[0, i]:+.3f}, {W[1, i]:+.3f}]   ({angle:.0f}°)")

    gram = W.T @ W
    print("\nInterference matrix W^T W  (diagonal=1, off-diagonal=crosstalk):")
    labels = [f"f{i}" for i in range(N_FEATURES)]
    print("        " + "  ".join(f"{l:>7}" for l in labels))
    for i in range(N_FEATURES):
        row = "  ".join(f"{gram[i, j]:+.3f}" for j in range(N_FEATURES))
        print(f"  {labels[i]}   [{row}]")

    section("SINGLE ACTIVE FEATURE")
    print("""One feature active at a time. ReLU zeroes the large negatives
(features pointing opposite), leaving only small positive crosstalk
from neighbors. Recovery is good.\n""")

    for i in range(N_FEATURES):
        f_true = one_hot(i, N_FEATURES)
        h = encode(W, f_true)
        f_hat = decode(W, h)
        error = np.abs(f_true - f_hat)
        worst_idx = int(np.argmax(error))
        print(f"  f{i} active  →  decoded: [{', '.join(f'{v:.3f}' for v in f_hat)}]"
              f"   max error: {error.max():.3f} (on f{worst_idx})")

    section("TWO ACTIVE FEATURES")
    print("""Two features active simultaneously. The hidden vector is now a
sum of two directions; interference between them creates phantom
activations on features that weren't active.\n""")

    pairs = [(0, 1), (0, 2), (1, 3)]
    for i, j in pairs:
        angle_ij = 360 * abs(i - j) / N_FEATURES
        f_true = one_hot(i, N_FEATURES) + one_hot(j, N_FEATURES)
        h = encode(W, f_true)
        f_hat = decode(W, h)
        error = np.abs(f_true - f_hat)
        phantoms = [f"f{k}" for k in range(N_FEATURES) if f_hat[k] > 0.05 and f_true[k] == 0]
        print(f"  f{i} + f{j} ({angle_ij:.0f}° apart):")
        print(f"    decoded: [{', '.join(f'{v:.3f}' for v in f_hat)}]")
        print(f"    phantom activations: {phantoms or 'none'}   max error: {error.max():.3f}")
        print()

    section("THE TRADEOFF")
    print("""Sparsity is the key variable:

  • If features are sparse (rarely co-active), errors stay small.
    Each token has only a few active features; the rest are off.
    → superposition is almost free.

  • If features are dense (many active at once), interference compounds.
    → superposition becomes expensive.

This is why gradient descent discovers superposition spontaneously
when features are sparse: the compression gain exceeds the error cost.

Implications for real models:
  1. Features aren't stored in individual neurons — they're directions
     spread across many neurons. Reading a single neuron tells you little.

  2. A model can confuse concepts that share superposition "space" in a
     specific layer, even if those concepts seem unrelated semantically.

  3. Sparsifying activations (e.g. with ReLU, or MoE routing) reduces
     interference — part of why these techniques work beyond raw compute.

The deeper puzzle I don't yet have a satisfying answer for: how do
attention heads learn to extract the right features from a superposed
residual stream? The QK and OV circuits must somehow be "aware" of the
superposition structure. I don't know how that actually works.
""")


if __name__ == "__main__":
    main()
