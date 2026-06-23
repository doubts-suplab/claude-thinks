#!/usr/bin/env python3
"""
Induction heads: the simplest nontrivial algorithm in transformers.

Two heads compose across layers to implement in-context bigram copying:
if [A, B] appeared earlier in the sequence, predict B when A appears again.
No training — the weight matrices are manually constructed to implement the circuit.

Run it:  python demo.py
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)

# 3-token vocabulary
VOCAB = {'A': 0, 'B': 1, 'C': 2}
N_VOCAB = 3

# Residual stream: 6 dimensions, split into two subspaces.
# dims 0-2: token embedding space  (written at initialization, read by Q and V)
# dims 3-5: predecessor info space (written by layer 1, read by K in layer 2)
D_MODEL = 6
D_HEAD = 3


def embed(token):
    e = np.zeros(D_MODEL)
    e[VOCAB[token]] = 1.0
    return e


def softmax(v):
    v = v - v.max()
    e = np.exp(v)
    return e / e.sum()


def causal_attention(Q, K, V, temperature=4.0):
    """Scaled dot-product attention with causal mask and temperature."""
    n = len(Q)
    scores = (Q @ K.T) * temperature  # (n, n); temperature sharpens softmax
    A = np.zeros((n, n))
    for i in range(n):
        causal_scores = np.where(np.arange(n) <= i, scores[i], -1e9)
        A[i] = softmax(causal_scores)
    return A @ V, A


def token_name(idx):
    return [k for k, v in VOCAB.items() if v == idx][0]


def section(title):
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}")


def main():
    sequence = ['A', 'B', 'C', 'A']
    n = len(sequence)

    section("THE TASK")
    print(f"""
Sequence:  {' '.join(f'{t}[{i}]' for i, t in enumerate(sequence))}
Goal:      predict the next token after A[3] (the second A)
Answer:    B  — because A→B appeared at positions 0→1

How? Two attention heads cooperate across two layers.
""")

    # ─── Residual stream initialization ─────────────────────────────
    X = np.array([embed(t) for t in sequence])  # (n, D_MODEL)

    section("INITIAL RESIDUAL STREAM")
    print("Token embeddings in dims 0-2. Dims 3-5 are zero (not yet written).\n")
    print("         [A  B  C] [pd pd pd]   ← subspaces")
    for i, t in enumerate(sequence):
        print(f"  pos {i} ({t}): {X[i]}")

    # ─── Layer 1: Previous Token Head ───────────────────────────────
    section("LAYER 1: Previous Token Head")
    print("""
Attention pattern: each position i attends to position i-1.
(The QK circuit implements this via position-based matching — not shown here.)

OV circuit:
  W_V  reads  dims 0-2  (the current token embedding at source position)
  W_O  writes dims 3-5  (the predecessor slot of the destination)

Effect: after layer 1, position j's dims 3-5 hold the embedding of token[j-1].
""")

    # Attention matrix: subdiagonal (position 0 has no predecessor → zeros)
    A1 = np.zeros((n, n))
    for i in range(1, n):
        A1[i, i - 1] = 1.0

    print("Attention matrix (previous token head):")
    print("         " + "  ".join(f"p{j}({sequence[j]})" for j in range(n)))
    for i, t in enumerate(sequence):
        row = "  ".join(f"  {A1[i,j]:.1f}  " for j in range(n))
        print(f"  p{i}({t})  [{row}]")

    # OV: reads token subspace (dims 0-2), writes to predecessor subspace (dims 3-5)
    attended = A1 @ X[:, :3]       # (n, 3): weighted sum of token embeddings
    layer1_out = np.zeros((n, D_MODEL))
    layer1_out[:, 3:6] = attended  # write into predecessor dims

    X1 = X + layer1_out            # residual connection

    print("\nResidual stream after layer 1:")
    print("         [A  B  C] [pd pd pd]   token / predecessor")
    for i, t in enumerate(sequence):
        prev = sequence[i - 1] if i > 0 else "∅"
        print(f"  pos {i} ({t}): {X1[i]}   ({t} / {prev})")

    print("""
Read the right block: position 1's dims 3-5 = [1,0,0] = A ✓
Position 3's dims 3-5 = [0,0,1] = C (C preceded A[3]).
These are the keys for layer 2.
""")

    # ─── Layer 2: Induction Head ─────────────────────────────────────
    section("LAYER 2: Induction Head")
    print("""
The key insight — Q and K read from DIFFERENT subspaces:

  Q reads dims 0-2: "what is my current token?"
  K reads dims 3-5: "what token preceded me?" (written by layer 1)

High attention score at (i, j) when:  token[i] == token[j-1]
  → "my current token was j's predecessor"
  → j is a position right after a previous occurrence of my current token
  → the value at j tells us what followed my current token last time
""")

    # W_Q: extract dims 0-2 (current token)
    W_Q = np.zeros((D_MODEL, D_HEAD))
    W_Q[:3, :] = np.eye(D_HEAD)

    # W_K: extract dims 3-5 (predecessor from layer 1)
    W_K = np.zeros((D_MODEL, D_HEAD))
    W_K[3:, :] = np.eye(D_HEAD)

    # W_V: extract dims 0-2 (copy current token embedding as prediction signal)
    W_V = np.zeros((D_MODEL, D_HEAD))
    W_V[:3, :] = np.eye(D_HEAD)

    Q = X1 @ W_Q   # (n, D_HEAD): current token vectors
    K = X1 @ W_K   # (n, D_HEAD): predecessor token vectors
    V = X1 @ W_V   # (n, D_HEAD): token embeddings to copy

    print("Query vectors  Q  (from dims 0-2, current token):")
    for i, t in enumerate(sequence):
        print(f"  pos {i} ({t}): {Q[i]}")

    print("\nKey vectors  K  (from dims 3-5, predecessor):")
    for i, t in enumerate(sequence):
        prev = sequence[i - 1] if i > 0 else "∅"
        print(f"  pos {i} (prev={prev}): {K[i]}")

    print("\nRaw dot products  Q @ K.T  (before temperature and softmax):")
    raw = Q @ K.T
    header = "         " + "  ".join(f"p{j}(prev={sequence[j-1] if j>0 else '∅'})" for j in range(n))
    print(header)
    for i, t in enumerate(sequence):
        row = "  ".join(f"   {raw[i,j]:+.1f}   " for j in range(n))
        print(f"  p{i}({t})  [{row}]")

    print("""
Matching positions have dot product 1.0; non-matching have 0.0.
Row p3(A): dot product is 1.0 only at p1, because p1's predecessor is A.
""")

    output_l2, A2 = causal_attention(Q, K, V, temperature=4.0)

    print("Attention matrix (induction head, after softmax + causal mask):")
    print(header)
    for i, t in enumerate(sequence):
        row = "  ".join(f"  {A2[i,j]:.3f}  " for j in range(n))
        print(f"  p{i}({t})  [{row}]")

    # ─── Result ──────────────────────────────────────────────────────
    section("RESULT")

    target = n - 1
    top_attended = int(np.argmax(A2[target]))
    top_weight = A2[target, top_attended]
    copied_value = V[top_attended]
    predicted_token = token_name(int(np.argmax(copied_value)))

    print(f"""
At position {target} (second A):
  Layer 2 attention weights: {np.round(A2[target], 3)}
  Strongest: position {top_attended} ({sequence[top_attended]}) with weight {top_weight:.3f}

  Value copied from position {top_attended}:  {copied_value}  → token {predicted_token}
  This is added to position {target}'s residual stream.
  → Logit for token B is boosted.
  → Prediction: B  ✓

Why position {top_attended}?
  Q[{target}] = {Q[target]}  ("my current token is A")
  K[{top_attended}]  = {K[top_attended]}  ("I was preceded by A")
  These match. No other position has a predecessor = A.
""")

    section("THE FULL CIRCUIT")
    print("""
Layer 1 (previous token head):
  - For each position j, writes token[j-1]'s embedding into j's residual stream (dims 3-5)
  - Acts as a "memory relay": positions now carry info about their history

Layer 2 (induction head):
  - Q reads current token (dims 0-2)
  - K reads predecessor info (dims 3-5, written by layer 1)
  - Match: "my current token == your predecessor"
  - V copies the matched position's token embedding
  - Effect: if you saw [A, B] before and you're now at A, copy B into your stream

Neither head alone implements induction:
  - Layer 1 alone: just shifts previous token into each position. No matching.
  - Layer 2 alone: has no predecessor info to match against (dims 3-5 are zero).
  - Together: information flows from layer 1's writes to layer 2's keys.

This is the simplest example of a multi-layer circuit. The residual stream is
the communication channel. Layer 1 writes a message; layer 2 reads it.

In real models (Olsson et al. 2022), induction heads generalize beyond bigrams:
they implement a form of in-context learning, matching ANY pattern seen earlier
in the context window. This is part of how transformers learn from examples
provided in the prompt — not just from training.
""")


if __name__ == "__main__":
    main()
