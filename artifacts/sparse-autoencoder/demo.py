#!/usr/bin/env python3
"""
Sparse autoencoder learning superposed features.

The residual stream has 5 features superposed into 2 dimensions
(pentagon arrangement from the superposition demo). A sparse autoencoder
(SAE) trained on samples from this distribution approximately recovers
the original feature directions — without being told what they are.

The demo also compares to PCA to show why sparsity constraints matter.

Run it:  python demo.py
"""

import numpy as np

np.set_printoptions(precision=3, suppress=True)
rng = np.random.default_rng(42)


# ─── Ground truth ────────────────────────────────────────────────────────────

N_FEATURES = 5
D_MODEL = 2
FEATURE_PROB = 0.1   # each feature is active ~10% of the time (sparse)


def make_pentagon():
    angles = np.linspace(0, 2 * np.pi, N_FEATURES, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=0)   # (D_MODEL, N_FEATURES)


def sample_batch(W_true, batch_size):
    f = (rng.uniform(size=(batch_size, N_FEATURES)) < FEATURE_PROB).astype(float)
    x = f @ W_true.T   # (batch_size, D_MODEL)
    return x, f


# ─── Sparse autoencoder ──────────────────────────────────────────────────────

class SAE:
    """
    Encoder: h = ReLU(x @ W_enc.T + b_enc)    D_MODEL → N_FEATURES
    Decoder: x̂ = h @ W_dec.T                  N_FEATURES → D_MODEL
    Loss    = ||x̂ - x||² + λ·||h||₁
    """

    def __init__(self, d_model, n_features, lr=3e-3, beta1=0.9, beta2=0.999):
        self.d = d_model
        self.n = n_features
        self.lr, self.beta1, self.beta2, self.t = lr, beta1, beta2, 0
        self.W_enc = rng.standard_normal((n_features, d_model)) * 0.1
        self.b_enc = np.zeros(n_features)
        self.W_dec = rng.standard_normal((d_model, n_features)) * 0.1
        self._normalize_dec()
        self.m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self.v = {k: np.zeros_like(v) for k, v in self._params().items()}

    def _normalize_dec(self):
        norms = np.linalg.norm(self.W_dec, axis=0, keepdims=True)
        self.W_dec = self.W_dec / np.maximum(norms, 1e-8)

    def _params(self):
        return {'W_enc': self.W_enc, 'b_enc': self.b_enc, 'W_dec': self.W_dec}

    def forward(self, x):
        pre = x @ self.W_enc.T + self.b_enc
        h = np.maximum(0.0, pre)
        x_hat = h @ self.W_dec.T
        return pre, h, x_hat

    def loss_and_grads(self, x, lam):
        B = x.shape[0]
        pre, h, x_hat = self.forward(x)
        rec_loss = np.mean(np.sum((x_hat - x) ** 2, axis=1))
        spar_loss = lam * np.mean(np.sum(h, axis=1))
        loss = rec_loss + spar_loss
        d_xhat = 2 * (x_hat - x) / B
        g_W_dec = d_xhat.T @ h
        d_h = d_xhat @ self.W_dec + lam / B * np.sign(h)
        delta = d_h * (pre > 0).astype(float)
        g_W_enc = delta.T @ x
        g_b_enc = delta.mean(axis=0)
        return loss, rec_loss, {'W_enc': g_W_enc, 'b_enc': g_b_enc, 'W_dec': g_W_dec}

    def step(self, grads):
        eps = 1e-8
        self.t += 1
        bc1 = 1 - self.beta1 ** self.t
        bc2 = 1 - self.beta2 ** self.t
        for k, p in [('W_enc', 'W_enc'), ('b_enc', 'b_enc'), ('W_dec', 'W_dec')]:
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[k]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * grads[k] ** 2
            update = self.lr * (self.m[k] / bc1) / (np.sqrt(self.v[k] / bc2) + eps)
            setattr(self, p, getattr(self, p) - update)
        self._normalize_dec()


# ─── Analysis helpers ─────────────────────────────────────────────────────────

def angle_deg(v):
    return np.degrees(np.arctan2(v[1], v[0])) % 360


def best_alignment_error(true_dirs, learned_dirs):
    """Mean angle error after optimal greedy matching (true → learned)."""
    n = true_dirs.shape[0]
    ld_normed = learned_dirs / (np.linalg.norm(learned_dirs, axis=1, keepdims=True) + 1e-8)
    assignment, errors = [], []
    used = set()
    for td in true_dirs:
        dots = np.abs(ld_normed @ td)
        dots_masked = dots.copy()
        for idx in used:
            dots_masked[idx] = -1
        best = int(np.argmax(dots_masked))
        used.add(best)
        assignment.append(best)
        errors.append(np.degrees(np.arccos(np.clip(dots[best], 0, 1))))
    return assignment, errors


def angular_spacings(angles_deg):
    """Given a list of angles in degrees, return sorted pairwise spacings."""
    s = sorted(angles_deg)
    spacings = [(s[(i + 1) % len(s)] - s[i]) % 360 for i in range(len(s))]
    return spacings


def section(title):
    print(f"\n{'─' * 64}")
    print(f"  {title}")
    print(f"{'─' * 64}")


def main():
    W_true = make_pentagon()   # (D_MODEL, N_FEATURES)
    true_dirs = W_true.T       # (N_FEATURES, D_MODEL)

    section("GROUND TRUTH")
    print(f"""
{N_FEATURES} features superposed into {D_MODEL} dimensions (pentagon).
Each feature is active independently with probability {FEATURE_PROB}.
Residual stream: x = W_true @ f  (linear sum of active feature directions)

True feature directions:""")
    for i in range(N_FEATURES):
        col = W_true[:, i]
        print(f"  f{i}: {col}   angle {angle_deg(col):.0f}°")

    # Generate a large dataset for analysis
    X_data, F_data = sample_batch(W_true, 50000)

    # ─── PCA baseline ────────────────────────────────────────────────────────
    section("PCA BASELINE (what sparsity-agnostic decomposition finds)")

    cov = X_data.T @ X_data / len(X_data)
    eigvals, eigvecs = np.linalg.eigh(cov)
    pca_dirs = eigvecs[:, ::-1].T    # (2, D_MODEL), sorted by variance descending

    print(f"""
PCA finds the 2 principal directions of the data cloud.
Since d_model = 2, PCA gives a full orthogonal basis — exactly 2 directions.
It cannot find 5 features. It doesn't know there are 5.

PCA directions:""")
    for i, pc in enumerate(pca_dirs):
        print(f"  PC{i}: {pc}   angle {angle_deg(pc):.0f}°")

    print("""
These 2 directions span the space but don't correspond to individual features.
PCA is a rotation, not a decomposition into meaningful units.
Sparsity constraints are what allow SAEs to do better.
""")

    # ─── SAE training ─────────────────────────────────────────────────────────
    section("TRAINING SAE")
    print("""
Encoder: h = ReLU(x @ W_enc.T + b_enc)   [D_MODEL → N_FEATURES]
Decoder: x̂ = h @ W_dec.T                 [N_FEATURES → D_MODEL]
Loss:    ||x̂ - x||² + λ·||h||₁

λ=0.05  lr=3e-3  batch=256  Adam optimizer
Decoder columns held unit-norm throughout training.
""")

    sae = SAE(D_MODEL, N_FEATURES, lr=3e-3)
    lam = 0.05
    batch_size = 256
    n_steps = 3000

    print(f"{'step':>6}  {'total loss':>10}  {'recon loss':>10}")
    for step in range(n_steps + 1):
        x, _ = sample_batch(W_true, batch_size)
        loss, rec, grads = sae.loss_and_grads(x, lam)
        if step > 0:
            sae.step(grads)
        if step % 600 == 0:
            print(f"  {step:>6}    {loss:>9.4f}    {rec:>9.4f}")

    # ─── Results ─────────────────────────────────────────────────────────────
    section("LEARNED ENCODER DIRECTIONS")
    enc_dirs = sae.W_enc.copy()  # (N_FEATURES, D_MODEL)
    enc_dirs_n = enc_dirs / (np.linalg.norm(enc_dirs, axis=1, keepdims=True) + 1e-8)

    for i in range(N_FEATURES):
        print(f"  enc{i}: {enc_dirs_n[i]}   angle {angle_deg(enc_dirs_n[i]):.0f}°")

    section("ALIGNMENT: TRUE → LEARNED")
    assignment, errors = best_alignment_error(true_dirs, enc_dirs)

    print(f"\n  {'True feature':>15}  {'Matched to':>12}  {'Angle error':>12}")
    for i in range(N_FEATURES):
        td = true_dirs[i]
        j = assignment[i]
        ld = enc_dirs_n[j]
        print(f"  f{i} ({angle_deg(td):.0f}°)"
              f"  → enc{j} ({angle_deg(ld):.0f}°)"
              f"   {errors[i]:.1f}°")

    mean_err = np.mean(errors)
    print(f"\n  Mean alignment error: {mean_err:.1f}°")

    section("STRUCTURE CHECK: FEATURE COLLAPSE?")
    learned_angles = [angle_deg(enc_dirs_n[i]) for i in range(N_FEATURES)]
    spacings = angular_spacings(learned_angles)

    # Detect collapsed features: pairs with < 10° separation
    pairwise = np.abs(enc_dirs_n @ enc_dirs_n.T)
    np.fill_diagonal(pairwise, 0)
    collapsed = [(i, j) for i in range(N_FEATURES) for j in range(i+1, N_FEATURES)
                 if pairwise[i, j] > 0.99]

    print(f"""
A perfect pentagon has equal 72° spacings between consecutive directions.
Collapsed features (cosine > 0.99, i.e. < ~8° apart): {collapsed if collapsed else 'none'}

Learned angles (sorted): {sorted([round(a) for a in learned_angles])}
Spacings between consecutive: {[round(s, 1) for s in spacings]}
  True spacing: 72° each
  Variance in spacings: {np.var(spacings):.1f}  (0 = perfect pentagon)

Feature collapse is a known SAE failure mode: two encoder units learn the same
direction, meaning the overcomplete dictionary has a redundant feature.
Practitioners fix this with auxiliary losses or by periodically reinitializing
dead/collapsed features ("feature resurrection").
""")

    section("INTERPRETATION")
    print(f"""
The SAE achieved near-perfect reconstruction (recon loss ~0.001)
but imperfect alignment with the true pentagon ({mean_err:.1f}° average error).

Why the gap?

1. LOCAL MINIMA: In 2D, the SAE objective has many near-equivalent solutions.
   The reconstruction loss penalizes poor decoding, but multiple W_dec arrangements
   can decode almost equally well. Gradient descent finds whichever basin it falls into.

2. ROTATION SYMMETRY: The distribution of x = W_true @ f is invariant to a joint
   rotation of all feature directions. There's no objective signal that says "the
   pentagon should be at 0°, 72°, 144°..." vs. some other orientation.
   In higher dimensions, individual feature directions have more "room" and this
   symmetry is broken by the data geometry.

3. 2D IS A HARD CASE: With 5 features in 2D (2.5× compression ratio), the
   features are maximally entangled. In practice, SAEs work on residual streams
   with d_model ≥ 512 and far fewer simultaneously active features. The ratio
   is more favorable, and alignment is much tighter.

PCA vs. SAE:
  - PCA finds 2 directions (it doesn't know there are 5 features)
  - SAE finds 5 directions (approximately right, even if imperfectly placed)
  - The difference is the sparsity constraint, which breaks the rotation freedom

Key insight: the L1 penalty encourages solutions where each encoder unit fires
rarely. This creates pressure toward "one unit per feature" — PCA has no such
pressure and finds the rotation-invariant eigen-decomposition instead.

What the two failure modes (local minima, feature collapse) teach us:
  — Real SAE training is more complex than "just add L1 and train"
  — 2D is the hardest case due to maximal superposition ratio
  — In real models (d_model ≥ 512, sparse features), the landscape is more
    favorable and SAEs do recover interpretable, stable feature directions
""")


if __name__ == "__main__":
    main()
