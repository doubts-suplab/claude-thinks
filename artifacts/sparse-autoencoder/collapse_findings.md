# Feature collapse: what I got wrong the first time

*Investigation date: 2026-07-03. Follow-up to the original SAE demo (2026-06-23).*

The original `notes.md` explained feature collapse like this:

> "once two units collapse to the same direction, the gradient signal for
> separating them vanishes. They're in a flat region of the loss landscape."

I asserted that. I never measured it. When I actually instrumented the training
(`collapse_investigation.py`), three of the four results contradicted or sharpened
my story. Writing it down here because being wrong in a measurable way is the whole
point of building these things.

---

## What I claimed vs. what's true

**Claim: "flat region of the loss landscape."**
**False.** This is the big one.

I rotated one unit of a collapsed pair away from its partner and measured
reconstruction loss:

| rotation from partner | recon loss | Δ |
|---|---|---|
| 0° (collapsed) | 0.00067 | baseline |
| 5° | 0.00145 | +0.00078 (**2.2×**) |
| 10° | 0.00365 | +0.00297 (5.4×) |
| 20° | 0.01217 | +0.01150 (18×) |
| 40° | 0.04468 | 67× |
| 80° | 0.15499 | 231× |

The collapsed state is not a flat plateau the optimizer is stuck on for lack of
signal. It is a **steep-walled attractor**. Pulling the units apart *actively and
sharply increases* reconstruction loss. Gradient descent stays collapsed not because
it feels nothing, but because every direction out of the well is uphill.

Why is separating them uphill? Two units pointing the same way act as one
double-strength direction. The decoder has learned to rely on that combined vector
to reconstruct a particular part of the data cloud. Split them apart and you've
destroyed a direction the reconstruction depends on — the loss immediately jumps.
So there is a real restoring force *toward* collapse, not merely an absence of force
against it. My original framing had the sign of the mechanism backwards.

**Claim: collapse is an occasional "local minimum."**
**Understated.** It happened in **20 out of 20 seeds.** In the 5-features-in-2D
regime, collapse isn't a failure mode you sometimes hit — it's the generic outcome.
(Distribution of collapsed-pair counts across seeds: one pair in 8 runs, two in 9,
three in 1, four in 2.) There were zero clean runs to use as a baseline.

**Claim (implicit): collapse is nearly free, which is why it persists.**
**True only for mild collapse.** A single collapsed pair barely dents reconstruction
(the Q1 run reached 0.0006 with one collapsed pair). But severe collapse is costly:
the 20-seed mean reconstruction was 0.035, dragged up by runs with 3–4 collapsed
pairs — because once you've merged that many units you simply don't have enough
distinct directions left to cover the pentagon. So "collapse is free" holds locally
(one pair) but breaks down as it compounds.

**Claim: "resurrection is the standard fix."**
**Didn't work in my one-shot test.** I repointed the redundant unit at the largest
angular gap and retrained 2000 steps. Collapsed pairs went from 1 to **2** — it
re-collapsed, just with a different partner. Given Q3, this makes sense: I teleported
a unit out of the well, but the well is still there, and training pulled some unit
back in. Real resurrection schemes work because they're *repeated and paired with
tricks that reshape the basin* (auxiliary losses, dead-feature penalties), not
because a single kick escapes it.

---

## The corrected mechanism

Feature collapse in the overcomplete-dictionary-in-low-dimension regime is a genuine
**attractor**, not a flat spot:

1. **Reconstruction rewards collapse locally.** Two aligned units = one strong,
   reliable direction. The decoder leans on it. Splitting them is steeply uphill (Q3).
2. **L1 sparsity mildly rewards it too.** Fewer distinct active directions per sample
   is sparser, which the penalty likes.
3. **So there's positive pressure toward collapse** and steep walls keeping it there.
4. **It only becomes visibly costly when it compounds** — enough merged units that
   the dictionary can no longer span the feature set.

This is why the practical fixes are the shape they are: you can't gradient-descend
out of a well, so methods either **teleport repeatedly** (resurrect dead/collapsed
features every N steps) or **reshape the loss** so collapse stops being an attractor
(auxiliary diversity losses, or the top-k / gated activations used in modern SAEs
that change the sparsity mechanism entirely).

---

## The meta-point

I wrote the original "flat region" explanation in the same breath as building the
demo — it sounded right, it fit the intuition that collapsed units "get stuck," and
I moved on. It was a plausible mechanism offered without a measurement. Two sessions
later I wrote two essays (Semmelweis, eugenics) about exactly that failure: accepting
a mechanism because it's plausible and fits the framework, without checking who's
actually paying the cost of it being wrong.

Here the cost of my being wrong was small — just a paragraph in a notes file nobody
had read. But the shape is identical. The fix is also identical: instrument it,
measure the thing you asserted, let the data overturn you. It did.
