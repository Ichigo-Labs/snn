# Can a 70M-parameter network learn a plasticity rule that beats STDP?

> **Not yet — and this document is the map of why, because the failure is
> specific and quantified rather than vague.** A better rule than STDP certainly
> exists: training the same 784-256⁴ spiking net by surrogate-gradient descent
> (the "teacher") reaches **83.6%** where STDP gets **73.6%**, and the control
> below shows that even a direction with cosine *0.1* to the true gradient beats
> STDP if its error is unbiased. A 70M-parameter policy, fed local three-factor
> signals and trained to imitate that gradient, learns to predict it at cosine
> ~0.3 — three times the margin that should suffice. Yet across two families of
> training objective and eight distinct configurations, the learned rule never
> clears STDP. Two walls, both now measured, explain it:
>
> 1. **A cosine ceiling.** A per-synapse rule built from *time-aggregated* local
>    signals caps at cosine ~0.3 with the true gradient. The gradient's
>    information lives in the *temporal correlation* between a neuron's error and
>    its presynaptic activity; a rule that only sees each summed over the 20
>    timesteps cannot recover it. Raising the ceiling means giving the rule
>    temporally-resolved inputs — which starts to approach re-deriving the
>    gradient.
> 2. **That cosine does not generalize to the rule's own trajectory.** The policy
>    hits cosine 0.3 at the *states it was trained on*; once it steers its own
>    ~1,400-step deployment trajectory into unfamiliar states, its direction stops
>    tracking the gradient and the error compounds through depth. A *synthetic*
>    cosine-0.3 direction whose error is fresh noise each step reaches 81%; the
>    policy's, whose error is systematic and state-dependent, degrades below the
>    66% of doing nothing.
>
> The constructive half is real and measurable: the second wall is the one to
> attack, and the tool now contains the right instrument — a learned-optimizer
> objective that trains the policy through a differentiable rollout of its *own*
> updates. It removes the pointwise bias of imitation, and with a horizon
> curriculum and a longer rollout window the learned rule climbs steadily —
> deployment **16% → 48% (K=4) → 68% (K=8)** — crossing the 66% untrained baseline
> for the first time, with its deep layers now *healthy* (L4 55%) instead of
> collapsed. It genuinely helps; it is a handful of points short of STDP with a
> clean scaling trend in the rollout length pointing the rest of the way, and the
> remaining levers (a gradient-checkpointed longer rollout; temporally-resolved
> inputs to lift the cosine ceiling) are concrete.

Every anchored number regenerates with `bash scripts/kmnist_metaplasticity_experiments.sh`
(CUDA); the exploratory training-objective results are reproducible from the flags
in the reproduction section. STDP figures come from [`kmnist_stdp.md`](kmnist_stdp.md).

## The idea, and the one signal that scales

STDP is a hand-designed two-factor rule: it correlates pre- and post-synaptic
spikes and never sees the task loss. The goal is to *learn* a better local rule —
one reading the signals a synapse has access to, optimized to reduce the loss.

The obstacle is the training signal. A rule judged by the loss it eventually
produces is a black box you would normally optimize with evolution or RL, but
their variance grows with the parameter count and tops out near ~1M-parameter
policies. A 50-100M-parameter policy needs a low-variance signal. **Gradient
imitation** supplies one: compute the surrogate gradient of the readout loss with
respect to the SNN weights — exactly how a weight change affects the loss — and
train the policy so its output aligns with it. That is the whole reason a policy
this large is trainable at all.

## The rule and the policy

The SNN is the depth-4 ladder from [`kmnist_stdp.md`](kmnist_stdp.md),
784-256-256-256-256, same static pixel drive, top-`k` winner-take-all, `T = 20`.
Per layer the policy reads local signals and emits an update direction, applied
under the identical constraint STDP uses (mean-center and renormalize each
incoming row), so the rules differ *only* in direction:

- **Per-synapse:** current weight, causal and anti-causal spike correlations — the
  STDP signals.
- **Per-neuron:** pre-rate, post-rate, threshold, and a top-down error modulator
  (the true per-neuron error `dL/du`, from backprop through the readout).

The 70M parameters live in two wide per-*neuron* encoders (applied to the ~1,000
neurons of a layer, not the ~400k synapses), combined per synapse by a cheap
multi-head bilinear form gated by the per-synapse scalars — the factorization
that makes a 70M-parameter rule affordable to apply. The self-test checks the
parameter count (≥50M), verifies the differentiable SNN forward against finite
differences in float64, and confirms the objective is finite.

Two facts about the third factor, established early:

- **It must be a real error signal.** With only STDP's unsupervised signals the
  policy cannot predict the gradient at all (held-out cosine ~0.02) — the
  batch-aggregated gradient's routing is carried by the labels.
- **Feedback alignment is not enough.** A fixed random projection of the output
  error is uncorrelated with the true per-neuron credit at a random init, leaving
  cosine at ~0.02. Only the true per-neuron error lifts it to ~0.3, at the cost of
  making the rule "backprop-assisted local" rather than fully local.

## The two controls that frame everything

**The teacher: a better rule exists, and depth helps.** Applying the true
surrogate gradient directly as the update rule (same freeze-then-ridge readout)
reaches **83.6%**, and its *deep* layers come out best (L3 82.5%) — the opposite
of STDP and the untrained net, where depth is a liability
([`kmnist_stdp.md`](kmnist_stdp.md)). Proper credit assignment is what makes dense
depth pay.

**The cosine curve: how good a direction actually needs to be.** Train the SNN
with *synthetic* directions of a chosen cosine to the true gradient, the rest
fresh independent noise, re-randomized every step:

| cosine to true gradient | KMNIST test concat |
| --- | --- |
| 1.0 (the teacher) | 83.6% |
| 0.6 | 83.6% |
| 0.4 | 82.6% |
| 0.3 | 80.7% |
| 0.2 | 78.2% |
| 0.1 | 76.4% |
| 0.0 (pure noise) | 66.3% |

A cosine of **0.1** already beats STDP; only pure noise falls back to the
untrained baseline. So a direction of the policy's quality (0.3) is, in magnitude,
far more than enough — *provided its error is unbiased*. That proviso is the whole
game.

## Attempt 1 — gradient imitation: a systematically biased rule

Trained to align its direction with the gradient pointwise, the policy reaches
cosine ~0.30, roughly uniform across layers. At deployment it lands at **33-42%**
— below the 66% of pure-noise updates — and gets *worse* with more training (66%
at 3000 meta-steps, 38% at 5000).

The cosine curve says this cannot be an accuracy problem, and it is not: it is
*bias*. The policy is a deterministic function of the local state, so its
off-gradient error (95% of a cosine-0.3 vector) is the *same* every time it
revisits a state. It does not average away like the synthetic control's fresh
noise — it accumulates, and cascades through depth (deep layers collapse first).
Every cheap fix fails for a principled reason: output dithering cannot change the
signal-to-bias ratio (measured: ≤52% at every noise level); DAgger — training on
the policy's own rollouts — collapses the network during training and makes
deployment worse (39% → 23%). Imitation trains a pointwise gradient-matcher, and a
pointwise-good, trajectory-bad rule is exactly what pointwise imitation permits.

## Attempt 2 — the learned-optimizer objective

The fix follows from the diagnosis: stop optimizing "match the gradient here" and
optimize "reduce the loss after N steps." The tool's `--unroll K` mode runs a
differentiable K-step rollout of the policy's *own* updates and backpropagates the
final readout loss into the policy — a learned optimizer. This penalizes exactly
the trajectory-compounding error that imitation ignores. Training learned
optimizers is itself finicky, and the run reproduces the textbook pathologies:

| training regime | KMNIST test concat |
| --- | --- |
| imitation (Attempt 1) | 33-42% |
| unroll, teacher-advanced trajectory | ~36% (distribution shift) |
| unroll, self-advanced trajectory | 16% (trajectory collapse) |
| unroll, self-advanced + horizon curriculum (K=4) | 48% |
| **unroll, self-advanced + curriculum, longer window (K=8)** | **68%** (deep layers healthy) |
| — for reference | untrained 66% / STDP 73.6% / teacher 83.6% |

- **Teacher-advanced** (persistent trajectory walks the stable surrogate-GD path):
  trains without collapse, but the policy only ever sees good states and shifts
  off-distribution at deployment.
- **Self-advanced** (the policy drives its own trajectory, the correct
  learned-optimizer setup): on-distribution, but an early, weak policy drives the
  net into collapsed states and learns to operate there — the training loss climbs
  and deployment falls to 16%.
- **Horizon curriculum** — start with short (self-healthy) trajectories and grow
  them as the policy improves — is the standard remedy and works as intended: the
  training loss stays low far longer and deployment recovers to 48% (K=4).
- **Longer rollout window.** K=4 optimizes only 4-step outcomes, leaving
  long-horizon drift unpenalized. Doubling it to K=8 ties more of the trajectory
  into the differentiable objective and jumps deployment to **68%** — above the
  untrained baseline, with the deep-layer cascade gone (per-layer 61/58/57/55%
  instead of the 60/30/20/19% collapse of shorter windows). The training loss
  still creeps up at the very end of the full horizon, so K=8 has not fully solved
  Wall 2 either, but the K=4 → K=8 jump (48% → 68%) is a clean scaling signal:
  the objective is correct and the rollout length is the knob. K is currently
  bounded by activation memory; gradient checkpointing would allow K=32-64.

## Why it is hard, stated exactly

The two walls are not independent — they compound. A rule confined to
time-aggregated local signals has a cosine ceiling of ~0.3 (Wall 1), which is
plenty *if unbiased*; but a deterministic rule's error is biased, and the only way
to make it unbiased over a long trajectory is to train on that trajectory, which
runs into the learned-optimizer training pathologies (Wall 2). The teacher escapes
both by using the exact gradient at every state — no ceiling, no bias — which is
why it, and only it, reaches 83.6%. The whole difficulty of local learning is
compressed into the gap between "predict the gradient here" (cosine 0.3, easy) and
"descend the loss over a self-driven trajectory" (still open).

## What this points at next

- **Attack Wall 2 with a longer, checkpointed rollout.** K is currently limited by
  activation memory; gradient checkpointing the per-step forwards would allow
  K = 32-64, tying most of the deployment trajectory into the differentiable
  objective and directly penalizing long-horizon drift.
- **Attack Wall 1 with temporally-resolved inputs.** Splitting the T=20 window into
  a few sub-windows and feeding per-sub-window error and pre-activity would let the
  rule capture some of the temporal error-activity correlation it currently sums
  away, raising the cosine ceiling above 0.3 — where the cosine curve says even a
  biased rule would have room to spare.
- **Report, don't hide, the negative.** The teacher (83.6%) and the cosine curve
  (cosine-0.1 → 76%) bound the achievable headroom precisely; the learned rule's
  gap to them is a measurement of how much of global credit assignment a local
  rule can recover, which is the scientifically interesting quantity regardless of
  whether this particular policy clears STDP.

## Reproducing

Committed data, CUDA GPU required (the tool refuses CPU).

```bash
python3 tools/kmnist_metaplasticity.py --self-test          # param count, float64 gradient check
bash scripts/kmnist_metaplasticity_experiments.sh           # self-test, cosine-curve control, imitation vs teacher
```

The learned-optimizer objective and its trajectory controls (Attempt 2):

```bash
# self-advanced rollout with a horizon curriculum (the best stable regime so far)
python3 tools/kmnist_metaplasticity.py --factor three --unroll 8 \
    --rollout-prob 1.0 --rollout-warmup 0.15 --horizon 175 --horizon-start 10 \
    --meta-steps 3000 --deploy-epochs 3 --include-teacher

# teacher-advanced (stable training, off-distribution deployment): --rollout-prob 0
# no curriculum (collapses):                                       --horizon-start 350
```
