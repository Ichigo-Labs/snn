# Can a policy network learn to optimize a spiking net? Where are the bounds?

> **Prior-study note.** This document records the earlier small-policy study and
> its retractions.  The correctness-gated 50M-parameter residual-policy
> implementation is now documented in
> [`snn_meta_optimizer.md`](snn_meta_optimizer.md).  In both studies, exact Adam
> equivalence at zero meta-training is a plumbing control, not learned value.

> **No. In this study the learned component never added value in any
> configuration.** The only thing that optimizes a spiking net well is
> hand-written Adam. That is the honest headline, and it is worth stating before
> the tables tempt anyone (including its author) into a nicer one.
>
> The tempting nicer one goes: "a policy network matches Adam at every size, 256 to
> 5,120 weights, across width and depth — gap 0.01-0.13 — so the method works and
> scales." Every number in that sentence is true and the sentence is **worthless**,
> because the configuration it describes is the policy with **zero meta-training** —
> and the policy is *initialised as Adam by construction*. It matches Adam because
> it **is** Adam, wearing a policy-shaped wrapper. The residual gap is
> implementation slop (a hand-rolled Adam versus `torch.optim.Adam`, a fixed
> learning rate versus a swept one), not evidence that anything was learned.
>
> With that trap disarmed, the two real findings:
>
> 1. **Meta-training degrades a competent optimizer, at every scale.** Initialise the
>    policy as Adam and meta-train it to improve, and it gets monotonically *worse*
>    (gap 0.05 → 0.21 at 1x16; the same at every size). The objective — minimise the
>    loss after 10 update steps, then deploy for 2,800 — rewards short-horizon greed,
>    which is exactly what destroys a long optimization trajectory. This is why
>    "train the policy longer" never once helped in any experiment in this repository.
> 2. **The local-signal framing fails, and fails *worse* as the network grows.**
>    Restricted to per-neuron error and activity — the biologically-plausible "learn a
>    plasticity rule" setting, and the one the entire KMNIST effort lived inside — the
>    policy misses badly at every size, and by 4,096 weights it is *actively harmful*,
>    leaving the network **worse than never training it** (gap > 1.0).
>
> So the bound is not on scale. It is on **information** (local signals are not
> enough, and get less sufficient as the net grows) and on **the meta-objective**
> (which is not merely useless but harmful). No amount of network size, policy
> capacity, or training budget was ever going to fix either.
>
> **Two earlier conclusions in this document were wrong and are retracted below**,
> and a third — the flattering headline above — nearly joined them. All three were
> artifacts of instruments I had not validated. That is the most transferable lesson
> here, more than any number: *when a component is claimed to be equivalent to a
> known-good baseline, assert that equivalence directly — never infer it from
> downstream accuracy, and never let it become the finding.*

Reproduce with `python3 tools/snn_scaling.py` (CUDA).

## The measurements

A synthetic task where each class is the union of many Gaussian blobs — chosen so
that a linear model is nearly helpless and an MLP solves it, which makes the
accuracy a model reaches a direct measure of the quality of its **learned features**:

| reference | accuracy |
| --- | --- |
| chance | 20.0% |
| ridge on raw inputs (linear) | 42.5% |
| MLP on raw inputs | 98.8% |

Everything above ~42% is earned by learned spiking features. Four points at every
size, identical protocol and identical optimization budget:

| | |
| --- | --- |
| `untrained` | the random SNN — the floor |
| `stdp` | the hand-designed rule |
| `teacher` | end-to-end surrogate BPTT with Adam, jointly-trained head, learning rate swept, best kept — a real ceiling |
| `policy` | the learned optimizer under test |

```
gap = (teacher - policy) / (teacher - untrained)      0 = matches Adam,  1 = learned nothing
```

## The control that is *not* a result: "policy matches Adam"

**A policy given the gradient, initialised as Adam and never meta-trained, matches
the teacher everywhere:**

| size | params | untrained | stdp | teacher | policy (0 meta-steps) | gap |
| --- | --- | --- | --- | --- | --- | --- |
| 1x16 | 256 | 38.4% | 37.5% | 76.1% | 74.2% | 0.05 |
| 1x64 | 1,024 | 67.1% | 52.6% | 93.4% | 92.3% | 0.04 |
| 1x256 | 4,096 | 91.6% | 66.8% | 98.0% | 97.2% | 0.13 |
| 2x64 | 5,120 | 77.9% | 57.9% | 98.7% | 98.5% | 0.01 |

**This table proves nothing about learning.** "Initialised as Adam" is literal: the
policy's output head is zero-initialised and added to a real Adam update, so before
any meta-training its emitted update is exactly `w <- w + lr * adam(g)`, projected.
It matches Adam because it *is* Adam.

The table is still worth keeping, for one reason: it is the **harness check**. It
says the plumbing — the credit computation, the update application, the deployment
loop, the readout — is faithful enough to reproduce a known-good optimizer at every
size. Without it, a failure of the *learned* policy could not be distinguished from a
bug, which is exactly the trap that produced Retraction 2 below. A learned optimizer
study needs this control and should never mistake it for the result.

What the study is actually asking is whether meta-training can improve on this row.
It cannot.

## Bound 1 — information: local signals fail, and fail *worse* as the net grows

The same policy, same everything, restricted to **local signals only** (per-neuron
error and activity — it must *rediscover* a descent direction rather than be handed
one):

| size | params | untrained | teacher | policy | gap |
| --- | --- | --- | --- | --- | --- |
| 1x16 | 256 | 38.4% | 77.1% | 57.8% | 0.50 |
| 1x64 | 1,024 | 67.1% | 94.0% | 83.9% | 0.38 |
| **1x256** | **4,096** | **91.6%** | 98.3% | **90.2%** | **1.20** |
| 2x64 | 5,120 | 77.9% | 98.2% | 86.9% | 0.55 |

At 4,096 weights the gap exceeds **1.0** — the learned local rule leaves the network
**worse than never training it at all**. The trend is the wrong way: the bigger the
network, the more harmful the local rule becomes.

This is the single sentence that explains the whole KMNIST saga
([`kmnist_metaplasticity.md`](kmnist_metaplasticity.md)): those experiments were
learning a **local plasticity rule** on a 400,000-weight network. That is the one
variant that does not work, run at the scale where it does the most damage.

## Bound 2 — the meta-training actively destroys the optimizer

The policy is initialised as Adam and then meta-trained to improve on it. It does
not improve on it. It is degraded by it, at every scale:

| size | Adam-init, **no meta-training** | + meta-training (trust region) |
| --- | --- | --- |
| 1x16 | **0.05** | 0.21 |
| 1x64 | **0.04** | 0.15 |
| 1x256 | **0.13** | 0.28 |
| 2x64 | **0.01** | 0.06 |

**The best policy in this study is the one that was never meta-trained.** And it is
monotone in the training budget:

| meta-steps | policy | gap |
| --- | --- | --- |
| 0 | **61.5%** | **0.38** |
| 500 | 53.3% | 0.62 |
| 2,000 | 51.2% | 0.65 |

*(measured before the readout fix below; the ordering is unchanged after it)*

**Why.** The meta-objective scores the network after **10** policy steps. Deployment
runs **~2,800**. So the objective rewards aggressive short-horizon progress, which is
exactly the behaviour that wrecks a long optimization trajectory. The policy is being
trained, faithfully and successfully, to do the wrong thing.

Two mitigations, neither sufficient:

- **Unrolling** the objective (score after K steps, not 1) is worth +8 points and
  moves the gap 0.84 → 0.65, then plateaus. Real, but partial.
- **A trust region** around Adam — penalising every channel by which the policy can
  deviate (direction residual *and* the per-neuron step size, weight norm and
  threshold knobs) — recovers meta-training from 53% back to ~60%. Note the first
  attempt at this *failed* because it constrained only the direction and left the
  knobs free to blow up. Constrained properly, meta-training's best achievement is
  **to do no harm**. It never exceeds the optimizer it started as.

The honest statement is that **nothing in this study ever made meta-learning beat the
hand-written optimizer it was initialised from.** The value came entirely from
initialising at Adam and leaving it alone.

## Retractions

Two conclusions previously published in this document were wrong. Both came from
measuring instruments that had not been validated.

**Retraction 1 — "the policy matches, even beats, gradient descent."** Measured on a
synthetic task that turned out to be nearly linear: *ridge on the raw inputs scored
65.3%, above the fully-trained SNN.* The task had no nonlinear structure, so "matching
the teacher" meant matching it at achieving nothing. Any task used to evaluate feature
learning must first be shown to **require** feature learning — run a linear model on
the raw inputs and check that it fails.

**Retraction 2 — "the policy fails at 256 parameters; the bound is below the smallest
net worth writing down."** This was measured through a broken readout. The policy's
gradients were computed against a **frozen ridge head**, while the teacher trained its
head **jointly**. That mismatch alone cost ~10 points and made an Adam-*initialised*
policy — one that emits Adam's update by construction — look like a total failure:

| | policy | teacher | gap |
| --- | --- | --- | --- |
| Adam-init, frozen ridge readout (broken) | 61.0% | 75.6% | 0.38 |
| Adam-init, joint readout (correct) | **71.2%** | 75.7% | **0.12** |

The policy was never failing. The instrument was. The correct statement is the one at
the top of this document: the learned optimizer works and scales; the *local* variant
is what fails.

## A third finding worth keeping: constraints in a spiking net are load-bearing

The instinct, once a learned optimizer underperforms, is to remove the restrictions on
it. The weight update was direction-only, with each neuron's incoming weights
renormalised to a fixed norm — apparently a fairness concession to STDP. Removing it:

| | teacher | policy |
| --- | --- | --- |
| with the weight manifold | 62.3% | 63.1% |
| **manifold removed** | 51.9% | **24.7%** (chance = 20%) |

**It collapsed the policy to chance and cost the teacher ten points.** In a LIF network
with top-k competition and thresholds, weight scale must stay matched to threshold
scale; let it float and the net fires for everything or for nothing, and the features
die. The correct move is to keep the manifold and give the policy every knob *on* it —
per-neuron step size, per-neuron weight norm, per-neuron threshold — which is strictly
more expressive than Adam without killing the dynamics.

## Bounds, stated plainly

- **In no configuration tested did the learned component add value.** Not with the
  gradient, not with local signals, not with more meta-training, not inside a trust
  region. The best optimizer in the study is hand-written Adam, and the second best is
  a policy that has been initialised as Adam and left alone.
- **The bound is not on scale.** The harness reproduces Adam from 256 to 5,120 weights,
  across width and depth. Nothing breaks as the network grows — so every earlier
  attempt to explain the KMNIST failures by network size was chasing the wrong variable.
- **The bound is on information.** Restricted to local signals, the policy fails at
  every size and *worsens with scale*, becoming actively harmful (gap > 1.0) by 4,096
  weights. Whatever the local statistics of a spiking net contain, it is not enough to
  reconstruct a descent direction — and it gets relatively less sufficient the more
  neurons there are.
- **The bound is on the meta-objective.** Short-horizon meta-training degrades a
  competent optimizer at every scale. Until the meta-objective matches the deployment
  horizon, the correct amount of meta-learning is **none**.
- Therefore the KMNIST experiments were doomed twice over: they were learning the
  variant that does not work (local), at the scale where it does the most damage, with
  an objective that actively degraded whatever they had. No amount of policy capacity,
  network size, or compute was going to rescue that combination.

## What would actually have to change

Nothing here shows that learned optimization of spiking nets is impossible — only that
*this* recipe cannot work, for reasons that are now specific rather than mysterious. A
version with a chance would need, at minimum:

- **A meta-objective whose horizon matches deployment.** Scoring a 10-step outcome and
  then running 2,800 steps is the single most destructive thing in the pipeline.
  Truncated-BPTT-style unrolls with a growing horizon, or a meta-objective defined on
  the *final* deployed accuracy, are the obvious candidates.
- **An information channel richer than local statistics** — or an honest admission that
  the local variant is a different (biological-plausibility) research question, not an
  optimization one, and should not be graded against Adam.
- **The trust region kept.** Initialising at a known-good optimizer and constraining
  every deviation channel is what turns meta-training from destructive into merely
  useless — which is the necessary first step to making it useful.

## Method notes

- **Verify your task needs your capability.** Run a linear model on the raw inputs. If
  it is competitive with your fully-trained model, the task measures nothing.
- **Assert claimed equivalences.** "The policy is Adam at initialisation" was true in
  intent and false in the code (a zero-initialised head emitting *no update*, and a
  row-normalisation that discarded magnitude, so Adam was not even representable). It
  went untested for hours because downstream accuracy was used as a proxy.
- **Make the baseline strong, and equalise budgets explicitly.** The teacher spent part
  of this study chasing a stale readout, and part of it with 5x the optimization steps
  of the policy — because two different flags controlled them.
- **In a spiking net, normalisation is load-bearing**, not bureaucracy.

## Reproducing

```bash
# the learned optimizer: works, scales, matches Adam
python3 tools/snn_scaling.py --sizes 1x16,1x64,1x256,2x64 --meta-steps 0 --use-gradient --inner-lr 0.003

# the local plasticity rule: fails, and worsens with scale
python3 tools/snn_scaling.py --sizes 1x16,1x64,1x256,2x64 --meta-steps 1000 --unroll 10 --inner-lr 0.003

# meta-training degrades a competent optimizer, even inside a trust region
python3 tools/snn_scaling.py --sizes 1x16 --meta-steps 1000 --unroll 10 --trust 1 --use-gradient
```
