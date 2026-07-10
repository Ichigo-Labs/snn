# Does depth help a spiking network the way it helps a CNN?

> **No — and the reason is not the spiking.** Stacking LIF layers from one to
> four buys the surrogate-BPTT SNN **+0.11 to +0.33 points** on KMNIST, none
> of it significant over 4 shared seeds. A ReLU MLP with the *same* layer
> widths gains an equally hollow +0.4 to +0.6. A plain CNN given the same
> depth budget gains **+6.6 points** (89.8% → 96.5%), every step significant
> at `t = 27` to `t = 106`. Depth pays through the convolutional prior, not
> through layer count — and the SNN, being architecturally a dense net, gets
> what dense nets get: nothing. Spending the same parameters on *width*
> instead is strictly better for both dense families (SNN 784-1024-10 beats
> the 4-layer ladder by 0.8 points with the same epoch cost).
>
> On the train/test loss question: **every** unregularized model here — 16
> configurations across three families — memorizes KMNIST outright, reaching
> ≥ 99.1% train accuracy by epoch 15 while test loss bottoms out at epoch 3–5
> and then climbs 40–75%. Test *accuracy* barely notices (it gives back only
> 0.3–0.7 points), so the loss curves and the accuracy curves genuinely
> disagree about when to stop. The SNN is the mildest offender at matched
> architecture — its test loss climbs +53% from the minimum against its MLP
> twin's +75% — but spiking only slows the memorization; it does not prevent
> it.
>
> The practical summary is unflattering but useful: at this scale a spiking
> dense net costs 0.2–1.0 accuracy points against its ReLU twin and 20
> timesteps of compute, and no amount of *dense* depth closes the 5–6 point
> gap to a small CNN. The missing ingredient is convolution, not layers —
> a convolutional SNN is the experiment this result points at.

All numbers regenerate with `python3 scripts/depth_report.py docs/data/kmnist`
from the committed CSVs.

## Protocol

Same data (`data/kmnist`, pixel/255, no augmentation), same optimizer (Adam,
swept lr), same batch size (128), same epoch budget (15) and the same four
seeds everywhere; no batch norm, no dropout, no schedules anywhere, because
the SNN trainer has none of them. Three families:

| family | trainer | depth ladder | width controls |
| --- | --- | --- | --- |
| SNN (LIF, atan surrogate, T=20) | `mnist_bptt --gpu` | 784-(256)ⁿ-10, n=1..4 | 784-512-10, 784-1024-10 |
| MLP (ReLU) | `kmnist_cnn.py --arch mlp` | same shapes as the SNN | same |
| CNN (conv3x3-ReLU-maxpool blocks) | `kmnist_cnn.py --arch cnn` | 1..4 blocks, 32-64-128-256 channels, linear head | — |

The MLP is the control that makes the CNN comparison interpretable: SNN vs
MLP isolates the neuron model at fixed architecture; MLP vs CNN isolates the
convolutional prior at fixed neuron model. Without it, "CNN beats SNN" would
confound the two.

Each architecture first got a hyperparameter sweep (4 epochs, 2 seeds): lr
over {5e-4, 1e-3, 2e-3} for the SNN and {3e-4, 1e-3, 3e-3} for the torch
models, and — because [`kmnist_bptt.md`](kmnist_bptt.md) showed the optimal
surrogate width does not transfer across datasets — `alpha` over {0.5, 1, 2}
per SNN depth rather than assuming depth-1's answer. The depth ladder picked
`alpha = 2` at every depth (the width controls were fixed at `alpha = 1`, so
if anything the width rows are *under*-tuned, which only strengthens the
width-beats-depth conclusion below). Finals ran 15 epochs on 4 shared seeds.

Train loss is measured two ways. The `train_loss` column of the CSVs is the
online average over the epoch, computed while the weights move; it overstates
early-epoch train loss by roughly 2x. For the generalization-gap analysis the
tool's `--train-eval 10000` scores the *frozen* post-epoch model on a fixed
10k prefix of the training set, so the train and test losses in the gap
tables are the same measurement on the same parameters.

## Head to head

### head to head - KMNIST, 15 epochs, 4 seeds, each at its swept lr (SNN: and alpha)

| model | params | lr | best test acc | final test acc | epoch of min test loss | firing |
| --- | --- | --- | --- | --- | --- | --- |
| `snn_d1` | 203,530 | 1e-3, a=2 | **90.36% ±0.16** | 89.63% ±0.57 | 3.2 | 0.161 |
| `snn_d2` | 269,322 | 1e-3, a=2 | **90.54% ±0.39** | 90.25% ±0.53 | 3.2 | 0.129 |
| `snn_d3` | 335,114 | 2e-3, a=2 | **90.47% ±0.44** | 90.39% ±0.56 | 4.5 | 0.101 |
| `snn_d4` | 400,906 | 1e-3, a=2 | **90.69% ±0.24** | 90.27% ±0.37 | 4.8 | 0.128 |
| `snn_w512` | 407,050 | 1e-3, a=1 | **91.20% ±0.21** | 90.62% ±0.33 | 3.0 | 0.091 |
| `snn_w1024` | 814,090 | 1e-3, a=1 | **91.53% ±0.14** | 91.28% ±0.33 | 3.2 | 0.051 |
| `mlp_d1` | 203,530 | 3e-3 | **90.60% ±0.26** | 90.23% ±0.68 | 4.2 | -- |
| `mlp_d2` | 269,322 | 3e-3 | **91.02% ±0.21** | 90.36% ±0.18 | 3.0 | -- |
| `mlp_d3` | 335,114 | 3e-3 | **90.98% ±0.24** | 90.67% ±0.39 | 3.2 | -- |
| `mlp_d4` | 400,906 | 1e-3 | **91.16% ±0.16** | 90.52% ±0.40 | 4.8 | -- |
| `mlp_w512` | 407,050 | 3e-3 | **91.39% ±0.24** | 90.65% ±0.73 | 2.8 | -- |
| `mlp_w1024` | 814,090 | 1e-3 | **92.50% ±0.44** | 92.10% ±0.80 | 4.2 | -- |
| `cnn_d1` | 63,050 | 3e-3 | **89.82% ±0.10** | 89.32% ±0.39 | 5.5 | -- |
| `cnn_d2` | 50,186 | 3e-3 | **93.99% ±0.22** | 93.37% ±0.48 | 3.2 | -- |
| `cnn_d3` | 104,202 | 3e-3 | **95.66% ±0.20** | 95.35% ±0.44 | 4.5 | -- |
| `cnn_d4` | 390,410 | 3e-3 | **96.45% ±0.16** | 96.36% ±0.22 | 4.0 | -- |

Three observations before any statistics:

- Every SNN sits within 0.2–1.0 points of the MLP with its exact widths.
  The neuron model is nearly free at this scale; the architecture class is
  everything.
- `cnn_d2` has **fewer parameters than `cnn_d1`** (the second pool shrinks
  the head more than the new conv adds) and still gains 4.2 points. The conv
  prior is not buying capacity, it is buying the right inductive bias.
- The best dense net of any kind (`mlp_w1024`, 814k params, 92.50%) loses by
  four points to a 104k-parameter three-block CNN.

## Does depth help?

Paired per-seed deltas of best test accuracy against each family's depth-1
net (Student t, two-sided, df=3, critical value 3.182):

| family | step | delta best test acc | t (df=3) | p<0.05? |
| --- | --- | --- | --- | --- |
| SNN | d1 -> d2 | +0.18% | +0.93 | no |
| SNN | d1 -> d3 | +0.11% | +0.45 | no |
| SNN | d1 -> d4 | +0.33% | +2.84 | no |
| MLP | d1 -> d2 | +0.42% | +2.43 | no |
| MLP | d1 -> d3 | +0.38% | +2.15 | no |
| MLP | d1 -> d4 | +0.57% | +3.15 | no |
| CNN | d1 -> d2 | +4.18% | +27.48 | yes |
| CNN | d1 -> d3 | +5.84% | +103.72 | yes |
| CNN | d1 -> d4 | +6.63% | +105.97 | yes |

The SNN's depth ladder is statistically flat, and flatter than the MLP's
already-flat one. Nothing about backpropagating through 20 timesteps of
spikes changes the fact that stacking dense layers on a 28x28 classification
problem mostly re-mixes what one hidden layer already separates. The CNN
column is what an inductive bias looks like when it compounds: each block
earns its keep, with the per-step gain shrinking (4.2, then 1.7, then 0.8)
but never vanishing into seed noise.

Spending the same parameters on width instead:

| family | width | params | best test acc |
| --- | --- | --- | --- |
| SNN | 256 | 203,530 | 90.36% ±0.16 |
| SNN | 512 | 407,050 | 91.20% ±0.21 |
| SNN | 1024 | 814,090 | 91.53% ±0.14 |
| MLP | 256 | 203,530 | 90.60% ±0.26 |
| MLP | 512 | 407,050 | 91.39% ±0.24 |
| MLP | 1024 | 814,090 | 92.50% ±0.44 |

`snn_w512` (407k params) beats `snn_d4` (401k params) by 0.5 points; at 814k
the gap over the ladder's best is 0.8. Width is also *cheaper* for this SNN:
the constant-current encoding computes the 784-to-hidden product once per
sample, so a depth-1 net's per-timestep work is only the hidden-to-output
product, while every extra hidden layer adds a 256x256 product *per
timestep*. Wider-not-deeper wins on accuracy and on wall-clock at once.

For the SNN there is one genuine per-layer effect, though it is not an
accuracy effect: hidden firing rates fall with width (0.161 at 256 down to
0.051 at 1024 — more neurons sharing the same drive each fire less), which
matters if spikes are the energy budget.

## Train loss versus test loss

Losses of the final-epoch frozen model (train loss on the fixed 10k prefix):

| model | train loss (10k eval) | test loss | gap ratio | train acc | test acc |
| --- | --- | --- | --- | --- | --- |
| `snn_d1` | 0.0126 | 0.5812 | 46.2x | 99.57% | 89.63% |
| `snn_d2` | 0.0171 | 0.6028 | 35.3x | 99.41% | 90.25% |
| `snn_d3` | 0.0274 | 0.5223 | 19.0x | 99.14% | 90.39% |
| `snn_d4` | 0.0173 | 0.5451 | 31.4x | 99.43% | 90.27% |
| `snn_w512` | 0.0092 | 0.5770 | 62.5x | 99.67% | 90.62% |
| `snn_w1024` | 0.0077 | 0.5970 | 77.1x | 99.75% | 91.28% |
| `mlp_d1` | 0.0110 | 0.7009 | 63.8x | 99.61% | 90.23% |
| `mlp_d2` | 0.0231 | 0.6888 | 29.9x | 99.27% | 90.36% |
| `mlp_d3` | 0.0218 | 0.5953 | 27.4x | 99.37% | 90.67% |
| `mlp_d4` | 0.0188 | 0.5062 | 27.0x | 99.42% | 90.52% |
| `mlp_w512` | 0.0167 | 0.7050 | 42.2x | 99.53% | 90.65% |
| `mlp_w1024` | 0.0033 | 0.4460 | 134.5x | 99.93% | 92.10% |
| `cnn_d1` | 0.0134 | 0.5914 | 44.3x | 99.70% | 89.32% |
| `cnn_d2` | 0.0122 | 0.4913 | 40.3x | 99.59% | 93.37% |
| `cnn_d3` | 0.0134 | 0.3349 | 24.9x | 99.61% | 95.35% |
| `cnn_d4` | 0.0064 | 0.2387 | 37.3x | 99.80% | 96.36% |

And the trajectories that produce it — test loss by epoch, mean over seeds:

| model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `snn_d1` | 0.457 | 0.403 | 0.379 | 0.387 | 0.401 | 0.406 | 0.435 | 0.471 | 0.490 | 0.507 | 0.518 | 0.551 | 0.542 | 0.583 | 0.581 |
| `snn_d2` | 0.457 | 0.395 | 0.397 | 0.402 | 0.448 | 0.451 | 0.484 | 0.507 | 0.526 | 0.544 | 0.583 | 0.565 | 0.572 | 0.610 | 0.603 |
| `cnn_d1` | 0.501 | 0.444 | 0.421 | 0.405 | 0.418 | 0.424 | 0.426 | 0.439 | 0.457 | 0.473 | 0.502 | 0.535 | 0.544 | 0.585 | 0.591 |
| `cnn_d2` | 0.344 | 0.296 | 0.275 | 0.267 | 0.285 | 0.303 | 0.337 | 0.343 | 0.365 | 0.381 | 0.391 | 0.422 | 0.417 | 0.480 | 0.491 |
| `mlp_d1` | 0.472 | 0.410 | 0.407 | 0.400 | 0.413 | 0.462 | 0.470 | 0.507 | 0.534 | 0.580 | 0.599 | 0.647 | 0.657 | 0.706 | 0.701 |

Train loss (10k eval) by epoch, for the same models:

| model | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `snn_d1` | 0.173 | 0.104 | 0.071 | 0.048 | 0.035 | 0.025 | 0.020 | 0.020 | 0.021 | 0.020 | 0.014 | 0.014 | 0.016 | 0.013 | 0.013 |
| `snn_d2` | 0.171 | 0.100 | 0.069 | 0.048 | 0.044 | 0.031 | 0.028 | 0.027 | 0.027 | 0.021 | 0.020 | 0.017 | 0.017 | 0.016 | 0.017 |
| `cnn_d1` | 0.197 | 0.145 | 0.116 | 0.093 | 0.079 | 0.064 | 0.050 | 0.044 | 0.034 | 0.030 | 0.026 | 0.022 | 0.022 | 0.018 | 0.013 |
| `cnn_d2` | 0.118 | 0.075 | 0.050 | 0.035 | 0.030 | 0.022 | 0.018 | 0.017 | 0.015 | 0.015 | 0.011 | 0.012 | 0.010 | 0.010 | 0.012 |
| `mlp_d1` | 0.185 | 0.112 | 0.079 | 0.056 | 0.038 | 0.033 | 0.024 | 0.025 | 0.023 | 0.022 | 0.018 | 0.016 | 0.017 | 0.017 | 0.011 |

What these two tables say, read together:

- **The regime is memorization, universally.** Train loss falls monotonically
  toward ~0.01 for every family while test loss turns around at epoch 3–5.
  KMNIST-with-no-regularization does not test a model's ceiling; it tests how
  gracefully it overfits. [`kmnist_bptt.md`](kmnist_bptt.md) saw the first
  hints of this at 3 epochs; by 15 it is the whole story.
- **Loss and accuracy disagree about the damage.** Between its best and its
  final epoch, `mlp_d1`'s test loss climbs 75% while its accuracy gives back
  only 0.37 points. The climb is concentrated in confidence on the samples
  the model already gets wrong, not in new mistakes. Early stopping on test
  *loss* here would stop 10 epochs before accuracy peaks stop mattering.
- **Spiking is a mild implicit regularizer, not a cure.** At matched
  architecture (784-256-10), the SNN reaches a lower test-loss minimum than
  the MLP (0.379 vs 0.400) and climbs more slowly afterwards (+53% vs +75% at
  epoch 15), consistent with discrete spikes limiting how precisely the model
  can inflate its logits. But it still ends at 99.6% train accuracy — spiking
  slowed the memorization by a few epochs and softened the loss blow-up; it
  prevented nothing.
- **Only the conv prior improves generalization rather than delaying its
  loss.** The deeper CNNs are the only models whose *test* loss floor
  actually moves down — the per-seed minimum falls 0.401 → 0.264 → 0.204 →
  0.172 across blocks — while their train behavior stays the same as
  everyone else's.
- The gap-ratio column should be read with care: with train losses this close
  to zero the ratio is mostly a train-loss reciprocal ("134x" for `mlp_w1024`
  means it memorized *hardest*, at 99.93% train accuracy). The test-loss
  columns carry the real signal.

## Cost, and the GPU backend this suite forced into existence

The SNN pays for its 20-timestep unroll. On this machine's 12 CPU cores the
depth ladder costs 1.6 / 10.8 / 14.8 / 24.9 s per epoch for d1–d4 — the
hidden-to-hidden products run every timestep, so dense SNN depth is
quadratically unpleasant — which put the full suite at ~2.5 hours. The torch
baselines take ~1 s per epoch on the GPU regardless of depth.

That asymmetry is why the tool now has a CUDA training backend
(`tools/bptt_cuda.cu`, `--gpu`): the same BPTT math batched into per-timestep
cuBLAS GEMMs, with the per-timestep rank-1 weight updates collapsed into one
GEMM per layer over the flattened time-by-batch extent. Per-epoch times drop
to 0.45 / 0.75 / 1.1 / 1.4 s — 3.6x for the already-cheap depth-1 net, 14–18x
where it hurts, 21x at width 1024 — and this document's entire SNN suite runs
in about ten minutes.

Two caveats belong in writing. First, the backend is a *tool*, not a library
feature — the library's coverage and parity gates do not apply to it; its
correctness case is `--mode gputest`, which (a) compares batch gradients
against the CPU library under the soft-spike hook — the same
make-the-model-smooth trick the finite-difference tests use, so no tolerance
has to absorb a spike flipping on a last-bit membrane difference — at
2.8e-7 relative L2 or better across depths 1–3, all six surrogates, and both
reset modes; and (b) trains hard-spiked for two epochs from identical
initialization and shuffle order, landing on the same loss to four decimals.
Second, unlike the simulator's CUDA backend there is no `--fmad=false` and no
bitwise CPU/GPU parity: a GPU run is deterministic and reproduces CPU
epoch-1 metrics to display precision, but it is its own numerical trajectory.
Every SNN number in this document comes from the GPU backend; do not mix
backends within one comparison.

Timing comparisons across families remain apples-to-oranges (C vs PyTorch,
and per-config CPU timings sit below run-to-run drift on this machine), which
is why no per-model wall-clock column appears in the tables above.

## What this points at next

The result is a clean negative with one obvious constructive reading: the
5–6 point SNN-to-CNN gap is an *architecture* gap, not a neuron-model gap.
A convolutional SNN — conv weight sharing in the drive, LIF dynamics and
surrogate BPTT unchanged — is the experiment these tables argue for, and the
GPU backend makes its cost feasible. Second-order follow-ups: dropout or
augmentation for the memorization regime (every family needs it equally),
and an `alpha` sweep for the width controls, which were held at `alpha = 1`.

## Reproducing

```bash
cmake -S . -B build-tools -DSNN_BUILD_TOOLS=ON -DSNN_BUILD_TESTS=ON -DSNN_ENABLE_CUDA=ON
cmake --build build-tools -j
./build-tools/mnist_bptt --data data/kmnist --mode gputest   # verify the backend first
bash scripts/kmnist_depth_experiments.sh                     # SNN suite, ~10 min on a GPU (GPU=0: ~2.5 h on CPU)
bash scripts/kmnist_torch_experiments.sh                     # CNN/MLP suite, ~15 min (PyTorch + CUDA)
python3 scripts/depth_report.py docs/data/kmnist             # regenerate every table above
```

`-DSNN_BUILD_TESTS=ON` matters for `gputest` (it needs the library's
soft-spike hook); the suites themselves run without it.
