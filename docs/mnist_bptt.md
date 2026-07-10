# BPTT with surrogate gradients on MNIST

> **The best surrogate is `atan`** — `phi(x) = 1 / (1 + (alpha*x)^2)` with
> `alpha = 2`. Not because it is the most accurate: at 8 seeds the four smooth
> surrogates are statistically indistinguishable (97.91–97.98%, every pairwise
> gap under 0.07% and none significant). It wins on the two things that *are*
> separable. It has the **widest usable `alpha` band** — across a 50x range of
> gradient-window widths it never loses more than 0.27% accuracy, against 0.74%
> for `triangle` — and at equal accuracy it is the **sparsest**, firing 20.7%
> fewer spikes than `fast_sigmoid` (paired `t = 10.7` over 8 shared seeds).
>
> A 784-1000-10 network unrolled over 25 steps reaches ~97% test accuracy after
> one epoch and **97.95% ± 0.18** after eight, at a 5.4% hidden firing rate, in
> about 10 s/epoch on 12 CPU cores. The single best run of any configuration was
> 98.24%.
>
> The uncomfortable finding is that **accuracy alone cannot rank surrogates on
> MNIST**, and neither can MNIST rank them on their temporal behaviour: a `T=1`
> unroll — no recurrence at all — already scores 97.48%.

## What is being trained

A fully-connected spiking network, unrolled over `T` timesteps and trained end
to end by backpropagation through time. The neuron is a discrete
leaky-integrate-and-fire unit that resets by subtraction:

```
pre[0][t] = image                        (the same current at every timestep)
pre[j][t] = s[j-1][t]                    (j >= 1, same timestep)
I[j][t]   = W[j] * pre[j][t] + b[j]

hidden:   U[j][t] = beta*U[j][t-1] + I[j][t] - threshold*s[j][t-1]
          s[j][t] = H(U[j][t] - threshold)
output:   U[j][t] = beta*U[j][t-1] + I[j][t]           (no spike, no reset)

logits    z = (1/T) * sum_t U[out][t]
loss      = softmax cross-entropy(z, label)
```

**Encoding** is constant-current ("direct"): the normalized 784-pixel vector is
injected unchanged at every timestep rather than being converted to Poisson
spike trains. This is the standard choice for static images, and it has a
pleasant consequence — the input layer's drive `W[0]*image + b[0]` does not
depend on `t`, so it is computed once instead of `T` times, and its weight
gradient collapses from `T` rank-1 updates into one. On 784-1000-10 that single
matrix-vector product is most of the arithmetic in a training step, which is why
a 30-step unroll costs about 50% more per epoch than a 1-step one rather than
30x (see the depth ablation).

**Readout** is the time-averaged membrane potential of a non-spiking output
layer. Because that layer never fires, the readout is differentiable and no
surrogate is applied to it.

**Backward** is exact everywhere except at the spike, where the Heaviside's
derivative (a Dirac delta) is replaced by a surrogate `phi`. In particular the
gradient does flow back through the `-threshold*s[j][t-1]` reset term and
through the same-timestep coupling between layers. The reset path therefore
runs *through* the surrogate, which is why every surrogate here is
**peak-normalized** so that `phi(0) == 1` regardless of `alpha`: Zenke & Vogels
(2021) show that surrogates whose peak grows with steepness explode or vanish
the gradient once the spike reset is differentiable. With the peak pinned,
`alpha` is purely the *width* of the gradient window, and two shapes can be
compared at one learning rate without the comparison secretly measuring a gain
difference.

| surrogate | `phi(x; alpha)`, at `x = U - threshold` | tail |
| --- | --- | --- |
| `fast_sigmoid` | `1 / (1 + alpha*abs(x))^2` | `~x^-2` |
| `atan` | `1 / (1 + (alpha*x)^2)` | `~x^-2` |
| `sigmoid` | `4*sig(alpha*x)*(1 - sig(alpha*x))` | `~e^-x` |
| `gaussian` | `exp(-(alpha*x)^2 / 2)` | `~e^-x^2` |
| `triangle` | `max(0, 1 - alpha*abs(x))` | compact |
| `rectangular` | `1` if `alpha*abs(x) < 1` else `0` | compact |

## Is the gradient actually right?

Nothing below means anything if the backward pass is wrong, and a spiking
network is unusually good at hiding a wrong gradient: it will still train, just
slightly worse. Three independent checks are in `tests/test_bptt.c`.

1. **Transpose (adjoint) test.** The backward pass must be the exact transpose
   of the forward-mode linearization of the taped forward. The test contains an
   independent tangent (JVP) propagation written in *forward* mode — structurally
   nothing like the library's reverse-mode code — and asserts
   `<dL/dz, J*d> == <dL/dparams, d>` for random directions `d`, across every
   surrogate, both input modes, `detach_reset` on and off, `T = 1` and `T = 6`,
   `beta = 0`, and networks with zero, one and two hidden layers.

2. **Finite differences against a real loss.** Differentiating the hard-spike
   forward is meaningless — it is piecewise constant in the weights, so a
   perturbation either changes nothing or flips a spike. But the surrogate
   backward *is* the exact gradient of the model in which `H` is replaced by
   `S = snn_surrogate_primitive`, the antiderivative of `phi`. A test hook makes
   the forward emit exactly that, so central differences of the cross-entropy
   must reproduce the analytic gradient. This grounds readout, reset path,
   cross-layer coupling and surrogate against an actual scalar loss.

3. **The `layer_count == 2` network** has no hidden layer, no spike and no
   surrogate, so it is exactly differentiable as written. Its finite
   differences validate the readout and the output recurrence with no hook at
   all.

The backward equations were also derived three ways independently (direct chain
rule, reverse-mode graph traversal, and the Lagrangian adjoint method) before
being implemented; all three agreed.

**Mutation testing.** A passing gradient test proves nothing until you know it
can fail. Twelve deliberate bugs were injected into the backward pass — reset
term sign-flipped, reset term deleted, the `beta` membrane carry dropped in
either layer, the `1/T` readout factor rescaled, the surrogate evaluated at the
wrong timestep, the cross-layer adjoint taken from the pre-update `gu`, the
static-input gradient collapsed over the wrong range, the bias gradient
sign-flipped, `out_correct` inverted, the loss scaled by 0.9, and `atan`
silently replaced by a Gaussian. All twelve are caught. The last three were
found *because* they slipped past an earlier version of the suite, which is
what prompted the tests that pin each surrogate's closed form and pin the loss
value and `out_correct` against independent recomputations.

## Results

Every run: Adam (`beta1=0.9`, `beta2=0.999`, `eps=1e-8`), batch 128, membrane
decay `beta = 0.95`, threshold `1.0`, Kaiming-uniform init with gain `0.577`
(i.e. PyTorch's `nn.Linear` default), pixels scaled to `[0, 1]`. Accuracy is on
the full 10 000-image test set; "firing rate" is spikes per hidden neuron per
timestep. Uncertainties are the standard deviation across seeds.

### head to head - 784-1000-10, T=25, 8 epochs, 8 seeds, lr=1e-3, each at its own best alpha

| surrogate | alpha | final test accuracy | best single run | firing rate |
| --- | --- | --- | --- | --- |
| `fast_sigmoid` | 5 | **97.98% ±0.19** | 98.24% | 0.066 |
| `atan` | 2 | **97.95% ±0.18** | 98.07% | 0.054 |
| `gaussian` | 1 | **97.92% ±0.14** | 98.11% | 0.058 |
| `sigmoid` | 2 | **97.91% ±0.15** | 98.09% | 0.061 |
| `triangle` | 1 | **97.83% ±0.18** | 98.13% | 0.068 |
| `rectangular` | 0.5 | **97.82% ±0.22** | 98.16% | 0.055 |

Paired difference against `fast_sigmoid` over the 8 shared seeds (Student t, two-sided, df=7):

| surrogate | mean gap | sd | t | p<0.05? |
| --- | --- | --- | --- | --- |
| `atan` | +0.029% | 0.224 | +0.36 | no |
| `gaussian` | +0.063% | 0.183 | +0.97 | no |
| `sigmoid` | +0.066% | 0.186 | +1.01 | no |
| `triangle` | +0.154% | 0.181 | +2.40 | yes |
| `rectangular` | +0.159% | 0.211 | +2.13 | no |

Test accuracy after each epoch (mean over seeds):

| surrogate | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fast_sigmoid` | 97.04 | 97.55 | 97.79 | 97.79 | 97.84 | 97.79 | 97.91 | 97.98 |
| `atan` | 96.93 | 97.43 | 97.72 | 97.80 | 97.87 | 97.75 | 97.90 | 97.95 |
| `sigmoid` | 96.92 | 97.44 | 97.67 | 97.81 | 97.96 | 97.78 | 97.95 | 97.91 |
| `triangle` | 97.02 | 97.56 | 97.66 | 97.73 | 97.88 | 97.66 | 97.89 | 97.83 |
| `gaussian` | 96.86 | 97.38 | 97.61 | 97.74 | 97.90 | 97.78 | 97.84 | 97.92 |
| `rectangular` | 96.78 | 97.32 | 97.64 | 97.78 | 97.93 | 97.76 | 97.80 | 97.82 |

No surrogate wins on accuracy. The top four span 0.07 percentage points and
every pairwise gap is inside the seed noise; only `triangle` falls significantly
short of the leader, and `rectangular` misses significance only narrowly. Nor is
convergence speed a discriminator — after one epoch the six are spread across
0.26 points, and the ordering there does not match the ordering at eight epochs.

The firing rate does discriminate, sharply and reproducibly. At statistically
identical accuracy `atan` emits **20.7% fewer spikes** than `fast_sigmoid`
(0.0544 ± 0.0027 against 0.0656 ± 0.0022; paired `t = 10.7` over the 8 shared
seeds, df = 7). On a neuromorphic target that is the number that matters, and it
is the number with the largest effect size in this entire document.

Wall-clock per epoch is not reported per surrogate. Repeated timings on this
machine reorder the six — `atan` measured 10.0 s/epoch in one batch and 13.0 in
another — so the arithmetic cost of the surrogate itself sits below run-to-run
drift, even though a Gaussian or a sigmoid evaluates a transcendental where a
boxcar evaluates a compare. Everything here runs at 10-13 s/epoch on 12 cores.

### The `alpha` sweep: width matters, shape barely does

Comparing shapes at one shared `alpha` would just measure slope mismatch, so
each surrogate is swept over its own gradient-window width first. 784-256-10,
`T=20`, 4 epochs, 3 seeds, `lr=2e-3`.

### alpha sweep - 784-256-10, T=20, 4 epochs, 3 seeds, lr=2e-3

| surrogate | a=0.5 | a=1 | a=2 | a=5 | a=10 | a=25 | best a | spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fast_sigmoid` | 97.50 ±0.15 | 97.43 ±0.21 | 97.46 ±0.04 | 97.63 ±0.11 | 97.30 ±0.43 | 97.21 ±0.46 | **5** | 0.42 |
| `atan` | 97.30 ±0.24 | 97.45 ±0.24 | 97.57 ±0.21 | 97.47 ±0.37 | 97.34 ±0.36 | 97.31 ±0.26 | **2** | 0.27 |
| `sigmoid` | 97.35 ±0.19 | 97.43 ±0.24 | 97.52 ±0.18 | 97.29 ±0.37 | 97.38 ±0.21 | 97.22 ±0.18 | **2** | 0.31 |
| `triangle` | 97.44 ±0.15 | 97.59 ±0.17 | 97.34 ±0.42 | 97.16 ±0.32 | 97.06 ±0.18 | 96.85 ±0.40 | **1** | 0.74 |
| `gaussian` | 97.54 ±0.21 | 97.64 ±0.09 | 97.35 ±0.26 | 97.38 ±0.26 | 97.37 ±0.11 | 97.04 ±0.46 | **1** | 0.59 |
| `rectangular` | 97.43 ±0.14 | 97.39 ±0.34 | 97.33 ±0.12 | 97.25 ±0.51 | 96.96 ±0.94 | 96.80 ±0.16 | **0.5** | 0.63 |

Hidden firing rate, averaged over surrogates and seeds: a=0.5 -> 0.121, a=1 -> 0.132, a=2 -> 0.144, a=5 -> 0.155, a=10 -> 0.163, a=25 -> 0.175

Two things fall out, and they are the real result of this document.

**Every surrogate works, and they all work about equally well.** Across 36
configurations the entire spread of final accuracy is 96.80% to 97.64% — under
one point. This is exactly what Zenke & Vogels report: once the peak is pinned,
surrogate-gradient learning is remarkably insensitive to the shape of the
surrogate.

**What separates the shapes is how forgiving they are about `alpha`, and that
tracks their tails.** The `spread` column is the accuracy lost by picking the
worst `alpha` instead of the best:

| tail of `phi` | surrogates | spread over `alpha` |
| --- | --- | --- |
| heavy, `~x^-2` | `atan`, `fast_sigmoid` | 0.27%, 0.42% |
| exponential | `sigmoid` | 0.31% |
| Gaussian | `gaussian` | 0.59% |
| compact support | `rectangular`, `triangle` | 0.63%, 0.74% |

A neuron far from threshold gets *some* gradient from a heavy-tailed surrogate
and exactly *zero* from a compact one. When `alpha` is large the compact kernels
switch off learning for most of the network, so they degrade fastest and their
seed-to-seed variance grows the most (`rectangular` at `alpha=10` has a standard
deviation of 0.94%, four times its own best). Choosing `atan` or `fast_sigmoid`
means being wrong about `alpha` costs you almost nothing.

**Firing rate rises monotonically with `alpha`** for every surrogate, from
0.121 at `alpha=0.5` to 0.175 at `alpha=25`. A narrow gradient window only
rewards neurons that sit near threshold, and the network answers by pushing
more of them across it. If spike sparsity is the objective — and on
neuromorphic hardware it is the objective — a *wide* window is 45% cheaper at
equal accuracy.

### The reset path is worth ~0.3%

The gradient of a spike with respect to its own future membrane potential — the
`-threshold*s[j][t-1]` term — is the piece most often dropped, because
detaching it is cheaper and simpler. It is not free to drop it.

### the reset path - 784-256-10, T=20, 4 epochs, 3 seeds

| surrogate | reset gradient | test accuracy | firing rate |
| --- | --- | --- | --- |
| `fast_sigmoid` | backpropagated | 97.63% ±0.11 | 0.147 |
| `fast_sigmoid` | detached | 97.31% ±0.11 | 0.163 |
| `gaussian` | backpropagated | 97.64% ±0.09 | 0.139 |
| `gaussian` | detached | 97.39% ±0.38 | 0.143 |

Detaching costs about 0.3 points for both surrogates tested and *raises* the
firing rate. This is also the sharpest unit test in the suite: with the
soft-spike hook enabled, the attached gradient matches central differences to
better than `1e-3`, while the detached one is off by more than twenty times
that — the path is real, and it is not cancelling itself out.

### Unrolled depth buys almost nothing on MNIST

### unrolled depth - 784-256-10, gaussian a=1, 4 epochs, 2 seeds

| timesteps | test accuracy | firing rate | s/epoch |
| --- | --- | --- | --- |
| 1 | 97.48% ±0.11 | 0.389 | 1.57 |
| 2 | 97.74% ±0.04 | 0.361 | 1.60 |
| 5 | 97.72% ±0.00 | 0.268 | 1.63 |
| 10 | 97.58% ±0.03 | 0.195 | 1.77 |
| 20 | 97.66% ±0.12 | 0.139 | 2.31 |
| 30 | 97.40% ±0.08 | 0.120 | 2.36 |

`T=1` already reaches 97.48%. A static image injected as a constant current has
no temporal structure to integrate, so the recurrence has nothing to do; what
`T` really buys is a finer rate code at the output and, as the firing column
shows, a much sparser one (0.389 spikes/neuron/step at `T=1` versus 0.120 at
`T=30`). Accuracy peaks around `T=2` and slowly declines — at `T=30` the reset
subtractions accumulate faster than the readout averages them.

The honest reading is that **MNIST does not exercise BPTT's temporal
dimension**; it exercises the surrogate. That is a good reason to be suspicious
of surrogate rankings drawn only from MNIST, and a good reason to report the
`alpha` robustness above rather than a single accuracy number. It is also why
the depth ablation is cheap: because the input is static, going from `T=1` to
`T=30` costs about 50% more time per epoch rather than 30x. (Those six timings
were taken in one contiguous run, so the monotone trend is meaningful even
though absolute wall-clock drifts between batches on this machine.)

### The ranking is not a learning-rate artifact

### learning-rate sensitivity - 784-256-10, T=20, 3 epochs, 2 seeds

| surrogate | lr=0.0005 | lr=0.001 | lr=0.002 | lr=0.004 |
| --- | --- | --- | --- | --- |
| `gaussian` | 97.31% ±0.13 | 97.69% ±0.13 | 97.42% ±0.06 | 96.32% ±0.08 |
| `rectangular` | 97.17% ±0.12 | 97.64% ±0.04 | 97.44% ±0.03 | 96.71% ±0.05 |

Both surrogates peak at `lr=1e-3` and fall off identically on either side, so
the head-to-head learning rate is at the optimum for both and the ordering is
not an artifact of picking one lr that happens to suit one shape.

## Reproducing

The dataset is committed (`data/mnist/`, 11.6 MB, checksums in its README), so
this needs no network access.

```bash
cmake -S . -B build-tools -DSNN_BUILD_TOOLS=ON -DSNN_BUILD_TESTS=OFF
cmake --build build-tools -j
bash scripts/mnist_experiments.sh          # ~45 min on 12 cores
python3 scripts/mnist_report.py docs/data/*.csv
```

`scripts/mnist_report.py` regenerates every table above from the raw per-epoch
records in `docs/data/*.csv`. A single configuration:

```bash
./build-tools/mnist_bptt --mode single --hidden 1000 --timesteps 25 --epochs 8 \
    --surrogate atan --alpha 2 --lr 1e-3
```

Training parallelizes over the minibatch: each thread owns a workspace and a
gradient accumulator, and the accumulators are reduced in a fixed thread order,
so the parameter update is bit-reproducible for a fixed thread count. (The
reported loss uses an OpenMP reduction whose order is unspecified, so it can
differ in the last decimal without the training trajectory differing at all.)
