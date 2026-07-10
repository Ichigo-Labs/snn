# Does the surrogate ranking survive a harder dataset?

> **No.** Ten points of headroom do not make accuracy discriminate. The
> best-to-worst spread widens from 0.16% on MNIST to 0.41% on KMNIST, but the
> seed noise widens with it, so *not one* pairwise difference reaches
> significance on either dataset — the pooled smooth-versus-compact contrast
> gives `t = 1.94` on MNIST and `t = 1.97` on KMNIST. Headroom scales signal and
> noise together.
>
> Two things do sharpen, and both vindicate the MNIST conclusion while changing
> its reasoning:
>
> - **Reliability.** Compact-support surrogates (`triangle`, `rectangular`) have
>   **4.8x the across-seed accuracy variance** of the smooth ones on KMNIST
>   (sd 0.61-0.74% against 0.25-0.37%). On MNIST that ratio was 1.3x — invisible.
> - **Sparsity is a shape effect after all, but not the one I claimed.** At
>   *matched* `alpha`, compact-support kernels fire 8-28% more than smooth ones
>   on both datasets (pooled `t = 9.6` / `6.6`).
>
> And one published claim is **wrong and is corrected here**: `mnist_bptt.md`
> attributed `atan`'s 20.7% sparsity advantage over `fast_sigmoid` to its shape.
> It is an `alpha` artifact. At matched `alpha` the two are indistinguishable
> (`t = -0.85` on MNIST, `t = +0.42` on KMNIST); the gap existed only because
> `fast_sigmoid`'s best `alpha` on MNIST was 5 and `atan`'s was 2, and a narrower
> gradient window drives more neurons across threshold.
>
> `atan` remains the recommendation, on firmer ground: it is the only surrogate
> that never ranks worse than **second on any axis — accuracy, `alpha`
> robustness, firing rate — on either dataset**. But the choice that actually
> matters is smooth versus compact support, not which smooth kernel.

## Why KMNIST

[`mnist_bptt.md`](mnist_bptt.md) ended with an uncomfortable admission: on MNIST
**accuracy cannot rank surrogate gradients**. Six of them landed inside 0.16
percentage points, eight seeds could not separate the top four, and the ranking
had to be made on two secondary criteria — tolerance to the `alpha` window width
and spike sparsity. That is a defensible answer, but it is an answer forced by a
saturated benchmark, and it invites the obvious objection: *maybe the surrogates
really do differ and MNIST simply cannot see it.*

Kuzushiji-MNIST is the cheapest way to test that objection. It is the same shape
(28x28, ten classes, 60 000 train / 10 000 test), the same IDX file format, the
same filenames, and the same cost per epoch — the training tool reads it with
**no code change at all**, just `--data data/kmnist`. But it is about ten points
harder, and harder for a real reason: each of its ten classes collapses several
distinct historical forms of one cursive hiragana character, so the within-class
variation is genuinely larger rather than merely noisier.

Same config, same binary, three epochs of 784-256-10:

| dataset | test accuracy | train loss | test loss |
| --- | --- | --- | --- |
| MNIST | 97.71% | 0.061 | 0.079 |
| KMNIST | 88.01% | 0.105 | 0.432 |

Note the loss columns, not just the accuracy. KMNIST's test loss is 5.5x its
train loss and *rises* between epochs 2 and 3 while accuracy still improves: this
model overfits KMNIST almost immediately, which MNIST never provoked. So the
harder dataset does not merely stretch the accuracy axis — it changes the regime.

The protocol below is byte-for-byte the one used on MNIST
(`scripts/kmnist_experiments.sh` mirrors `scripts/mnist_experiments.sh`), so the
two are directly comparable. The only thing not carried over is each surrogate's
`alpha`, which is re-derived from KMNIST's own sweep rather than inherited — and
that turns out to matter.

## The optimal `alpha` does not transfer

### alpha sweep - 784-256-10, T=20, 4 epochs, 3 seeds, lr=2e-3

| surrogate | a=0.5 | a=1 | a=2 | a=5 | a=10 | a=25 | best a | spread |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fast_sigmoid` | 88.66 ±0.68 | 88.80 ±0.25 | 88.33 ±0.49 | 87.93 ±0.62 | 87.77 ±0.36 | 87.60 ±0.57 | **1** | 1.20 |
| `atan` | 88.67 ±0.28 | 89.18 ±0.41 | 88.64 ±0.73 | 88.20 ±0.06 | 88.13 ±0.81 | 87.60 ±0.38 | **1** | 1.58 |
| `sigmoid` | 88.47 ±0.76 | 88.67 ±0.51 | 88.46 ±0.91 | 88.22 ±0.47 | 88.03 ±0.50 | 86.94 ±0.51 | **1** | 1.73 |
| `triangle` | 88.40 ±1.14 | 88.30 ±0.41 | 87.72 ±0.60 | 87.16 ±0.34 | 86.96 ±0.46 | 85.98 ±0.53 | **0.5** | 2.41 |
| `gaussian` | 88.25 ±0.76 | 88.58 ±0.81 | 88.39 ±0.35 | 87.46 ±1.12 | 87.23 ±0.36 | 86.41 ±0.67 | **1** | 2.17 |
| `rectangular` | 88.58 ±0.66 | 88.51 ±0.43 | 87.87 ±0.28 | 87.28 ±0.47 | 87.01 ±0.10 | 85.84 ±0.71 | **0.5** | 2.75 |

Hidden firing rate, averaged over surrogates and seeds: a=0.5 -> 0.113, a=1 -> 0.117, a=2 -> 0.122, a=5 -> 0.130, a=10 -> 0.141, a=25 -> 0.162

Every surrogate wants a **wider** gradient window on the harder dataset. Four of
the six move their optimum, and `fast_sigmoid` moves it by a factor of five:

| surrogate | best `alpha` on MNIST | best `alpha` on KMNIST |
| --- | --- | --- |
| `fast_sigmoid` | 5 | 1 |
| `atan` | 2 | 1 |
| `sigmoid` | 2 | 1 |
| `triangle` | 1 | 0.5 |
| `gaussian` | 1 | 1 |
| `rectangular` | 0.5 | 0.5 |

This is the single most practically useful result here. A narrow window only
delivers gradient to neurons already sitting near threshold; when the task is
harder and the decision boundary is less well served by the initial weights,
more of the network needs to be reachable. Anyone who copies a slope constant
out of a paper — or out of this repository's MNIST tables — onto a new dataset
is very likely leaving accuracy on the floor.

The ranking by `alpha` robustness replicates almost perfectly, and it lines up
with the weight of each surrogate's tail:

| `phi` tail | surrogate | MNIST spread | KMNIST spread |
| --- | --- | --- | --- |
| `~x^-2` | `atan` | 0.27% | 1.58% |
| `~x^-2` | `fast_sigmoid` | 0.42% | 1.20% |
| `~e^-x` | `sigmoid` | 0.31% | 1.73% |
| `~e^-x^2` | `gaussian` | 0.59% | 2.17% |
| compact | `triangle` | 0.74% | 2.41% |
| compact | `rectangular` | 0.63% | 2.75% |

The three smooth kernels take the top three places on **both** datasets, the
Gaussian is fourth on **both**, and the two compact kernels are last on **both**
(Spearman `rho = +0.77` across the six). The order *within* the smooth group
shuffles between datasets, so nothing should be read into `atan` beating
`fast_sigmoid` on one and losing on the other.

The mechanism is the same one the MNIST document argued, now visible three times
larger: a neuron far from threshold receives *some* gradient from a heavy-tailed
surrogate and exactly *zero* from a compact one. Narrow the window and the
compact kernels stop teaching most of the network.

## Head to head, with room to breathe

### head to head - 784-1000-10, T=25, 8 epochs, 8 seeds, lr=1e-3, each at its own best alpha

| surrogate | alpha | final test accuracy | best single run | firing rate |
| --- | --- | --- | --- | --- |
| `atan` | 1 | **90.76% ±0.30** | 91.17% | 0.044 |
| `fast_sigmoid` | 1 | **90.72% ±0.25** | 91.10% | 0.046 |
| `sigmoid` | 1 | **90.63% ±0.37** | 91.17% | 0.049 |
| `triangle` | 0.5 | **90.44% ±0.61** | 91.51% | 0.062 |
| `gaussian` | 1 | **90.39% ±0.54** | 91.21% | 0.059 |
| `rectangular` | 0.5 | **90.35% ±0.74** | 90.95% | 0.055 |

Paired difference against `atan` over the 8 shared seeds (Student t, two-sided, df=7):

| surrogate | mean gap | sd | t | p<0.05? |
| --- | --- | --- | --- | --- |
| `fast_sigmoid` | +0.037% | 0.488 | +0.22 | no |
| `sigmoid` | +0.132% | 0.372 | +1.01 | no |
| `triangle` | +0.316% | 0.786 | +1.14 | no |
| `gaussian` | +0.371% | 0.615 | +1.71 | no |
| `rectangular` | +0.411% | 0.694 | +1.68 | no |

Test accuracy after each epoch (mean over seeds):

| surrogate | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `fast_sigmoid` | 87.22 | 89.60 | 90.17 | 90.75 | 90.62 | 90.78 | 90.99 | 90.72 |
| `atan` | 86.64 | 89.35 | 90.05 | 90.57 | 90.59 | 90.51 | 90.81 | 90.76 |
| `sigmoid` | 86.41 | 89.33 | 90.02 | 90.34 | 90.39 | 90.73 | 90.66 | 90.63 |
| `triangle` | 87.90 | 89.85 | 90.32 | 90.59 | 90.34 | 90.48 | 90.55 | 90.44 |
| `gaussian` | 87.81 | 89.73 | 90.09 | 90.17 | 90.49 | 90.57 | 90.25 | 90.39 |
| `rectangular` | 87.57 | 89.80 | 90.24 | 90.55 | 90.31 | 90.61 | 90.70 | 90.35 |

Nothing here is significant. The largest paired gap, `rectangular` trailing
`atan` by 0.41 points, reaches only `t = 1.68` against a critical value of 2.365.
Pooling the three smooth surrogates against the two compact ones gives
`+0.307% ± 0.441`, `t = 1.97` — still short, and almost exactly the `t = 1.94`
the same contrast produced on MNIST at a quarter of the effect size. The
variance grew in step with the signal.

What did emerge is **reliability**. Look at the standard deviations rather than
the means:

| | smooth (`fast_sigmoid`, `atan`, `sigmoid`) | compact (`triangle`, `rectangular`) |
| --- | --- | --- |
| seed sd, MNIST | 0.15-0.19% | 0.18-0.22% |
| seed sd, KMNIST | 0.25-0.37% | 0.61-0.74% |

On KMNIST the compact kernels carry **4.8x the variance** (sd ratio 2.2x). They
are not less accurate on average; they are less *dependable*, and a single lucky
seed of `triangle` produced the best run of any configuration in this document
(91.51%). That is the signature of a surrogate that switches off learning for
whichever neurons initialization happens to place outside its window — and it is
exactly the same defect as the `alpha` fragility above, seen from another angle.

Firing rate discriminates too, and this is where MNIST misled me. At each
surrogate's own best `alpha`, `atan` fires 0.044 against `triangle`'s 0.062. But
the honest test is at *matched* `alpha`, where the smooth-versus-compact split
survives (compact fires 8-28% more, pooled `t = 9.6` on MNIST and `6.6` on
KMNIST) while the `atan`-versus-`fast_sigmoid` gap evaporates entirely
(`t = -0.85` and `t = +0.42`). Firing rate is set by the *width* of the gradient
window and by whether the tail is compact — not by which smooth shape you pick.

## The reset path, again

The MNIST result was that backpropagating through the spike's own reset term is
worth about 0.3 points. If that is a property of the gradient rather than of the
dataset, it should reappear.

### the reset path - 784-256-10, T=20, 4 epochs, 3 seeds

| surrogate | reset gradient | test accuracy | firing rate |
| --- | --- | --- | --- |
| `atan` | backpropagated | 89.18% ±0.41 | 0.110 |
| `atan` | detached | 88.86% ±0.37 | 0.119 |
| `fast_sigmoid` | backpropagated | 88.80% ±0.25 | 0.109 |
| `fast_sigmoid` | detached | 88.75% ±0.46 | 0.120 |

It does, but the MNIST document over-read it. Per dataset and per surrogate the
accuracy effect is a consistent `+0.05%` to `+0.32%`, yet with three seeds per
cell none of the four cells clears significance on its own. Pooling all **twelve
paired runs** across both datasets and both surrogates:

| effect | mean | t (df=11) | sign test |
| --- | --- | --- | --- |
| accuracy, attached − detached | **+0.233%** | +2.43, significant | 9/12 positive |
| firing rate, detached − attached | **+0.0106** | +5.62, significant | **12/12 positive** |

So the reset path is worth roughly a fifth of a point of accuracy — the MNIST
document's "about 0.3 points" was the point estimate of an under-powered cell,
and the pooled figure is smaller. The robust effect is on **sparsity**: cutting
the gradient through a spike's own reset makes the network fire more, in every
one of the twelve runs.

That the path carries real gradient is not in question either way — the
finite-difference test in `tests/test_bptt.c` proves it as a matter of
arithmetic, independent of whether it helps accuracy.

## Reproducing

Both datasets are committed, so this needs no network access.

```bash
cmake -S . -B build-tools -DSNN_BUILD_TOOLS=ON -DSNN_BUILD_TESTS=OFF
cmake --build build-tools -j
bash scripts/kmnist_experiments.sh        # ~90 min on 12 cores
python3 scripts/mnist_report.py docs/data/kmnist/*.csv
```

A single run, showing the drop-in property — the same binary, one flag:

```bash
./build-tools/mnist_bptt --data data/kmnist --mode single \
    --hidden 1000 --timesteps 25 --epochs 8 --surrogate atan --alpha 1 --lr 1e-3
```

`scripts/kmnist_experiments.sh` re-derives each surrogate's best `alpha` from its
own sweep rather than hardcoding MNIST's, which — per the table above — is the
whole point.
