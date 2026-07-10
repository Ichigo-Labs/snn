# Can local, unsupervised STDP learn KMNIST features without a teacher?

> **Yes, and it is not noise — but only about a third as far as backprop, and
> the deep stack it runs on actively works against it.** Four LIF layers
> (784-256-256-256-256) trained with nothing but local pair-STDP and a
> winner-take-all gate — no labels reach the weights — produce features that a
> linear (ridge) readout turns into KMNIST classes. Read out the best hidden
> layer and STDP beats a linear classifier on the raw pixels by **+7.33 points**
> (67.17% vs 59.84%, `t = 41.6` over 4 seeds); concatenate all four layers and
> it wins by **+13.77** (73.61%). Every one of those gains clears significance
> comfortably. Local Hebbian plasticity does carve label-relevant structure out
> of unlabeled spikes.
>
> But two ceilings sit right above it. First, **depth is a liability here, not
> an asset.** In the standard "read the last layer" protocol the tool used to
> report, accuracy *falls* monotonically with depth — trained L1 67.17%, L2
> 65.69%, L3 64.25%, L4 61.77% — and the L1-over-L4 gap of **+5.40 points**
> (`t = 12.6`) is as significant as the learning itself. A single random LIF
> layer already preserves the pixels' linear separability almost exactly
> (untrained L1 60.63% vs pixels 59.84%); each further untrained layer throws
> 4-8 points away, and STDP lifts every layer by a near-constant ~6.5 points
> without ever undoing that stacking loss. The layers are *complementary* — the
> concatenation beats any single one — but deeper is never individually better.
> This is the same verdict [`kmnist_snn_vs_cnn.md`](kmnist_snn_vs_cnn.md)
> reached for *supervised* dense SNNs (depth ≈ free), sharpened into a real
> penalty by the fact that an unsupervised layer cannot repair information its
> input already discarded.
>
> Second, **the wall it hits is representation, not overfitting.** The best
> readout memorizes only ~82% of the *training* set; the identical
> 784-256⁴ network trained end-to-end with surrogate BPTT reaches 90.69% test
> and >99% train. So the ~17-point gap to supervised is not STDP generalizing
> worse — it is STDP building a feature space too linearly poor to even fit the
> training labels. That is the opposite regime from the one every supervised
> model in this repository lives in, where the danger is memorizing KMNIST
> outright. Unsupervised local learning here *underfits*; backprop *overfits*.
> The teacher is buying representational richness, and no amount of it leaks in
> through a Hebbian rule.

All numbers regenerate with `python3 scripts/summarize_stdp.py docs/data/kmnist/stdp_final.csv`
from the committed CSV, itself produced by `bash scripts/kmnist_stdp_experiments.sh`
(GPU, ~5 min for the self-test plus 4 seeds).

## The setup

The network is the depth-4 ladder from
[`kmnist_snn_vs_cnn.md`](kmnist_snn_vs_cnn.md), 784-256-256-256-256, driven the
same way (constant pixel current for `T = 20` steps, `beta = 0.95` leak). Every
piece that touches a weight is local and label-free:

- **Competition.** Each layer keeps a top-`k` (k = 32 of 256) winner-take-all
  gate; a neuron spikes only if it is a winner *and* its membrane clears
  threshold. This is what makes an unsupervised layer do anything but saturate.
- **Plasticity.** A standard pair-STDP rule with exponential pre/post traces
  (`trace_decay = 0.5`): coincident pre-before-post pairs potentiate, post-before-pre
  depress (`ltd_ratio = 0.5`). The per-batch eligibility is mean-centered and
  L2-normalized per postsynaptic neuron, and each incoming weight vector is
  renormalized after the step, so the rule moves *directions*, not scale.
- **Homeostasis.** A slow per-neuron threshold adjuster nudges toward a target
  firing rate (`target_rate = 0.08`).
- **Readout.** After training the hidden weights are frozen and a closed-form
  ridge regression maps a low-pass-filtered spike statistic to ten classes.
  The readout is the *only* place a label is ever used, and it is linear, so it
  measures what the features already make linearly separable — it cannot invent
  structure the STDP layers did not build.

The self-test (`--self-test`) pins the mechanism independently of any accuracy:
a single causal spike pair produces pure potentiation, a single anti-causal
pair pure depression, quiet tapes move no weight, the CPU and CUDA correlation
kernels agree to `1e-7`, and a four-layer CUDA update leaves every layer
changed, finite, and spiking while a frozen evaluation touches nothing.

### The controls that make the number mean something

A single "STDP gets 62%" would be almost content-free. Three baselines fix
what it is being compared against, all with the same ridge readout:

- **Raw pixels** — ridge straight off the 784 pixels. The "no features at all"
  floor: **59.84%**.
- **Untrained network** — the same LIF stack with its random QR-orthogonal init,
  never trained. Isolates what the *architecture* contributes for free (a random
  nonlinear projection, i.e. a reservoir / extreme-learning-machine expansion)
  from what *learning* contributes.
- **Per layer, and concatenated** — the original tool read only the last hidden
  layer. Reading every layer is what exposes that this was the worst possible
  choice.

## What the readout sees (KMNIST, 4 seeds, 3 epochs)

Test accuracy, mean ± sd over 4 seeds:

| readout source | untrained | STDP-trained | STDP gain |
| --- | --- | --- | --- |
| raw pixels | 59.84 | — | — |
| hidden L1 | 60.63 | **67.17 ± 0.35** | +6.54 |
| hidden L2 | 56.27 | 65.69 ± 0.61 | +9.42 |
| hidden L3 | 52.38 | 64.25 ± 0.75 | +11.87 |
| hidden L4 (old default) | 47.76 | 61.77 ± 1.09 | +14.01 |
| concat L1–L4 | 66.99 | **73.61 ± 0.83** | +6.62 |

Paired per-seed contrasts (Student t, two-sided, df = 3, critical value 3.182):

| contrast | mean | t | significant? |
| --- | --- | --- | --- |
| STDP learning: trained − untrained, L1 | +6.54% | +14.6 | yes |
| STDP L1 − raw pixels | +7.33% | +41.6 | yes |
| STDP concat − raw pixels | +13.77% | +33.4 | yes |
| **untrained** concat − raw pixels | +7.15% | +23.8 | yes |
| depth cost: trained L1 − L4 | +5.40% | +12.6 | yes |

Two facts jump out of the paired column and neither is subtle.

**Learning is real and roughly layer-independent.** The trained-minus-untrained
gain is +6.5 points at L1 and rises down the stack only because the untrained
baseline is collapsing — STDP adds a near-constant amount of linearly-readable
signal to whatever its input layer preserved. Read as an absolute best-layer
result, STDP beats a raw-pixel linear classifier by 7.3 points with `t = 41.6`.
There is no interpretation of that under which the Hebbian rule is doing nothing.

**A lot of the headline is the architecture, not the learning.** The untrained
concatenation already scores 66.99% — +7.15 over raw pixels — purely as a random
feature expansion. So of the trained concat's +13.77 over pixels, roughly half is
the reservoir the random projection hands you for free and half is STDP. The old
single-number report (L4, 61.77%) hid both effects at once: it sat 2 points over
pixels and looked like a weak win, when in fact it was reading the *worst* of
four layers a much stronger representation had produced.

## Depth is a liability for a linearly-read spiking stack

Line the layers up and the trend is monotone in both the untrained and trained
rows: **shallower is linearly better.** A single random LIF layer (L1 untrained,
60.63%) reproduces the raw pixels' separability to within a rounding error —
top-`k` competition and a leaky membrane are a near-lossless recoding of one
projection. Every layer after that is a fresh random projection *of an
already-compressed code*, and each throws away 4-8 points a linear reader can no
longer recover. STDP shifts the whole curve up by its ~6.5-point learning
increment but does not bend it: trained L4 (61.77%) still trails trained L1
(67.17%) by 5.40 points at `t = 12.6`.

This lands exactly where the supervised depth study did.
[`kmnist_snn_vs_cnn.md`](kmnist_snn_vs_cnn.md) found dense-SNN depth worth ≤ 0.33
points to *backprop* (not significant) — depth is free, neither help nor harm,
because the global gradient can at least route information through the extra
layers untouched. Strip the teacher away and the same dense depth turns from
free to costly: a purely local rule has no mechanism to preserve, across a layer,
information the layer's input already lacks. The one thing depth does buy is
*diversity* — the layers are complementary, so their concatenation (73.61%) beats
the best single layer (67.17%) by 6.4 points — but that is an argument for reading
the whole stack, not for making it deep. The repo's standing recommendation holds
and hardens: for a dense spiking net on 28×28 images, spend parameters on width,
not depth.

## The ceiling is representation, not generalization

Every unregularized supervised model in this repository fails the same way:
it memorizes KMNIST (>99% train accuracy) and the interesting question is how
gracefully its *test* loss degrades. STDP is in a different regime entirely.

| | STDP best layer (L1) | supervised SNN, same 784-256⁴ net |
| --- | --- | --- |
| train accuracy | ~82% | >99% |
| test accuracy | 67.2% | 90.69% |

The STDP readout cannot even fit the training set — 82% train against a
supervised 99%+. So the ~17-point test gap to the matched-architecture BPTT net
(`snn_d4`, 90.69% in [`kmnist_snn_vs_cnn.md`](kmnist_snn_vs_cnn.md)) is not STDP
generalizing worse; it is STDP building a feature space too linearly impoverished
to separate the classes in the first place. The generalization gap it *does* have
(82% train → 67% test) is ordinary and smaller than the supervised nets'. What
the label-carrying gradient buys, and a local Hebbian rule cannot, is a
representation rich enough to overfit — the very thing that gets the supervised
models into their (different) trouble.

And this is not an undertraining artifact. Running one seed to eight epochs
instead of three moves the concatenated readout *down* slightly (74.13% → 73.59%)
and the firing rates converge exactly onto the top-`k` ceiling (0.125 at every
layer). More STDP helps only the shallowest layer (L1 67.4% → 68.7%) and actively
degrades the deepest (L4 61.8% → 56.2%, L3 64.8% → 60.7%): with more passes the
deep layers over-align onto their sparse input and shed the very diversity that
made the concatenation worth reading. Three epochs is already the plateau, and
training past it *sharpens* the depth penalty rather than paying it down.

## The knobs that actually bind

Per-layer physical state after training (mean over seeds):

| layer | firing rate | dead units | ‖ΔW‖ / ‖W₀‖ | threshold |
| --- | --- | --- | --- | --- |
| L1 | 0.125 | 0.0% | 25.7% | 1.26 |
| L2 | 0.124 | 0.0% | 23.5% | 1.25 |
| L3 | 0.122 | 0.0% | 22.2% | 1.24 |
| L4 | 0.121 | 0.0% | 22.0% | 1.22 |

- **The winner-take-all gate sets the sparsity, not the homeostasis.** Firing
  rate pins to ~0.125 = 32/256 — the top-`k` fraction — not to the `target_rate`
  of 0.08 the threshold controller aims at. Because the plasticity schedule
  scales the threshold learning rate down with dataset size (`--update-reference`),
  the homeostat only crept the threshold from 1.0 to ~1.25 over three epochs and
  never caught up to its target. Over this budget, `top_k` *is* the sparsity
  knob; the homeostasis is a slow trim, not the governor. Anyone tuning firing
  rate should reach for `--top-k` first.
- **No neuron dies.** 0.0% dead fraction at every layer — the normalize-per-row
  weight constraint plus WTA keeps all 256 units in play, which is the failure
  mode (a few units winning everything) the QR-orthogonal init was chosen to
  avoid.
- **Weights move a real but bounded amount** — a 22-26% relative change, largest
  at L1 (closest to the strong pixel drive) and tapering with depth, consistent
  with the deeper layers receiving progressively sparser, weaker input to learn
  from.

## What this points at next

The clean reading is that **the limiting reagent is the objective, not the
locality.** A local rule can extract label-relevant structure (the +7.3 points
over pixels proves it) but plateaus where an unlabeled objective plateaus, and
dense depth only makes it worse. Three follow-ups the tables argue for:

- **Give locality the prior that helped the CNN.** The supervised story ended by
  pointing at convolution; the same weight-sharing drive under a local STDP rule
  is the unsupervised analogue, and — unlike dense depth — a convolutional layer
  *does* preserve spatial structure a linear reader can use.
- **Read the whole stack, or read it shallow.** The concatenation is the honest
  headline for this architecture; a width-256 *single* STDP layer is the honest
  cheap option. Deep-and-read-the-top is the one configuration to avoid.
- **A supervised local rule.** The gap is representational, so the interesting
  test is whether a local rule with *some* label signal (a three-factor /
  reward-modulated STDP) closes it, or whether global credit assignment is doing
  something locality structurally cannot.

## Reproducing

The dataset is committed, so this needs no network access; it does need a CUDA
GPU (the tool refuses CPU, like the other experiment backends here).

```bash
python3 tools/kmnist_stdp.py --self-test          # mechanism checks, no data needed
bash scripts/kmnist_stdp_experiments.sh           # self-test + 4 seeds, ~5 min on a GPU
python3 scripts/summarize_stdp.py docs/data/kmnist/stdp_final.csv   # regenerate every table above
```

A single run, showing the per-layer / concat / pixel readouts the tool prints:

```bash
python3 tools/kmnist_stdp.py --data data/kmnist --epochs 3 --seeds 1
```

The readout is deliberately linear and closed-form: it reports what the frozen,
label-free features already separate, and nothing more. `--top-k` is the sparsity
knob; `--stdp-lr`, `--trace-decay`, and `--ltd-ratio` shape the rule; and the
per-layer columns in `stdp_final.csv` are what make the depth penalty visible
rather than hidden behind a single last-layer number.
