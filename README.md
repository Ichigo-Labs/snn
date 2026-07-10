# SNN: C + CUDA Spiking Neural Network library

[![CI](https://github.com/Ichigo-Labs/snn/actions/workflows/ci.yml/badge.svg)](https://github.com/Ichigo-Labs/snn/actions/workflows/ci.yml)

A compact, performance-oriented leaky-integrate-and-fire SNN library in C with an optional CUDA backend.

## Highlights

- 64-bit neuron/synapse indexing (`snn_size_t`) for very large sparse networks.
- CSR/SoA topology (`row_ptr`, `col_idx`, `weights`) for cache-friendly CPU traversal and coalesced GPU uploads.
- CUDA backend with two VRAM policies:
  - **Full**: topology + state resident on the GPU when it fits.
  - **Streaming**: neuron state stays resident while topology is transferred in bounded CSR chunks when topology does not fit the configured VRAM budget.
- Architecture builders:
  - Custom CSR.
  - Feed-forward layered networks, dense or fixed fanout.
  - Random recurrent pools with configurable fanout and weight range.
- LIF hyperparameters are runtime configurable: `dt_ms`, membrane time constant, rest/reset/threshold voltages, input scale, refractory steps.
- Sparse input events (`snn_state_inject_current` / `snn_cuda_inject_current`):
  drive only the active neurons instead of streaming a dense per-neuron buffer
  each step — on CUDA this replaces the dominant per-step host-to-device
  upload with a transfer proportional to input activity.
- Memory-planning API for dry-run sizing before allocating massive networks.
- Training by backpropagation through time with six peak-normalized surrogate
  gradients (`<snn/snn_bptt.h>`), validated against MNIST — see below.
- Optional OpenMP parallelization of the CPU path (`-DSNN_ENABLE_OPENMP=ON`):
  the per-neuron membrane update splits across threads directly, and synaptic
  propagation uses per-thread scatter buffers reduced in fixed thread order —
  results stay reproducible for a fixed thread count, though the summation
  order (and thus last-bit rounding) may differ from the serial build's.

## Build

```bash
cmake -S . -B build -DSNN_ENABLE_CUDA=ON -DSNN_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

If CUDA is disabled or unavailable, the CPU library builds with API-compatible CUDA stubs.
The build type defaults to Release when unset, so the simulation loops are optimized out of the box.

For a multi-core CPU speedup on large networks, enable OpenMP:

```bash
cmake -S . -B build -DSNN_ENABLE_CUDA=ON -DSNN_ENABLE_OPENMP=ON -DSNN_BUILD_TESTS=ON
cmake --build build -j
```

`-DSNN_ENABLE_NATIVE_ARCH=ON` additionally compiles the CPU library with
`-march=native` (wider SIMD for the vectorized membrane update, non-portable
binaries; may relax the bit-exact CPU/GPU spike parity through FMA contraction).

## Training: BPTT with surrogate gradients

`<snn/snn_bptt.h>` trains layered LIF networks by backpropagation through time.
The spike's Heaviside step has a zero derivative almost everywhere and an
infinite one at threshold, so the backward pass substitutes a **surrogate
derivative**; every other edge of the unrolled graph — including the reset term
and the same-timestep cross-layer coupling — is differentiated exactly.

Six surrogates are provided, all **peak-normalized** (`phi(0) == 1` for every
`alpha`), so `alpha` is purely the width of the gradient window and carries no
implicit gain:

| surrogate | `phi(x; alpha)` |
| --- | --- |
| `fast_sigmoid` | `1 / (1 + alpha*abs(x))^2` |
| `atan` | `1 / (1 + (alpha*x)^2)` |
| `sigmoid` | `4*sig(alpha*x)*(1 - sig(alpha*x))` |
| `triangle` | `max(0, 1 - alpha*abs(x))` |
| `gaussian` | `exp(-(alpha*x)^2 / 2)` |
| `rectangular` | `1` when `alpha*abs(x) < 1`, else `0` |

`snn_surrogate_primitive` returns the antiderivative `S` of each `phi` — the
smooth spike function whose exact derivative is the surrogate. It is what gives
a surrogate gradient its meaning (the backward pass computes the exact gradient
of the network in which `H` is replaced by `S`) and it is what the
finite-difference gradient tests differentiate.

The trainable neuron is deliberately **not** `snn_step_cpu`'s: it drops `v_rest`
and the refractory counter and resets by subtraction, leaving a graph that is
differentiable everywhere except at the spike. `snn_bptt_beta_from_lif` bridges
the simulator's `dt_ms`/`membrane_tau_ms` to the trainable decay.

A network is read-only during forward and backward, so training parallelizes
over the minibatch: give each thread a workspace and a gradient accumulator,
then reduce with `snn_bptt_grads_add`.

```c
snn_size_t layers[] = {784, 256, 10};
/* defaults: atan, alpha 2, beta 0.95, threshold 1.0 -- the MNIST-best config */
snn_bptt_config_t cfg = snn_bptt_default_config(layers, 3, /*timesteps=*/20);

snn_bptt_network_t *net = NULL;
snn_bptt_workspace_t *ws = NULL;
snn_bptt_grads_t *grads = NULL;
snn_bptt_optimizer_t *adam = NULL;
snn_bptt_network_create(&cfg, &net);
snn_bptt_workspace_create(net, &ws);
snn_bptt_grads_create(net, &grads);
snn_bptt_optimizer_create(net, 2e-3f, 0.9f, 0.999f, 1e-8f, &adam);

snn_bptt_grads_zero(grads);
for (int i = 0; i < batch_size; ++i) {
    /* static_input=1: one 784-current frame injected at every timestep */
    snn_bptt_forward_backward(net, ws, image[i], 1, label[i], grads, NULL, NULL);
}
snn_bptt_optimizer_step(adam, net, grads, batch_size);
```

### MNIST and Kuzushiji-MNIST

Both datasets are committed (`data/mnist/`, `data/kmnist/` — the original idx.gz
files), so the results below reproduce with no network access.

```bash
cmake -S . -B build-tools -DSNN_BUILD_TOOLS=ON -DSNN_BUILD_TESTS=OFF
cmake --build build-tools -j
./build-tools/mnist_bptt --mode single --hidden 1000 --timesteps 25 --epochs 3
```

A 784-1000-10 network unrolled over 25 steps reaches **~97% test accuracy after
one epoch** and **97.95% ± 0.18 after eight** (8 seeds), at a 5.4% hidden firing
rate, in 10-13 s/epoch on 12 CPU cores.

`atan` is the surrogate to reach for — it is the only one that never ranks worse
than second on accuracy, `alpha` robustness or firing rate, on either MNIST or
the harder Kuzushiji-MNIST. Accuracy alone cannot rank surrogates on either
dataset: every pairwise difference sits inside the seed noise. What separates
them is that compact-support kernels (`triangle`, `rectangular`) are fragile to
the choice of `alpha`, carry ~5x the across-seed variance, and fire 8-28% more
spikes at matched `alpha`.

Beware transplanting `alpha`: its optimum does not transfer between datasets
(`fast_sigmoid` wants 5 on MNIST and 1 on KMNIST). Sweep it.

- [docs/mnist_bptt.md](docs/mnist_bptt.md) — the comparison, the `alpha` sweep,
  and the reset-path and unrolled-depth ablations.
- [docs/kmnist_bptt.md](docs/kmnist_bptt.md) — the same protocol on a dataset ten
  points harder, which replicates the ranking and corrects one MNIST claim.

Kuzushiji-MNIST is committed too (`data/kmnist/`, 20.3 MB) and is a drop-in:
`--data data/kmnist`, no code change.

## Benchmarks

```bash
cmake -S . -B build-bench -DSNN_BUILD_BENCHMARKS=ON -DSNN_BUILD_TESTS=OFF
cmake --build build-bench -j
./build-bench/step_throughput
```

`step_throughput` measures ms/step on an integrate-bound workload (2M neurons,
no synapses) and a propagation-bound one (200k-neuron random pool, fanout 64,
~30% of neurons spiking per step), on CPU and, when a device is present, GPU.

## Coverage

The gcov gate enforces **100% line coverage** on all host-side library code.

CPU-only configuration (measures `src/snn.c`, `src/snn_bptt.c` and the CUDA stub
`src/snn_cuda_stub.c`):

```bash
cmake -S . -B build-coverage -DSNN_ENABLE_CUDA=OFF -DSNN_ENABLE_COVERAGE=ON -DSNN_BUILD_TESTS=ON
cmake --build build-coverage -j
ctest --test-dir build-coverage --output-on-failure
python3 scripts/coverage.py build-coverage 100
```

CUDA configuration (additionally measures the **host-side control flow of the
CUDA backend** `src/snn_cuda.cu`, instrumented via `nvcc -Xcompiler=--coverage`):

```bash
cmake -S . -B build-cudacov -DSNN_ENABLE_CUDA=ON -DSNN_ENABLE_COVERAGE=ON -DSNN_BUILD_TESTS=ON
cmake --build build-cudacov -j
ctest --test-dir build-cudacov --output-on-failure
python3 scripts/coverage.py build-cudacov 100
```

Both configurations report 100% for every gated source file. Coverage of the CUDA
error/edge paths (device allocation failures, driver errors, streaming clamps) is
achieved with a compile-time fault-injection layer that is enabled only under
`SNN_ENABLE_TEST_HOOKS`. The hooks come along with `SNN_BUILD_TESTS=ON` (the
default, so development builds can always run the suite); a production library
without them is a `-DSNN_BUILD_TESTS=OFF` build.

CI (GitHub Actions) runs the CPU-side matrix on every push: Release tests
(serial and OpenMP), the 100% coverage gate, ASan+UBSan (serial and OpenMP),
ThreadSanitizer over the parallel step, a GPU-less nvcc compile of the CUDA
backend, the production configuration (hooks off, benchmarks and examples
built), and a coverage-guided libFuzzer smoke of the builder/step API with a
persisted corpus (`fuzz/fuzz_api.c` — build commands in its header).
The CUDA runtime configurations (GPU tests, the CUDA coverage gate, CPU/GPU
bitwise parity, and `compute-sanitizer`) require a device and remain a local
pre-push gate.

Note on scope: gcov measures the **host** code of the `.cu` translation unit
(control flow, VRAM policy, chunk scheduling, error handling). Code that runs
*on the GPU* (the `__global__` kernels) is not expressed as gcov line counts;
kernel correctness is instead verified by the test suite's CPU-vs-GPU
differential checks, which assert bit-identical spikes across large random-pool
and feed-forward networks in both FULL and STREAMING modes.

## Minimal example

```c
#include <snn/snn.h>

int main(void) {
    snn_size_t layers[] = {128, 256, 10};
    snn_feedforward_config_t cfg = snn_default_feedforward_config(layers, 3);
    cfg.fanout_per_neuron = 32;
    cfg.weight = 0.2f;

    snn_network_t *net = NULL;
    snn_state_t *state = NULL;
    snn_build_feedforward(&cfg, NULL, &net);
    snn_state_create(net, &state);

    float input[394] = {0};
    uint8_t spikes[394] = {0};
    input[0] = 2.0f;
    snn_step_cpu(net, state, input, spikes);

    snn_state_free(state);
    snn_network_free(net);
}
```
