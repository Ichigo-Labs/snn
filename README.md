# SNN: C + CUDA Spiking Neural Network library

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
- Memory-planning API for dry-run sizing before allocating massive networks.
- Optional OpenMP parallelization of the CPU membrane-integration phase
  (`-DSNN_ENABLE_OPENMP=ON`); the CPU path stays bit-exact and deterministic
  because only the race-free per-neuron phase is parallelized.

## Build

```bash
cmake -S . -B build -DSNN_ENABLE_CUDA=ON -DSNN_BUILD_TESTS=ON
cmake --build build -j
ctest --test-dir build --output-on-failure
```

If CUDA is disabled or unavailable, the CPU library builds with API-compatible CUDA stubs.

For a multi-core CPU speedup on large networks, enable OpenMP:

```bash
cmake -S . -B build -DSNN_ENABLE_CUDA=ON -DSNN_ENABLE_OPENMP=ON -DSNN_BUILD_TESTS=ON
cmake --build build -j
```

## Coverage

The gcov gate enforces **100% line coverage** on all host-side library code.

CPU-only configuration (measures `src/snn.c` and the CUDA stub `src/snn_cuda_stub.c`):

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
`SNN_ENABLE_TEST_HOOKS` and compiled out of production builds.

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
