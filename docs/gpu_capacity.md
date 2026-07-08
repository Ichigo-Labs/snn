# SNN GPU Capacity — How Large Can We Go?

**GPU:** NVIDIA GeForce RTX 4060 Ti (16 GB) · **VRAM:** 15.999 GiB total, ~14.86 GiB free after CUDA init
**Host:** ~12.4 GiB free system RAM · **Measured:** July 8, 2026
**Question:** how large can these SNNs get along two axes — **neuron count** and **connection (weight) count**.

---

## TL;DR

| Axis | Mode | Held fixed | Maximum | Bottleneck |
|---|---|---:|---:|---|
| **Neurons** | FULL (all on GPU) | 0 connections | **≈ 551 million** | GPU VRAM (29 B/neuron) |
| **Connections** | FULL (all on GPU) | 1,024 neurons | **≈ 1.33 billion** | GPU VRAM (12 B/edge) |
| **Connections** | STREAMING (state on GPU, topology chunked) | small N | **≈ 1.1 billion** | **Host RAM** (12 B/edge), *not* VRAM |

Both FULL-mode figures are hard VRAM ceilings, **independently reproduced 3×** and **cross-checked against the real library API**. Streaming decouples the connection count from VRAM, moving the limit to host storage.

---

## The memory model (verified against `src/snn_cuda.cu`)

Topology is stored in CSR; per-neuron state is separate device arrays. Indices are 64-bit (`snn_size_t = uint64_t`).

```text
row_ptr = 8·(N+1)   col_idx = 8·E   weights = 4·E                              (topology)
voltage/current/next_current/external = 4·N each   refractory = 4·N   spikes = 1·N   (state = 21·N)
```

**FULL mode** keeps everything resident:

```text
VRAM_full(N, E) = 29·N + 12·E + 8   bytes
                  |__ 21·N state + 8·(N+1) row_ptr __| + |__ 12·E topology __|
```

**STREAMING mode** keeps only state + one bounded topology chunk resident:

```text
VRAM_stream(N) ~= 21·N + 8·(chunk_rows+1) + 12·chunk_synapses      (independent of total E)
```

---

## Axis 1 — Neuron count (FULL mode)

| Result | Value |
|---|---:|
| Max neurons (0 connections) | **551,357,768** |
| Device VRAM used | 14.891 GiB |
| Dominant cost | 29 B/neuron (21 B state + 8 B `row_ptr`) |

**Real-API check:** `snn_build_custom_csr` + `snn_cuda_create` successfully created a **540,000,000-neuron** context (free VRAM fell to 0.166 GiB), confirming the ceiling is genuine and reachable through the actual library — not just a raw-`cudaMalloc` proxy.

## Axis 2 — Connection count

### FULL mode (all topology on GPU)

| Result | Value |
|---|---:|
| Max connections (1,024 neurons) | **1,331,031,274** |
| Device VRAM used | 14.875 GiB |
| Dominant cost | 12 B/edge (8 B `col_idx` + 4 B `weights`) |

**Real-API check:** a 200 M-edge network built through the library reported **2.238 GiB** measured device VRAM vs **2.235 GiB** predicted — agreement within CUDA's allocation granularity.

### STREAMING mode (state resident, topology chunked)

This is the library's "massive size / smart VRAM" path, and it changes the answer: **device VRAM becomes independent of the total edge count.** Verified with the real API:

| Network | Chunk cap | Measured device VRAM |
|---|---|---:|
| N=100 k, **E=300 million** | 1 M synapses | **0.016 GiB** |
| N=100 k, **E=300 million** | 8 M synapses | **0.096 GiB** |

300 million connections were held with **16–96 MB** of VRAM. The limit therefore moves to **host RAM** for the CSR (12 B/edge):

| Streaming connection limit | Value |
|---|---:|
| Host-RAM ceiling (12 B/edge, ~12.6 GiB free) | **≈ 1.13 billion** |
| Buildable in one process (2× transient during build) | **≈ 560 million** |

> ⚠️ **Default streaming does _not_ save VRAM.** With `prefer_streaming` and auto-sizing, the chunk expands to fill free VRAM (a 200 M-edge net still used 2.238 GiB). To get the decoupling above you **must bound** `max_stream_synapses` (and optionally `max_stream_rows`).

---

## How this was measured

1. **Analytic VRAM search** — `benchmarks/gpu_capacity.cu` binary-searches the largest `N` and `E` for which every buffer in the FULL-mode layout is successfully `cudaMalloc`'d, and reports the streaming/host-RAM ceilings. Reproduced 3× with identical results.
2. **Real-API validation** — networks were built with `snn_build_custom_csr` and instantiated with `snn_cuda_create`; device usage was measured via `cudaMemGetInfo` deltas and matched the formula (checks above).

```bash
nvcc -O2 benchmarks/gpu_capacity.cu -o /tmp/snn_gpu_capacity && /tmp/snn_gpu_capacity
```

```json
{
  "device": "NVIDIA GeForce RTX 4060 Ti",
  "total_vram_bytes": 17175150592,
  "free_vram_at_start_bytes": 15960375296,
  "full_mode_layout": "29*N + 12*E + 8 bytes",
  "max_neurons_with_zero_connections": 551357768,
  "max_neuron_axis_vram_gib": 14.891,
  "max_connections_with_1024_neurons": 1331031274,
  "max_connection_axis_vram_gib": 14.875,
  "streaming_host_limited_edge_ceiling": 1125130922,
  "streaming_host_limited_edge_buildable_single_process": 562565461
}
```


Related: for neuron ceilings at common average fanouts (8, 16, 32, ... connections/neuron), see [`reasonable_fanout_capacity.md`](reasonable_fanout_capacity.md).

---

## Caveats & uncertainties

- **Ceilings, not operating points.** These are the largest successful allocations. Real runs need headroom for CUDA context, kernel scratch, fragmentation, and other processes — plan for **~90–95%** of these maxima.
- **Runtime-dependent.** Free VRAM after context init was 15,960,375,296 B this run; the successful `cudaMalloc` search is the source of truth, while `cudaMemGetInfo` is a point-in-time estimate.
- **Host RAM is the real streaming limit here.** With only ~12.4 GiB free system RAM, a *single process* can build ~560 M edges, even though the GPU in streaming mode could reference far more if the CSR were memory-mapped or produced incrementally.
- **FULL connection ceiling is not host-buildable in this box.** 1.33 B edges need ~16 GB of host RAM for the CSR before upload; 1.33 B is the GPU-side ceiling. The largest end-to-end FULL build here is host-RAM-bound (~560 M edges).
- The two axes were measured independently (neurons with E=0; connections with small N). A network large on *both* axes shares one budget via `VRAM_full = 29·N + 12·E + 8`.
