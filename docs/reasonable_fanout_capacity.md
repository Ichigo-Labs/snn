# Max Neurons with Reasonable Connection Counts

**GPU:** NVIDIA GeForce RTX 4060 Ti 16 GB  
**Question:** if each neuron has a reasonable average number of outgoing connections, how many neurons fit?

## Answer: FULL mode, all state + weights on GPU

For an average out-degree / fanout `F`, the CUDA FULL footprint is:

```text
VRAM = (29 + 12F) · N + 8 bytes
```

Using the measured FULL-mode GPU budget from this card (**14.875 GiB**), the maximum neuron counts are:

| Avg. connections / neuron | Max neurons | Total connections | 90% headroom target |
|---:|---:|---:|---:|
| 8 | **127,779,239** | 1,022,233,912 | 115,001,315 |
| 16 | **72,273,325** | 1,156,373,200 | 65,045,993 |
| 32 | **38,674,104** | 1,237,571,328 | 34,806,693 |
| 64 | **20,040,658** | 1,282,602,112 | 18,036,592 |
| 128 | **10,206,009** | 1,306,369,152 | 9,185,408 |
| 256 | **5,150,727** | 1,318,586,112 | 4,635,654 |

**Best practical one-line summary:** on this 16 GB RTX 4060 Ti, expect roughly
**20M neurons at 64 connections/neuron** or **39M neurons at 32 connections/neuron** if the whole network lives on the GPU.

## Streaming mode: more edges, but host-RAM-bound

With bounded CUDA STREAMING, GPU memory is mostly:

```text
21 · N + stream_chunk_bytes
```

Total edges live in host CSR memory instead:

```text
host_resident ≈ (8 + 12F) · N bytes
current_builder_peak ≈ 2 · (8 + 12F) · N bytes
```

On this machine's current host-RAM headroom (~12.1 GiB free), streaming-mode neuron limits are therefore:

| Avg. connections / neuron | Host-resident ceiling | Buildable now with current 2× builder copy |
|---:|---:|---:|
| 8 | 125,308,691 | **62,654,345** |
| 16 | 65,160,519 | **32,580,259** |
| 32 | 33,245,163 | **16,622,581** |
| 64 | 16,793,948 | **8,396,974** |
| 128 | 8,440,481 | **4,220,240** |
| 256 | 4,231,202 | **2,115,601** |

So, on this box, **FULL mode gives the highest neuron count** for typical sparse fanouts because VRAM is larger than available host build headroom. Streaming is still the right path when the goal is **very high total connection count** at a smaller neuron count, or when the topology is generated/memory-mapped instead of duplicated during build.

## Benchmark command

```bash
nvcc -O2 benchmarks/gpu_fanout_capacity.cu -o /tmp/snn_gpu_fanout_capacity
/tmp/snn_gpu_fanout_capacity
```

Raw output from the run:

```text
fanout,full_max_neurons,full_max_edges,full_90pct_neurons,stream_host_resident_neurons,stream_buildable_neurons
1,389570853,389570853,350613767,651605196,325802598
2,301366131,602732262,271229518,407253248,203626624
4,207433830,829735320,186690447,232716141,116358070
8,127779239,1022233912,115001315,125308691,62654345
16,72273325,1156373200,65045993,65160519,32580259
32,38674104,1237571328,34806693,33245163,16622581
64,20040658,1282602112,18036592,16793948,8396974
128,10206009,1306369152,9185408,8440481,4220240
256,5150727,1318586112,4635654,4231202,2115601
512,2587462,1324780544,2328716,2118352,1059176
1024,1296777,1327899648,1167099,1059865,529932
```

## Caveat

These are capacity ceilings, not recommended production targets. The **90% headroom target** is the better planning number for long runs, larger kernels, fragmentation, or other GPU processes.
