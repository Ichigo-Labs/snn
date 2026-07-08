#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#if defined(__linux__)
#include <unistd.h>
#endif

struct AllocSet {
    void *row_ptr;
    void *col_idx;
    void *weights;
    void *voltage;
    void *current;
    void *next_current;
    void *external;
    void *refractory;
    void *spikes;
};

static void free_set(struct AllocSet *a) {
    cudaFree(a->row_ptr);
    cudaFree(a->col_idx);
    cudaFree(a->weights);
    cudaFree(a->voltage);
    cudaFree(a->current);
    cudaFree(a->next_current);
    cudaFree(a->external);
    cudaFree(a->refractory);
    cudaFree(a->spikes);
    *a = (struct AllocSet){0};
}

static int alloc_bytes(void **ptr, uint64_t bytes) {
    *ptr = 0;
    if (bytes == 0) return 1;
    return cudaMalloc(ptr, (size_t)bytes) == cudaSuccess;
}

static int try_full(uint64_t neurons, uint64_t edges) {
    struct AllocSet a = {0};
    int ok = 0;
    if (!alloc_bytes(&a.row_ptr, (neurons + 1ull) * 8ull)) goto done;
    if (!alloc_bytes(&a.col_idx, edges * 8ull)) goto done;
    if (!alloc_bytes(&a.weights, edges * 4ull)) goto done;
    if (!alloc_bytes(&a.voltage, neurons * 4ull)) goto done;
    if (!alloc_bytes(&a.current, neurons * 4ull)) goto done;
    if (!alloc_bytes(&a.next_current, neurons * 4ull)) goto done;
    if (!alloc_bytes(&a.external, neurons * 4ull)) goto done;
    if (!alloc_bytes(&a.refractory, neurons * 4ull)) goto done;
    if (!alloc_bytes(&a.spikes, neurons)) goto done;
    ok = 1;
done:
    free_set(&a);
    cudaGetLastError(); /* clear any OOM status from failed cudaMalloc */
    return ok;
}

static uint64_t bsearch_axis(uint64_t fixed_neurons, uint64_t fixed_edges,
                             int search_neurons, uint64_t hi) {
    uint64_t lo = 0;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo + 1ull) / 2ull;
        uint64_t n = search_neurons ? mid : fixed_neurons;
        uint64_t e = search_neurons ? fixed_edges : mid;
        if (try_full(n, e)) {
            lo = mid;
        } else {
            hi = mid - 1ull;
        }
    }
    return lo;
}

static double gib(uint64_t bytes) {
    return (double)bytes / (1024.0 * 1024.0 * 1024.0);
}

/* Best-effort host free-RAM query (Linux). Returns 0 if unavailable. */
static uint64_t host_free_bytes(void) {
#if defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page > 0) {
        return (uint64_t)pages * (uint64_t)page;
    }
#endif
    return 0;
}

int main(void) {
    int device = 0;
    cudaDeviceProp prop;
    size_t free0 = 0, total0 = 0;

    cudaError_t st = cudaGetDevice(&device);
    if (st != cudaSuccess) {
        fprintf(stderr, "cudaGetDevice failed: %s\n", cudaGetErrorString(st));
        return 2;
    }
    st = cudaGetDeviceProperties(&prop, device);
    if (st != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(st));
        return 2;
    }
    cudaFree(0); /* initialize context */
    st = cudaMemGetInfo(&free0, &total0);
    if (st != cudaSuccess) {
        fprintf(stderr, "cudaMemGetInfo failed: %s\n", cudaGetErrorString(st));
        return 2;
    }

    /* CUDA FULL mode exact layout:
       row_ptr 8*(N+1), col_idx 8*E, weights 4*E,
       state: voltage/current/next_current/external/refractory/spikes = 21*N. */
    uint64_t neuron_hi = (uint64_t)(free0 / 29ull) + 1000000ull;
    uint64_t edge_hi = (uint64_t)(free0 / 12ull) + 1000000ull;

    uint64_t max_neurons = bsearch_axis(0, 0, 1, neuron_hi);
    uint64_t max_edges_at_1024_neurons = bsearch_axis(1024, 0, 0, edge_hi);

    size_t free1 = 0, total1 = 0;
    cudaMemGetInfo(&free1, &total1);

    uint64_t neuron_axis_bytes = 29ull * max_neurons + 8ull;
    uint64_t edge_axis_bytes = 29ull * 1024ull + 8ull + 12ull * max_edges_at_1024_neurons;

    /* CUDA STREAMING mode: neuron state (21*N) stays resident; topology is
       transferred in a bounded chunk of `chunk_syn` edges + (chunk_rows+1)
       row offsets. Device VRAM is therefore decoupled from the *total* edge
       count E, so the connection ceiling in streaming mode is set by HOST
       storage for the CSR (col_idx 8*E + weights 4*E = 12*E), not VRAM. */
    uint64_t hfree = host_free_bytes();
    uint64_t stream_host_edge_ceiling = hfree ? (hfree / 12ull) : 0ull;
    /* Buildable in a single process (transient duplicate during build): ~half. */
    uint64_t stream_host_edge_buildable = hfree ? (hfree / 24ull) : 0ull;

    printf("{\n");
    printf("  \"device\": \"%s\",\n", prop.name);
    printf("  \"total_vram_bytes\": %llu,\n", (unsigned long long)total0);
    printf("  \"free_vram_at_start_bytes\": %llu,\n", (unsigned long long)free0);
    printf("  \"free_vram_at_end_bytes\": %llu,\n", (unsigned long long)free1);
    printf("  \"full_mode_layout\": \"29*N + 12*E + 8 bytes\",\n");
    printf("  \"max_neurons_with_zero_connections\": %llu,\n", (unsigned long long)max_neurons);
    printf("  \"max_neuron_axis_vram_gib\": %.3f,\n", gib(neuron_axis_bytes));
    printf("  \"max_connections_with_1024_neurons\": %llu,\n", (unsigned long long)max_edges_at_1024_neurons);
    printf("  \"max_connection_axis_vram_gib\": %.3f,\n", gib(edge_axis_bytes));
    printf("  \"streaming_mode_note\": \"state 21*N resident; topology chunked; device VRAM independent of total E\",\n");
    printf("  \"host_free_bytes\": %llu,\n", (unsigned long long)hfree);
    printf("  \"streaming_host_limited_edge_ceiling\": %llu,\n", (unsigned long long)stream_host_edge_ceiling);
    printf("  \"streaming_host_limited_edge_buildable_single_process\": %llu\n", (unsigned long long)stream_host_edge_buildable);
    printf("}\n");
    return 0;
}
