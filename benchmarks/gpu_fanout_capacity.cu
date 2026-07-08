#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#if defined(__linux__)
#include <unistd.h>
#endif

struct AllocSet {
    void *row_ptr, *col_idx, *weights;
    void *voltage, *current, *next_current, *external, *refractory, *spikes;
};

static void free_set(struct AllocSet *a) {
    cudaFree(a->row_ptr); cudaFree(a->col_idx); cudaFree(a->weights);
    cudaFree(a->voltage); cudaFree(a->current); cudaFree(a->next_current);
    cudaFree(a->external); cudaFree(a->refractory); cudaFree(a->spikes);
    *a = (struct AllocSet){0};
}

static int alloc_bytes(void **p, uint64_t bytes) {
    *p = 0;
    return bytes == 0 || cudaMalloc(p, (size_t)bytes) == cudaSuccess;
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
    cudaGetLastError();
    return ok;
}

static uint64_t bsearch_edges_at_1024(uint64_t hi) {
    uint64_t lo = 0;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo + 1ull) / 2ull;
        if (try_full(1024ull, mid)) lo = mid;
        else hi = mid - 1ull;
    }
    return lo;
}

static uint64_t host_free_bytes(void) {
#if defined(_SC_AVPHYS_PAGES) && defined(_SC_PAGESIZE)
    long pages = sysconf(_SC_AVPHYS_PAGES);
    long page = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page > 0) return (uint64_t)pages * (uint64_t)page;
#endif
    return 0;
}

static double gib(uint64_t bytes) { return (double)bytes / (1024.0 * 1024.0 * 1024.0); }

int main(void) {
    const uint64_t fanouts[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    cudaDeviceProp prop;
    size_t free0 = 0, total0 = 0;
    cudaFree(0);
    cudaGetDeviceProperties(&prop, 0);
    cudaMemGetInfo(&free0, &total0);
    uint64_t host_free = host_free_bytes();

    /* Measure a faithful FULL-mode allocation budget using the connection axis. */
    uint64_t edge_hi = (uint64_t)(free0 / 12ull) + 1000000ull;
    uint64_t max_edges = bsearch_edges_at_1024(edge_hi);
    uint64_t full_budget = 29ull * 1024ull + 8ull + 12ull * max_edges;
    uint64_t safe_budget = (uint64_t)((long double)full_budget * 0.90L);

    printf("# Max neurons for average out-degree / fanout\n");
    printf("device=%s\n", prop.name);
    printf("full_budget_bytes=%llu\n", (unsigned long long)full_budget);
    printf("full_budget_gib=%.3f\n", gib(full_budget));
    printf("host_free_bytes=%llu\n", (unsigned long long)host_free);
    printf("formula_full=Nmax=floor((budget-8)/(29+12*fanout)); edges=fanout*Nmax\n");
    printf("formula_streaming_host=host_resident_N=floor(host_free/(8+12*fanout)); buildable_current_builder_N=floor(host_free/(2*(8+12*fanout)))\n");
    printf("fanout,full_max_neurons,full_max_edges,full_90pct_neurons,stream_host_resident_neurons,stream_buildable_neurons\n");

    for (size_t i = 0; i < sizeof(fanouts) / sizeof(fanouts[0]); ++i) {
        uint64_t f = fanouts[i];
        uint64_t full_bpn = 29ull + 12ull * f;
        uint64_t n_full = (full_budget - 8ull) / full_bpn;
        uint64_t n_safe = (safe_budget - 8ull) / full_bpn;
        uint64_t host_bpn = 8ull + 12ull * f;
        uint64_t n_stream_host = host_free ? host_free / host_bpn : 0ull;
        uint64_t n_stream_build = host_free ? host_free / (2ull * host_bpn) : 0ull;
        printf("%llu,%llu,%llu,%llu,%llu,%llu\n",
               (unsigned long long)f,
               (unsigned long long)n_full,
               (unsigned long long)(n_full * f),
               (unsigned long long)n_safe,
               (unsigned long long)n_stream_host,
               (unsigned long long)n_stream_build);
    }
    return 0;
}
