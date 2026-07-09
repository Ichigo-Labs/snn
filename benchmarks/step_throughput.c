/*
 * Step-throughput benchmark for the CPU and CUDA simulation paths.
 *
 * Build:
 *   cmake -S . -B build-bench -DSNN_BUILD_BENCHMARKS=ON -DSNN_BUILD_TESTS=OFF
 *   cmake --build build-bench -j
 * Run:
 *   ./build-bench/step_throughput
 *
 * Workloads:
 *   - integrate-bound: 2M neurons, 0 synapses (isolates the membrane update)
 *   - propagation-bound: 200k-neuron random pool, fanout 64 (12.8M synapses,
 *     ~8% of neurons spiking per step)
 *
 * A short instrumented probe reports the spike rate and synapse events per
 * step; the timed loop then runs the bare step (no spike readback on CPU,
 * spike download included on GPU) so numbers are comparable across changes.
 */
#include <snn/snn.h>

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define FRAME_COUNT 8u

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

/* FRAME_COUNT deterministic input frames driving ~drive_percent% of neurons. */
static float *make_frames(snn_size_t n, unsigned drive_percent, float amplitude) {
    float *frames = (float *)calloc((size_t)n * FRAME_COUNT, sizeof(float));
    if (frames == NULL) {
        return NULL;
    }
    for (unsigned f = 0; f < FRAME_COUNT; ++f) {
        for (snn_size_t i = 0; i < n; ++i) {
            if (mix64(i * UINT64_C(0x9e3779b97f4a7c15) + f) % 100u < drive_percent) {
                frames[(size_t)f * n + i] = amplitude;
            }
        }
    }
    return frames;
}

static const float *frame_at(const float *frames, snn_size_t n, int step) {
    return frames + (size_t)((unsigned)step % FRAME_COUNT) * n;
}

/* Probe a few steps to report the workload's spike rate and synapse events. */
static void probe_workload(const snn_network_t *net, snn_state_t *state,
                           const float *frames, int probe_steps,
                           double *out_spike_rate, double *out_events_per_step) {
    const snn_size_t n = snn_network_neuron_count(net);
    const snn_size_t *row_ptr = snn_network_row_ptr(net);
    uint8_t *spikes = (uint8_t *)malloc((size_t)n);
    uint64_t spike_total = 0;
    uint64_t event_total = 0;
    int s = 0;
    if (spikes == NULL) {
        fprintf(stderr, "probe alloc failed\n");
        exit(1);
    }
    for (s = 0; s < probe_steps; ++s) {
        snn_size_t i = 0;
        if (snn_step_cpu(net, state, frame_at(frames, n, s), spikes) != SNN_OK) {
            fprintf(stderr, "probe step failed\n");
            exit(1);
        }
        for (i = 0; i < n; ++i) {
            if (spikes[i] != 0u) {
                ++spike_total;
                event_total += row_ptr[i + 1u] - row_ptr[i];
            }
        }
    }
    *out_spike_rate = (double)spike_total / ((double)probe_steps * (double)n);
    *out_events_per_step = (double)event_total / (double)probe_steps;
    free(spikes);
}

static void bench_cpu(const char *name, const snn_network_t *net,
                      const float *frames, int warmup, int steps) {
    const snn_size_t n = snn_network_neuron_count(net);
    snn_state_t *state = NULL;
    double spike_rate = 0.0;
    double events_per_step = 0.0;
    double t0 = 0.0;
    double t1 = 0.0;
    double ms_per_step = 0.0;
    int s = 0;
    if (snn_state_create(net, &state) != SNN_OK) {
        fprintf(stderr, "state create failed\n");
        exit(1);
    }
    probe_workload(net, state, frames, 10, &spike_rate, &events_per_step);
    if (snn_state_reset(net, state) != SNN_OK) {
        fprintf(stderr, "state reset failed\n");
        exit(1);
    }
    for (s = 0; s < warmup; ++s) {
        (void)snn_step_cpu(net, state, frame_at(frames, n, s), NULL);
    }
    t0 = now_sec();
    for (s = 0; s < steps; ++s) {
        (void)snn_step_cpu(net, state, frame_at(frames, n, s), NULL);
    }
    t1 = now_sec();
    ms_per_step = (t1 - t0) * 1000.0 / (double)steps;
    printf("%-26s cpu  %10.3f ms/step %10.1f steps/s  %5.2f%% spikes  %8.2f Mevents/s\n",
           name, ms_per_step, 1000.0 / ms_per_step, spike_rate * 100.0,
           events_per_step / (ms_per_step * 1000.0));
    snn_state_free(state);
}

static void bench_gpu(const char *name, const snn_network_t *net,
                      const float *frames, int warmup, int steps,
                      double spike_rate, double events_per_step) {
    const snn_size_t n = snn_network_neuron_count(net);
    snn_cuda_context_t *ctx = NULL;
    uint8_t *spikes = (uint8_t *)malloc((size_t)n);
    double t0 = 0.0;
    double t1 = 0.0;
    double ms_per_step = 0.0;
    int s = 0;
    if (spikes == NULL || snn_cuda_create(net, NULL, &ctx) != SNN_OK) {
        fprintf(stderr, "cuda create failed\n");
        exit(1);
    }
    for (s = 0; s < warmup; ++s) {
        (void)snn_cuda_step(ctx, frame_at(frames, n, s), spikes);
    }
    t0 = now_sec();
    for (s = 0; s < steps; ++s) {
        (void)snn_cuda_step(ctx, frame_at(frames, n, s), spikes);
    }
    t1 = now_sec();
    ms_per_step = (t1 - t0) * 1000.0 / (double)steps;
    printf("%-26s gpu  %10.3f ms/step %10.1f steps/s  %5.2f%% spikes  %8.2f Mevents/s\n",
           name, ms_per_step, 1000.0 / ms_per_step, spike_rate * 100.0,
           events_per_step / (ms_per_step * 1000.0));
    snn_cuda_free(ctx);
    free(spikes);
}

static void run_workload(const char *name, const snn_network_t *net,
                         unsigned drive_percent, int cpu_steps, int gpu_steps) {
    const snn_size_t n = snn_network_neuron_count(net);
    float *frames = make_frames(n, drive_percent, 1.5f);
    snn_state_t *probe_state = NULL;
    double spike_rate = 0.0;
    double events_per_step = 0.0;
    if (frames == NULL) {
        fprintf(stderr, "frame alloc failed\n");
        exit(1);
    }
    bench_cpu(name, net, frames, 5, cpu_steps);
    if (snn_cuda_available()) {
        if (snn_state_create(net, &probe_state) != SNN_OK) {
            fprintf(stderr, "probe state create failed\n");
            exit(1);
        }
        probe_workload(net, probe_state, frames, 10, &spike_rate, &events_per_step);
        snn_state_free(probe_state);
        bench_gpu(name, net, frames, 30, gpu_steps, spike_rate, events_per_step);
    }
    free(frames);
}

int main(void) {
    snn_network_t *integrate_net = NULL;
    snn_network_t *pool_net = NULL;
    snn_random_pool_config_t rp;

    rp = snn_default_random_pool_config(2000000u, 0u);
    rp.allow_self_connections = 1;
    if (snn_build_random_pool(&rp, NULL, &integrate_net) != SNN_OK) {
        fprintf(stderr, "integrate net build failed\n");
        return 1;
    }
    run_workload("integrate 2M x0", integrate_net, 8u, 100, 300);
    snn_network_free(integrate_net);

    rp = snn_default_random_pool_config(200000u, 64u);
    rp.weight_min = 0.03f;
    rp.weight_max = 0.06f;
    rp.seed = UINT64_C(4242);
    if (snn_build_random_pool(&rp, NULL, &pool_net) != SNN_OK) {
        fprintf(stderr, "pool net build failed\n");
        return 1;
    }
    run_workload("pool 200k x64", pool_net, 8u, 100, 300);
    snn_network_free(pool_net);
    return 0;
}
