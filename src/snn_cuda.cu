#include "snn_internal.h"

#include <cuda_runtime.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct snn_cuda_context {
    const snn_network_t *network;
    snn_cuda_mode_t mode;
    snn_size_t neuron_count;
    snn_size_t synapse_count;
    snn_lif_params_t lif;
    float decay;
    snn_size_t *d_row_ptr;
    snn_size_t *d_col_idx;
    float *d_weights;
    snn_size_t *d_chunk_row_ptr;
    snn_size_t *d_chunk_col_idx;
    float *d_chunk_weights;
    snn_size_t *h_chunk_row_ptr;
    snn_size_t max_stream_rows;
    snn_size_t max_stream_synapses;
    float *d_voltage;
    float *d_current;
    float *d_next_current;
    float *d_external;
    uint32_t *d_refractory;
    uint8_t *d_spikes;
};

static snn_status_t cuda_status(cudaError_t err) {
    return err == cudaSuccess ? SNN_OK : SNN_ERR_CUDA;
}

/*
 * Fault injection for deterministic coverage of CUDA/allocation error paths.
 * In production builds (no SNN_ENABLE_TEST_HOOKS) cuda_inject_failure() is a
 * constant 0 and is fully optimized out.
 */
#ifdef SNN_ENABLE_TEST_HOOKS
static int64_t g_cuda_fail_after = -1;
static int g_cuda_force_unavailable = 0;
static int g_cuda_force_meminfo_fail = 0;

extern "C" void snn_test_cuda_set_fail_after(int64_t calls_before_failure) {
    g_cuda_fail_after = calls_before_failure;
}

extern "C" void snn_test_cuda_disable_failure(void) {
    g_cuda_fail_after = -1;
    g_cuda_force_unavailable = 0;
    g_cuda_force_meminfo_fail = 0;
}

extern "C" void snn_test_cuda_force_unavailable(int enable) {
    g_cuda_force_unavailable = enable;
}

extern "C" void snn_test_cuda_force_meminfo_fail(int enable) {
    g_cuda_force_meminfo_fail = enable;
}

static int cuda_inject_failure(void) {
    if (g_cuda_fail_after < 0) {
        return 0;
    }
    if (g_cuda_fail_after == 0) {
        return 1;
    }
    --g_cuda_fail_after;
    return 0;
}

static int cuda_force_unavailable(void) {
    return g_cuda_force_unavailable;
}

static int cuda_force_meminfo_fail(void) {
    return g_cuda_force_meminfo_fail;
}

/* Host allocation seam so OOM cleanup paths are deterministically testable. */
static void *cuda_host_malloc(size_t bytes) {
    if (cuda_inject_failure()) {
        return 0;
    }
    return malloc(bytes);
}

static void *cuda_host_calloc(size_t count, size_t elem) {
    if (cuda_inject_failure()) {
        return 0;
    }
    return calloc(count, elem);
}
#else
static int cuda_inject_failure(void) {
    return 0;
}
static int cuda_force_unavailable(void) {
    return 0;
}
static int cuda_force_meminfo_fail(void) {
    return 0;
}
#define cuda_host_malloc malloc
#define cuda_host_calloc calloc
#endif

/*
 * Kernel launches report asynchronously via cudaGetLastError(); this seam lets
 * the test suite deterministically exercise the launch-failure return paths.
 */
static snn_status_t cuda_check_launch(void) {
    if (cuda_inject_failure()) {
        return SNN_ERR_CUDA;
    }
    return cuda_status(cudaGetLastError());
}

/*
 * Byte sizing for device buffers. All counts reaching these helpers are bounded
 * by snn_cuda_create()'s snn_network_memory_plan() validation (topology/state)
 * or by the streaming chunk clamps (rows <= 65536, synapses <= synapse_count),
 * so on an LP64 host the product always fits in size_t; no per-call overflow
 * re-check is needed here (it is validated once, at create time).
 */
static size_t bytes_for_cuda(uint64_t count, uint64_t elem_size) {
    return (size_t)(count * elem_size);
}

static snn_status_t cuda_malloc_array(void **ptr, uint64_t count, uint64_t elem_size) {
    *ptr = 0;
    if (count == 0) {
        return SNN_OK;
    }
    if (cuda_inject_failure()) {
        return SNN_ERR_CUDA;
    }
    return cuda_status(cudaMalloc(ptr, bytes_for_cuda(count, elem_size)));
}

static snn_status_t cuda_copy(void *dst, const void *src, uint64_t count, uint64_t elem_size,
                              enum cudaMemcpyKind kind) {
    if (count == 0) {
        return SNN_OK;
    }
    if (cuda_inject_failure()) {
        return SNN_ERR_CUDA;
    }
    return cuda_status(cudaMemcpy(dst, src, bytes_for_cuda(count, elem_size), kind));
}

static snn_status_t cuda_copy_to_device(void *dst, const void *src, uint64_t count, uint64_t elem_size) {
    return cuda_copy(dst, src, count, elem_size, cudaMemcpyHostToDevice);
}

static snn_status_t cuda_copy_to_host(void *dst, const void *src, uint64_t count, uint64_t elem_size) {
    return cuda_copy(dst, src, count, elem_size, cudaMemcpyDeviceToHost);
}

static unsigned int launch_blocks(snn_size_t n) {
    const snn_size_t blocks = (n + 255u) / 256u;
    return blocks > 65535u ? 65535u : (unsigned int)blocks;
}

__global__ static void init_state_kernel(snn_size_t n,
                                         float v_rest,
                                         float *voltage,
                                         float *current,
                                         float *next_current,
                                         float *external,
                                         uint32_t *refractory,
                                         uint8_t *spikes) {
    const snn_size_t stride = (snn_size_t)blockDim.x * (snn_size_t)gridDim.x;
    for (snn_size_t i = (snn_size_t)blockIdx.x * (snn_size_t)blockDim.x + (snn_size_t)threadIdx.x; i < n; i += stride) {
        voltage[i] = v_rest;
        current[i] = 0.0f;
        next_current[i] = 0.0f;
        external[i] = 0.0f;
        refractory[i] = 0u;
        spikes[i] = 0u;
    }
}

__global__ static void integrate_kernel(snn_size_t n,
                                        snn_lif_params_t lif,
                                        float decay,
                                        float *voltage,
                                        float *current,
                                        const float *external,
                                        uint32_t *refractory,
                                        uint8_t *spikes) {
    const snn_size_t stride = (snn_size_t)blockDim.x * (snn_size_t)gridDim.x;
    for (snn_size_t i = (snn_size_t)blockIdx.x * (snn_size_t)blockDim.x + (snn_size_t)threadIdx.x; i < n; i += stride) {
        const float current_i = current[i];
        const float ext = external == 0 ? 0.0f : external[i] * lif.input_scale;
        uint8_t spiked = 0u;
        current[i] = 0.0f;
        if (refractory[i] != 0u) {
            refractory[i] -= 1u;
            voltage[i] = lif.v_reset;
        } else {
            const float v = lif.v_rest + (voltage[i] - lif.v_rest) * decay + current_i + ext;
            if (v >= lif.v_threshold) {
                spiked = 1u;
                voltage[i] = lif.v_reset;
                refractory[i] = lif.refractory_steps;
            } else {
                voltage[i] = v;
            }
        }
        spikes[i] = spiked;
    }
}

__global__ static void propagate_full_kernel(snn_size_t n,
                                             const uint8_t *spikes,
                                             const snn_size_t *row_ptr,
                                             const snn_size_t *col_idx,
                                             const float *weights,
                                             float *next_current) {
    const snn_size_t stride = (snn_size_t)blockDim.x * (snn_size_t)gridDim.x;
    for (snn_size_t pre = (snn_size_t)blockIdx.x * (snn_size_t)blockDim.x + (snn_size_t)threadIdx.x; pre < n; pre += stride) {
        if (spikes[pre] == 0u) {
            continue;
        }
        for (snn_size_t edge = row_ptr[pre]; edge < row_ptr[pre + 1u]; ++edge) {
            atomicAdd(&next_current[col_idx[edge]], weights[edge]);
        }
    }
}

__global__ static void propagate_chunk_kernel(snn_size_t row0,
                                              snn_size_t rows,
                                              const uint8_t *spikes,
                                              const snn_size_t *row_ptr,
                                              const snn_size_t *col_idx,
                                              const float *weights,
                                              float *next_current) {
    const snn_size_t stride = (snn_size_t)blockDim.x * (snn_size_t)gridDim.x;
    for (snn_size_t local = (snn_size_t)blockIdx.x * (snn_size_t)blockDim.x + (snn_size_t)threadIdx.x; local < rows; local += stride) {
        const snn_size_t pre = row0 + local;
        if (spikes[pre] == 0u) {
            continue;
        }
        for (snn_size_t edge = row_ptr[local]; edge < row_ptr[local + 1u]; ++edge) {
            atomicAdd(&next_current[col_idx[edge]], weights[edge]);
        }
    }
}

static snn_status_t allocate_device_state(snn_cuda_context_t *ctx) {
    snn_status_t st = SNN_OK;
#define ALLOC_OR_RETURN(PTR, COUNT, TYPE)                  \
    do {                                                   \
        st = cuda_malloc_array((void **)&(PTR), (COUNT), sizeof(TYPE)); \
        if (st != SNN_OK) {                                \
            return st;                                     \
        }                                                  \
    } while (0)
    ALLOC_OR_RETURN(ctx->d_voltage, ctx->neuron_count, float);
    ALLOC_OR_RETURN(ctx->d_current, ctx->neuron_count, float);
    ALLOC_OR_RETURN(ctx->d_next_current, ctx->neuron_count, float);
    ALLOC_OR_RETURN(ctx->d_external, ctx->neuron_count, float);
    ALLOC_OR_RETURN(ctx->d_refractory, ctx->neuron_count, uint32_t);
    ALLOC_OR_RETURN(ctx->d_spikes, ctx->neuron_count, uint8_t);
#undef ALLOC_OR_RETURN
    init_state_kernel<<<launch_blocks(ctx->neuron_count), 256>>>(ctx->neuron_count,
                                                                 ctx->lif.v_rest,
                                                                 ctx->d_voltage,
                                                                 ctx->d_current,
                                                                 ctx->d_next_current,
                                                                 ctx->d_external,
                                                                 ctx->d_refractory,
                                                                 ctx->d_spikes);
    return cuda_check_launch();
}

static snn_status_t allocate_full_topology(snn_cuda_context_t *ctx) {
    snn_status_t st = SNN_OK;
    st = cuda_malloc_array((void **)&ctx->d_row_ptr, ctx->neuron_count + 1u, sizeof(snn_size_t));
    if (st != SNN_OK) {
        return st;
    }
    st = cuda_malloc_array((void **)&ctx->d_col_idx, ctx->synapse_count, sizeof(snn_size_t));
    if (st != SNN_OK) {
        return st;
    }
    st = cuda_malloc_array((void **)&ctx->d_weights, ctx->synapse_count, sizeof(float));
    if (st != SNN_OK) {
        return st;
    }
    st = cuda_copy_to_device(ctx->d_row_ptr, ctx->network->row_ptr, ctx->neuron_count + 1u, sizeof(snn_size_t));
    if (st != SNN_OK) {
        return st;
    }
    st = cuda_copy_to_device(ctx->d_col_idx, ctx->network->col_idx, ctx->synapse_count, sizeof(snn_size_t));
    if (st != SNN_OK) {
        return st;
    }
    return cuda_copy_to_device(ctx->d_weights, ctx->network->weights, ctx->synapse_count, sizeof(float));
}

static snn_size_t find_max_degree(const snn_network_t *network) {
    snn_size_t max_degree = 0;
    for (snn_size_t i = 0; i < network->neuron_count; ++i) {
        const snn_size_t degree = network->row_ptr[i + 1u] - network->row_ptr[i];
        if (degree > max_degree) {
            max_degree = degree;
        }
    }
    return max_degree;
}

static snn_status_t allocate_streaming_topology(snn_cuda_context_t *ctx, const snn_cuda_config_t *config, uint64_t vram_budget) {
    const uint64_t per_neuron_state = (uint64_t)(4u * sizeof(float) + sizeof(uint32_t) + sizeof(uint8_t));
    const uint64_t edge_bytes = (uint64_t)(sizeof(snn_size_t) + sizeof(float));
    snn_size_t max_degree = find_max_degree(ctx->network);
    snn_size_t desired_rows = 0;
    snn_size_t desired_synapses = 0;
    /* neuron_count * per_neuron_state and (rows+1)*8 cannot overflow: the state
     * size was already validated by snn_network_memory_plan() in snn_cuda_create
     * and rows is clamped to 65536, so plain 64-bit arithmetic is safe here. */
    uint64_t state_bytes = (uint64_t)ctx->neuron_count * per_neuron_state;
    uint64_t available_for_chunks = vram_budget > state_bytes ? vram_budget - state_bytes : 0u;
    uint64_t row_bytes = 0;
    snn_status_t st = SNN_OK;

    desired_rows = (config->max_stream_rows != 0u) ? config->max_stream_rows : ctx->neuron_count;
    if (desired_rows > 65536u) {
        desired_rows = 65536u;
    }
    if (desired_rows > ctx->neuron_count) {
        desired_rows = ctx->neuron_count;
    }
    row_bytes = (uint64_t)(desired_rows + 1u) * sizeof(snn_size_t);

    if (config->max_stream_synapses != 0u) {
        /* An explicit chunk that cannot hold the densest row is rejected. */
        if (config->max_stream_synapses < max_degree) {
            return SNN_ERR_INVALID_ARGUMENT;
        }
        desired_synapses = config->max_stream_synapses;
    } else {
        /* Auto-size the synapse chunk to the remaining VRAM budget. */
        desired_synapses = (available_for_chunks > row_bytes + edge_bytes)
                               ? (snn_size_t)((available_for_chunks - row_bytes) / edge_bytes)
                               : 0u;
        if (desired_synapses > ctx->synapse_count) {
            desired_synapses = ctx->synapse_count;
        }
        if (desired_synapses < max_degree) {
            desired_synapses = max_degree; /* guarantee the densest row still fits */
        }
        if (desired_synapses == 0u) {
            desired_synapses = 1u; /* zero-synapse network: keep a valid buffer */
        }
    }

    ctx->h_chunk_row_ptr = (snn_size_t *)cuda_host_malloc((size_t)row_bytes);
    if (ctx->h_chunk_row_ptr == 0) {
        return SNN_ERR_OUT_OF_MEMORY;
    }
    ctx->max_stream_rows = desired_rows;
    ctx->max_stream_synapses = desired_synapses;

    st = cuda_malloc_array((void **)&ctx->d_chunk_row_ptr, desired_rows + 1u, sizeof(snn_size_t));
    if (st != SNN_OK) {
        return st;
    }
    st = cuda_malloc_array((void **)&ctx->d_chunk_col_idx, desired_synapses, sizeof(snn_size_t));
    if (st != SNN_OK) {
        return st;
    }
    return cuda_malloc_array((void **)&ctx->d_chunk_weights, desired_synapses, sizeof(float));
}

snn_cuda_config_t snn_cuda_default_config(void) {
    snn_cuda_config_t cfg;
    cfg.max_vram_bytes = 0u;
    cfg.max_stream_synapses = 0u;
    cfg.max_stream_rows = 0u;
    cfg.prefer_streaming = 0;
    return cfg;
}

int snn_cuda_available(void) {
    int count = 0;
    if (cuda_force_unavailable()) {
        return 0;
    }
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

snn_status_t snn_cuda_create(const snn_network_t *network,
                             const snn_cuda_config_t *config,
                             snn_cuda_context_t **out_context) {
    snn_cuda_context_t *ctx = 0;
    snn_cuda_config_t effective;
    snn_memory_plan_t plan;
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    uint64_t vram_budget = 0;
    snn_status_t st = SNN_OK;

    if (network == 0 || out_context == 0) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    *out_context = 0;
    if (!snn_cuda_available()) {
        return SNN_ERR_CUDA;
    }
    effective = config == 0 ? snn_cuda_default_config() : *config;
    if (cuda_force_meminfo_fail() || cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
        return SNN_ERR_CUDA;
    }
    vram_budget = effective.max_vram_bytes == 0u || effective.max_vram_bytes > (uint64_t)free_bytes
                      ? (uint64_t)free_bytes
                      : effective.max_vram_bytes;
    /* For a validly-built network the plan cannot overflow (its counts were
     * already allocated), so this only fills in plan.* and always succeeds. */
    (void)snn_network_memory_plan(network, &plan);
    ctx = (snn_cuda_context_t *)cuda_host_calloc(1u, sizeof(*ctx));
    if (ctx == 0) {
        return SNN_ERR_OUT_OF_MEMORY;
    }
    ctx->network = network;
    ctx->neuron_count = network->neuron_count;
    ctx->synapse_count = network->synapse_count;
    ctx->lif = network->lif;
    ctx->decay = network->decay;

    st = allocate_device_state(ctx);
    if (st != SNN_OK) {
        snn_cuda_free(ctx);
        return st;
    }

    if (!effective.prefer_streaming && plan.device_total_full_bytes <= vram_budget) {
        ctx->mode = SNN_CUDA_MODE_FULL;
        st = allocate_full_topology(ctx);
    } else {
        ctx->mode = SNN_CUDA_MODE_STREAMING;
        st = allocate_streaming_topology(ctx, &effective, vram_budget);
    }
    if (st != SNN_OK) {
        snn_cuda_free(ctx);
        return st;
    }
    *out_context = ctx;
    (void)total_bytes;
    return SNN_OK;
}

void snn_cuda_free(snn_cuda_context_t *context) {
    if (context != 0) {
        cudaFree(context->d_row_ptr);
        cudaFree(context->d_col_idx);
        cudaFree(context->d_weights);
        cudaFree(context->d_chunk_row_ptr);
        cudaFree(context->d_chunk_col_idx);
        cudaFree(context->d_chunk_weights);
        cudaFree(context->d_voltage);
        cudaFree(context->d_current);
        cudaFree(context->d_next_current);
        cudaFree(context->d_external);
        cudaFree(context->d_refractory);
        cudaFree(context->d_spikes);
        free(context->h_chunk_row_ptr);
        free(context);
    }
}

snn_cuda_mode_t snn_cuda_context_mode(const snn_cuda_context_t *context) {
    return context == 0 ? SNN_CUDA_MODE_NONE : context->mode;
}

static snn_status_t upload_external_if_present(snn_cuda_context_t *ctx, const float *host_external_current) {
    size_t bytes = bytes_for_cuda(ctx->neuron_count, sizeof(float));
    if (host_external_current == 0) {
        return SNN_OK;
    }
    if (cuda_inject_failure()) {
        return SNN_ERR_CUDA;
    }
    return cuda_status(cudaMemcpy(ctx->d_external, host_external_current, bytes, cudaMemcpyHostToDevice));
}

static snn_status_t stream_propagate(snn_cuda_context_t *ctx) {
    const snn_network_t *network = ctx->network;
    snn_size_t row_begin = 0;
    while (row_begin < network->neuron_count) {
        const snn_size_t edge_begin = network->row_ptr[row_begin];
        snn_size_t row_end = row_begin;
        snn_size_t edge_count = 0;
        snn_size_t rows = 0;
        snn_status_t st = SNN_OK;
        /*
         * Grow the chunk row-by-row. snn_cuda_create guarantees
         * max_stream_synapses >= max_degree, so a single row always fits and the
         * first row of every chunk is always admitted (rows >= 1 on exit).
         */
        while (row_end < network->neuron_count && rows < ctx->max_stream_rows) {
            const snn_size_t degree = network->row_ptr[row_end + 1u] - network->row_ptr[row_end];
            if (rows != 0u && edge_count + degree > ctx->max_stream_synapses) {
                break;
            }
            edge_count += degree;
            ++row_end;
            ++rows;
            if (edge_count == ctx->max_stream_synapses) {
                break;
            }
        }
        for (snn_size_t i = 0; i <= rows; ++i) {
            ctx->h_chunk_row_ptr[i] = network->row_ptr[row_begin + i] - edge_begin;
        }
        st = cuda_copy_to_device(ctx->d_chunk_row_ptr, ctx->h_chunk_row_ptr, rows + 1u, sizeof(snn_size_t));
        if (st != SNN_OK) {
            return st;
        }
        if (edge_count != 0u) {
            st = cuda_copy_to_device(ctx->d_chunk_col_idx, network->col_idx + edge_begin, edge_count, sizeof(snn_size_t));
            if (st != SNN_OK) {
                return st;
            }
            st = cuda_copy_to_device(ctx->d_chunk_weights, network->weights + edge_begin, edge_count, sizeof(float));
            if (st != SNN_OK) {
                return st;
            }
        }
        propagate_chunk_kernel<<<launch_blocks(rows), 256>>>(row_begin,
                                                             rows,
                                                             ctx->d_spikes,
                                                             ctx->d_chunk_row_ptr,
                                                             ctx->d_chunk_col_idx,
                                                             ctx->d_chunk_weights,
                                                             ctx->d_next_current);
        st = cuda_check_launch();
        if (st != SNN_OK) {
            return st;
        }
        row_begin = row_end;
    }
    return SNN_OK;
}

snn_status_t snn_cuda_step(snn_cuda_context_t *context,
                           const float *host_external_current,
                           uint8_t *host_out_spikes) {
    float *tmp = 0;
    snn_status_t st = SNN_OK;
    if (context == 0) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    st = upload_external_if_present(context, host_external_current);
    if (st != SNN_OK) {
        return st;
    }
    integrate_kernel<<<launch_blocks(context->neuron_count), 256>>>(context->neuron_count,
                                                                    context->lif,
                                                                    context->decay,
                                                                    context->d_voltage,
                                                                    context->d_current,
                                                                    host_external_current == 0 ? 0 : context->d_external,
                                                                    context->d_refractory,
                                                                    context->d_spikes);
    st = cuda_check_launch();
    if (st != SNN_OK) {
        return st;
    }
    if (context->mode == SNN_CUDA_MODE_FULL) {
        propagate_full_kernel<<<launch_blocks(context->neuron_count), 256>>>(context->neuron_count,
                                                                            context->d_spikes,
                                                                            context->d_row_ptr,
                                                                            context->d_col_idx,
                                                                            context->d_weights,
                                                                            context->d_next_current);
        st = cuda_check_launch();
    } else {
        /* A successfully created context is always FULL or STREAMING. */
        st = stream_propagate(context);
    }
    if (st != SNN_OK) {
        return st;
    }
    tmp = context->d_current;
    context->d_current = context->d_next_current;
    context->d_next_current = tmp;
    if (host_out_spikes != 0) {
        return cuda_copy_to_host(host_out_spikes, context->d_spikes, context->neuron_count, sizeof(uint8_t));
    }
    return cuda_status(cudaDeviceSynchronize());
}

snn_status_t snn_cuda_download_voltage(const snn_cuda_context_t *context,
                                       float *host_voltage,
                                       snn_size_t count) {
    if (context == 0 || host_voltage == 0 || count < context->neuron_count) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    return cuda_copy_to_host(host_voltage, context->d_voltage, context->neuron_count, sizeof(float));
}


#ifdef SNN_ENABLE_TEST_HOOKS
extern "C" snn_cuda_context_t *snn_test_nonnull_cuda_context(void) {
    return (snn_cuda_context_t *)(uintptr_t)1u;
}
#endif
