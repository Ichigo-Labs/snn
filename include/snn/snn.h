#ifndef SNN_SNN_H
#define SNN_SNN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SNN_VERSION_MAJOR 0
#define SNN_VERSION_MINOR 1
#define SNN_VERSION_PATCH 0

typedef uint64_t snn_size_t;

typedef enum snn_status {
    SNN_OK = 0,
    SNN_ERR_INVALID_ARGUMENT = -1,
    SNN_ERR_OUT_OF_MEMORY = -2,
    SNN_ERR_OVERFLOW = -3,
    SNN_ERR_CUDA = -4,
    SNN_ERR_UNSUPPORTED = -5
} snn_status_t;

typedef enum snn_architecture {
    SNN_ARCH_CUSTOM_CSR = 0,
    SNN_ARCH_FEED_FORWARD = 1,
    SNN_ARCH_RANDOM_POOL = 2
} snn_architecture_t;

typedef struct snn_lif_params {
    float dt_ms;
    float membrane_tau_ms;
    float v_rest;
    float v_reset;
    float v_threshold;
    float input_scale;
    uint32_t refractory_steps;
} snn_lif_params_t;

typedef struct snn_feedforward_config {
    const snn_size_t *layer_sizes;
    size_t layer_count;
    snn_size_t fanout_per_neuron; /* 0 means dense all-to-all into the next layer. */
    float weight;
    uint64_t seed;
} snn_feedforward_config_t;

typedef struct snn_random_pool_config {
    snn_size_t neuron_count;
    snn_size_t fanout_per_neuron;
    float weight_min;
    float weight_max;
    uint64_t seed;
    int allow_self_connections;
} snn_random_pool_config_t;

/*
 * Dry-run byte sizing for a network plus one simulation state. Two additions
 * are intentionally not included: OpenMP builds allocate an extra
 * omp_get_max_threads() * neuron_count floats of propagation scratch per
 * state, and CUDA event injection keeps device staging sized to the largest
 * batch passed to snn_cuda_inject_current.
 *
 * device_streaming_min_bytes: computed from counts alone
 * (snn_estimate_memory_for_counts) it assumes a minimal one-edge topology
 * chunk; snn_network_memory_plan refines it to the real backend minimum,
 * whose chunk must hold the network's densest row.
 */
typedef struct snn_memory_plan {
    snn_size_t neuron_count;
    snn_size_t synapse_count;
    uint64_t row_ptr_bytes;
    uint64_t col_index_bytes;
    uint64_t weight_bytes;
    uint64_t host_topology_bytes;
    uint64_t host_state_bytes;
    uint64_t host_total_bytes;
    uint64_t device_topology_bytes;
    uint64_t device_state_bytes;
    uint64_t device_total_full_bytes;
    uint64_t device_streaming_min_bytes;
    int overflowed;
} snn_memory_plan_t;

typedef struct snn_network snn_network_t;
typedef struct snn_state snn_state_t;

typedef enum snn_cuda_mode {
    SNN_CUDA_MODE_NONE = 0,
    SNN_CUDA_MODE_FULL = 1,
    SNN_CUDA_MODE_STREAMING = 2
} snn_cuda_mode_t;

typedef struct snn_cuda_config {
    /* 0 means use all currently free VRAM reported by CUDA. Best effort: the
     * resident per-neuron state and a minimal topology chunk are always
     * allocated, even when they alone exceed the cap. */
    uint64_t max_vram_bytes;
    /* 0 means auto-size the streamed synapse chunk. A nonzero value must be
     * at least the network's densest row (max out-degree) or snn_cuda_create
     * fails with SNN_ERR_INVALID_ARGUMENT; values beyond the total synapse
     * count are clamped. */
    snn_size_t max_stream_synapses;
    snn_size_t max_stream_rows; /* 0 means auto-size the streamed row chunk. */
    int prefer_streaming;
} snn_cuda_config_t;

typedef struct snn_cuda_context snn_cuda_context_t;

const char *snn_status_string(snn_status_t status);
const char *snn_architecture_string(snn_architecture_t architecture);
snn_lif_params_t snn_default_lif_params(void);
snn_status_t snn_lif_params_validate(const snn_lif_params_t *params);

snn_feedforward_config_t snn_default_feedforward_config(const snn_size_t *layer_sizes, size_t layer_count);
snn_random_pool_config_t snn_default_random_pool_config(snn_size_t neuron_count, snn_size_t fanout_per_neuron);

snn_status_t snn_estimate_memory_for_counts(snn_size_t neuron_count,
                                            snn_size_t synapse_count,
                                            snn_memory_plan_t *out_plan);
snn_status_t snn_network_memory_plan(const snn_network_t *network, snn_memory_plan_t *out_plan);

snn_status_t snn_build_custom_csr(snn_size_t neuron_count,
                                  snn_size_t synapse_count,
                                  const snn_size_t *row_ptr,
                                  const snn_size_t *col_idx,
                                  const float *weights,
                                  const snn_lif_params_t *params,
                                  snn_network_t **out_network);
snn_status_t snn_build_feedforward(const snn_feedforward_config_t *config,
                                   const snn_lif_params_t *params,
                                   snn_network_t **out_network);
snn_status_t snn_build_random_pool(const snn_random_pool_config_t *config,
                                   const snn_lif_params_t *params,
                                   snn_network_t **out_network);
void snn_network_free(snn_network_t *network);

snn_size_t snn_network_neuron_count(const snn_network_t *network);
snn_size_t snn_network_synapse_count(const snn_network_t *network);
snn_architecture_t snn_network_architecture(const snn_network_t *network);
const snn_size_t *snn_network_row_ptr(const snn_network_t *network);
const snn_size_t *snn_network_col_idx(const snn_network_t *network);
const float *snn_network_weights(const snn_network_t *network);
snn_lif_params_t snn_network_lif_params(const snn_network_t *network);
/* Applies to subsequent CPU steps. CUDA contexts capture the parameters at
 * snn_cuda_create time; recreate the context to apply changes on the GPU. */
snn_status_t snn_network_set_lif_params(snn_network_t *network, const snn_lif_params_t *params);

snn_status_t snn_state_create(const snn_network_t *network, snn_state_t **out_state);
void snn_state_free(snn_state_t *state);
snn_status_t snn_state_reset(const snn_network_t *network, snn_state_t *state);
snn_status_t snn_state_copy_voltage(const snn_state_t *state, float *out_voltage, snn_size_t count);
snn_status_t snn_state_copy_spikes(const snn_state_t *state, uint8_t *out_spikes, snn_size_t count);

/*
 * Sparse input events: adds values[k] * input_scale into the current consumed
 * by the next step at neuron indices[k]. Combined with a NULL external_current
 * step this drives only the given neurons instead of streaming a dense
 * n-float buffer (and, on CUDA, uploads only the event arrays). Duplicate
 * indices accumulate. Nothing is applied if any index is out of range.
 */
snn_status_t snn_state_inject_current(const snn_network_t *network,
                                      snn_state_t *state,
                                      const snn_size_t *indices,
                                      const float *values,
                                      snn_size_t count);

snn_status_t snn_step_cpu(const snn_network_t *network,
                          snn_state_t *state,
                          const float *external_current,
                          uint8_t *out_spikes);
snn_status_t snn_run_cpu(const snn_network_t *network,
                         snn_state_t *state,
                         const float *external_current_by_step,
                         snn_size_t step_count,
                         snn_size_t input_stride,
                         uint8_t *out_spikes_by_step,
                         snn_size_t output_stride);

snn_cuda_config_t snn_cuda_default_config(void);
int snn_cuda_available(void);
/* The network must outlive the context: STREAMING mode reads its topology on
 * every step (FULL mode copies it to the device at create time). When CUDA is
 * unusable, the status distinguishes the cause: SNN_ERR_CUDA means the CUDA
 * backend found no usable device; SNN_ERR_UNSUPPORTED means the library was
 * built without the CUDA backend (stubs). */
snn_status_t snn_cuda_create(const snn_network_t *network,
                             const snn_cuda_config_t *config,
                             snn_cuda_context_t **out_context);
void snn_cuda_free(snn_cuda_context_t *context);
snn_cuda_mode_t snn_cuda_context_mode(const snn_cuda_context_t *context);
/* CUDA twin of snn_state_inject_current. Device event buffers are allocated
 * lazily and grow to the largest count seen. Accumulation order for duplicate
 * indices is nondeterministic (atomic scatter). */
snn_status_t snn_cuda_inject_current(snn_cuda_context_t *context,
                                     const snn_size_t *host_indices,
                                     const float *host_values,
                                     snn_size_t count);
/* On any error other than SNN_ERR_INVALID_ARGUMENT the device state may be
 * partially advanced and the context must be freed and recreated; if only the
 * final spike download fails, the step itself has already been applied. */
snn_status_t snn_cuda_step(snn_cuda_context_t *context,
                           const float *host_external_current,
                           uint8_t *host_out_spikes);
snn_status_t snn_cuda_download_voltage(const snn_cuda_context_t *context,
                                       float *host_voltage,
                                       snn_size_t count);

#ifdef __cplusplus
}
#endif

#endif /* SNN_SNN_H */
