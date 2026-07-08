#include "snn_internal.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef SNN_ENABLE_TEST_HOOKS
static int64_t g_alloc_fail_after = -1;

void snn_test_set_alloc_fail_after(int64_t successful_allocations_before_failure) {
    g_alloc_fail_after = successful_allocations_before_failure;
}

void snn_test_disable_alloc_failure(void) {
    g_alloc_fail_after = -1;
}

static int snn_test_should_fail_alloc(void) {
    if (g_alloc_fail_after < 0) {
        return 0;
    }
    if (g_alloc_fail_after == 0) {
        return 1;
    }
    --g_alloc_fail_after;
    return 0;
}

static void *snn_malloc_impl(size_t bytes) {
    if (snn_test_should_fail_alloc()) {
        return NULL;
    }
    return malloc(bytes);
}

static void *snn_calloc_impl(size_t count, size_t elem_size) {
    if (snn_test_should_fail_alloc()) {
        return NULL;
    }
    return calloc(count, elem_size);
}
#else
#define snn_malloc_impl malloc
#define snn_calloc_impl calloc
#endif

static int checked_add_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (out == NULL) {
        return 0;
    }
    if (UINT64_MAX - a < b) {
        return 0;
    }
    *out = a + b;
    return 1;
}

static int checked_mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (out == NULL) {
        return 0;
    }
    if (a != 0 && b > UINT64_MAX / a) {
        return 0;
    }
    *out = a * b;
    return 1;
}

static int bytes_for_array_u64(uint64_t count, uint64_t elem_size, uint64_t *out) {
    return checked_mul_u64(count, elem_size, out);
}

static void *calloc_u64(uint64_t count, uint64_t elem_size) {
    size_t c = (size_t)count;
    size_t e = (size_t)elem_size;
    return snn_calloc_impl(c, e);
}

static void *malloc_bytes_u64(uint64_t bytes) {
    size_t b = (size_t)bytes;
    return snn_malloc_impl(b);
}

static uint64_t splitmix64_next(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static float rng_uniform_float(uint64_t *state) {
    const uint64_t r = splitmix64_next(state);
    const double unit = (double)(r >> 11) * (1.0 / 9007199254740992.0);
    return (float)unit;
}

static snn_lif_params_t effective_params(const snn_lif_params_t *params) {
    return params != NULL ? *params : snn_default_lif_params();
}

static snn_status_t allocate_network(snn_size_t neuron_count,
                                     snn_size_t synapse_count,
                                     snn_architecture_t architecture,
                                     const snn_lif_params_t *params,
                                     snn_network_t **out_network) {
    uint64_t row_count = 0;
    uint64_t row_bytes = 0;
    uint64_t col_bytes = 0;
    uint64_t weight_bytes = 0;
    snn_network_t *network = NULL;
    snn_lif_params_t lif = effective_params(params);

    *out_network = NULL;
    if (snn_lif_params_validate(&lif) != SNN_OK) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    if (!checked_add_u64(neuron_count, 1u, &row_count) ||
        !bytes_for_array_u64(row_count, sizeof(snn_size_t), &row_bytes) ||
        !bytes_for_array_u64(synapse_count, sizeof(snn_size_t), &col_bytes) ||
        !bytes_for_array_u64(synapse_count, sizeof(float), &weight_bytes)) {
        return SNN_ERR_OVERFLOW;
    }

    network = (snn_network_t *)snn_calloc_impl(1u, sizeof(*network));
    if (network == NULL) {
        return SNN_ERR_OUT_OF_MEMORY;
    }
    network->row_ptr = (snn_size_t *)calloc_u64(row_count, sizeof(snn_size_t));
    network->col_idx = synapse_count == 0 ? NULL : (snn_size_t *)malloc_bytes_u64(col_bytes);
    network->weights = synapse_count == 0 ? NULL : (float *)malloc_bytes_u64(weight_bytes);
    if (network->row_ptr == NULL || (synapse_count != 0 && (network->col_idx == NULL || network->weights == NULL))) {
        snn_network_free(network);
        return SNN_ERR_OUT_OF_MEMORY;
    }
    network->neuron_count = neuron_count;
    network->synapse_count = synapse_count;
    network->architecture = architecture;
    network->lif = lif;
    (void)row_bytes;
    *out_network = network;
    return SNN_OK;
}

static snn_status_t validate_csr(snn_size_t neuron_count,
                                 snn_size_t synapse_count,
                                 const snn_size_t *row_ptr,
                                 const snn_size_t *col_idx,
                                 const float *weights) {
    snn_size_t i = 0;
    if (neuron_count == 0 || row_ptr == NULL || (synapse_count != 0 && (col_idx == NULL || weights == NULL))) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    if (row_ptr[0] != 0 || row_ptr[neuron_count] != synapse_count) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < neuron_count; ++i) {
        if (row_ptr[i] > row_ptr[i + 1u] || row_ptr[i + 1u] > synapse_count) {
            return SNN_ERR_INVALID_ARGUMENT;
        }
    }
    for (i = 0; i < synapse_count; ++i) {
        if (col_idx[i] >= neuron_count || !isfinite(weights[i])) {
            return SNN_ERR_INVALID_ARGUMENT;
        }
    }
    return SNN_OK;
}

static snn_status_t count_feedforward_synapses(const snn_feedforward_config_t *config,
                                               snn_size_t *out_neurons,
                                               snn_size_t *out_synapses) {
    uint64_t neurons = 0;
    uint64_t synapses = 0;
    size_t layer = 0;
    if (config == NULL || out_neurons == NULL || out_synapses == NULL ||
        config->layer_sizes == NULL || config->layer_count < 2u) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    for (layer = 0; layer < config->layer_count; ++layer) {
        if (config->layer_sizes[layer] == 0) {
            return SNN_ERR_INVALID_ARGUMENT;
        }
        if (!checked_add_u64(neurons, config->layer_sizes[layer], &neurons)) {
            return SNN_ERR_OVERFLOW;
        }
    }
    for (layer = 0; layer + 1u < config->layer_count; ++layer) {
        const snn_size_t pre = config->layer_sizes[layer];
        const snn_size_t post = config->layer_sizes[layer + 1u];
        const snn_size_t fanout = (config->fanout_per_neuron == 0 || config->fanout_per_neuron >= post)
                                      ? post
                                      : config->fanout_per_neuron;
        uint64_t layer_edges = 0;
        if (!checked_mul_u64(pre, fanout, &layer_edges) || !checked_add_u64(synapses, layer_edges, &synapses)) {
            return SNN_ERR_OVERFLOW;
        }
    }
    *out_neurons = neurons;
    *out_synapses = synapses;
    return SNN_OK;
}

static snn_status_t prefix_layer_offsets(const snn_size_t *sizes,
                                         size_t count,
                                         snn_size_t *offsets) {
    size_t i = 0;
    uint64_t running = 0;
    if (sizes == NULL || offsets == NULL || count == 0) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    offsets[0] = 0;
    for (i = 0; i < count; ++i) {
        if (!checked_add_u64(running, sizes[i], &running)) {
            return SNN_ERR_OVERFLOW;
        }
        offsets[i + 1u] = running;
    }
    return SNN_OK;
}

#ifdef SNN_ENABLE_TEST_HOOKS
int snn_test_exercise_internal_guards(void) {
    int failures = 0;
    failures += checked_add_u64(1u, 1u, NULL) == 0;
    failures += checked_mul_u64(1u, 1u, NULL) == 0;
    return failures == 2;
}

snn_status_t snn_test_prefix_layer_offsets(const snn_size_t *sizes, size_t count, snn_size_t *offsets) {
    return prefix_layer_offsets(sizes, count, offsets);
}
#endif

const char *snn_status_string(snn_status_t status) {
    switch (status) {
    case SNN_OK:
        return "ok";
    case SNN_ERR_INVALID_ARGUMENT:
        return "invalid argument";
    case SNN_ERR_OUT_OF_MEMORY:
        return "out of memory";
    case SNN_ERR_OVERFLOW:
        return "integer overflow";
    case SNN_ERR_CUDA:
        return "cuda error";
    case SNN_ERR_UNSUPPORTED:
        return "unsupported";
    default:
        return "unknown status";
    }
}

const char *snn_architecture_string(snn_architecture_t architecture) {
    switch (architecture) {
    case SNN_ARCH_CUSTOM_CSR:
        return "custom_csr";
    case SNN_ARCH_FEED_FORWARD:
        return "feed_forward";
    case SNN_ARCH_RANDOM_POOL:
        return "random_pool";
    default:
        return "unknown";
    }
}

snn_lif_params_t snn_default_lif_params(void) {
    snn_lif_params_t p;
    p.dt_ms = 1.0f;
    p.membrane_tau_ms = 20.0f;
    p.v_rest = 0.0f;
    p.v_reset = 0.0f;
    p.v_threshold = 1.0f;
    p.input_scale = 1.0f;
    p.refractory_steps = 1u;
    return p;
}

snn_status_t snn_lif_params_validate(const snn_lif_params_t *params) {
    if (params == NULL || !isfinite(params->dt_ms) || !isfinite(params->membrane_tau_ms) ||
        !isfinite(params->v_rest) || !isfinite(params->v_reset) || !isfinite(params->v_threshold) ||
        !isfinite(params->input_scale) || params->dt_ms <= 0.0f || params->membrane_tau_ms <= 0.0f ||
        params->v_threshold <= params->v_reset) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    return SNN_OK;
}

snn_feedforward_config_t snn_default_feedforward_config(const snn_size_t *layer_sizes, size_t layer_count) {
    snn_feedforward_config_t cfg;
    cfg.layer_sizes = layer_sizes;
    cfg.layer_count = layer_count;
    cfg.fanout_per_neuron = 0;
    cfg.weight = 1.0f;
    cfg.seed = UINT64_C(0x534e4e5ffeedf00d);
    return cfg;
}

snn_random_pool_config_t snn_default_random_pool_config(snn_size_t neuron_count, snn_size_t fanout_per_neuron) {
    snn_random_pool_config_t cfg;
    cfg.neuron_count = neuron_count;
    cfg.fanout_per_neuron = fanout_per_neuron;
    cfg.weight_min = 0.5f;
    cfg.weight_max = 1.0f;
    cfg.seed = UINT64_C(0x534e4e5fc001d00d);
    cfg.allow_self_connections = 0;
    return cfg;
}

snn_status_t snn_estimate_memory_for_counts(snn_size_t neuron_count,
                                            snn_size_t synapse_count,
                                            snn_memory_plan_t *out_plan) {
    uint64_t n_plus_one = 0;
    uint64_t row = 0;
    uint64_t col = 0;
    uint64_t weights = 0;
    uint64_t topology = 0;
    uint64_t state_floats = 0;
    uint64_t state_float_bytes = 0;
    uint64_t refractory_bytes = 0;
    uint64_t spike_bytes = 0;
    uint64_t state_bytes = 0;
    uint64_t streaming_min = 0;
    if (out_plan == NULL || neuron_count == 0) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    memset(out_plan, 0, sizeof(*out_plan));
    out_plan->neuron_count = neuron_count;
    out_plan->synapse_count = synapse_count;

    if (!checked_add_u64(neuron_count, 1u, &n_plus_one) ||
        !bytes_for_array_u64(n_plus_one, sizeof(snn_size_t), &row) ||
        !bytes_for_array_u64(synapse_count, sizeof(snn_size_t), &col) ||
        !bytes_for_array_u64(synapse_count, sizeof(float), &weights) ||
        !checked_add_u64(row, col, &topology) ||
        !checked_add_u64(topology, weights, &topology) ||
        !bytes_for_array_u64(neuron_count, 4u, &state_floats) ||
        !bytes_for_array_u64(state_floats, sizeof(float), &state_float_bytes) ||
        !bytes_for_array_u64(neuron_count, sizeof(uint32_t), &refractory_bytes) ||
        !bytes_for_array_u64(neuron_count, sizeof(uint8_t), &spike_bytes) ||
        !checked_add_u64(state_float_bytes, refractory_bytes, &state_bytes) ||
        !checked_add_u64(state_bytes, spike_bytes, &state_bytes) ||
        !checked_add_u64(state_bytes, sizeof(snn_size_t) * 2u + sizeof(snn_size_t) + sizeof(float), &streaming_min)) {
        out_plan->overflowed = 1;
        return SNN_ERR_OVERFLOW;
    }
    out_plan->row_ptr_bytes = row;
    out_plan->col_index_bytes = col;
    out_plan->weight_bytes = weights;
    out_plan->host_topology_bytes = topology;
    out_plan->device_topology_bytes = topology;
    out_plan->device_state_bytes = state_bytes;
    out_plan->device_streaming_min_bytes = streaming_min;
    if (!checked_add_u64(topology, state_bytes, &out_plan->device_total_full_bytes)) {
        out_plan->overflowed = 1;
        return SNN_ERR_OVERFLOW;
    }
    return SNN_OK;
}

snn_status_t snn_network_memory_plan(const snn_network_t *network, snn_memory_plan_t *out_plan) {
    if (network == NULL) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    return snn_estimate_memory_for_counts(network->neuron_count, network->synapse_count, out_plan);
}

snn_status_t snn_build_custom_csr(snn_size_t neuron_count,
                                  snn_size_t synapse_count,
                                  const snn_size_t *row_ptr,
                                  const snn_size_t *col_idx,
                                  const float *weights,
                                  const snn_lif_params_t *params,
                                  snn_network_t **out_network) {
    snn_network_t *network = NULL;
    snn_status_t st = SNN_OK;
    uint64_t row_bytes = 0;
    uint64_t col_bytes = 0;
    uint64_t weight_bytes = 0;
    if (out_network == NULL) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    st = validate_csr(neuron_count, synapse_count, row_ptr, col_idx, weights);
    if (st != SNN_OK) {
        return st;
    }
    st = allocate_network(neuron_count, synapse_count, SNN_ARCH_CUSTOM_CSR, params, &network);
    if (st != SNN_OK) {
        return st;
    }
    (void)bytes_for_array_u64(neuron_count + 1u, sizeof(snn_size_t), &row_bytes);
    (void)bytes_for_array_u64(synapse_count, sizeof(snn_size_t), &col_bytes);
    (void)bytes_for_array_u64(synapse_count, sizeof(float), &weight_bytes);
    memcpy(network->row_ptr, row_ptr, (size_t)row_bytes);
    if (synapse_count != 0) {
        memcpy(network->col_idx, col_idx, (size_t)col_bytes);
        memcpy(network->weights, weights, (size_t)weight_bytes);
    }
    *out_network = network;
    return SNN_OK;
}

snn_status_t snn_build_feedforward(const snn_feedforward_config_t *config,
                                   const snn_lif_params_t *params,
                                   snn_network_t **out_network) {
    snn_size_t neurons = 0;
    snn_size_t synapses = 0;
    snn_network_t *network = NULL;
    snn_size_t *offsets = NULL;
    snn_status_t st = count_feedforward_synapses(config, &neurons, &synapses);
    snn_size_t edge = 0;
    snn_size_t global_pre = 0;
    size_t layer = 0;
    uint64_t rng = 0;
    if (out_network == NULL) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    if (st != SNN_OK) {
        return st;
    }
    if (!isfinite(config->weight)) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    st = allocate_network(neurons, synapses, SNN_ARCH_FEED_FORWARD, params, &network);
    if (st != SNN_OK) {
        return st;
    }
    offsets = (snn_size_t *)snn_calloc_impl(config->layer_count + 1u, sizeof(*offsets));
    if (offsets == NULL) {
        snn_network_free(network);
        return SNN_ERR_OUT_OF_MEMORY;
    }
    (void)prefix_layer_offsets(config->layer_sizes, config->layer_count, offsets);
    rng = config->seed;
    for (layer = 0; layer + 1u < config->layer_count; ++layer) {
        const snn_size_t pre_count = config->layer_sizes[layer];
        const snn_size_t post_count = config->layer_sizes[layer + 1u];
        const snn_size_t fanout = (config->fanout_per_neuron == 0 || config->fanout_per_neuron >= post_count)
                                      ? post_count
                                      : config->fanout_per_neuron;
        snn_size_t local_pre = 0;
        for (local_pre = 0; local_pre < pre_count; ++local_pre, ++global_pre) {
            snn_size_t j = 0;
            network->row_ptr[global_pre] = edge;
            for (j = 0; j < fanout; ++j) {
                const snn_size_t post_local = (fanout == post_count) ? j : (snn_size_t)(splitmix64_next(&rng) % post_count);
                network->col_idx[edge] = offsets[layer + 1u] + post_local;
                network->weights[edge] = config->weight;
                ++edge;
            }
        }
    }
    while (global_pre <= neurons) {
        network->row_ptr[global_pre] = edge;
        ++global_pre;
    }
    free(offsets);
    *out_network = network;
    return SNN_OK;
}

snn_status_t snn_build_random_pool(const snn_random_pool_config_t *config,
                                   const snn_lif_params_t *params,
                                   snn_network_t **out_network) {
    uint64_t synapses = 0;
    snn_network_t *network = NULL;
    snn_status_t st = SNN_OK;
    uint64_t rng = 0;
    snn_size_t edge = 0;
    snn_size_t pre = 0;
    if (out_network == NULL) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    if (config == NULL || config->neuron_count == 0 ||
        (!config->allow_self_connections && config->neuron_count == 1u && config->fanout_per_neuron != 0u) ||
        !isfinite(config->weight_min) || !isfinite(config->weight_max) || config->weight_max < config->weight_min) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    if (!checked_mul_u64(config->neuron_count, config->fanout_per_neuron, &synapses)) {
        return SNN_ERR_OVERFLOW;
    }
    st = allocate_network(config->neuron_count, synapses, SNN_ARCH_RANDOM_POOL, params, &network);
    if (st != SNN_OK) {
        return st;
    }
    rng = config->seed;
    for (pre = 0; pre < config->neuron_count; ++pre) {
        snn_size_t j = 0;
        network->row_ptr[pre] = edge;
        for (j = 0; j < config->fanout_per_neuron; ++j) {
            snn_size_t post = (snn_size_t)(splitmix64_next(&rng) % config->neuron_count);
            const float u = rng_uniform_float(&rng);
            if (!config->allow_self_connections && post == pre) {
                post = (post + 1u) % config->neuron_count;
            }
            network->col_idx[edge] = post;
            network->weights[edge] = config->weight_min + (config->weight_max - config->weight_min) * u;
            ++edge;
        }
    }
    network->row_ptr[config->neuron_count] = edge;
    *out_network = network;
    return SNN_OK;
}

void snn_network_free(snn_network_t *network) {
    if (network != NULL) {
        free(network->row_ptr);
        free(network->col_idx);
        free(network->weights);
        free(network);
    }
}

snn_size_t snn_network_neuron_count(const snn_network_t *network) {
    return network == NULL ? 0u : network->neuron_count;
}

snn_size_t snn_network_synapse_count(const snn_network_t *network) {
    return network == NULL ? 0u : network->synapse_count;
}

snn_architecture_t snn_network_architecture(const snn_network_t *network) {
    return network == NULL ? SNN_ARCH_CUSTOM_CSR : network->architecture;
}

const snn_size_t *snn_network_row_ptr(const snn_network_t *network) {
    return network == NULL ? NULL : network->row_ptr;
}

const snn_size_t *snn_network_col_idx(const snn_network_t *network) {
    return network == NULL ? NULL : network->col_idx;
}

const float *snn_network_weights(const snn_network_t *network) {
    return network == NULL ? NULL : network->weights;
}

snn_lif_params_t snn_network_lif_params(const snn_network_t *network) {
    return network == NULL ? snn_default_lif_params() : network->lif;
}

snn_status_t snn_network_set_lif_params(snn_network_t *network, const snn_lif_params_t *params) {
    snn_lif_params_t lif = effective_params(params);
    if (network == NULL || snn_lif_params_validate(&lif) != SNN_OK) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    network->lif = lif;
    return SNN_OK;
}

snn_status_t snn_state_create(const snn_network_t *network, snn_state_t **out_state) {
    snn_state_t *state = NULL;
    if (network == NULL || out_state == NULL) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    *out_state = NULL;
    state = (snn_state_t *)snn_calloc_impl(1u, sizeof(*state));
    if (state == NULL) {
        return SNN_ERR_OUT_OF_MEMORY;
    }
    state->neuron_count = network->neuron_count;
    state->voltage = (float *)malloc_bytes_u64(network->neuron_count * (uint64_t)sizeof(float));
    state->current = (float *)calloc_u64(network->neuron_count, sizeof(float));
    state->next_current = (float *)calloc_u64(network->neuron_count, sizeof(float));
    state->refractory = (uint32_t *)calloc_u64(network->neuron_count, sizeof(uint32_t));
    state->spikes = (uint8_t *)calloc_u64(network->neuron_count, sizeof(uint8_t));
    if (state->voltage == NULL || state->current == NULL || state->next_current == NULL ||
        state->refractory == NULL || state->spikes == NULL) {
        snn_state_free(state);
        return SNN_ERR_OUT_OF_MEMORY;
    }
    *out_state = state;
    return snn_state_reset(network, state);
}

void snn_state_free(snn_state_t *state) {
    if (state != NULL) {
        free(state->voltage);
        free(state->current);
        free(state->next_current);
        free(state->refractory);
        free(state->spikes);
        free(state);
    }
}

snn_status_t snn_state_reset(const snn_network_t *network, snn_state_t *state) {
    snn_size_t i = 0;
    if (network == NULL || state == NULL || state->neuron_count != network->neuron_count) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < state->neuron_count; ++i) {
        state->voltage[i] = network->lif.v_rest;
        state->current[i] = 0.0f;
        state->next_current[i] = 0.0f;
        state->refractory[i] = 0u;
        state->spikes[i] = 0u;
    }
    return SNN_OK;
}

snn_status_t snn_state_copy_voltage(const snn_state_t *state, float *out_voltage, snn_size_t count) {
    uint64_t bytes = 0;
    if (state == NULL || out_voltage == NULL || count < state->neuron_count ||
        !bytes_for_array_u64(state->neuron_count, sizeof(float), &bytes)) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    memcpy(out_voltage, state->voltage, (size_t)bytes);
    return SNN_OK;
}

snn_status_t snn_state_copy_spikes(const snn_state_t *state, uint8_t *out_spikes, snn_size_t count) {
    uint64_t bytes = 0;
    if (state == NULL || out_spikes == NULL || count < state->neuron_count ||
        !bytes_for_array_u64(state->neuron_count, sizeof(uint8_t), &bytes)) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    memcpy(out_spikes, state->spikes, (size_t)bytes);
    return SNN_OK;
}

snn_status_t snn_step_cpu(const snn_network_t *network,
                          snn_state_t *state,
                          const float *external_current,
                          uint8_t *out_spikes) {
    const float decay = network == NULL ? 0.0f : expf(-network->lif.dt_ms / network->lif.membrane_tau_ms);
    snn_size_t i = 0;
    if (network == NULL || state == NULL || state->neuron_count != network->neuron_count) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    memset(state->next_current, 0, (size_t)(network->neuron_count * (uint64_t)sizeof(float)));
    /*
     * Membrane integration is embarrassingly parallel: iteration i reads and
     * writes only index i. It is parallelized when the library is built with
     * SNN_ENABLE_OPENMP. Synaptic propagation below is left serial so the CPU
     * path stays a deterministic, bit-exact reference for the CUDA backend
     * (scatter-add ordering would otherwise become nondeterministic).
     */
#ifdef SNN_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (i = 0; i < network->neuron_count; ++i) {
        const float ext = external_current == NULL ? 0.0f : external_current[i] * network->lif.input_scale;
        uint8_t spiked = 0u;
        if (state->refractory[i] != 0u) {
            state->refractory[i] -= 1u;
            state->voltage[i] = network->lif.v_reset;
        } else {
            state->voltage[i] = network->lif.v_rest + (state->voltage[i] - network->lif.v_rest) * decay + state->current[i] + ext;
            if (state->voltage[i] >= network->lif.v_threshold) {
                spiked = 1u;
                state->voltage[i] = network->lif.v_reset;
                state->refractory[i] = network->lif.refractory_steps;
            }
        }
        state->spikes[i] = spiked;
        if (out_spikes != NULL) {
            out_spikes[i] = spiked;
        }
    }
    for (i = 0; i < network->neuron_count; ++i) {
        snn_size_t edge = 0;
        if (state->spikes[i] == 0u) {
            continue;
        }
        for (edge = network->row_ptr[i]; edge < network->row_ptr[i + 1u]; ++edge) {
            state->next_current[network->col_idx[edge]] += network->weights[edge];
        }
    }
    {
        float *tmp = state->current;
        state->current = state->next_current;
        state->next_current = tmp;
    }
    return SNN_OK;
}

snn_status_t snn_run_cpu(const snn_network_t *network,
                         snn_state_t *state,
                         const float *external_current_by_step,
                         snn_size_t step_count,
                         snn_size_t input_stride,
                         uint8_t *out_spikes_by_step,
                         snn_size_t output_stride) {
    snn_size_t step = 0;
    if (network == NULL || state == NULL || state->neuron_count != network->neuron_count) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    if ((external_current_by_step != NULL && input_stride < network->neuron_count) ||
        (out_spikes_by_step != NULL && output_stride < network->neuron_count)) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    for (step = 0; step < step_count; ++step) {
        const float *input = external_current_by_step == NULL ? NULL : external_current_by_step + step * input_stride;
        uint8_t *out = out_spikes_by_step == NULL ? NULL : out_spikes_by_step + step * output_stride;
        (void)snn_step_cpu(network, state, input, out);
    }
    return SNN_OK;
}
