#include "snn/snn_bptt.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "snn_internal.h"

/* sqrt(pi/2) and 1/sqrt(2), for the Gaussian surrogate's primitive. */
#define SNN_SQRT_HALF_PI 1.2533141373155003f
#define SNN_INV_SQRT2 0.70710678118654752f

struct snn_bptt_network {
    size_t neuron_layers; /* L = layer_count - 1 */
    snn_size_t *sizes;    /* L + 1 entries: sizes[j] feeds layer j, sizes[j+1] is its width */
    /* 2L entries: weights of layer j at param_offsets[2j], biases at [2j+1]. */
    snn_size_t *param_offsets;
    float *params;
    snn_size_t parameter_count;
    snn_size_t timesteps;
    snn_size_t max_layer_size; /* max over j of sizes[j+1] */
    float beta;
    float threshold;
    float alpha;
    snn_surrogate_t surrogate;
    int detach_reset;
    /* Test hook (snn_test_bptt_set_soft_spikes): emit S(U - threshold) instead
     * of H(U - threshold), making the whole unrolled model differentiable so a
     * finite-difference check can validate the surrogate backward end to end.
     * Unconditional field so every translation unit sees one struct layout. */
    int soft_spikes;
};

struct snn_bptt_workspace {
    const snn_bptt_network_t *owner;
    float *arena;
    /* 3L entries: u[j] at offsets[j], s[j] at offsets[L+j], gs[j] at offsets[2L+j].
     * The s/gs slots of the output layer (j == L-1) are unused: it never spikes. */
    snn_size_t *offsets;
    snn_size_t gu_tape_off; /* T+1 rows of max_layer_size; row T is the all-zero seed */
    snn_size_t gsum_off;    /* max_layer_size */
    snn_size_t drive_off;   /* max_layer_size: layer 0's drive under static input */
    snn_size_t logits_off;  /* output_size */
    snn_size_t probs_off;   /* output_size */
    snn_size_t output_size;
    const float *input;
    int static_input;
    snn_size_t prediction;
    snn_size_t spike_count;
};

struct snn_bptt_grads {
    const snn_bptt_network_t *owner;
    float *g;
    snn_size_t count;
};

struct snn_bptt_optimizer {
    const snn_bptt_network_t *owner;
    float *m; /* first moment */
    float *v; /* second moment */
    snn_size_t count;
    float lr;
    float beta1;
    float beta2;
    float eps;
    uint64_t steps;
};

static int bptt_add(uint64_t a, uint64_t b, uint64_t *out) {
    if (UINT64_MAX - a < b) {
        return 0;
    }
    *out = a + b;
    return 1;
}

static int bptt_mul(uint64_t a, uint64_t b, uint64_t *out) {
    if (a != 0u && b > UINT64_MAX / a) {
        return 0;
    }
    *out = a * b;
    return 1;
}

/* The SIZE_MAX guards fail cleanly (as OOM) on ILP32 hosts instead of
 * truncating, and constant-fold away on LP64. Kept as one-line ternaries for
 * the same reason snn.c's calloc_u64 is. */
static float *bptt_calloc_floats(uint64_t count) {
    return count > (uint64_t)SIZE_MAX ? NULL : (float *)snn_internal_calloc((size_t)count, sizeof(float));
}

static snn_size_t *bptt_calloc_sizes(uint64_t count) {
    return count > (uint64_t)SIZE_MAX ? NULL : (snn_size_t *)snn_internal_calloc((size_t)count, sizeof(snn_size_t));
}

static uint64_t bptt_rng_next(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static float bptt_rng_uniform(uint64_t *state) {
    const uint64_t r = bptt_rng_next(state);
    return (float)((double)(r >> 11) * (1.0 / 9007199254740992.0));
}

static float sigmoid_f(float z) {
    return 1.0f / (1.0f + expf(-z));
}

/*
 * Static twin of snn_surrogate_grad so the backward pass inlines it: through
 * the public symbol it would cost a call per neuron per timestep per layer.
 */
static float surrogate_grad_at(snn_surrogate_t surrogate, float x, float alpha) {
    switch (surrogate) {
    case SNN_SURROGATE_FAST_SIGMOID: {
        const float d = 1.0f + alpha * fabsf(x);
        return 1.0f / (d * d);
    }
    case SNN_SURROGATE_ATAN: {
        const float a = alpha * x;
        return 1.0f / (1.0f + a * a);
    }
    case SNN_SURROGATE_SIGMOID: {
        const float s = sigmoid_f(alpha * x);
        return 4.0f * s * (1.0f - s);
    }
    case SNN_SURROGATE_TRIANGLE: {
        const float v = 1.0f - alpha * fabsf(x);
        return v > 0.0f ? v : 0.0f;
    }
    case SNN_SURROGATE_GAUSSIAN: {
        const float a = alpha * x;
        return expf(-0.5f * a * a);
    }
    case SNN_SURROGATE_RECTANGULAR:
        return alpha * fabsf(x) < 1.0f ? 1.0f : 0.0f;
    default:
        return 0.0f;
    }
}

const char *snn_surrogate_string(snn_surrogate_t surrogate) {
    switch (surrogate) {
    case SNN_SURROGATE_FAST_SIGMOID:
        return "fast_sigmoid";
    case SNN_SURROGATE_ATAN:
        return "atan";
    case SNN_SURROGATE_SIGMOID:
        return "sigmoid";
    case SNN_SURROGATE_TRIANGLE:
        return "triangle";
    case SNN_SURROGATE_GAUSSIAN:
        return "gaussian";
    case SNN_SURROGATE_RECTANGULAR:
        return "rectangular";
    case SNN_SURROGATE_COUNT:
    default:
        return "unknown";
    }
}

float snn_surrogate_grad(snn_surrogate_t surrogate, float x, float alpha) {
    return surrogate_grad_at(surrogate, x, alpha);
}

float snn_surrogate_primitive(snn_surrogate_t surrogate, float x, float alpha) {
    switch (surrogate) {
    case SNN_SURROGATE_FAST_SIGMOID:
        /* In float, alpha*|x| overflows to inf for |x| > FLT_MAX/alpha, and
         * x/inf collapses to 0 -- so S would fall back to exactly 1/2 right
         * where it should saturate at 1/2 + sign(x)/alpha, making it
         * non-monotone across one float boundary. A double cannot overflow. */
        return (float)(0.5 + (double)x / (1.0 + (double)alpha * fabs((double)x)));
    case SNN_SURROGATE_ATAN:
        return 0.5f + atanf(alpha * x) / alpha;
    case SNN_SURROGATE_SIGMOID:
        return 0.5f + (4.0f / alpha) * (sigmoid_f(alpha * x) - 0.5f);
    case SNN_SURROGATE_TRIANGLE: {
        /* Antiderivative of max(0, 1 - alpha|x|): a ramp with a quadratic knee
         * that flattens at +-1/alpha, where it reaches 1/2 +- 1/(2 alpha). */
        const float ax = fabsf(x);
        const float half_area = 0.5f / alpha;
        return ax * alpha >= 1.0f ? (x >= 0.0f ? 0.5f + half_area : 0.5f - half_area)
                                  : 0.5f + x - 0.5f * alpha * x * ax;
    }
    case SNN_SURROGATE_GAUSSIAN:
        return 0.5f + (SNN_SQRT_HALF_PI / alpha) * erff(alpha * x * SNN_INV_SQRT2);
    case SNN_SURROGATE_RECTANGULAR: {
        const float lim = 1.0f / alpha;
        return 0.5f + (x < -lim ? -lim : (x > lim ? lim : x));
    }
    default:
        return 0.0f;
    }
}

snn_bptt_config_t snn_bptt_default_config(const snn_size_t *layer_sizes, size_t layer_count, snn_size_t timesteps) {
    snn_bptt_config_t cfg;
    cfg.layer_sizes = layer_sizes;
    cfg.layer_count = layer_count;
    cfg.timesteps = timesteps;
    cfg.beta = 0.95f;
    cfg.threshold = 1.0f;
    cfg.surrogate = SNN_SURROGATE_ATAN;
    cfg.surrogate_alpha = 2.0f;
    cfg.detach_reset = 0;
    cfg.weight_init_gain = 1.0f;
    cfg.seed = UINT64_C(0x534e4e5f62707474);
    return cfg;
}

static int surrogate_known(snn_surrogate_t surrogate) {
    return (int)surrogate >= 0 && (int)surrogate < (int)SNN_SURROGATE_COUNT;
}

snn_status_t snn_bptt_config_validate(const snn_bptt_config_t *config) {
    size_t i = 0;
    if (config == NULL || config->layer_sizes == NULL || config->layer_count < 2u) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    if (config->timesteps == 0u) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    if (!isfinite(config->beta) || config->beta < 0.0f || config->beta >= 1.0f) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    if (!isfinite(config->threshold) || config->threshold <= 0.0f) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    if (!surrogate_known(config->surrogate)) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    if (!isfinite(config->surrogate_alpha) || config->surrogate_alpha <= 0.0f) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    if (!isfinite(config->weight_init_gain) || config->weight_init_gain <= 0.0f) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < config->layer_count; ++i) {
        if (config->layer_sizes[i] == 0u) {
            return SNN_ERR_INVALID_ARGUMENT;
        }
    }
    return SNN_OK;
}

float snn_bptt_beta_from_lif(const snn_lif_params_t *params) {
    return snn_lif_params_validate(params) != SNN_OK ? 0.0f : expf(-params->dt_ms / params->membrane_tau_ms);
}

snn_status_t snn_bptt_network_create(const snn_bptt_config_t *config, snn_bptt_network_t **out_network) {
    snn_bptt_network_t *net = NULL;
    size_t layers = 0;
    size_t j = 0;
    uint64_t total = 0;
    uint64_t running = 0;
    uint64_t rng = 0;
    snn_status_t st = SNN_OK;

    if (out_network == NULL) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    *out_network = NULL;
    st = snn_bptt_config_validate(config);
    if (st != SNN_OK) {
        return st;
    }
    layers = config->layer_count - 1u;
    for (j = 0; j < layers; ++j) {
        uint64_t weights = 0;
        if (!bptt_mul(config->layer_sizes[j + 1u], config->layer_sizes[j], &weights) ||
            !bptt_add(total, weights, &total) || !bptt_add(total, config->layer_sizes[j + 1u], &total)) {
            return SNN_ERR_OVERFLOW;
        }
    }

    net = (snn_bptt_network_t *)snn_internal_calloc(1u, sizeof(*net));
    if (net == NULL) {
        return SNN_ERR_OUT_OF_MEMORY;
    }
    net->neuron_layers = layers;
    net->sizes = bptt_calloc_sizes((uint64_t)layers + 1u);
    net->param_offsets = bptt_calloc_sizes((uint64_t)layers * 2u);
    net->params = bptt_calloc_floats(total);
    if (net->sizes == NULL || net->param_offsets == NULL || net->params == NULL) {
        snn_bptt_network_free(net);
        return SNN_ERR_OUT_OF_MEMORY;
    }

    for (j = 0; j <= layers; ++j) {
        net->sizes[j] = config->layer_sizes[j];
    }
    for (j = 0; j < layers; ++j) {
        const snn_size_t rows = net->sizes[j + 1u];
        const snn_size_t cols = net->sizes[j];
        net->param_offsets[2u * j] = (snn_size_t)running;
        running += (uint64_t)rows * (uint64_t)cols;
        net->param_offsets[2u * j + 1u] = (snn_size_t)running;
        running += rows;
        if (rows > net->max_layer_size) {
            net->max_layer_size = rows;
        }
    }
    net->parameter_count = (snn_size_t)total;
    net->timesteps = config->timesteps;
    net->beta = config->beta;
    net->threshold = config->threshold;
    net->alpha = config->surrogate_alpha;
    net->surrogate = config->surrogate;
    net->detach_reset = config->detach_reset;

    /* Kaiming-uniform: U(-limit, limit) with limit = gain*sqrt(3/fan_in), so
     * Var[w] = gain^2 / fan_in. Biases stay at the calloc'd zero. */
    rng = config->seed;
    for (j = 0; j < layers; ++j) {
        const snn_size_t rows = net->sizes[j + 1u];
        const snn_size_t cols = net->sizes[j];
        const float limit = config->weight_init_gain * sqrtf(3.0f / (float)cols);
        float *w = net->params + net->param_offsets[2u * j];
        const uint64_t count = (uint64_t)rows * (uint64_t)cols;
        uint64_t i = 0;
        for (i = 0; i < count; ++i) {
            w[i] = (2.0f * bptt_rng_uniform(&rng) - 1.0f) * limit;
        }
    }
    *out_network = net;
    return SNN_OK;
}

void snn_bptt_network_free(snn_bptt_network_t *network) {
    if (network != NULL) {
        free(network->sizes);
        free(network->param_offsets);
        free(network->params);
        free(network);
    }
}

size_t snn_bptt_layer_count(const snn_bptt_network_t *network) {
    return network == NULL ? 0u : network->neuron_layers + 1u;
}

snn_size_t snn_bptt_layer_size(const snn_bptt_network_t *network, size_t layer) {
    return (network == NULL || layer > network->neuron_layers) ? 0u : network->sizes[layer];
}

snn_size_t snn_bptt_input_size(const snn_bptt_network_t *network) {
    return network == NULL ? 0u : network->sizes[0];
}

snn_size_t snn_bptt_output_size(const snn_bptt_network_t *network) {
    return network == NULL ? 0u : network->sizes[network->neuron_layers];
}

snn_size_t snn_bptt_timesteps(const snn_bptt_network_t *network) {
    return network == NULL ? 0u : network->timesteps;
}

snn_size_t snn_bptt_parameter_count(const snn_bptt_network_t *network) {
    return network == NULL ? 0u : network->parameter_count;
}

snn_surrogate_t snn_bptt_network_surrogate(const snn_bptt_network_t *network) {
    return network == NULL ? SNN_SURROGATE_COUNT : network->surrogate;
}

float snn_bptt_network_alpha(const snn_bptt_network_t *network) {
    return network == NULL ? 0.0f : network->alpha;
}

int snn_bptt_network_detach_reset(const snn_bptt_network_t *network) {
    return network == NULL ? 0 : network->detach_reset;
}

snn_status_t snn_bptt_network_set_surrogate(snn_bptt_network_t *network, snn_surrogate_t surrogate, float alpha) {
    if (network == NULL || !surrogate_known(surrogate) || !isfinite(alpha) || alpha <= 0.0f) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    network->surrogate = surrogate;
    network->alpha = alpha;
    return SNN_OK;
}

snn_status_t snn_bptt_get_parameters(const snn_bptt_network_t *network, float *out_params, snn_size_t capacity) {
    if (network == NULL || out_params == NULL || capacity < network->parameter_count) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    memcpy(out_params, network->params, (size_t)network->parameter_count * sizeof(float));
    return SNN_OK;
}

snn_status_t snn_bptt_set_parameters(snn_bptt_network_t *network, const float *params, snn_size_t capacity) {
    snn_size_t i = 0;
    if (network == NULL || params == NULL || capacity < network->parameter_count) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < network->parameter_count; ++i) {
        if (!isfinite(params[i])) {
            return SNN_ERR_INVALID_ARGUMENT;
        }
    }
    memcpy(network->params, params, (size_t)network->parameter_count * sizeof(float));
    return SNN_OK;
}

/*
 * Assigns every arena offset and returns the total float count, or 0 on any
 * overflow. Written without early returns so that a single caller-side
 * SNN_ERR_OVERFLOW covers every checked add and multiply: the sizes and
 * timestep count are caller-controlled 64-bit values, but only one of the
 * dozen products can be made to overflow first for any given input.
 */
static int workspace_layout(const snn_bptt_network_t *net, snn_bptt_workspace_t *ws, uint64_t *out_floats) {
    const size_t layers = net->neuron_layers;
    const uint64_t timesteps = net->timesteps;
    const uint64_t max_size = net->max_layer_size;
    uint64_t running = 0;
    uint64_t block = 0;
    size_t j = 0;
    int ok = 1;

    for (j = 0; j < layers; ++j) {
        ws->offsets[j] = (snn_size_t)running;
        ok &= bptt_mul(timesteps, net->sizes[j + 1u], &block) && bptt_add(running, block, &running);
    }
    for (j = 0; j + 1u < layers; ++j) {
        ws->offsets[layers + j] = (snn_size_t)running;
        ok &= bptt_mul(timesteps, net->sizes[j + 1u], &block) && bptt_add(running, block, &running);
    }
    for (j = 0; j + 1u < layers; ++j) {
        ws->offsets[2u * layers + j] = (snn_size_t)running;
        ok &= bptt_mul(timesteps, net->sizes[j + 1u], &block) && bptt_add(running, block, &running);
    }
    ws->gu_tape_off = (snn_size_t)running;
    ok &= bptt_mul(timesteps, max_size, &block) && bptt_add(block, max_size, &block) &&
          bptt_add(running, block, &running);
    ws->gsum_off = (snn_size_t)running;
    ok &= bptt_add(running, max_size, &running);
    ws->drive_off = (snn_size_t)running;
    ok &= bptt_add(running, max_size, &running);
    ws->logits_off = (snn_size_t)running;
    ok &= bptt_add(running, ws->output_size, &running);
    ws->probs_off = (snn_size_t)running;
    ok &= bptt_add(running, ws->output_size, &running);
    *out_floats = running;
    return ok;
}

snn_status_t snn_bptt_workspace_create(const snn_bptt_network_t *network, snn_bptt_workspace_t **out_workspace) {
    snn_bptt_workspace_t *ws = NULL;
    uint64_t floats = 0;

    if (out_workspace == NULL) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    *out_workspace = NULL;
    if (network == NULL) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    ws = (snn_bptt_workspace_t *)snn_internal_calloc(1u, sizeof(*ws));
    if (ws == NULL) {
        return SNN_ERR_OUT_OF_MEMORY;
    }
    ws->owner = network;
    ws->output_size = network->sizes[network->neuron_layers];
    ws->offsets = bptt_calloc_sizes((uint64_t)network->neuron_layers * 3u);
    if (ws->offsets == NULL) {
        snn_bptt_workspace_free(ws);
        return SNN_ERR_OUT_OF_MEMORY;
    }
    if (!workspace_layout(network, ws, &floats)) {
        snn_bptt_workspace_free(ws);
        return SNN_ERR_OVERFLOW;
    }
    ws->arena = bptt_calloc_floats(floats);
    if (ws->arena == NULL) {
        snn_bptt_workspace_free(ws);
        return SNN_ERR_OUT_OF_MEMORY;
    }
    *out_workspace = ws;
    return SNN_OK;
}

void snn_bptt_workspace_free(snn_bptt_workspace_t *workspace) {
    if (workspace != NULL) {
        free(workspace->offsets);
        free(workspace->arena);
        free(workspace);
    }
}

static float *ws_u(const snn_bptt_workspace_t *ws, size_t j) {
    return ws->arena + ws->offsets[j];
}

static float *ws_s(const snn_bptt_workspace_t *ws, size_t j) {
    return ws->arena + ws->offsets[ws->owner->neuron_layers + j];
}

static float *ws_gs(const snn_bptt_workspace_t *ws, size_t j) {
    return ws->arena + ws->offsets[2u * ws->owner->neuron_layers + j];
}

snn_status_t snn_bptt_grads_create(const snn_bptt_network_t *network, snn_bptt_grads_t **out_grads) {
    snn_bptt_grads_t *grads = NULL;
    if (out_grads == NULL) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    *out_grads = NULL;
    if (network == NULL) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    grads = (snn_bptt_grads_t *)snn_internal_calloc(1u, sizeof(*grads));
    if (grads == NULL) {
        return SNN_ERR_OUT_OF_MEMORY;
    }
    grads->owner = network;
    grads->count = network->parameter_count;
    grads->g = bptt_calloc_floats(grads->count);
    if (grads->g == NULL) {
        snn_bptt_grads_free(grads);
        return SNN_ERR_OUT_OF_MEMORY;
    }
    *out_grads = grads;
    return SNN_OK;
}

void snn_bptt_grads_free(snn_bptt_grads_t *grads) {
    if (grads != NULL) {
        free(grads->g);
        free(grads);
    }
}

snn_status_t snn_bptt_grads_zero(snn_bptt_grads_t *grads) {
    if (grads == NULL) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    memset(grads->g, 0, (size_t)grads->count * sizeof(float));
    return SNN_OK;
}

snn_status_t snn_bptt_grads_add(snn_bptt_grads_t *dst, const snn_bptt_grads_t *src) {
    snn_size_t i = 0;
    if (dst == NULL || src == NULL || dst->owner != src->owner) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    for (i = 0; i < dst->count; ++i) {
        dst->g[i] += src->g[i];
    }
    return SNN_OK;
}

snn_status_t snn_bptt_grads_copy_out(const snn_bptt_grads_t *grads, float *out_grads, snn_size_t capacity) {
    if (grads == NULL || out_grads == NULL || capacity < grads->count) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    memcpy(out_grads, grads->g, (size_t)grads->count * sizeof(float));
    return SNN_OK;
}

/* out = W*pre + b, with W row-major (rows x cols). */
static void matvec(const float *restrict w,
                   const float *restrict b,
                   const float *restrict pre,
                   float *restrict out,
                   snn_size_t rows,
                   snn_size_t cols) {
    snn_size_t r = 0;
    for (r = 0; r < rows; ++r) {
        const float *restrict row = w + (size_t)r * (size_t)cols;
        float acc = b[r];
        snn_size_t c = 0;
        for (c = 0; c < cols; ++c) {
            acc += row[c] * pre[c];
        }
        out[r] = acc;
    }
}

/* gw += outer(g, pre). */
static void rank1_accumulate(float *restrict gw,
                             const float *restrict g,
                             const float *restrict pre,
                             snn_size_t rows,
                             snn_size_t cols) {
    snn_size_t r = 0;
    for (r = 0; r < rows; ++r) {
        float *restrict row = gw + (size_t)r * (size_t)cols;
        const float gr = g[r];
        snn_size_t c = 0;
        for (c = 0; c < cols; ++c) {
            row[c] += gr * pre[c];
        }
    }
}

/* out = W^T * g, accumulated row-wise so W is walked in memory order. */
static void transposed_matvec(const float *restrict w,
                              const float *restrict g,
                              float *restrict out,
                              snn_size_t rows,
                              snn_size_t cols) {
    snn_size_t r = 0;
    snn_size_t c = 0;
    for (c = 0; c < cols; ++c) {
        out[c] = 0.0f;
    }
    for (r = 0; r < rows; ++r) {
        const float *restrict row = w + (size_t)r * (size_t)cols;
        const float gr = g[r];
        for (c = 0; c < cols; ++c) {
            out[c] += row[c] * gr;
        }
    }
}

static int all_finite(const float *values, uint64_t count) {
    uint64_t i = 0;
    for (i = 0; i < count; ++i) {
        if (!isfinite(values[i])) {
            return 0;
        }
    }
    return 1;
}

snn_status_t snn_bptt_forward(const snn_bptt_network_t *network,
                              snn_bptt_workspace_t *workspace,
                              const float *input,
                              int static_input) {
    size_t layers = 0;
    size_t j = 0;
    snn_size_t t = 0;
    snn_size_t k = 0;
    snn_size_t in_size = 0;
    snn_size_t out_size = 0;
    snn_size_t timesteps = 0;
    snn_size_t spikes = 0;
    snn_size_t best = 0;
    float threshold = 0.0f;
    float beta = 0.0f;
    int soft = 0;
    const float *u_out = NULL;
    float *logits = NULL;

    if (network == NULL || workspace == NULL || input == NULL || workspace->owner != network) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    layers = network->neuron_layers;
    timesteps = network->timesteps;
    in_size = network->sizes[0];
    out_size = network->sizes[layers];
    threshold = network->threshold;
    beta = network->beta;
    soft = network->soft_spikes;
    /* Validated one frame at a time: the whole-tape element count
     * timesteps*in_size is a product of two caller-controlled 64-bit values,
     * and a wrapped count would silently scan less than it claimed to. */
    {
        const uint64_t frames = static_input ? 1u : (uint64_t)timesteps;
        uint64_t f = 0;
        for (f = 0; f < frames; ++f) {
            if (!all_finite(input + (size_t)f * (size_t)in_size, in_size)) {
                return SNN_ERR_INVALID_ARGUMENT;
            }
        }
    }
    workspace->input = input;
    workspace->static_input = static_input;

    /* Under a static input the layer-0 drive W[0]*input + b[0] does not depend
     * on t: compute it once instead of T times. On 784-N-10 that single matvec
     * is most of the forward pass. */
    if (static_input) {
        matvec(network->params + network->param_offsets[0], network->params + network->param_offsets[1], input,
               workspace->arena + workspace->drive_off, network->sizes[1], in_size);
    }

    for (t = 0; t < timesteps; ++t) {
        for (j = 0; j < layers; ++j) {
            const snn_size_t rows = network->sizes[j + 1u];
            const snn_size_t cols = network->sizes[j];
            float *u_t = ws_u(workspace, j) + (size_t)t * (size_t)rows;
            snn_size_t i = 0;

            if (j == 0u) {
                if (static_input) {
                    memcpy(u_t, workspace->arena + workspace->drive_off, (size_t)rows * sizeof(float));
                } else {
                    matvec(network->params + network->param_offsets[0], network->params + network->param_offsets[1],
                           input + (size_t)t * (size_t)in_size, u_t, rows, cols);
                }
            } else {
                matvec(network->params + network->param_offsets[2u * j],
                       network->params + network->param_offsets[2u * j + 1u],
                       ws_s(workspace, j - 1u) + (size_t)t * (size_t)cols, u_t, rows, cols);
            }

            if (t > 0u) {
                const float *u_prev = ws_u(workspace, j) + (size_t)(t - 1u) * (size_t)rows;
                for (i = 0; i < rows; ++i) {
                    u_t[i] += beta * u_prev[i];
                }
                if (j + 1u < layers) {
                    const float *s_prev = ws_s(workspace, j) + (size_t)(t - 1u) * (size_t)rows;
                    for (i = 0; i < rows; ++i) {
                        u_t[i] -= threshold * s_prev[i];
                    }
                }
            }

            if (j + 1u < layers) {
                float *s_t = ws_s(workspace, j) + (size_t)t * (size_t)rows;
                for (i = 0; i < rows; ++i) {
                    const int fired = u_t[i] - threshold >= 0.0f;
                    spikes += (snn_size_t)fired;
                    s_t[i] = fired ? 1.0f : 0.0f;
                }
                if (soft) {
                    for (i = 0; i < rows; ++i) {
                        s_t[i] = snn_surrogate_primitive(network->surrogate, u_t[i] - threshold, network->alpha);
                    }
                }
            }
        }
    }

    /* logits = mean over time of the output layer's membrane potential. */
    u_out = ws_u(workspace, layers - 1u);
    logits = workspace->arena + workspace->logits_off;
    for (k = 0; k < out_size; ++k) {
        float acc = 0.0f;
        for (t = 0; t < timesteps; ++t) {
            acc += u_out[(size_t)t * (size_t)out_size + k];
        }
        logits[k] = acc / (float)timesteps;
    }
    for (k = 1; k < out_size; ++k) {
        if (logits[k] > logits[best]) {
            best = k;
        }
    }
    workspace->prediction = best;
    workspace->spike_count = spikes;
    return SNN_OK;
}

snn_status_t snn_bptt_copy_logits(const snn_bptt_workspace_t *workspace, float *out_logits, snn_size_t capacity) {
    if (workspace == NULL || out_logits == NULL || capacity < workspace->output_size) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    memcpy(out_logits, workspace->arena + workspace->logits_off, (size_t)workspace->output_size * sizeof(float));
    return SNN_OK;
}

snn_status_t snn_bptt_cross_entropy(const snn_bptt_workspace_t *workspace, snn_size_t label, float *out_loss) {
    const float *logits = NULL;
    float max_logit = 0.0f;
    float sum = 0.0f;
    snn_size_t k = 0;
    if (workspace == NULL || out_loss == NULL || label >= workspace->output_size) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    logits = workspace->arena + workspace->logits_off;
    max_logit = logits[0];
    for (k = 1; k < workspace->output_size; ++k) {
        if (logits[k] > max_logit) {
            max_logit = logits[k];
        }
    }
    for (k = 0; k < workspace->output_size; ++k) {
        sum += expf(logits[k] - max_logit);
    }
    /* logsumexp form: no probability is formed that could underflow to zero. */
    *out_loss = (max_logit + logf(sum)) - logits[label];
    return SNN_OK;
}

snn_size_t snn_bptt_prediction(const snn_bptt_workspace_t *workspace) {
    return workspace == NULL ? 0u : workspace->prediction;
}

snn_size_t snn_bptt_spike_count(const snn_bptt_workspace_t *workspace) {
    return workspace == NULL ? 0u : workspace->spike_count;
}

snn_status_t snn_bptt_forward_backward(const snn_bptt_network_t *network,
                                       snn_bptt_workspace_t *workspace,
                                       const float *input,
                                       int static_input,
                                       snn_size_t label,
                                       snn_bptt_grads_t *grads,
                                       float *out_loss,
                                       int *out_correct) {
    size_t layers = 0;
    size_t j = 0;
    snn_size_t k = 0;
    snn_size_t t = 0;
    snn_size_t timesteps = 0;
    snn_size_t out_size = 0;
    snn_size_t in_size = 0;
    snn_size_t max_size = 0;
    float threshold = 0.0f;
    float beta = 0.0f;
    float alpha = 0.0f;
    float inv_t = 0.0f;
    float max_logit = 0.0f;
    float sum = 0.0f;
    int detach = 0;
    snn_surrogate_t surrogate = SNN_SURROGATE_ATAN;
    const float *logits = NULL;
    float *gz = NULL;
    float *gu_tape = NULL;
    snn_status_t st = SNN_OK;

    if (network == NULL || grads == NULL || grads->owner != network) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    if (label >= network->sizes[network->neuron_layers]) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    st = snn_bptt_forward(network, workspace, input, static_input);
    if (st != SNN_OK) {
        return st;
    }

    layers = network->neuron_layers;
    timesteps = network->timesteps;
    out_size = network->sizes[layers];
    in_size = network->sizes[0];
    max_size = network->max_layer_size;
    threshold = network->threshold;
    beta = network->beta;
    alpha = network->alpha;
    surrogate = network->surrogate;
    detach = network->detach_reset;
    inv_t = 1.0f / (float)timesteps;
    logits = workspace->arena + workspace->logits_off;
    gz = workspace->arena + workspace->probs_off;
    gu_tape = workspace->arena + workspace->gu_tape_off;

    /* dLoss/dlogits = softmax(logits) - onehot(label). */
    max_logit = logits[0];
    for (k = 1; k < out_size; ++k) {
        if (logits[k] > max_logit) {
            max_logit = logits[k];
        }
    }
    for (k = 0; k < out_size; ++k) {
        gz[k] = expf(logits[k] - max_logit);
        sum += gz[k];
    }
    for (k = 0; k < out_size; ++k) {
        gz[k] /= sum;
    }
    if (out_loss != NULL) {
        *out_loss = (max_logit + logf(sum)) - logits[label];
    }
    if (out_correct != NULL) {
        *out_correct = workspace->prediction == label;
    }
    gz[label] -= 1.0f;

    j = layers;
    while (j-- > 0u) {
        const snn_size_t rows = network->sizes[j + 1u];
        const snn_size_t cols = network->sizes[j];
        float *gw = grads->g + network->param_offsets[2u * j];
        float *gb = grads->g + network->param_offsets[2u * j + 1u];
        const float *w = network->params + network->param_offsets[2u * j];
        const int is_output = (j + 1u == layers);

        /* Row T of the gu tape is the zero seed for dLoss/dU[j][T]. */
        memset(gu_tape + (size_t)timesteps * (size_t)max_size, 0, (size_t)rows * sizeof(float));

        t = timesteps;
        while (t-- > 0u) {
            float *gu_t = gu_tape + (size_t)t * (size_t)max_size;
            const float *gu_next = gu_tape + (size_t)(t + 1u) * (size_t)max_size;
            snn_size_t i = 0;

            if (is_output) {
                for (i = 0; i < rows; ++i) {
                    gu_t[i] = gz[i] * inv_t + beta * gu_next[i];
                }
            } else {
                const float *u_t = ws_u(workspace, j) + (size_t)t * (size_t)rows;
                const float *gs_t = ws_gs(workspace, j) + (size_t)t * (size_t)rows;
                for (i = 0; i < rows; ++i) {
                    /* s[j][t] reaches the loss two ways: through the next layer
                     * at the same t (already summed into gs_t), and through its
                     * own reset of U[j][t+1], whose coefficient is -threshold. */
                    const float gs = detach ? gs_t[i] : gs_t[i] - threshold * gu_next[i];
                    const float phi = surrogate_grad_at(surrogate, u_t[i] - threshold, alpha);
                    gu_t[i] = gs * phi + beta * gu_next[i];
                }
            }
            if (j > 0u) {
                transposed_matvec(w, gu_t, ws_gs(workspace, j - 1u) + (size_t)t * (size_t)cols, rows, cols);
            }
        }

        if (j == 0u && workspace->static_input) {
            /* pre[0][t] is the same vector at every t, so sum_t outer(gu[t], input)
             * collapses T rank-1 updates into one. */
            float *gsum = workspace->arena + workspace->gsum_off;
            snn_size_t i = 0;
            memset(gsum, 0, (size_t)rows * sizeof(float));
            for (t = 0; t < timesteps; ++t) {
                const float *gu_t = gu_tape + (size_t)t * (size_t)max_size;
                for (i = 0; i < rows; ++i) {
                    gsum[i] += gu_t[i];
                }
            }
            for (i = 0; i < rows; ++i) {
                gb[i] += gsum[i];
            }
            rank1_accumulate(gw, gsum, workspace->input, rows, cols);
        } else {
            for (t = 0; t < timesteps; ++t) {
                const float *gu_t = gu_tape + (size_t)t * (size_t)max_size;
                const float *pre = j == 0u ? workspace->input + (size_t)t * (size_t)in_size
                                           : ws_s(workspace, j - 1u) + (size_t)t * (size_t)cols;
                snn_size_t i = 0;
                for (i = 0; i < rows; ++i) {
                    gb[i] += gu_t[i];
                }
                rank1_accumulate(gw, gu_t, pre, rows, cols);
            }
        }
    }
    return SNN_OK;
}

snn_status_t snn_bptt_optimizer_create(const snn_bptt_network_t *network,
                                       float lr,
                                       float beta1,
                                       float beta2,
                                       float eps,
                                       snn_bptt_optimizer_t **out_optimizer) {
    snn_bptt_optimizer_t *opt = NULL;
    if (out_optimizer == NULL) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    *out_optimizer = NULL;
    if (network == NULL || !isfinite(lr) || lr <= 0.0f || !isfinite(beta1) || beta1 < 0.0f || beta1 >= 1.0f ||
        !isfinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f || !isfinite(eps) || eps <= 0.0f) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    opt = (snn_bptt_optimizer_t *)snn_internal_calloc(1u, sizeof(*opt));
    if (opt == NULL) {
        return SNN_ERR_OUT_OF_MEMORY;
    }
    opt->owner = network;
    opt->count = network->parameter_count;
    opt->lr = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps = eps;
    opt->m = bptt_calloc_floats(opt->count);
    opt->v = bptt_calloc_floats(opt->count);
    if (opt->m == NULL || opt->v == NULL) {
        snn_bptt_optimizer_free(opt);
        return SNN_ERR_OUT_OF_MEMORY;
    }
    *out_optimizer = opt;
    return SNN_OK;
}

void snn_bptt_optimizer_free(snn_bptt_optimizer_t *optimizer) {
    if (optimizer != NULL) {
        free(optimizer->m);
        free(optimizer->v);
        free(optimizer);
    }
}

snn_status_t snn_bptt_optimizer_set_lr(snn_bptt_optimizer_t *optimizer, float lr) {
    if (optimizer == NULL || !isfinite(lr) || lr <= 0.0f) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    optimizer->lr = lr;
    return SNN_OK;
}

snn_status_t snn_bptt_optimizer_step(snn_bptt_optimizer_t *optimizer,
                                     snn_bptt_network_t *network,
                                     const snn_bptt_grads_t *grads,
                                     snn_size_t batch_size) {
    snn_size_t i = 0;
    float scale = 0.0f;
    float bias1 = 0.0f;
    float bias2 = 0.0f;
    if (optimizer == NULL || network == NULL || grads == NULL || optimizer->owner != network ||
        grads->owner != network || batch_size == 0u) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    /* Reject a poisoned batch before it touches any state. The moments are
     * exponential moving averages, so a single non-finite gradient would never
     * decay back out of them: one overflowing forward would quietly destroy
     * the optimizer for the rest of training. */
    if (!all_finite(grads->g, grads->count)) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    optimizer->steps += 1u;
    scale = 1.0f / (float)batch_size;
    /* beta in [0,1) and steps >= 1, so both correction terms lie in (0, 1]. */
    bias1 = 1.0f - powf(optimizer->beta1, (float)optimizer->steps);
    bias2 = 1.0f - powf(optimizer->beta2, (float)optimizer->steps);
    for (i = 0; i < optimizer->count; ++i) {
        const float g = grads->g[i] * scale;
        /* Raw moments; the bias correction divides them in the update below,
         * and eps sits outside the sqrt, as in Adam's algorithm 1. */
        const float m = optimizer->beta1 * optimizer->m[i] + (1.0f - optimizer->beta1) * g;
        const float v = optimizer->beta2 * optimizer->v[i] + (1.0f - optimizer->beta2) * g * g;
        optimizer->m[i] = m;
        optimizer->v[i] = v;
        network->params[i] -= optimizer->lr * (m / bias1) / (sqrtf(v / bias2) + optimizer->eps);
    }
    return SNN_OK;
}

#ifdef SNN_ENABLE_TEST_HOOKS
void snn_test_bptt_set_soft_spikes(snn_bptt_network_t *network, int enable) {
    if (network != NULL) {
        network->soft_spikes = enable;
    }
}
#endif
