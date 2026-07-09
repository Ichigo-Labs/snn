/*
 * Fuzz harness for the host-side API surface: builders (custom CSR,
 * feedforward, random pool), parameter validation, memory planning, state
 * lifecycle, sparse injection, and stepping. Input bytes are parsed into API
 * arguments — including deliberately malformed CSR topology, out-of-range
 * indices, and raw (NaN/Inf) floats — with sizes bounded so allocator OOM
 * does not mask real bugs.
 *
 * Coverage-guided libFuzzer target (clang):
 *   clang -O1 -g -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all \
 *       -Iinclude -Isrc src/snn.c src/snn_cuda_stub.c fuzz/fuzz_api.c -lm -o fuzz_api
 *   ./fuzz_api -runs=1000000 -max_len=1024
 *
 * Standalone randomized driver (any compiler; iterations via argv[1]):
 *   gcc -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
 *       -DSNN_FUZZ_STANDALONE -Iinclude -Isrc \
 *       src/snn.c src/snn_cuda_stub.c fuzz/fuzz_api.c -lm -o fuzz_api
 *   ./fuzz_api 1000000
 */
#include <snn/snn.h>

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const uint8_t *data;
    size_t size;
    size_t pos;
} reader_t;

static uint8_t rd_u8(reader_t *r) {
    return r->pos < r->size ? r->data[r->pos++] : 0u;
}

static uint64_t rd_u64(reader_t *r) {
    uint64_t v = 0;
    int i = 0;
    for (i = 0; i < 8; ++i) {
        v = (v << 8) | rd_u8(r);
    }
    return v;
}

/* Raw bit pattern: NaN, Inf, and denormals are all fair inputs. */
static float rd_f32(reader_t *r) {
    uint32_t bits = 0;
    float f = 0.0f;
    int i = 0;
    for (i = 0; i < 4; ++i) {
        bits = (bits << 8) | rd_u8(r);
    }
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static snn_lif_params_t rd_lif(reader_t *r) {
    snn_lif_params_t p = snn_default_lif_params();
    /* Half the time keep the (valid) defaults, half the time fuzz raw. */
    if (rd_u8(r) & 1u) {
        p.dt_ms = rd_f32(r);
        p.membrane_tau_ms = rd_f32(r);
        p.v_rest = rd_f32(r);
        p.v_reset = rd_f32(r);
        p.v_threshold = rd_f32(r);
        p.input_scale = rd_f32(r);
        p.refractory_steps = rd_u8(r);
    }
    return p;
}

static int status_ok(snn_status_t st) {
    return st == SNN_OK || st == SNN_ERR_INVALID_ARGUMENT || st == SNN_ERR_OUT_OF_MEMORY ||
           st == SNN_ERR_OVERFLOW || st == SNN_ERR_CUDA || st == SNN_ERR_UNSUPPORTED;
}

/* Drive a successfully built network through the full state API. */
static void exercise_network(reader_t *r, snn_network_t *net) {
    const snn_size_t n = snn_network_neuron_count(net);
    snn_memory_plan_t plan;
    snn_state_t *state = NULL;
    float *ext = NULL;
    float *voltage = NULL;
    uint8_t *spikes = NULL;
    snn_size_t inject_idx[8];
    float inject_val[8];
    int step = 0;

    if (!status_ok(snn_network_memory_plan(net, &plan))) {
        abort();
    }
    if (snn_state_create(net, &state) != SNN_OK) {
        return;
    }
    ext = (float *)malloc((size_t)n * sizeof(float));
    voltage = (float *)malloc((size_t)n * sizeof(float));
    spikes = (uint8_t *)malloc((size_t)n);
    if (ext == NULL || voltage == NULL || spikes == NULL) {
        goto done;
    }
    for (step = 0; step < 4; ++step) {
        const uint8_t mode = rd_u8(r);
        snn_size_t i = 0;
        if (mode & 1u) {
            /* Sparse injection with fuzzed (possibly out-of-range) indices. */
            const snn_size_t count = rd_u8(r) % 8u;
            for (i = 0; i < count; ++i) {
                inject_idx[i] = rd_u64(r) % (n + 3u); /* sometimes >= n */
                inject_val[i] = rd_f32(r);
            }
            if (!status_ok(snn_state_inject_current(net, state, inject_idx, inject_val, count))) {
                abort();
            }
        }
        if (mode & 2u) {
            /* Dense external input from raw bits (NaN/Inf reach the step). */
            for (i = 0; i < n; ++i) {
                ext[i] = rd_f32(r);
            }
        }
        if (snn_step_cpu(net, state, (mode & 2u) ? ext : NULL, (mode & 4u) ? spikes : NULL) != SNN_OK) {
            abort(); /* a valid net + state must always step */
        }
    }
    if (snn_state_copy_voltage(state, voltage, n) != SNN_OK ||
        snn_state_copy_spikes(state, spikes, n) != SNN_OK ||
        snn_state_reset(net, state) != SNN_OK) {
        abort();
    }
    if (snn_run_cpu(net, state, NULL, 2, 0, NULL, 0) != SNN_OK) {
        abort();
    }
done:
    free(ext);
    free(voltage);
    free(spikes);
    snn_state_free(state);
}

static void fuzz_custom_csr(reader_t *r) {
    const snn_size_t n = 1u + rd_u8(r) % 48u;
    const snn_size_t s = rd_u8(r) % 96u;
    snn_lif_params_t p = rd_lif(r);
    snn_network_t *net = NULL;
    snn_size_t *row_ptr = (snn_size_t *)malloc((size_t)(n + 1u) * sizeof(snn_size_t));
    snn_size_t *col_idx = (snn_size_t *)malloc((size_t)(s ? s : 1u) * sizeof(snn_size_t));
    float *weights = (float *)malloc((size_t)(s ? s : 1u) * sizeof(float));
    snn_size_t i = 0;
    if (row_ptr == NULL || col_idx == NULL || weights == NULL) {
        goto done;
    }
    if (rd_u8(r) & 1u) {
        /* Well-formed monotonic row_ptr ending at s... */
        snn_size_t edge = 0;
        for (i = 0; i < n; ++i) {
            row_ptr[i] = edge;
            edge += rd_u8(r) % (s - edge + 1u);
        }
        row_ptr[0] = 0;
        row_ptr[n] = s;
    } else {
        /* ...or raw garbage the validator must reject safely. */
        for (i = 0; i <= n; ++i) {
            row_ptr[i] = rd_u64(r) % (s + 4u);
        }
    }
    for (i = 0; i < s; ++i) {
        col_idx[i] = rd_u64(r) % (n + 2u); /* sometimes out of range */
        weights[i] = rd_f32(r);            /* sometimes non-finite */
    }
    if (snn_build_custom_csr(n, s, row_ptr, col_idx, weights, &p, &net) == SNN_OK) {
        exercise_network(r, net);
        snn_network_free(net);
    }
done:
    free(row_ptr);
    free(col_idx);
    free(weights);
}

static void fuzz_feedforward(reader_t *r) {
    snn_size_t layers[6];
    snn_feedforward_config_t cfg;
    snn_lif_params_t p = rd_lif(r);
    snn_network_t *net = NULL;
    const size_t layer_count = rd_u8(r) % 6u; /* includes invalid 0 and 1 */
    size_t i = 0;
    for (i = 0; i < layer_count; ++i) {
        layers[i] = rd_u8(r) % 33u; /* includes invalid 0 */
    }
    cfg = snn_default_feedforward_config(layers, layer_count);
    cfg.fanout_per_neuron = rd_u8(r) % 40u;
    cfg.weight = rd_f32(r);
    cfg.seed = rd_u64(r);
    if (snn_build_feedforward(&cfg, &p, &net) == SNN_OK) {
        exercise_network(r, net);
        snn_network_free(net);
    }
}

static void fuzz_random_pool(reader_t *r) {
    snn_random_pool_config_t cfg = snn_default_random_pool_config(rd_u8(r) % 64u, rd_u8(r) % 24u);
    snn_lif_params_t p = rd_lif(r);
    snn_network_t *net = NULL;
    cfg.weight_min = rd_f32(r);
    cfg.weight_max = rd_f32(r);
    cfg.seed = rd_u64(r);
    cfg.allow_self_connections = rd_u8(r) & 1;
    if (snn_build_random_pool(&cfg, &p, &net) == SNN_OK) {
        exercise_network(r, net);
        snn_network_free(net);
    }
}

static void fuzz_plans_and_validation(reader_t *r) {
    snn_memory_plan_t plan;
    snn_lif_params_t p = rd_lif(r);
    /* Raw u64 counts reach the overflow-checked sizing arithmetic. */
    if (!status_ok(snn_estimate_memory_for_counts(rd_u64(r), rd_u64(r), &plan))) {
        abort();
    }
    if (!status_ok(snn_lif_params_validate(&p))) {
        abort();
    }
    (void)snn_status_string((snn_status_t)(int)rd_u64(r));
    (void)snn_architecture_string((snn_architecture_t)(int)rd_u64(r));
}

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size);

int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
    reader_t r;
    r.data = data;
    r.size = size;
    r.pos = 0;
    switch (rd_u8(&r) % 4u) {
    case 0:
        fuzz_custom_csr(&r);
        break;
    case 1:
        fuzz_feedforward(&r);
        break;
    case 2:
        fuzz_random_pool(&r);
        break;
    default:
        fuzz_plans_and_validation(&r);
        break;
    }
    return 0;
}

#ifdef SNN_FUZZ_STANDALONE
#include <stdio.h>

static uint64_t xorshift(uint64_t *s) {
    uint64_t x = *s;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *s = x;
    return x;
}

int main(int argc, char **argv) {
    const long iterations = argc > 1 ? strtol(argv[1], NULL, 10) : 100000L;
    uint64_t seed = argc > 2 ? (uint64_t)strtoull(argv[2], NULL, 10) : UINT64_C(0x5eedf00d);
    uint8_t buf[768];
    long it = 0;
    for (it = 0; it < iterations; ++it) {
        const size_t len = (size_t)(xorshift(&seed) % (sizeof(buf) + 1u));
        size_t i = 0;
        for (i = 0; i < len; i += 8u) {
            const uint64_t v = xorshift(&seed);
            const size_t chunk = len - i < 8u ? len - i : 8u;
            memcpy(buf + i, &v, chunk);
        }
        (void)LLVMFuzzerTestOneInput(buf, len);
    }
    printf("fuzz_api standalone: %ld iterations, no findings\n", iterations);
    return 0;
}
#endif
