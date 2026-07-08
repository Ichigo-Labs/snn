#include <snn/snn.h>
#include <snn/snn_test.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ASSERT_TRUE(EXPR)                                                                 \
    do {                                                                                  \
        if (!(EXPR)) {                                                                    \
            fprintf(stderr, "ASSERT_TRUE failed at %s:%d: %s\n", __FILE__, __LINE__, #EXPR); \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

#define ASSERT_EQ_U64(A, B)                                                               \
    do {                                                                                  \
        unsigned long long va_ = (unsigned long long)(A);                                 \
        unsigned long long vb_ = (unsigned long long)(B);                                 \
        if (va_ != vb_) {                                                                 \
            fprintf(stderr, "ASSERT_EQ_U64 failed at %s:%d: %s=%llu %s=%llu\n", __FILE__, __LINE__, #A, va_, #B, vb_); \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

#define ASSERT_EQ_INT(A, B)                                                               \
    do {                                                                                  \
        int va_ = (int)(A);                                                               \
        int vb_ = (int)(B);                                                               \
        if (va_ != vb_) {                                                                 \
            fprintf(stderr, "ASSERT_EQ_INT failed at %s:%d: %s=%d %s=%d\n", __FILE__, __LINE__, #A, va_, #B, vb_); \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

#define ASSERT_NEAR(A, B, EPS)                                                            \
    do {                                                                                  \
        float va_ = (float)(A);                                                           \
        float vb_ = (float)(B);                                                           \
        float eps_ = (float)(EPS);                                                        \
        if (fabsf(va_ - vb_) > eps_) {                                                    \
            fprintf(stderr, "ASSERT_NEAR failed at %s:%d: %s=%f %s=%f\n", __FILE__, __LINE__, #A, va_, #B, vb_); \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

static void test_strings_and_defaults(void) {
    snn_lif_params_t p = snn_default_lif_params();
    snn_feedforward_config_t ff;
    snn_random_pool_config_t rp;
    snn_size_t layers[] = {1, 2};

    ASSERT_TRUE(strcmp(snn_status_string(SNN_OK), "ok") == 0);
    ASSERT_TRUE(strcmp(snn_status_string(SNN_ERR_INVALID_ARGUMENT), "invalid argument") == 0);
    ASSERT_TRUE(strcmp(snn_status_string(SNN_ERR_OUT_OF_MEMORY), "out of memory") == 0);
    ASSERT_TRUE(strcmp(snn_status_string(SNN_ERR_OVERFLOW), "integer overflow") == 0);
    ASSERT_TRUE(strcmp(snn_status_string(SNN_ERR_CUDA), "cuda error") == 0);
    ASSERT_TRUE(strcmp(snn_status_string(SNN_ERR_UNSUPPORTED), "unsupported") == 0);
    ASSERT_TRUE(strcmp(snn_status_string((snn_status_t)999), "unknown status") == 0);
    ASSERT_TRUE(strcmp(snn_architecture_string(SNN_ARCH_CUSTOM_CSR), "custom_csr") == 0);
    ASSERT_TRUE(strcmp(snn_architecture_string(SNN_ARCH_FEED_FORWARD), "feed_forward") == 0);
    ASSERT_TRUE(strcmp(snn_architecture_string(SNN_ARCH_RANDOM_POOL), "random_pool") == 0);
    ASSERT_TRUE(strcmp(snn_architecture_string((snn_architecture_t)999), "unknown") == 0);
    ASSERT_EQ_INT(snn_lif_params_validate(&p), SNN_OK);

    p.dt_ms = 0.0f;
    ASSERT_EQ_INT(snn_lif_params_validate(&p), SNN_ERR_INVALID_ARGUMENT);
    p = snn_default_lif_params();
    p.membrane_tau_ms = 0.0f;
    ASSERT_EQ_INT(snn_lif_params_validate(&p), SNN_ERR_INVALID_ARGUMENT);
    p = snn_default_lif_params();
    p.v_threshold = p.v_reset;
    ASSERT_EQ_INT(snn_lif_params_validate(&p), SNN_ERR_INVALID_ARGUMENT);
    p = snn_default_lif_params();
    p.input_scale = NAN;
    ASSERT_EQ_INT(snn_lif_params_validate(&p), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_lif_params_validate(NULL), SNN_ERR_INVALID_ARGUMENT);

    ff = snn_default_feedforward_config(layers, 2);
    ASSERT_TRUE(ff.layer_sizes == layers);
    ASSERT_EQ_U64(ff.layer_count, 2);
    ASSERT_EQ_U64(ff.fanout_per_neuron, 0);
    ASSERT_NEAR(ff.weight, 1.0f, 0.0f);
    ASSERT_TRUE(ff.seed != 0u);

    rp = snn_default_random_pool_config(7, 3);
    ASSERT_EQ_U64(rp.neuron_count, 7);
    ASSERT_EQ_U64(rp.fanout_per_neuron, 3);
    ASSERT_NEAR(rp.weight_min, 0.5f, 0.0f);
    ASSERT_NEAR(rp.weight_max, 1.0f, 0.0f);
    ASSERT_EQ_INT(rp.allow_self_connections, 0);
}

static void test_memory_plans(void) {
    snn_memory_plan_t plan;
    snn_network_t *net = NULL;
    snn_size_t row[] = {0, 1, 1};
    snn_size_t col[] = {1};
    float w[] = {0.25f};

    ASSERT_EQ_INT(snn_estimate_memory_for_counts(2, 1, &plan), SNN_OK);
    ASSERT_EQ_U64(plan.neuron_count, 2);
    ASSERT_EQ_U64(plan.synapse_count, 1);
    ASSERT_TRUE(plan.row_ptr_bytes == 3u * sizeof(snn_size_t));
    ASSERT_TRUE(plan.col_index_bytes == sizeof(snn_size_t));
    ASSERT_TRUE(plan.weight_bytes == sizeof(float));
    ASSERT_TRUE(plan.host_topology_bytes == plan.row_ptr_bytes + plan.col_index_bytes + plan.weight_bytes);
    ASSERT_TRUE(plan.device_total_full_bytes > plan.device_topology_bytes);
    ASSERT_TRUE(plan.device_streaming_min_bytes < plan.device_total_full_bytes);
    ASSERT_EQ_INT(plan.overflowed, 0);
    ASSERT_EQ_INT(snn_estimate_memory_for_counts(0, 0, &plan), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_estimate_memory_for_counts(2, 1, NULL), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_estimate_memory_for_counts(UINT64_MAX, UINT64_MAX, &plan), SNN_ERR_OVERFLOW);
    ASSERT_EQ_INT(plan.overflowed, 1);

    ASSERT_EQ_INT(snn_network_memory_plan(NULL, &plan), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_build_custom_csr(2, 1, row, col, w, NULL, &net), SNN_OK);
    ASSERT_EQ_INT(snn_network_memory_plan(net, &plan), SNN_OK);
    ASSERT_EQ_U64(plan.neuron_count, 2);
    snn_network_free(net);
}

static void test_overflow_and_allocation_failures(void) {
#ifdef SNN_ENABLE_TEST_HOOKS
    snn_network_t *net = NULL;
    snn_state_t *state = NULL;
    snn_size_t row[] = {0, 1, 1};
    snn_size_t col[] = {1};
    float w[] = {1.0f};
    snn_size_t huge_layers1[] = {UINT64_MAX, 1};
    snn_size_t huge_layers2[] = {UINT64_MAX / 2u + 1u, 3};
    snn_size_t huge_layers3[] = {UINT64_MAX, 1};
    snn_feedforward_config_t ff;
    snn_random_pool_config_t rp;
    snn_size_t offsets[3] = {0, 0, 0};

    ASSERT_EQ_INT(snn_test_exercise_internal_guards(), 1);
    ASSERT_EQ_INT(snn_test_prefix_layer_offsets(NULL, 2, offsets), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_test_prefix_layer_offsets(huge_layers3, 2, offsets), SNN_ERR_OVERFLOW);

    ff = snn_default_feedforward_config(huge_layers1, 2);
    ASSERT_EQ_INT(snn_build_feedforward(&ff, NULL, &net), SNN_ERR_OVERFLOW);
    ff = snn_default_feedforward_config(huge_layers2, 2);
    ASSERT_EQ_INT(snn_build_feedforward(&ff, NULL, &net), SNN_ERR_OVERFLOW);
    {
        snn_size_t huge_layers_alloc[] = {1, UINT64_MAX / 8u + 1u};
        ff = snn_default_feedforward_config(huge_layers_alloc, 2);
        ASSERT_EQ_INT(snn_build_feedforward(&ff, NULL, &net), SNN_ERR_OVERFLOW);
    }
    {
        snn_size_t small_layers[] = {1, 1};
        snn_lif_params_t bad = snn_default_lif_params();
        bad.dt_ms = 0.0f;
        ff = snn_default_feedforward_config(small_layers, 2);
        ASSERT_EQ_INT(snn_build_feedforward(&ff, &bad, &net), SNN_ERR_INVALID_ARGUMENT);
    }

    rp = snn_default_random_pool_config(UINT64_MAX, 0);
    rp.allow_self_connections = 1;
    ASSERT_EQ_INT(snn_build_random_pool(&rp, NULL, &net), SNN_ERR_OVERFLOW);
    rp = snn_default_random_pool_config(UINT64_MAX, 2);
    rp.allow_self_connections = 1;
    ASSERT_EQ_INT(snn_build_random_pool(&rp, NULL, &net), SNN_ERR_OVERFLOW);
    {
        snn_lif_params_t bad = snn_default_lif_params();
        bad.dt_ms = 0.0f;
        rp = snn_default_random_pool_config(2, 1);
        ASSERT_EQ_INT(snn_build_random_pool(&rp, &bad, &net), SNN_ERR_INVALID_ARGUMENT);
    }
    {
        snn_memory_plan_t plan;
        snn_size_t syn = (UINT64_MAX - 24u) / 12u;
        ASSERT_EQ_INT(snn_estimate_memory_for_counts(2, syn, &plan), SNN_ERR_OVERFLOW);
        ASSERT_EQ_INT(plan.overflowed, 1);
    }

    snn_test_set_alloc_fail_after(0);
    ASSERT_EQ_INT(snn_build_custom_csr(2, 1, row, col, w, NULL, &net), SNN_ERR_OUT_OF_MEMORY);
    snn_test_disable_alloc_failure();

    snn_test_set_alloc_fail_after(1);
    ASSERT_EQ_INT(snn_build_custom_csr(2, 1, row, col, w, NULL, &net), SNN_ERR_OUT_OF_MEMORY);
    snn_test_disable_alloc_failure();

    snn_test_set_alloc_fail_after(2);
    ASSERT_EQ_INT(snn_build_custom_csr(2, 1, row, col, w, NULL, &net), SNN_ERR_OUT_OF_MEMORY);
    snn_test_disable_alloc_failure();

    snn_test_set_alloc_fail_after(3);
    ASSERT_EQ_INT(snn_build_custom_csr(2, 1, row, col, w, NULL, &net), SNN_ERR_OUT_OF_MEMORY);
    snn_test_disable_alloc_failure();

    ASSERT_EQ_INT(snn_build_custom_csr(2, 1, row, col, w, NULL, &net), SNN_OK);
    snn_test_set_alloc_fail_after(0);
    ASSERT_EQ_INT(snn_state_create(net, &state), SNN_ERR_OUT_OF_MEMORY);
    snn_test_disable_alloc_failure();

    snn_test_set_alloc_fail_after(2);
    ASSERT_EQ_INT(snn_state_create(net, &state), SNN_ERR_OUT_OF_MEMORY);
    snn_test_disable_alloc_failure();
    snn_network_free(net);
    net = NULL;

    {
        snn_size_t layers[] = {2, 2};
        ff = snn_default_feedforward_config(layers, 2);
        snn_test_set_alloc_fail_after(4);
        ASSERT_EQ_INT(snn_build_feedforward(&ff, NULL, &net), SNN_ERR_OUT_OF_MEMORY);
        snn_test_disable_alloc_failure();
    }
#endif
}

static void test_custom_csr_validation_and_accessors(void) {
    snn_network_t *net = NULL;
    snn_lif_params_t p = snn_default_lif_params();
    snn_size_t row[] = {0, 2, 2, 3};
    snn_size_t col[] = {1, 2, 0};
    float w[] = {0.5f, -0.25f, 1.25f};
    snn_size_t bad_row0[] = {1, 2, 2, 3};
    snn_size_t bad_rown[] = {0, 2, 2, 2};
    snn_size_t bad_row_order[] = {0, 2, 1, 3};
    snn_size_t bad_col[] = {1, 3, 0};
    float bad_w[] = {1.0f, INFINITY, 0.0f};

    p.v_threshold = 2.0f;
    ASSERT_EQ_INT(snn_build_custom_csr(0, 0, row, col, w, &p, &net), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_build_custom_csr(3, 3, row, col, w, &p, NULL), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_build_custom_csr(3, 3, NULL, col, w, &p, &net), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_build_custom_csr(3, 3, bad_row0, col, w, &p, &net), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_build_custom_csr(3, 3, bad_rown, col, w, &p, &net), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_build_custom_csr(3, 3, bad_row_order, col, w, &p, &net), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_build_custom_csr(3, 3, row, bad_col, w, &p, &net), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_build_custom_csr(3, 3, row, col, bad_w, &p, &net), SNN_ERR_INVALID_ARGUMENT);
    p.dt_ms = -1.0f;
    ASSERT_EQ_INT(snn_build_custom_csr(3, 3, row, col, w, &p, &net), SNN_ERR_INVALID_ARGUMENT);

    p = snn_default_lif_params();
    ASSERT_EQ_INT(snn_build_custom_csr(3, 3, row, col, w, &p, &net), SNN_OK);
    ASSERT_TRUE(net != NULL);
    ASSERT_EQ_U64(snn_network_neuron_count(net), 3);
    ASSERT_EQ_U64(snn_network_synapse_count(net), 3);
    ASSERT_EQ_INT(snn_network_architecture(net), SNN_ARCH_CUSTOM_CSR);
    ASSERT_TRUE(snn_network_row_ptr(net) != row);
    ASSERT_TRUE(snn_network_col_idx(net) != col);
    ASSERT_TRUE(snn_network_weights(net) != w);
    ASSERT_EQ_U64(snn_network_row_ptr(net)[1], 2);
    ASSERT_EQ_U64(snn_network_col_idx(net)[2], 0);
    ASSERT_NEAR(snn_network_weights(net)[2], 1.25f, 0.0f);
    ASSERT_NEAR(snn_network_lif_params(net).v_threshold, 1.0f, 0.0f);

    p.v_threshold = 1.5f;
    ASSERT_EQ_INT(snn_network_set_lif_params(net, &p), SNN_OK);
    ASSERT_NEAR(snn_network_lif_params(net).v_threshold, 1.5f, 0.0f);
    p.v_threshold = 0.0f;
    ASSERT_EQ_INT(snn_network_set_lif_params(net, &p), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_network_set_lif_params(NULL, NULL), SNN_ERR_INVALID_ARGUMENT);

    ASSERT_EQ_U64(snn_network_neuron_count(NULL), 0);
    ASSERT_EQ_U64(snn_network_synapse_count(NULL), 0);
    ASSERT_EQ_INT(snn_network_architecture(NULL), SNN_ARCH_CUSTOM_CSR);
    ASSERT_TRUE(snn_network_row_ptr(NULL) == NULL);
    ASSERT_TRUE(snn_network_col_idx(NULL) == NULL);
    ASSERT_TRUE(snn_network_weights(NULL) == NULL);
    ASSERT_NEAR(snn_network_lif_params(NULL).v_threshold, 1.0f, 0.0f);
    snn_network_free(net);
    snn_network_free(NULL);
}

static void test_zero_synapse_csr(void) {
    snn_network_t *net = NULL;
    snn_state_t *state = NULL;
    snn_size_t row[] = {0, 0};
    float input[] = {2.0f};
    uint8_t spike[] = {0};
    ASSERT_EQ_INT(snn_build_custom_csr(1, 0, row, NULL, NULL, NULL, &net), SNN_OK);
    ASSERT_EQ_U64(snn_network_synapse_count(net), 0);
    ASSERT_TRUE(snn_network_col_idx(net) == NULL);
    ASSERT_TRUE(snn_network_weights(net) == NULL);
    ASSERT_EQ_INT(snn_state_create(net, &state), SNN_OK);
    ASSERT_EQ_INT(snn_step_cpu(net, state, input, spike), SNN_OK);
    ASSERT_EQ_INT(spike[0], 1);
    snn_state_free(state);
    snn_network_free(net);
}

static void test_feedforward_builders(void) {
    snn_size_t layers[] = {2, 3, 1};
    snn_size_t bad_zero[] = {2, 0};
    snn_network_t *dense = NULL;
    snn_network_t *fanout = NULL;
    snn_feedforward_config_t cfg = snn_default_feedforward_config(layers, 3);

    ASSERT_EQ_INT(snn_build_feedforward(NULL, NULL, &dense), SNN_ERR_INVALID_ARGUMENT);
    cfg = snn_default_feedforward_config(layers, 3);
    ASSERT_EQ_INT(snn_build_feedforward(&cfg, NULL, NULL), SNN_ERR_INVALID_ARGUMENT);
    cfg.layer_sizes = NULL;
    ASSERT_EQ_INT(snn_build_feedforward(&cfg, NULL, &dense), SNN_ERR_INVALID_ARGUMENT);
    cfg.layer_sizes = layers;
    cfg.layer_count = 1;
    ASSERT_EQ_INT(snn_build_feedforward(&cfg, NULL, &dense), SNN_ERR_INVALID_ARGUMENT);
    cfg.layer_sizes = bad_zero;
    cfg.layer_count = 2;
    ASSERT_EQ_INT(snn_build_feedforward(&cfg, NULL, &dense), SNN_ERR_INVALID_ARGUMENT);
    cfg.layer_sizes = layers;
    cfg.layer_count = 3;
    cfg.weight = NAN;
    ASSERT_EQ_INT(snn_build_feedforward(&cfg, NULL, &dense), SNN_ERR_INVALID_ARGUMENT);

    cfg.weight = 0.75f;
    cfg.fanout_per_neuron = 0;
    ASSERT_EQ_INT(snn_build_feedforward(&cfg, NULL, &dense), SNN_OK);
    ASSERT_EQ_INT(snn_network_architecture(dense), SNN_ARCH_FEED_FORWARD);
    ASSERT_EQ_U64(snn_network_neuron_count(dense), 6);
    ASSERT_EQ_U64(snn_network_synapse_count(dense), 2u * 3u + 3u * 1u);
    ASSERT_EQ_U64(snn_network_row_ptr(dense)[0], 0);
    ASSERT_EQ_U64(snn_network_row_ptr(dense)[1], 3);
    ASSERT_EQ_U64(snn_network_row_ptr(dense)[2], 6);
    ASSERT_EQ_U64(snn_network_row_ptr(dense)[3], 7);
    ASSERT_EQ_U64(snn_network_row_ptr(dense)[4], 8);
    ASSERT_EQ_U64(snn_network_row_ptr(dense)[5], 9);
    ASSERT_EQ_U64(snn_network_row_ptr(dense)[6], 9);
    ASSERT_EQ_U64(snn_network_col_idx(dense)[0], 2);
    ASSERT_EQ_U64(snn_network_col_idx(dense)[2], 4);
    ASSERT_EQ_U64(snn_network_col_idx(dense)[8], 5);
    ASSERT_NEAR(snn_network_weights(dense)[0], 0.75f, 0.0f);

    cfg.fanout_per_neuron = 1;
    cfg.seed = 123;
    ASSERT_EQ_INT(snn_build_feedforward(&cfg, NULL, &fanout), SNN_OK);
    ASSERT_EQ_U64(snn_network_synapse_count(fanout), 5);
    ASSERT_EQ_U64(snn_network_row_ptr(fanout)[2], 2);
    ASSERT_EQ_U64(snn_network_row_ptr(fanout)[5], 5);
    ASSERT_TRUE(snn_network_col_idx(fanout)[0] >= 2 && snn_network_col_idx(fanout)[0] < 5);
    ASSERT_EQ_U64(snn_network_col_idx(fanout)[4], 5);

    snn_network_free(dense);
    snn_network_free(fanout);
}

static void test_random_pool_builder(void) {
    snn_random_pool_config_t cfg = snn_default_random_pool_config(4, 3);
    snn_network_t *net = NULL;

    ASSERT_EQ_INT(snn_build_random_pool(NULL, NULL, &net), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_build_random_pool(&cfg, NULL, NULL), SNN_ERR_INVALID_ARGUMENT);
    cfg.neuron_count = 0;
    ASSERT_EQ_INT(snn_build_random_pool(&cfg, NULL, &net), SNN_ERR_INVALID_ARGUMENT);
    cfg.neuron_count = 1;
    cfg.fanout_per_neuron = 1;
    cfg.allow_self_connections = 0;
    ASSERT_EQ_INT(snn_build_random_pool(&cfg, NULL, &net), SNN_ERR_INVALID_ARGUMENT);
    cfg.neuron_count = 4;
    cfg.fanout_per_neuron = 3;
    cfg.weight_min = 2.0f;
    cfg.weight_max = 1.0f;
    ASSERT_EQ_INT(snn_build_random_pool(&cfg, NULL, &net), SNN_ERR_INVALID_ARGUMENT);
    cfg.weight_min = 0.1f;
    cfg.weight_max = 0.3f;
    cfg.allow_self_connections = 0;
    cfg.seed = 44;
    ASSERT_EQ_INT(snn_build_random_pool(&cfg, NULL, &net), SNN_OK);
    ASSERT_EQ_INT(snn_network_architecture(net), SNN_ARCH_RANDOM_POOL);
    ASSERT_EQ_U64(snn_network_neuron_count(net), 4);
    ASSERT_EQ_U64(snn_network_synapse_count(net), 12);
    for (snn_size_t i = 0; i < 4; ++i) {
        ASSERT_EQ_U64(snn_network_row_ptr(net)[i], i * 3u);
        for (snn_size_t e = snn_network_row_ptr(net)[i]; e < snn_network_row_ptr(net)[i + 1u]; ++e) {
            ASSERT_TRUE(snn_network_col_idx(net)[e] < 4);
            ASSERT_TRUE(snn_network_col_idx(net)[e] != i);
            ASSERT_TRUE(snn_network_weights(net)[e] >= 0.1f && snn_network_weights(net)[e] <= 0.3f);
        }
    }
    snn_network_free(net);

    cfg = snn_default_random_pool_config(1, 2);
    cfg.allow_self_connections = 1;
    ASSERT_EQ_INT(snn_build_random_pool(&cfg, NULL, &net), SNN_OK);
    ASSERT_EQ_U64(snn_network_synapse_count(net), 2);
    ASSERT_EQ_U64(snn_network_col_idx(net)[0], 0);
    snn_network_free(net);
}

static void test_cpu_state_and_steps(void) {
    snn_network_t *net = NULL;
    snn_state_t *state = NULL;
    snn_state_t *other_state = NULL;
    snn_network_t *other_net = NULL;
    snn_lif_params_t p = snn_default_lif_params();
    snn_size_t row[] = {0, 1, 2, 2};
    snn_size_t col[] = {1, 2};
    float w[] = {0.8f, 1.1f};
    float input0[] = {1.2f, 0.0f, 0.0f};
    float input1[] = {0.0f, 0.0f, 0.0f};
    uint8_t spikes[3] = {0};
    float voltage[3] = {0};
    uint8_t copied_spikes[3] = {0};
    float small_voltage[2] = {0};
    uint8_t small_spikes[2] = {0};

    p.refractory_steps = 1;
    ASSERT_EQ_INT(snn_build_custom_csr(3, 2, row, col, w, &p, &net), SNN_OK);
    ASSERT_EQ_INT(snn_state_create(NULL, &state), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_state_create(net, NULL), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_state_create(net, &state), SNN_OK);
    ASSERT_TRUE(state != NULL);
    ASSERT_EQ_INT(snn_state_copy_voltage(state, voltage, 3), SNN_OK);
    ASSERT_NEAR(voltage[0], p.v_rest, 0.0f);
    ASSERT_EQ_INT(snn_state_copy_voltage(NULL, voltage, 3), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_state_copy_voltage(state, NULL, 3), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_state_copy_voltage(state, small_voltage, 2), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_state_copy_spikes(state, copied_spikes, 3), SNN_OK);
    ASSERT_EQ_INT(snn_state_copy_spikes(NULL, copied_spikes, 3), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_state_copy_spikes(state, NULL, 3), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_state_copy_spikes(state, small_spikes, 2), SNN_ERR_INVALID_ARGUMENT);

    ASSERT_EQ_INT(snn_step_cpu(NULL, state, input0, spikes), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_step_cpu(net, NULL, input0, spikes), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_step_cpu(net, state, input0, spikes), SNN_OK);
    ASSERT_EQ_INT(spikes[0], 1);
    ASSERT_EQ_INT(spikes[1], 0);
    ASSERT_EQ_INT(spikes[2], 0);
    ASSERT_EQ_INT(snn_state_copy_voltage(state, voltage, 3), SNN_OK);
    ASSERT_NEAR(voltage[0], 0.0f, 0.0f);

    ASSERT_EQ_INT(snn_step_cpu(net, state, input1, spikes), SNN_OK);
    ASSERT_EQ_INT(spikes[0], 0);
    ASSERT_EQ_INT(spikes[1], 0);
    ASSERT_EQ_INT(spikes[2], 0);

    ASSERT_EQ_INT(snn_step_cpu(net, state, input1, spikes), SNN_OK);
    ASSERT_EQ_INT(spikes[0], 0);
    ASSERT_EQ_INT(spikes[1], 0);
    ASSERT_EQ_INT(spikes[2], 0);

    {
        float input2[] = {0.0f, 0.3f, 0.0f};
        ASSERT_EQ_INT(snn_step_cpu(net, state, input2, spikes), SNN_OK);
        ASSERT_EQ_INT(spikes[0], 0);
        ASSERT_EQ_INT(spikes[1], 1);
        ASSERT_EQ_INT(spikes[2], 0);
    }

    ASSERT_EQ_INT(snn_step_cpu(net, state, NULL, NULL), SNN_OK);

    ASSERT_EQ_INT(snn_state_reset(net, state), SNN_OK);
    ASSERT_EQ_INT(snn_state_copy_spikes(state, copied_spikes, 3), SNN_OK);
    ASSERT_EQ_INT(copied_spikes[1], 0);
    ASSERT_EQ_INT(snn_state_reset(NULL, state), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_state_reset(net, NULL), SNN_ERR_INVALID_ARGUMENT);

    {
        snn_size_t row2[] = {0, 0, 0, 0, 0};
        ASSERT_EQ_INT(snn_build_custom_csr(4, 0, row2, NULL, NULL, NULL, &other_net), SNN_OK);
        ASSERT_EQ_INT(snn_state_create(other_net, &other_state), SNN_OK);
        ASSERT_EQ_INT(snn_state_reset(net, other_state), SNN_ERR_INVALID_ARGUMENT);
        ASSERT_EQ_INT(snn_step_cpu(net, other_state, input1, spikes), SNN_ERR_INVALID_ARGUMENT);
        snn_state_free(other_state);
        snn_network_free(other_net);
    }

    snn_state_free(state);
    snn_network_free(net);
    snn_state_free(NULL);
}

static void test_run_cpu(void) {
    snn_network_t *net = NULL;
    snn_state_t *state = NULL;
    snn_size_t row[] = {0, 1, 1};
    snn_size_t col[] = {1};
    float w[] = {1.0f};
    float inputs[6] = {1.1f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    uint8_t outputs[6] = {0};

    ASSERT_EQ_INT(snn_build_custom_csr(2, 1, row, col, w, NULL, &net), SNN_OK);
    ASSERT_EQ_INT(snn_state_create(net, &state), SNN_OK);
    ASSERT_EQ_INT(snn_run_cpu(NULL, state, inputs, 3, 2, outputs, 2), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_run_cpu(net, NULL, inputs, 3, 2, outputs, 2), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_run_cpu(net, state, inputs, 3, 1, outputs, 2), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_run_cpu(net, state, inputs, 3, 2, outputs, 1), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_run_cpu(net, state, inputs, 3, 2, outputs, 2), SNN_OK);
    ASSERT_EQ_INT(outputs[0], 1);
    ASSERT_EQ_INT(outputs[1], 0);
    ASSERT_EQ_INT(outputs[2], 0);
    ASSERT_EQ_INT(outputs[3], 1);
    ASSERT_EQ_INT(outputs[4], 0);
    ASSERT_EQ_INT(outputs[5], 0);
    ASSERT_EQ_INT(snn_run_cpu(net, state, NULL, 0, 0, NULL, 0), SNN_OK);
    snn_state_free(state);
    snn_network_free(net);
}

static int spikes_match(const uint8_t *a, const uint8_t *b, snn_size_t n) {
    snn_size_t i = 0;
    for (i = 0; i < n; ++i) {
        if (a[i] != b[i]) {
            return 0;
        }
    }
    return 1;
}

/* Run a network on CPU and on a given CUDA config; assert per-step spike parity. */
static void cuda_parity_run(snn_network_t *net, snn_cuda_config_t cfg,
                            snn_cuda_mode_t expect_mode, int steps, unsigned seed) {
    snn_size_t n = snn_network_neuron_count(net);
    snn_state_t *cpu = NULL;
    snn_cuda_context_t *ctx = NULL;
    float *in = (float *)malloc((size_t)n * sizeof(float));
    uint8_t *cs = (uint8_t *)malloc((size_t)n);
    uint8_t *gs = (uint8_t *)malloc((size_t)n);
    int s = 0;
    ASSERT_TRUE(in != NULL && cs != NULL && gs != NULL);
    ASSERT_EQ_INT(snn_state_create(net, &cpu), SNN_OK);
    ASSERT_EQ_INT(snn_cuda_create(net, &cfg, &ctx), SNN_OK);
    ASSERT_EQ_INT(snn_cuda_context_mode(ctx), expect_mode);
    srand(seed);
    for (s = 0; s < steps; ++s) {
        snn_size_t i = 0;
        for (i = 0; i < n; ++i) {
            in[i] = (rand() % 100 < 8) ? 1.5f : 0.0f;
        }
        ASSERT_EQ_INT(snn_step_cpu(net, cpu, in, cs), SNN_OK);
        ASSERT_EQ_INT(snn_cuda_step(ctx, in, gs), SNN_OK);
        ASSERT_TRUE(spikes_match(cs, gs, n));
    }
    snn_cuda_free(ctx);
    snn_state_free(cpu);
    free(in);
    free(cs);
    free(gs);
}

static void test_cuda_api(void) {
    snn_cuda_config_t cfg = snn_cuda_default_config();
    snn_cuda_context_t *ctx = NULL;
    snn_network_t *net = NULL;
    snn_state_t *cpu_state = NULL;
    snn_size_t row[] = {0, 1, 2, 2};
    snn_size_t col[] = {1, 2};
    float w[] = {0.75f, 1.25f};
    float input[] = {1.2f, 0.0f, 0.0f};
    uint8_t cuda_spikes[3] = {0};
    uint8_t cpu_spikes[3] = {0};
    float voltage[3] = {0};

    ASSERT_EQ_U64(cfg.max_vram_bytes, 0);
    ASSERT_EQ_U64(cfg.max_stream_synapses, 0);
    ASSERT_EQ_U64(cfg.max_stream_rows, 0);
    ASSERT_EQ_INT(cfg.prefer_streaming, 0);
    ASSERT_EQ_INT(snn_cuda_context_mode(NULL), SNN_CUDA_MODE_NONE);
    ASSERT_EQ_INT(snn_cuda_step(NULL, NULL, NULL), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_cuda_download_voltage(NULL, voltage, 3), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_cuda_create(NULL, &cfg, &ctx), SNN_ERR_INVALID_ARGUMENT);

    ASSERT_EQ_INT(snn_build_custom_csr(3, 2, row, col, w, NULL, &net), SNN_OK);
    ASSERT_EQ_INT(snn_cuda_create(net, &cfg, NULL), SNN_ERR_INVALID_ARGUMENT);

    if (!snn_cuda_available()) {
        snn_status_t st = snn_cuda_create(net, &cfg, &ctx);
        ASSERT_TRUE(st == SNN_ERR_CUDA || st == SNN_ERR_UNSUPPORTED);
        ASSERT_TRUE(ctx == NULL);
#ifdef SNN_WITH_CUDA
        if (!SNN_WITH_CUDA) {
            ASSERT_EQ_INT(snn_cuda_step(snn_test_nonnull_cuda_context(), NULL, NULL), SNN_ERR_UNSUPPORTED);
            ASSERT_EQ_INT(snn_cuda_download_voltage(snn_test_nonnull_cuda_context(), voltage, 3), SNN_ERR_UNSUPPORTED);
        }
#endif
#ifdef SNN_ENABLE_TEST_HOOKS
        /* Exercise the fault-injection controls even without a device so their
         * definitions are covered in the CPU-only build. They are pure setters
         * (no-ops in the stub) and must not affect subsequent behavior. */
        snn_test_cuda_set_fail_after(0);
        snn_test_cuda_force_unavailable(1);
        snn_test_cuda_force_meminfo_fail(1);
        snn_test_cuda_disable_failure();
        ASSERT_TRUE(snn_test_nonnull_cuda_context() != NULL);
#endif
        snn_cuda_free(NULL);
        snn_network_free(net);
        return;
    }

    /* Small-net FULL mode: exact CPU parity + basic accessors/errors. */
    ASSERT_EQ_INT(snn_state_create(net, &cpu_state), SNN_OK);
    ASSERT_EQ_INT(snn_cuda_create(net, &cfg, &ctx), SNN_OK);
    ASSERT_EQ_INT(snn_cuda_context_mode(ctx), SNN_CUDA_MODE_FULL);
    ASSERT_EQ_INT(snn_step_cpu(net, cpu_state, input, cpu_spikes), SNN_OK);
    ASSERT_EQ_INT(snn_cuda_step(ctx, input, cuda_spikes), SNN_OK);
    ASSERT_TRUE(spikes_match(cpu_spikes, cuda_spikes, 3));
    ASSERT_EQ_INT(snn_cuda_step(ctx, NULL, NULL), SNN_OK); /* null external + sync path */
    ASSERT_EQ_INT(snn_cuda_download_voltage(ctx, voltage, 2), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_cuda_download_voltage(ctx, voltage, 3), SNN_OK);
    snn_cuda_free(ctx);
    ctx = NULL;
    snn_state_free(cpu_state);
    snn_network_free(net);
    net = NULL;

    /* Zero-synapse network on GPU (FULL): empty topology must be well-behaved. */
    {
        snn_size_t zrow[3] = {0, 0, 0};
        snn_network_t *znet = NULL;
        float zin[2] = {2.0f, 2.0f};
        uint8_t zsp[2] = {0};
        ASSERT_EQ_INT(snn_build_custom_csr(2, 0, zrow, NULL, NULL, NULL, &znet), SNN_OK);
        ASSERT_EQ_INT(snn_cuda_create(znet, &cfg, &ctx), SNN_OK);
        ASSERT_EQ_INT(snn_cuda_context_mode(ctx), SNN_CUDA_MODE_FULL);
        ASSERT_EQ_INT(snn_cuda_step(ctx, zin, zsp), SNN_OK);
        snn_cuda_free(ctx);
        ctx = NULL;
        /* Zero-synapse in STREAMING: exercises max_degree==0 -> chunk clamp. */
        {
            snn_cuda_config_t sc = snn_cuda_default_config();
            sc.prefer_streaming = 1;
            sc.max_stream_rows = 1;
            ASSERT_EQ_INT(snn_cuda_create(znet, &sc, &ctx), SNN_OK);
            ASSERT_EQ_INT(snn_cuda_context_mode(ctx), SNN_CUDA_MODE_STREAMING);
            ASSERT_EQ_INT(snn_cuda_step(ctx, zin, zsp), SNN_OK);
            snn_cuda_free(ctx);
            ctx = NULL;
        }
        snn_network_free(znet);
    }

    /* Scale + parity: random pool, FULL and STREAMING (multi-chunk). */
    {
        snn_lif_params_t p = snn_default_lif_params();
        snn_random_pool_config_t rp = snn_default_random_pool_config(20000, 48);
        snn_network_t *pool = NULL;
        snn_cuda_config_t sc;
        p.refractory_steps = 2;
        rp.seed = 4242;
        rp.weight_min = 0.03f;
        rp.weight_max = 0.06f;
        ASSERT_EQ_INT(snn_build_random_pool(&rp, &p, &pool), SNN_OK);
        cuda_parity_run(pool, snn_cuda_default_config(), SNN_CUDA_MODE_FULL, 12, 11);
        sc = snn_cuda_default_config();
        sc.prefer_streaming = 1;
        sc.max_stream_rows = 512;
        sc.max_stream_synapses = 30000; /* forces many chunks incl. exact-fill break */
        cuda_parity_run(pool, sc, SNN_CUDA_MODE_STREAMING, 12, 11);
        /* Auto-sized streaming chunk (max_stream_synapses == 0 path). */
        sc = snn_cuda_default_config();
        sc.prefer_streaming = 1;
        sc.max_stream_rows = 300;
        cuda_parity_run(pool, sc, SNN_CUDA_MODE_STREAMING, 6, 11);
        /* Tiny VRAM budget forces small auto chunk via available_for_chunks path. */
        sc = snn_cuda_default_config();
        sc.prefer_streaming = 1;
        sc.max_vram_bytes = (uint64_t)2u << 20;
        cuda_parity_run(pool, sc, SNN_CUDA_MODE_STREAMING, 4, 11);
        /*
         * Budget just above the resident-state size: the auto-sized synapse
         * chunk computes below max_degree and must clamp up to max_degree so the
         * densest row still fits. 20000 neurons * 21 B ~= 420 KB of state.
         */
        sc = snn_cuda_default_config();
        sc.prefer_streaming = 1;
        sc.max_vram_bytes = (uint64_t)440u * 1024u;
        cuda_parity_run(pool, sc, SNN_CUDA_MODE_STREAMING, 3, 11);
        /* max_stream_rows clamp above 65536 (still valid, just clamped). */
        sc = snn_cuda_default_config();
        sc.prefer_streaming = 1;
        sc.max_stream_rows = 200000;
        cuda_parity_run(pool, sc, SNN_CUDA_MODE_STREAMING, 3, 11);
        snn_network_free(pool);
    }

    /* Scale + parity: feedforward, FULL and STREAMING. */
    {
        snn_lif_params_t p = snn_default_lif_params();
        snn_size_t layers[] = {1024, 2048, 512};
        snn_feedforward_config_t ff = snn_default_feedforward_config(layers, 3);
        snn_network_t *ffnet = NULL;
        snn_cuda_config_t sc;
        p.refractory_steps = 1;
        ff.fanout_per_neuron = 32;
        ff.weight = 0.09f;
        ff.seed = 5;
        ASSERT_EQ_INT(snn_build_feedforward(&ff, &p, &ffnet), SNN_OK);
        cuda_parity_run(ffnet, snn_cuda_default_config(), SNN_CUDA_MODE_FULL, 15, 2);
        sc = snn_cuda_default_config();
        sc.prefer_streaming = 1;
        sc.max_stream_rows = 128;
        sc.max_stream_synapses = 8000;
        cuda_parity_run(ffnet, sc, SNN_CUDA_MODE_STREAMING, 15, 2);
        snn_network_free(ffnet);
    }

    /* Streaming error: a single row's degree exceeds the synapse chunk. */
    {
        snn_size_t drow[3] = {0, 3, 3};
        snn_size_t dcol[3] = {0, 1, 1};
        float dw[3] = {0.1f, 0.1f, 0.1f};
        snn_network_t *dense = NULL;
        snn_cuda_config_t sc = snn_cuda_default_config();
        float din[2] = {2.0f, 0.0f};
        uint8_t dsp[2] = {0};
        ASSERT_EQ_INT(snn_build_custom_csr(2, 3, drow, dcol, dw, NULL, &dense), SNN_OK);
        sc.prefer_streaming = 1;
        sc.max_stream_rows = 4;
        sc.max_stream_synapses = 2; /* < degree(0)=3 -> reject at create */
        ASSERT_EQ_INT(snn_cuda_create(dense, &sc, &ctx), SNN_ERR_INVALID_ARGUMENT);
        ASSERT_TRUE(ctx == NULL);
        snn_network_free(dense);
    }

#ifdef SNN_ENABLE_TEST_HOOKS
    /* Fault injection: drive every CUDA/allocation error branch deterministically. */
    {
        snn_lif_params_t p = snn_default_lif_params();
        snn_random_pool_config_t rp = snn_default_random_pool_config(1000, 8);
        snn_network_t *fn = NULL;
        int k = 0;
        float *in = (float *)calloc(1000, sizeof(float));
        uint8_t *sp = (uint8_t *)calloc(1000, 1);
        float *v = (float *)calloc(1000, sizeof(float));
        ASSERT_TRUE(in != NULL && sp != NULL && v != NULL);
        rp.seed = 7;
        ASSERT_EQ_INT(snn_build_random_pool(&rp, &p, &fn), SNN_OK);

        /* The non-null sentinel helper is a stable, non-dereferenced marker. */
        ASSERT_TRUE(snn_test_nonnull_cuda_context() != NULL);

        /* Environment failures during create. */
        snn_test_cuda_force_unavailable(1);
        ASSERT_EQ_INT(snn_cuda_create(fn, &cfg, &ctx), SNN_ERR_CUDA);
        ASSERT_TRUE(ctx == NULL);
        snn_test_cuda_disable_failure();
        snn_test_cuda_force_meminfo_fail(1);
        ASSERT_EQ_INT(snn_cuda_create(fn, &cfg, &ctx), SNN_ERR_CUDA);
        ASSERT_TRUE(ctx == NULL);
        snn_test_cuda_disable_failure();

        /* Context host allocation (seam #0) fails with OOM. */
        snn_test_cuda_set_fail_after(0);
        ASSERT_EQ_INT(snn_cuda_create(fn, &cfg, &ctx), SNN_ERR_OUT_OF_MEMORY);
        ASSERT_TRUE(ctx == NULL);
        snn_test_cuda_disable_failure();

        /*
         * FULL create device seams after the context calloc (#0):
         *   #1..#6  device-state cudaMalloc x6
         *   #7      init_state kernel launch
         *   #8..#10 topology cudaMalloc x3
         *   #11..#13 topology cudaMemcpy x3
         * All report SNN_ERR_CUDA.
         */
        for (k = 1; k <= 13; ++k) {
            snn_test_cuda_set_fail_after(k);
            ASSERT_EQ_INT(snn_cuda_create(fn, &cfg, &ctx), SNN_ERR_CUDA);
            ASSERT_TRUE(ctx == NULL);
            snn_test_cuda_disable_failure();
        }

        /*
         * STREAMING create seams: #0 ctx calloc, #1..#6 device state, #7 init
         * launch, then host chunk-row malloc, then #8..#10 chunk cudaMalloc x3.
         * Exercise the host malloc OOM (seam #8, before device chunk mallocs)
         * and the three device chunk allocations (#8..#10 after the host malloc
         * consumes one injection slot -> #9..#11).
         */
        {
            snn_cuda_config_t sc = snn_cuda_default_config();
            sc.prefer_streaming = 1;
            sc.max_stream_rows = 64;
            sc.max_stream_synapses = 2000;
            /* Host chunk-row malloc is the 9th seam (index 8). */
            snn_test_cuda_set_fail_after(8);
            ASSERT_EQ_INT(snn_cuda_create(fn, &sc, &ctx), SNN_ERR_OUT_OF_MEMORY);
            ASSERT_TRUE(ctx == NULL);
            snn_test_cuda_disable_failure();
            /* The three device chunk cudaMalloc calls follow (indices 9,10,11). */
            for (k = 9; k <= 11; ++k) {
                snn_test_cuda_set_fail_after(k);
                ASSERT_EQ_INT(snn_cuda_create(fn, &sc, &ctx), SNN_ERR_CUDA);
                ASSERT_TRUE(ctx == NULL);
                snn_test_cuda_disable_failure();
            }
        }

        /* Now build a good context and fail each step-time seam in turn. */
        ASSERT_EQ_INT(snn_cuda_create(fn, &cfg, &ctx), SNN_OK);
        snn_test_cuda_set_fail_after(0); /* external upload */
        ASSERT_EQ_INT(snn_cuda_step(ctx, in, sp), SNN_ERR_CUDA);
        snn_test_cuda_disable_failure();
        snn_test_cuda_set_fail_after(1); /* zero_float launch */
        ASSERT_EQ_INT(snn_cuda_step(ctx, in, sp), SNN_ERR_CUDA);
        snn_test_cuda_disable_failure();
        snn_test_cuda_set_fail_after(2); /* integrate launch */
        ASSERT_EQ_INT(snn_cuda_step(ctx, in, sp), SNN_ERR_CUDA);
        snn_test_cuda_disable_failure();
        snn_test_cuda_set_fail_after(3); /* propagate_full launch */
        ASSERT_EQ_INT(snn_cuda_step(ctx, in, sp), SNN_ERR_CUDA);
        snn_test_cuda_disable_failure();
        snn_test_cuda_set_fail_after(0); /* download voltage copy */
        ASSERT_EQ_INT(snn_cuda_download_voltage(ctx, v, 1000), SNN_ERR_CUDA);
        snn_test_cuda_disable_failure();
        snn_cuda_free(ctx);
        ctx = NULL;

        /* Streaming step-time copy/launch failures inside stream_propagate. */
        {
            snn_cuda_config_t sc = snn_cuda_default_config();
            sc.prefer_streaming = 1;
            sc.max_stream_rows = 64;
            sc.max_stream_synapses = 2000;
            ASSERT_EQ_INT(snn_cuda_create(fn, &sc, &ctx), SNN_OK);
            snn_test_cuda_set_fail_after(3); /* first chunk row_ptr copy (after upload+2 launches) */
            ASSERT_EQ_INT(snn_cuda_step(ctx, in, sp), SNN_ERR_CUDA);
            snn_test_cuda_disable_failure();
            snn_test_cuda_set_fail_after(4); /* first chunk col_idx copy */
            ASSERT_EQ_INT(snn_cuda_step(ctx, in, sp), SNN_ERR_CUDA);
            snn_test_cuda_disable_failure();
            snn_test_cuda_set_fail_after(5); /* first chunk weights copy */
            ASSERT_EQ_INT(snn_cuda_step(ctx, in, sp), SNN_ERR_CUDA);
            snn_test_cuda_disable_failure();
            snn_test_cuda_set_fail_after(6); /* first chunk kernel launch */
            ASSERT_EQ_INT(snn_cuda_step(ctx, in, sp), SNN_ERR_CUDA);
            snn_test_cuda_disable_failure();
            snn_cuda_free(ctx);
            ctx = NULL;
        }
        snn_network_free(fn);
        free(in);
        free(sp);
        free(v);
    }
#endif
}

int main(void) {
    test_strings_and_defaults();
    test_memory_plans();
    test_overflow_and_allocation_failures();
    test_custom_csr_validation_and_accessors();
    test_zero_synapse_csr();
    test_feedforward_builders();
    test_random_pool_builder();
    test_cpu_state_and_steps();
    test_run_cpu();
    test_cuda_api();
    printf("all tests passed\n");
    return 0;
}
