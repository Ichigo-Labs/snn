#include "snn/snn.h"
#include <stdint.h>

snn_cuda_config_t snn_cuda_default_config(void) {
    snn_cuda_config_t cfg;
    cfg.max_vram_bytes = 0;
    cfg.max_stream_synapses = 0;
    cfg.max_stream_rows = 0;
    cfg.prefer_streaming = 0;
    return cfg;
}

int snn_cuda_available(void) {
    return 0;
}

snn_status_t snn_cuda_create(const snn_network_t *network,
                             const snn_cuda_config_t *config,
                             snn_cuda_context_t **out_context) {
    (void)config;
    if (out_context != 0) {
        *out_context = 0;
    }
    if (network == 0 || out_context == 0) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    return SNN_ERR_UNSUPPORTED;
}

void snn_cuda_free(snn_cuda_context_t *context) {
    (void)context;
}

snn_cuda_mode_t snn_cuda_context_mode(const snn_cuda_context_t *context) {
    (void)context;
    return SNN_CUDA_MODE_NONE;
}

snn_status_t snn_cuda_step(snn_cuda_context_t *context,
                           const float *host_external_current,
                           uint8_t *host_out_spikes) {
    (void)host_external_current;
    (void)host_out_spikes;
    if (context == 0) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    return SNN_ERR_UNSUPPORTED;
}

snn_status_t snn_cuda_download_voltage(const snn_cuda_context_t *context,
                                       float *host_voltage,
                                       snn_size_t count) {
    (void)count;
    if (context == 0 || host_voltage == 0) {
        return SNN_ERR_INVALID_ARGUMENT;
    }
    return SNN_ERR_UNSUPPORTED;
}


#ifdef SNN_ENABLE_TEST_HOOKS
snn_cuda_context_t *snn_test_nonnull_cuda_context(void) {
    return (snn_cuda_context_t *)(uintptr_t)1u;
}

void snn_test_cuda_set_fail_after(int64_t calls_before_failure) {
    (void)calls_before_failure;
}

void snn_test_cuda_disable_failure(void) {
}

void snn_test_cuda_force_unavailable(int enable) {
    (void)enable;
}

void snn_test_cuda_force_meminfo_fail(int enable) {
    (void)enable;
}
#endif
