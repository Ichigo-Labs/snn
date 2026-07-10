/*
 * CPU-only stand-in for tools/bptt_cuda.cu, linked when the tool is built
 * without a CUDA compiler. Mirrors the real backend's contract the way
 * src/snn_cuda_stub.c mirrors the simulator backend: available() says no,
 * create fails with a message, and every other entry point is a safe no-op,
 * so --gpu degrades into one clear error instead of a link failure.
 */

#include "bptt_cuda.h"

#include <stdio.h>

int bptt_gpu_available(void) {
    return 0;
}

bptt_gpu_t *bptt_gpu_create(const snn_size_t *layer_sizes,
                            size_t layer_count,
                            snn_size_t timesteps,
                            float beta,
                            float threshold,
                            snn_surrogate_t surrogate,
                            float alpha,
                            int detach_reset,
                            const float *params,
                            snn_size_t param_count,
                            const uint8_t *train_px,
                            const uint8_t *train_lb,
                            snn_size_t train_count,
                            const uint8_t *test_px,
                            const uint8_t *test_lb,
                            snn_size_t test_count,
                            snn_size_t batch,
                            snn_size_t eval_batch,
                            float lr,
                            float adam_beta1,
                            float adam_beta2,
                            float adam_eps,
                            char *err,
                            size_t err_len) {
    (void)layer_sizes;
    (void)layer_count;
    (void)timesteps;
    (void)beta;
    (void)threshold;
    (void)surrogate;
    (void)alpha;
    (void)detach_reset;
    (void)params;
    (void)param_count;
    (void)train_px;
    (void)train_lb;
    (void)train_count;
    (void)test_px;
    (void)test_lb;
    (void)test_count;
    (void)batch;
    (void)eval_batch;
    (void)lr;
    (void)adam_beta1;
    (void)adam_beta2;
    (void)adam_eps;
    if (err != NULL && err_len > 0u) {
        snprintf(err, err_len, "this binary was built without CUDA; rebuild with a CUDA toolkit for --gpu");
    }
    return NULL;
}

int bptt_gpu_train_epoch(bptt_gpu_t *gpu, const snn_size_t *order, snn_size_t count, float *out_mean_loss) {
    (void)gpu;
    (void)order;
    (void)count;
    (void)out_mean_loss;
    return -1;
}

int bptt_gpu_evaluate(bptt_gpu_t *gpu,
                      int use_test,
                      snn_size_t limit,
                      float *out_accuracy,
                      float *out_loss,
                      double *out_spikes) {
    (void)gpu;
    (void)use_test;
    (void)limit;
    (void)out_accuracy;
    (void)out_loss;
    (void)out_spikes;
    return -1;
}

int bptt_gpu_batch_grads(bptt_gpu_t *gpu,
                         const snn_size_t *indices,
                         snn_size_t count,
                         float *out_grads,
                         float *out_loss_sum) {
    (void)gpu;
    (void)indices;
    (void)count;
    (void)out_grads;
    (void)out_loss_sum;
    return -1;
}

void bptt_gpu_set_soft_spikes(bptt_gpu_t *gpu, int enable) {
    (void)gpu;
    (void)enable;
}

const char *bptt_gpu_last_error(const bptt_gpu_t *gpu) {
    (void)gpu;
    return "this binary was built without CUDA";
}

void bptt_gpu_destroy(bptt_gpu_t *gpu) {
    (void)gpu;
}
