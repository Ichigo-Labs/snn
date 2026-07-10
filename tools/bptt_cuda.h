#ifndef TOOLS_BPTT_CUDA_H
#define TOOLS_BPTT_CUDA_H

/*
 * GPU training backend for the MNIST/KMNIST BPTT tool.
 *
 * This is a batched reimplementation of the library's forward/backward/Adam
 * math (src/snn_bptt.c) on CUDA + cuBLAS: one minibatch is one set of GEMMs
 * per timestep instead of one sample per CPU thread. It is a *tool* backend,
 * not a library feature: it exists so the KMNIST experiment suites run in
 * minutes, it supports only the tool's configuration (static input, dense
 * layers), and unlike the simulator's CUDA backend it compiles with fmad on,
 * so it is numerically equivalent to the CPU trainer only up to float
 * summation order, not bitwise. `--mode gputest` verifies the gradients
 * against the library under the soft-spike test hook, where the model is
 * smooth and no spike can flip between the two implementations.
 *
 * The stub (bptt_cuda_stub.c) is linked when the tool is built without CUDA;
 * bptt_gpu_available() reports which one is present.
 */

#include <snn/snn.h>
#include <snn/snn_bptt.h>

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct bptt_gpu bptt_gpu_t;

/* 1 when the real backend is linked and a CUDA device is usable. */
int bptt_gpu_available(void);

/*
 * Uploads both datasets (raw idx pixels/labels) and the flat parameter vector
 * (layout of snn_bptt_get_parameters), and allocates tapes for `batch`
 * training samples and `eval_batch` evaluation samples plus Adam state.
 * Returns NULL with a message in err[err_len] on failure.
 */
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
                            size_t err_len);

/*
 * One epoch over the training samples order[0..count): forward/backward per
 * minibatch, one Adam step per minibatch, exactly as the CPU tool sequences
 * it. out_mean_loss is the per-sample average. Returns 0, or -1 with the
 * message in bptt_gpu_last_error() -- including when any minibatch produced a
 * non-finite gradient, which the CPU path treats as fatal too.
 */
int bptt_gpu_train_epoch(bptt_gpu_t *gpu, const snn_size_t *order, snn_size_t count, float *out_mean_loss);

/*
 * Scores the first `limit` samples (0 = the whole set) of the train or test
 * set on the current parameters. out_spikes is the mean hidden spike count
 * per sample, the same quantity evaluate() derives from
 * snn_bptt_spike_count on the CPU path.
 */
int bptt_gpu_evaluate(bptt_gpu_t *gpu,
                      int use_test,
                      snn_size_t limit,
                      float *out_accuracy,
                      float *out_loss,
                      double *out_spikes);

/*
 * Selftest hook: batch-summed gradients (layout of snn_bptt_grads_copy_out)
 * for the given training-set indices, without an optimizer step.
 */
int bptt_gpu_batch_grads(bptt_gpu_t *gpu,
                         const snn_size_t *indices,
                         snn_size_t count,
                         float *out_grads,
                         float *out_loss_sum);

/* Mirror of the library's snn_test_bptt_set_soft_spikes, for gputest. */
void bptt_gpu_set_soft_spikes(bptt_gpu_t *gpu, int enable);

const char *bptt_gpu_last_error(const bptt_gpu_t *gpu);

void bptt_gpu_destroy(bptt_gpu_t *gpu);

#ifdef __cplusplus
}
#endif

#endif /* TOOLS_BPTT_CUDA_H */
