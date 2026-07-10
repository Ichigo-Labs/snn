#ifndef SNN_SNN_TEST_H
#define SNN_SNN_TEST_H

#include <stdint.h>
#include "snn/snn.h"
#include "snn/snn_bptt.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifdef SNN_ENABLE_TEST_HOOKS
void snn_test_set_alloc_fail_after(int64_t successful_allocations_before_failure);
void snn_test_disable_alloc_failure(void);
int snn_test_exercise_internal_guards(void);
snn_status_t snn_test_prefix_layer_offsets(const snn_size_t *sizes, size_t count, snn_size_t *offsets);
/* Shrink a state's parallel-propagation thread budget (no-op in serial builds). */
void snn_test_state_limit_threads(snn_state_t *state, snn_size_t thread_count);
snn_cuda_context_t *snn_test_nonnull_cuda_context(void);
/* CUDA-backend fault injection: fail the Nth cudaMalloc/cudaMemcpy wrapper. */
void snn_test_cuda_set_fail_after(int64_t calls_before_failure);
void snn_test_cuda_disable_failure(void);
/* Force snn_cuda_available()/cudaMemGetInfo to report failure (create errors). */
void snn_test_cuda_force_unavailable(int enable);
void snn_test_cuda_force_meminfo_fail(int enable);
/*
 * Make the BPTT forward emit S(U - threshold) -- snn_surrogate_primitive --
 * instead of the Heaviside. The surrogate backward is then the *exact*
 * gradient of the resulting smooth model, so a finite-difference check
 * validates the whole unrolled backward (reset path, cross-layer coupling,
 * surrogate) against a real scalar loss. Differentiating the hard-spike
 * forward would be meaningless: it is piecewise constant in the weights.
 */
void snn_test_bptt_set_soft_spikes(snn_bptt_network_t *network, int enable);
#endif

#ifdef __cplusplus
}
#endif

#endif /* SNN_SNN_TEST_H */
