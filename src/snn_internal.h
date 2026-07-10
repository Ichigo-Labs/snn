#ifndef SNN_INTERNAL_H
#define SNN_INTERNAL_H

#include <stddef.h>

#include "snn/snn.h"

/*
 * calloc for library translation units other than snn.c. Defined in snn.c so
 * that every allocation in the library funnels through the one
 * SNN_ENABLE_TEST_HOOKS failure-injection counter, whichever file made it.
 */
void *snn_internal_calloc(size_t count, size_t elem_size);

struct snn_network {
    snn_size_t neuron_count;
    snn_size_t synapse_count;
    snn_architecture_t architecture;
    snn_lif_params_t lif;
    float decay;
    snn_size_t *row_ptr;
    snn_size_t *col_idx;
    float *weights;
};

struct snn_state {
    snn_size_t neuron_count;
    float *voltage;
    float *current;
    float *next_current;
    uint32_t *refractory;
    uint8_t *spikes;
    snn_size_t *spike_indices;
    snn_size_t spike_count;
    /*
     * OpenMP builds only: thread_count * neuron_count floats of per-thread
     * scatter buffers for parallel synaptic propagation, kept all-zero
     * between steps by the reduction. NULL/0 in serial builds. Unconditional
     * fields so every translation unit sees the same struct layout.
     */
    float *thread_partials;
    snn_size_t thread_count;
};

#endif /* SNN_INTERNAL_H */
