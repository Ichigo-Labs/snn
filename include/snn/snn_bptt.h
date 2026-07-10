#ifndef SNN_SNN_BPTT_H
#define SNN_SNN_BPTT_H

#include <stddef.h>
#include <stdint.h>

#include "snn/snn.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Backpropagation through time (BPTT) with surrogate gradients for layered
 * leaky-integrate-and-fire networks.
 *
 * The trainable model is deliberately NOT the simulator's LIF. snn_step_cpu's
 * neuron resets to v_reset and holds an integer refractory counter, both
 * piecewise-constant state that carries no useful gradient. The trainable
 * neuron drops v_rest and the refractory period and resets by subtraction,
 * leaving an unrolled graph that is differentiable everywhere except at the
 * spike itself:
 *
 *   L = layer_count - 1 weight matrices, neuron layers j = 0 .. L-1
 *   W[j] has shape (layer_sizes[j+1] x layer_sizes[j]); b[j] has layer_sizes[j+1]
 *
 *   pre[0][t] = input[t]                    (a real-valued current vector)
 *   pre[j][t] = s[j-1][t]                   (j >= 1, same timestep)
 *   I[j][t]   = W[j] * pre[j][t] + b[j]
 *
 *   hidden j < L-1:  U[j][t] = beta*U[j][t-1] + I[j][t] - threshold*s[j][t-1]
 *                    s[j][t] = H(U[j][t] - threshold)
 *   output j = L-1:  U[j][t] = beta*U[j][t-1] + I[j][t]      (no spike, no reset)
 *
 *   logits  z = (1/T) * sum_{t=0}^{T-1} U[L-1][t]
 *   loss      = softmax cross-entropy(z, label)
 *
 * with U[j][-1] = 0 and s[j][-1] = 0. H is the Heaviside step, 1 when its
 * argument is >= 0 (ties spike). The output layer is a non-spiking leaky
 * integrator, so the readout is differentiable and no surrogate touches it.
 *
 * Backward replaces the Heaviside's derivative (a Dirac delta) by a surrogate
 * derivative phi = snn_surrogate_grad(); every other edge is differentiated
 * exactly, including the same-timestep cross-layer coupling and the
 * -threshold*s[j][t-1] reset path. The reset path therefore runs *through* the
 * surrogate, which is what makes peak-normalization of phi load-bearing rather
 * than cosmetic (see below). Set detach_reset to cut that path instead.
 *
 * With layer_count == 2 there is no hidden layer, no spike, and no surrogate:
 * the model degenerates to an exactly differentiable leaky-integrator readout,
 * which is what the finite-difference gradient tests use as ground truth.
 *
 * Threading: a network is read-only during forward/backward. Give each thread
 * its own workspace and its own gradient accumulator, then reduce with
 * snn_bptt_grads_add. Workspaces, gradient accumulators and optimizers must
 * not outlive the network they were created from.
 */

/*
 * Surrogate derivatives phi(x; alpha), evaluated at x = U - threshold.
 *
 * Every one is PEAK-NORMALIZED: phi(0) == 1 regardless of alpha, so alpha is
 * purely a width knob (the gradient window narrows as 1/alpha) and carries no
 * implicit gain. This is deliberate. Zenke & Vogels (Neural Computation 33(4),
 * 2021) show that surrogates whose peak grows with steepness explode or vanish
 * the gradient once the spike reset is differentiable -- which it is here --
 * and that once the peak is pinned, final accuracy is remarkably insensitive
 * to the shape. It also means two surrogates can be compared at one learning
 * rate without the comparison secretly measuring a gain difference.
 *
 * Note this normalization rescales some library conventions: snntorch's ATan
 * carries pi/2 factors and its Sigmoid peaks at alpha/4.
 *
 *   FAST_SIGMOID  1 / (1 + alpha*|x|)^2            SuperSpike (Zenke & Ganguli)
 *   ATAN          1 / (1 + (alpha*x)^2)            derivative of an arctan step
 *   SIGMOID       4 * sig(alpha*x) * (1 - sig(alpha*x))
 *   TRIANGLE      max(0, 1 - alpha*|x|)            piecewise linear
 *   GAUSSIAN      exp(-(alpha*x)^2 / 2)
 *   RECTANGULAR   1 when alpha*|x| < 1, else 0     boxcar (Wu et al., STBP)
 */
typedef enum snn_surrogate {
    SNN_SURROGATE_FAST_SIGMOID = 0,
    SNN_SURROGATE_ATAN = 1,
    SNN_SURROGATE_SIGMOID = 2,
    SNN_SURROGATE_TRIANGLE = 3,
    SNN_SURROGATE_GAUSSIAN = 4,
    SNN_SURROGATE_RECTANGULAR = 5,
    SNN_SURROGATE_COUNT = 6
} snn_surrogate_t;

const char *snn_surrogate_string(snn_surrogate_t surrogate);

/* phi(x; alpha). Returns 0 for an unknown surrogate. alpha must be > 0. */
float snn_surrogate_grad(snn_surrogate_t surrogate, float x, float alpha);

/*
 * The antiderivative S of phi pinned to S(0) = 1/2: the smooth spike function
 * whose exact derivative is snn_surrogate_grad. Because phi is peak-normalized
 * rather than normalized to unit area, S is a proper 0->1 smooth step only when
 * phi happens to integrate to 1; otherwise it saturates at 1/2 +- (area of
 * phi)/2.
 *
 * It is public because it is what gives a surrogate gradient its meaning: the
 * backward pass computes the *exact* gradient of the network in which H is
 * replaced by S. The finite-difference gradient tests exploit exactly that --
 * they differentiate the S-forward and compare against the surrogate backward.
 * Returns 0 for an unknown surrogate.
 */
float snn_surrogate_primitive(snn_surrogate_t surrogate, float x, float alpha);

typedef struct snn_bptt_config {
    const snn_size_t *layer_sizes; /* layer_count entries, all nonzero */
    size_t layer_count;            /* >= 2: input size, hidden sizes..., output size */
    snn_size_t timesteps;          /* T >= 1 */
    float beta;                    /* membrane decay per step, in [0, 1) */
    float threshold;               /* spike threshold, > 0 */
    snn_surrogate_t surrogate;
    float surrogate_alpha; /* gradient-window steepness, > 0 */
    /* Nonzero cuts the gradient flowing back through the reset term, so a
     * spike's effect on its own future membrane is ignored. Cheaper, and the
     * usual default in spikingjelly; the honest full-BPTT gradient keeps it. */
    int detach_reset;
    /* Kaiming-uniform half-width: weights ~ U(-g*sqrt(3/fan_in), +g*sqrt(3/fan_in)). */
    float weight_init_gain; /* > 0 */
    uint64_t seed;
} snn_bptt_config_t;

typedef struct snn_bptt_network snn_bptt_network_t;
typedef struct snn_bptt_workspace snn_bptt_workspace_t;
typedef struct snn_bptt_grads snn_bptt_grads_t;
typedef struct snn_bptt_optimizer snn_bptt_optimizer_t;

snn_bptt_config_t snn_bptt_default_config(const snn_size_t *layer_sizes, size_t layer_count, snn_size_t timesteps);
snn_status_t snn_bptt_config_validate(const snn_bptt_config_t *config);

/* exp(-dt_ms / membrane_tau_ms): the decay the simulator's LIF uses, so a
 * trainable network can be given the membrane time constant of a network built
 * for snn_step_cpu. Returns 0 for parameters snn_lif_params_validate rejects. */
float snn_bptt_beta_from_lif(const snn_lif_params_t *params);

snn_status_t snn_bptt_network_create(const snn_bptt_config_t *config, snn_bptt_network_t **out_network);
void snn_bptt_network_free(snn_bptt_network_t *network);

size_t snn_bptt_layer_count(const snn_bptt_network_t *network);
snn_size_t snn_bptt_layer_size(const snn_bptt_network_t *network, size_t layer);
snn_size_t snn_bptt_input_size(const snn_bptt_network_t *network);
snn_size_t snn_bptt_output_size(const snn_bptt_network_t *network);
snn_size_t snn_bptt_timesteps(const snn_bptt_network_t *network);
snn_size_t snn_bptt_parameter_count(const snn_bptt_network_t *network);
snn_surrogate_t snn_bptt_network_surrogate(const snn_bptt_network_t *network);
float snn_bptt_network_alpha(const snn_bptt_network_t *network);
int snn_bptt_network_detach_reset(const snn_bptt_network_t *network);
snn_status_t snn_bptt_network_set_surrogate(snn_bptt_network_t *network, snn_surrogate_t surrogate, float alpha);

/* Flat parameter vector: W[0], b[0], W[1], b[1], ... with each W row-major.
 * Both counts are capacities: they must be at least snn_bptt_parameter_count,
 * and only that many entries are read or written. */
snn_status_t snn_bptt_get_parameters(const snn_bptt_network_t *network, float *out_params, snn_size_t capacity);
snn_status_t snn_bptt_set_parameters(snn_bptt_network_t *network, const float *params, snn_size_t capacity);

snn_status_t snn_bptt_workspace_create(const snn_bptt_network_t *network, snn_bptt_workspace_t **out_workspace);
void snn_bptt_workspace_free(snn_bptt_workspace_t *workspace);

snn_status_t snn_bptt_grads_create(const snn_bptt_network_t *network, snn_bptt_grads_t **out_grads);
void snn_bptt_grads_free(snn_bptt_grads_t *grads);
snn_status_t snn_bptt_grads_zero(snn_bptt_grads_t *grads);
snn_status_t snn_bptt_grads_add(snn_bptt_grads_t *dst, const snn_bptt_grads_t *src);
snn_status_t snn_bptt_grads_copy_out(const snn_bptt_grads_t *grads, float *out_grads, snn_size_t capacity);

/*
 * Unrolls the network over its timesteps and records the tape.
 *
 * static_input != 0: `input` is one frame of input_size currents, injected
 * unchanged at every timestep (constant-current encoding). This is not merely
 * a convenience -- the first layer's drive W[0]*input + b[0] is then constant
 * in t, so it is computed once instead of T times, and its weight gradient
 * collapses from T rank-1 updates to one. On 784-N-10 that is most of the
 * arithmetic in the whole step.
 *
 * static_input == 0: `input` is a timesteps x input_size tape, row t first.
 *
 * All input values must be finite.
 */
snn_status_t snn_bptt_forward(const snn_bptt_network_t *network,
                              snn_bptt_workspace_t *workspace,
                              const float *input,
                              int static_input);

/* Results of the last snn_bptt_forward on this workspace. A workspace on which
 * no forward has run holds the zero-initialized tape, so these report uniform
 * logits, prediction 0 and no spikes rather than failing. */
snn_status_t snn_bptt_copy_logits(const snn_bptt_workspace_t *workspace, float *out_logits, snn_size_t capacity);
snn_status_t snn_bptt_cross_entropy(const snn_bptt_workspace_t *workspace, snn_size_t label, float *out_loss);
snn_size_t snn_bptt_prediction(const snn_bptt_workspace_t *workspace);
/* Hard threshold crossings across all hidden layers and timesteps in the last
 * forward -- the sparsity metric. It counts H(U - threshold) on whichever
 * membrane trajectory ran; under the soft-spike test hook the trajectory
 * itself differs, because the reset feeds back S(x) rather than 1. */
snn_size_t snn_bptt_spike_count(const snn_bptt_workspace_t *workspace);

/*
 * Forward, then BPTT. Accumulates dLoss/dparam into `grads` (which the caller
 * zeroes once per batch). out_loss and out_correct may be NULL.
 */
snn_status_t snn_bptt_forward_backward(const snn_bptt_network_t *network,
                                       snn_bptt_workspace_t *workspace,
                                       const float *input,
                                       int static_input,
                                       snn_size_t label,
                                       snn_bptt_grads_t *grads,
                                       float *out_loss,
                                       int *out_correct);

/* Adam. lr > 0, beta1/beta2 in [0, 1), eps > 0. */
snn_status_t snn_bptt_optimizer_create(const snn_bptt_network_t *network,
                                       float lr,
                                       float beta1,
                                       float beta2,
                                       float eps,
                                       snn_bptt_optimizer_t **out_optimizer);
void snn_bptt_optimizer_free(snn_bptt_optimizer_t *optimizer);
snn_status_t snn_bptt_optimizer_set_lr(snn_bptt_optimizer_t *optimizer, float lr);
/* Applies grads/batch_size to the network's parameters. batch_size >= 1.
 * A gradient containing any non-finite entry is rejected with
 * SNN_ERR_INVALID_ARGUMENT and leaves the optimizer and network untouched:
 * the moments are exponential moving averages, so a NaN admitted once would
 * never decay out of them. */
snn_status_t snn_bptt_optimizer_step(snn_bptt_optimizer_t *optimizer,
                                     snn_bptt_network_t *network,
                                     const snn_bptt_grads_t *grads,
                                     snn_size_t batch_size);

#ifdef __cplusplus
}
#endif

#endif /* SNN_SNN_BPTT_H */
