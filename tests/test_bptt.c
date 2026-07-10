/*
 * Tests for the BPTT / surrogate-gradient trainer.
 *
 * The gradient is checked three independent ways, because no single check is
 * sufficient:
 *
 *  1. TRANSPOSE (adjoint) TEST. The backward pass must be the exact transpose
 *     of the forward-mode linearization of the taped forward. A reference
 *     tangent (JVP) propagation is written here from the model definition, in
 *     forward mode -- structurally nothing like the library's reverse-mode
 *     code -- and the identity <dLoss/dz, J*d> == <dLoss/dparams, d> is
 *     asserted for random directions d. This catches sign errors, index
 *     shifts, wrong timestep alignment and dropped gradient paths, to float
 *     roundoff.
 *
 *  2. FINITE DIFFERENCES against a real scalar loss. Differentiating the
 *     hard-spike forward is meaningless -- it is piecewise constant in the
 *     weights, so a perturbation either changes nothing or flips a spike. But
 *     the surrogate backward IS the exact gradient of the model in which the
 *     Heaviside is replaced by S = snn_surrogate_primitive (whose derivative
 *     is the surrogate by construction). The snn_test_bptt_set_soft_spikes
 *     hook makes the forward emit exactly that, so central differences of the
 *     cross-entropy must reproduce the analytic gradient. This grounds the
 *     whole chain -- readout, reset path, cross-layer coupling, surrogate --
 *     against an actual loss, catching a conceptual error that checks 1 and
 *     the implementation could share.
 *
 *  3. The layer_count == 2 network has no hidden layer, no spike, and no
 *     surrogate, so it is exactly differentiable as written. Its finite
 *     differences validate the readout and output-layer recurrence with no
 *     hook involved at all.
 *
 * Configurations are chosen against the ways a gradient check silently passes:
 * T > 1 (T == 1 hides every temporal error, including a missing reset term),
 * layer_count >= 4 (a two-hidden-layer net exercises the multi-hop
 * same-timestep hand-off), and parameters that make some neurons spike and
 * some not (an all-silent or all-firing net has degenerate gradients).
 */

#include <snn/snn.h>
#include <snn/snn_bptt.h>
#include <snn/snn_test.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "test_common.h"

#define REF_MAX_LAYERS 8
#define REF_MAX_OUT 32

/* ------------------------------------------------------------------ */
/* helpers                                                            */
/* ------------------------------------------------------------------ */

static uint64_t trng_next(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

/* Uniform in (-scale, scale). */
static float trng_signed(uint64_t *state, float scale) {
    const uint64_t r = trng_next(state);
    const double unit = (double)(r >> 11) * (1.0 / 9007199254740992.0);
    return (float)(2.0 * unit - 1.0) * scale;
}

static void fill_signed(float *values, snn_size_t count, uint64_t *state, float scale) {
    snn_size_t i = 0;
    for (i = 0; i < count; ++i) {
        values[i] = trng_signed(state, scale);
    }
}

static float dot(const float *a, const float *b, snn_size_t count) {
    snn_size_t i = 0;
    double acc = 0.0;
    for (i = 0; i < count; ++i) {
        acc += (double)a[i] * (double)b[i];
    }
    return (float)acc;
}

static void *xcalloc(size_t count, size_t size) {
    void *p = calloc(count, size);
    ASSERT_TRUE(p != NULL);
    return p;
}

/* ------------------------------------------------------------------ */
/* reference model, written straight from the header's definition      */
/* ------------------------------------------------------------------ */

typedef struct {
    snn_size_t sizes[REF_MAX_LAYERS + 1];
    size_t layers; /* L */
    snn_size_t timesteps;
    float beta;
    float threshold;
    float alpha;
    snn_surrogate_t surrogate;
    int detach;
    int soft;
    snn_size_t w_off[REF_MAX_LAYERS];
    snn_size_t b_off[REF_MAX_LAYERS];
    snn_size_t param_count;
} ref_net_t;

typedef struct {
    float *u[REF_MAX_LAYERS];
    float *s[REF_MAX_LAYERS];
    float logits[REF_MAX_OUT];
} ref_tape_t;

/* Derives the reference's view of the network purely from the public API and
 * the documented flat parameter layout; asserting param_count matches is
 * itself a check of that documented layout. */
static void ref_init(ref_net_t *r, const snn_bptt_network_t *net, float beta, float threshold, int soft) {
    size_t j = 0;
    snn_size_t run = 0;
    r->layers = snn_bptt_layer_count(net) - 1u;
    ASSERT_TRUE(r->layers >= 1u && r->layers <= (size_t)REF_MAX_LAYERS);
    for (j = 0; j <= r->layers; ++j) {
        r->sizes[j] = snn_bptt_layer_size(net, j);
    }
    ASSERT_TRUE(r->sizes[r->layers] <= (snn_size_t)REF_MAX_OUT);
    r->timesteps = snn_bptt_timesteps(net);
    r->beta = beta;
    r->threshold = threshold;
    r->alpha = snn_bptt_network_alpha(net);
    r->surrogate = snn_bptt_network_surrogate(net);
    r->detach = snn_bptt_network_detach_reset(net);
    r->soft = soft;
    for (j = 0; j < r->layers; ++j) {
        r->w_off[j] = run;
        run += r->sizes[j + 1u] * r->sizes[j];
        r->b_off[j] = run;
        run += r->sizes[j + 1u];
    }
    r->param_count = run;
    ASSERT_EQ_U64(r->param_count, snn_bptt_parameter_count(net));
}

static void ref_tape_alloc(const ref_net_t *r, ref_tape_t *tp) {
    size_t j = 0;
    for (j = 0; j < r->layers; ++j) {
        const size_t n = (size_t)(r->timesteps * r->sizes[j + 1u]);
        tp->u[j] = (float *)xcalloc(n, sizeof(float));
        tp->s[j] = (float *)xcalloc(n, sizeof(float));
    }
}

static void ref_tape_free(const ref_net_t *r, ref_tape_t *tp) {
    size_t j = 0;
    for (j = 0; j < r->layers; ++j) {
        free(tp->u[j]);
        free(tp->s[j]);
    }
}

static float ref_spike(const ref_net_t *r, float x) {
    return r->soft ? snn_surrogate_primitive(r->surrogate, x, r->alpha) : (x >= 0.0f ? 1.0f : 0.0f);
}

static const float *ref_pre(const ref_net_t *r, const ref_tape_t *tp, size_t j, snn_size_t t, const float *input,
                            int static_input) {
    if (j == 0u) {
        return static_input ? input : input + (size_t)t * (size_t)r->sizes[0];
    }
    return tp->s[j - 1u] + (size_t)t * (size_t)r->sizes[j];
}

static void ref_forward(const ref_net_t *r, const float *p, const float *input, int static_input, ref_tape_t *tp) {
    size_t j = 0;
    snn_size_t t = 0;
    snn_size_t i = 0;
    snn_size_t c = 0;
    for (t = 0; t < r->timesteps; ++t) {
        for (j = 0; j < r->layers; ++j) {
            const snn_size_t rows = r->sizes[j + 1u];
            const snn_size_t cols = r->sizes[j];
            const float *w = p + r->w_off[j];
            const float *b = p + r->b_off[j];
            const float *pre = ref_pre(r, tp, j, t, input, static_input);
            float *u = tp->u[j] + (size_t)t * (size_t)rows;
            for (i = 0; i < rows; ++i) {
                float acc = b[i];
                for (c = 0; c < cols; ++c) {
                    acc += w[(size_t)i * (size_t)cols + c] * pre[c];
                }
                if (t > 0u) {
                    acc += r->beta * tp->u[j][(size_t)(t - 1u) * (size_t)rows + i];
                    if (j + 1u < r->layers) {
                        acc -= r->threshold * tp->s[j][(size_t)(t - 1u) * (size_t)rows + i];
                    }
                }
                u[i] = acc;
            }
            if (j + 1u < r->layers) {
                float *s = tp->s[j] + (size_t)t * (size_t)rows;
                for (i = 0; i < rows; ++i) {
                    s[i] = ref_spike(r, u[i] - r->threshold);
                }
            }
        }
    }
    {
        const snn_size_t rows = r->sizes[r->layers];
        const float *u_out = tp->u[r->layers - 1u];
        for (i = 0; i < rows; ++i) {
            float acc = 0.0f;
            for (t = 0; t < r->timesteps; ++t) {
                acc += u_out[(size_t)t * (size_t)rows + i];
            }
            tp->logits[i] = acc / (float)r->timesteps;
        }
    }
}

/*
 * Forward-mode tangent propagation: given a parameter direction d, produce
 * dz = J*d, where J is the Jacobian of the logits with respect to the
 * parameters, linearized at the taped forward with the surrogate standing in
 * for the Heaviside's derivative. Reverse mode is what the library does; this
 * is the same linear map built the other way round.
 */
static void ref_jvp(const ref_net_t *r,
                    const float *p,
                    const float *d,
                    const float *input,
                    int static_input,
                    const ref_tape_t *tp,
                    float *dz) {
    float *du[REF_MAX_LAYERS];
    float *ds[REF_MAX_LAYERS];
    size_t j = 0;
    snn_size_t t = 0;
    snn_size_t i = 0;
    snn_size_t c = 0;

    for (j = 0; j < r->layers; ++j) {
        const size_t n = (size_t)(r->timesteps * r->sizes[j + 1u]);
        du[j] = (float *)xcalloc(n, sizeof(float));
        ds[j] = (float *)xcalloc(n, sizeof(float));
    }
    for (t = 0; t < r->timesteps; ++t) {
        for (j = 0; j < r->layers; ++j) {
            const snn_size_t rows = r->sizes[j + 1u];
            const snn_size_t cols = r->sizes[j];
            const float *w = p + r->w_off[j];
            const float *dw = d + r->w_off[j];
            const float *db = d + r->b_off[j];
            const float *pre = ref_pre(r, tp, j, t, input, static_input);
            /* The input is not a parameter, so its tangent is zero. */
            const float *dpre = j == 0u ? NULL : ds[j - 1u] + (size_t)t * (size_t)cols;
            for (i = 0; i < rows; ++i) {
                float acc = db[i];
                for (c = 0; c < cols; ++c) {
                    acc += dw[(size_t)i * (size_t)cols + c] * pre[c];
                }
                if (dpre != NULL) {
                    for (c = 0; c < cols; ++c) {
                        acc += w[(size_t)i * (size_t)cols + c] * dpre[c];
                    }
                }
                if (t > 0u) {
                    acc += r->beta * du[j][(size_t)(t - 1u) * (size_t)rows + i];
                    if (j + 1u < r->layers && !r->detach) {
                        acc -= r->threshold * ds[j][(size_t)(t - 1u) * (size_t)rows + i];
                    }
                }
                du[j][(size_t)t * (size_t)rows + i] = acc;
            }
            if (j + 1u < r->layers) {
                for (i = 0; i < rows; ++i) {
                    const float x = tp->u[j][(size_t)t * (size_t)rows + i] - r->threshold;
                    ds[j][(size_t)t * (size_t)rows + i] =
                        snn_surrogate_grad(r->surrogate, x, r->alpha) * du[j][(size_t)t * (size_t)rows + i];
                }
            }
        }
    }
    {
        const snn_size_t rows = r->sizes[r->layers];
        for (i = 0; i < rows; ++i) {
            float acc = 0.0f;
            for (t = 0; t < r->timesteps; ++t) {
                acc += du[r->layers - 1u][(size_t)t * (size_t)rows + i];
            }
            dz[i] = acc / (float)r->timesteps;
        }
    }
    for (j = 0; j < r->layers; ++j) {
        free(du[j]);
        free(ds[j]);
    }
}

/* Hard threshold crossings of whatever trajectory the tape recorded. */
static snn_size_t ref_count_crossings(const ref_net_t *r, const ref_tape_t *tp) {
    size_t j = 0;
    snn_size_t t = 0;
    snn_size_t i = 0;
    snn_size_t total = 0;
    for (j = 0; j + 1u < r->layers; ++j) {
        const snn_size_t rows = r->sizes[j + 1u];
        for (t = 0; t < r->timesteps; ++t) {
            for (i = 0; i < rows; ++i) {
                total += tp->u[j][(size_t)t * (size_t)rows + i] - r->threshold >= 0.0f;
            }
        }
    }
    return total;
}

/* dLoss/dlogits for softmax cross-entropy. */
static void ref_softmax_grad(const float *logits, snn_size_t count, snn_size_t label, float *gz) {
    snn_size_t k = 0;
    float max_logit = logits[0];
    float sum = 0.0f;
    for (k = 1; k < count; ++k) {
        if (logits[k] > max_logit) {
            max_logit = logits[k];
        }
    }
    for (k = 0; k < count; ++k) {
        gz[k] = expf(logits[k] - max_logit);
        sum += gz[k];
    }
    for (k = 0; k < count; ++k) {
        gz[k] /= sum;
    }
    gz[label] -= 1.0f;
}

/* ------------------------------------------------------------------ */
/* surrogate functions                                                */
/* ------------------------------------------------------------------ */

static void test_surrogate_functions(void) {
    static const snn_surrogate_t all[] = {SNN_SURROGATE_FAST_SIGMOID, SNN_SURROGATE_ATAN,     SNN_SURROGATE_SIGMOID,
                                          SNN_SURROGATE_TRIANGLE,     SNN_SURROGATE_GAUSSIAN, SNN_SURROGATE_RECTANGULAR};
    static const float alphas[] = {0.5f, 1.0f, 2.0f, 5.0f};
    int k = 0;
    int a = 0;

    ASSERT_TRUE(strcmp(snn_surrogate_string(SNN_SURROGATE_FAST_SIGMOID), "fast_sigmoid") == 0);
    ASSERT_TRUE(strcmp(snn_surrogate_string(SNN_SURROGATE_ATAN), "atan") == 0);
    ASSERT_TRUE(strcmp(snn_surrogate_string(SNN_SURROGATE_SIGMOID), "sigmoid") == 0);
    ASSERT_TRUE(strcmp(snn_surrogate_string(SNN_SURROGATE_TRIANGLE), "triangle") == 0);
    ASSERT_TRUE(strcmp(snn_surrogate_string(SNN_SURROGATE_GAUSSIAN), "gaussian") == 0);
    ASSERT_TRUE(strcmp(snn_surrogate_string(SNN_SURROGATE_RECTANGULAR), "rectangular") == 0);
    ASSERT_TRUE(strcmp(snn_surrogate_string(SNN_SURROGATE_COUNT), "unknown") == 0);
    ASSERT_TRUE(strcmp(snn_surrogate_string((snn_surrogate_t)999), "unknown") == 0);
    ASSERT_NEAR(snn_surrogate_grad((snn_surrogate_t)999, 0.0f, 1.0f), 0.0f, 0.0f);
    ASSERT_NEAR(snn_surrogate_primitive((snn_surrogate_t)999, 0.0f, 1.0f), 0.0f, 0.0f);

    for (k = 0; k < 6; ++k) {
        for (a = 0; a < 4; ++a) {
            const snn_surrogate_t s = all[k];
            const float alpha = alphas[a];
            int step = 0;

            /* Peak normalization: phi(0) == 1 for every surrogate and every
             * alpha. This is the invariant that makes alpha a pure width knob
             * and keeps the differentiable reset from exploding. */
            ASSERT_NEAR(snn_surrogate_grad(s, 0.0f, alpha), 1.0f, 1e-6f);
            /* S(0) == 1/2 and S is the antiderivative of phi. */
            ASSERT_NEAR(snn_surrogate_primitive(s, 0.0f, alpha), 0.5f, 1e-6f);

            for (step = -12; step <= 12; ++step) {
                const float x = (float)step * (0.25f / alpha);
                const float phi = snn_surrogate_grad(s, x, alpha);
                /* Even, bounded by the peak, non-negative. */
                ASSERT_NEAR(phi, snn_surrogate_grad(s, -x, alpha), 1e-6f);
                ASSERT_TRUE(phi >= 0.0f && phi <= 1.0f + 1e-6f);
                /* S is odd about (0, 1/2). */
                ASSERT_NEAR(snn_surrogate_primitive(s, x, alpha) + snn_surrogate_primitive(s, -x, alpha), 1.0f, 1e-5f);

                /* d/dx S(x) == phi(x). The rectangular kernel's S has a corner
                 * at |x| == 1/alpha, where a central difference straddles the
                 * discontinuity, so skip that one point. */
                if (!(s == SNN_SURROGATE_RECTANGULAR && step * step == 16)) {
                    const float h = 1e-3f / alpha;
                    const float numeric = (snn_surrogate_primitive(s, x + h, alpha) -
                                           snn_surrogate_primitive(s, x - h, alpha)) /
                                          (2.0f * h);
                    ASSERT_NEAR(numeric, phi, 2e-3f);
                }
            }
            /* Width scales as 1/alpha: the surrogate has decayed by 4/alpha. */
            ASSERT_TRUE(snn_surrogate_grad(s, 4.0f / alpha, alpha) < 0.2f);
        }
    }

    /*
     * Pin each surrogate's closed form. Everything above -- peak 1, evenness,
     * S' == phi, decay -- is satisfied by ANY peak-normalized even bump paired
     * with its own antiderivative, so without this block one surrogate could
     * silently be another's shape and the whole suite would still pass. The
     * probes are the values at x = 1/alpha, where all six differ (triangle and
     * rectangular both vanish there, so they are separated at x = 1/(2*alpha)).
     */
    for (a = 0; a < 4; ++a) {
        const float alpha = alphas[a];
        const float e = 1.0f / alpha;  /* the surrogate's natural width unit */
        const float h = 0.5f / alpha;
        /* 1/(1+alpha|x|)^2 at alpha|x| = 1 */
        ASSERT_NEAR(snn_surrogate_grad(SNN_SURROGATE_FAST_SIGMOID, e, alpha), 0.25f, 1e-6f);
        /* 1/(1+(alpha x)^2) at alpha x = 1 */
        ASSERT_NEAR(snn_surrogate_grad(SNN_SURROGATE_ATAN, e, alpha), 0.5f, 1e-6f);
        /* 4 sig(1) (1 - sig(1)) */
        ASSERT_NEAR(snn_surrogate_grad(SNN_SURROGATE_SIGMOID, e, alpha), 0.78644773f, 1e-6f);
        /* max(0, 1 - alpha|x|) */
        ASSERT_NEAR(snn_surrogate_grad(SNN_SURROGATE_TRIANGLE, e, alpha), 0.0f, 1e-6f);
        ASSERT_NEAR(snn_surrogate_grad(SNN_SURROGATE_TRIANGLE, h, alpha), 0.5f, 1e-6f);
        /* exp(-(alpha x)^2 / 2) at alpha x = 1 */
        ASSERT_NEAR(snn_surrogate_grad(SNN_SURROGATE_GAUSSIAN, e, alpha), 0.60653066f, 1e-6f);
        /* boxcar: open at alpha|x| == 1 */
        ASSERT_NEAR(snn_surrogate_grad(SNN_SURROGATE_RECTANGULAR, e, alpha), 0.0f, 0.0f);
        ASSERT_NEAR(snn_surrogate_grad(SNN_SURROGATE_RECTANGULAR, h, alpha), 1.0f, 0.0f);

        /* And the primitives, which the soft-spike forward actually evaluates.
         * (fast_sigmoid and triangle happen to agree at x = 1/alpha; the
         * derivative probes above separate them.) */
        ASSERT_NEAR(snn_surrogate_primitive(SNN_SURROGATE_FAST_SIGMOID, e, alpha), 0.5f + 0.5f * e, 1e-5f);
        ASSERT_NEAR(snn_surrogate_primitive(SNN_SURROGATE_ATAN, e, alpha), 0.5f + 0.78539816f * e, 1e-5f);
        ASSERT_NEAR(snn_surrogate_primitive(SNN_SURROGATE_SIGMOID, e, alpha), 0.5f + 0.92423431f * e, 1e-5f);
        ASSERT_NEAR(snn_surrogate_primitive(SNN_SURROGATE_TRIANGLE, e, alpha), 0.5f + 0.5f * e, 1e-5f);
        ASSERT_NEAR(snn_surrogate_primitive(SNN_SURROGATE_GAUSSIAN, e, alpha), 0.5f + 0.85562439f * e, 1e-5f);
        ASSERT_NEAR(snn_surrogate_primitive(SNN_SURROGATE_RECTANGULAR, e, alpha), 0.5f + e, 1e-5f);
    }
    /* The six shapes are mutually distinct, not aliases of one another. */
    {
        int p = 0;
        int q = 0;
        for (p = 0; p < 6; ++p) {
            for (q = p + 1; q < 6; ++q) {
                const float gp = snn_surrogate_grad(all[p], 0.5f, 2.0f);
                const float gq = snn_surrogate_grad(all[q], 0.5f, 2.0f);
                const float hp = snn_surrogate_grad(all[p], 0.25f, 2.0f);
                const float hq = snn_surrogate_grad(all[q], 0.25f, 2.0f);
                ASSERT_TRUE(fabsf(gp - gq) > 1e-4f || fabsf(hp - hq) > 1e-4f);
            }
        }
    }

    /*
     * Saturation at extreme x. The fast-sigmoid primitive would compute
     * alpha*|x| as inf and then x/inf == 0, collapsing to 1/2 exactly where it
     * must saturate at 1/2 + sign(x)/alpha -- a non-monotone step of 1/alpha
     * between two adjacent floats.
     */
    ASSERT_NEAR(snn_surrogate_primitive(SNN_SURROGATE_FAST_SIGMOID, 1e30f, 100.0f), 0.51f, 1e-5f);
    ASSERT_NEAR(snn_surrogate_primitive(SNN_SURROGATE_FAST_SIGMOID, 3.4e38f, 100.0f), 0.51f, 1e-5f);
    ASSERT_NEAR(snn_surrogate_primitive(SNN_SURROGATE_FAST_SIGMOID, -3.4e38f, 100.0f), 0.49f, 1e-5f);
    ASSERT_TRUE(snn_surrogate_primitive(SNN_SURROGATE_FAST_SIGMOID, 3.4e38f, 100.0f) >
                snn_surrogate_primitive(SNN_SURROGATE_FAST_SIGMOID, 1e30f, 100.0f) - 1e-6f);
    /* The other five saturate through library functions that already handle inf. */
    ASSERT_NEAR(snn_surrogate_grad(SNN_SURROGATE_FAST_SIGMOID, 3.4e38f, 100.0f), 0.0f, 0.0f);
    ASSERT_NEAR(snn_surrogate_grad(SNN_SURROGATE_ATAN, 3.4e38f, 100.0f), 0.0f, 0.0f);
    ASSERT_NEAR(snn_surrogate_grad(SNN_SURROGATE_GAUSSIAN, 3.4e38f, 100.0f), 0.0f, 0.0f);
    ASSERT_NEAR(snn_surrogate_grad(SNN_SURROGATE_TRIANGLE, 3.4e38f, 100.0f), 0.0f, 0.0f);
    ASSERT_NEAR(snn_surrogate_grad(SNN_SURROGATE_RECTANGULAR, 3.4e38f, 100.0f), 0.0f, 0.0f);
    ASSERT_NEAR(snn_surrogate_grad(SNN_SURROGATE_SIGMOID, 3.4e38f, 100.0f), 0.0f, 0.0f);

    /* Compact support of the two kernels that have it. */
    ASSERT_NEAR(snn_surrogate_grad(SNN_SURROGATE_TRIANGLE, 0.5f, 2.0f), 0.0f, 0.0f);
    ASSERT_NEAR(snn_surrogate_grad(SNN_SURROGATE_TRIANGLE, 0.25f, 2.0f), 0.5f, 1e-6f);
    ASSERT_NEAR(snn_surrogate_grad(SNN_SURROGATE_RECTANGULAR, 0.49f, 2.0f), 1.0f, 0.0f);
    ASSERT_NEAR(snn_surrogate_grad(SNN_SURROGATE_RECTANGULAR, 0.51f, 2.0f), 0.0f, 0.0f);
    /* ...and the saturation of their primitives beyond it. */
    ASSERT_NEAR(snn_surrogate_primitive(SNN_SURROGATE_TRIANGLE, 5.0f, 2.0f), 0.75f, 1e-6f);
    ASSERT_NEAR(snn_surrogate_primitive(SNN_SURROGATE_TRIANGLE, -5.0f, 2.0f), 0.25f, 1e-6f);
    ASSERT_NEAR(snn_surrogate_primitive(SNN_SURROGATE_RECTANGULAR, 5.0f, 2.0f), 1.0f, 1e-6f);
    ASSERT_NEAR(snn_surrogate_primitive(SNN_SURROGATE_RECTANGULAR, -5.0f, 2.0f), 0.0f, 1e-6f);
}

/* ------------------------------------------------------------------ */
/* config, lifecycle, accessors                                       */
/* ------------------------------------------------------------------ */

static void test_config_and_defaults(void) {
    snn_size_t layers[] = {4, 3, 2};
    snn_bptt_config_t cfg = snn_bptt_default_config(layers, 3, 7);
    snn_lif_params_t lif = snn_default_lif_params();
    snn_size_t zero_layer[] = {4, 0, 2};

    ASSERT_TRUE(cfg.layer_sizes == layers);
    ASSERT_EQ_U64(cfg.layer_count, 3);
    ASSERT_EQ_U64(cfg.timesteps, 7);
    ASSERT_NEAR(cfg.beta, 0.95f, 0.0f);
    ASSERT_NEAR(cfg.threshold, 1.0f, 0.0f);
    ASSERT_EQ_INT(cfg.surrogate, SNN_SURROGATE_ATAN);
    ASSERT_NEAR(cfg.surrogate_alpha, 2.0f, 0.0f);
    ASSERT_EQ_INT(cfg.detach_reset, 0);
    ASSERT_NEAR(cfg.weight_init_gain, 1.0f, 0.0f);
    ASSERT_TRUE(cfg.seed != 0u);
    ASSERT_EQ_INT(snn_bptt_config_validate(&cfg), SNN_OK);

    ASSERT_EQ_INT(snn_bptt_config_validate(NULL), SNN_ERR_INVALID_ARGUMENT);
    cfg.layer_sizes = NULL;
    ASSERT_EQ_INT(snn_bptt_config_validate(&cfg), SNN_ERR_INVALID_ARGUMENT);
    cfg = snn_bptt_default_config(layers, 1, 7);
    ASSERT_EQ_INT(snn_bptt_config_validate(&cfg), SNN_ERR_INVALID_ARGUMENT);
    cfg = snn_bptt_default_config(layers, 3, 0);
    ASSERT_EQ_INT(snn_bptt_config_validate(&cfg), SNN_ERR_INVALID_ARGUMENT);

    cfg = snn_bptt_default_config(layers, 3, 7);
    cfg.beta = 1.0f;
    ASSERT_EQ_INT(snn_bptt_config_validate(&cfg), SNN_ERR_INVALID_ARGUMENT);
    cfg.beta = -0.1f;
    ASSERT_EQ_INT(snn_bptt_config_validate(&cfg), SNN_ERR_INVALID_ARGUMENT);
    cfg.beta = NAN;
    ASSERT_EQ_INT(snn_bptt_config_validate(&cfg), SNN_ERR_INVALID_ARGUMENT);
    /* beta == 0 is a valid, memoryless membrane. */
    cfg.beta = 0.0f;
    ASSERT_EQ_INT(snn_bptt_config_validate(&cfg), SNN_OK);

    cfg = snn_bptt_default_config(layers, 3, 7);
    cfg.threshold = 0.0f;
    ASSERT_EQ_INT(snn_bptt_config_validate(&cfg), SNN_ERR_INVALID_ARGUMENT);
    cfg.threshold = INFINITY;
    ASSERT_EQ_INT(snn_bptt_config_validate(&cfg), SNN_ERR_INVALID_ARGUMENT);

    cfg = snn_bptt_default_config(layers, 3, 7);
    cfg.surrogate = (snn_surrogate_t)-1;
    ASSERT_EQ_INT(snn_bptt_config_validate(&cfg), SNN_ERR_INVALID_ARGUMENT);
    cfg.surrogate = SNN_SURROGATE_COUNT;
    ASSERT_EQ_INT(snn_bptt_config_validate(&cfg), SNN_ERR_INVALID_ARGUMENT);

    cfg = snn_bptt_default_config(layers, 3, 7);
    cfg.surrogate_alpha = 0.0f;
    ASSERT_EQ_INT(snn_bptt_config_validate(&cfg), SNN_ERR_INVALID_ARGUMENT);
    cfg.surrogate_alpha = NAN;
    ASSERT_EQ_INT(snn_bptt_config_validate(&cfg), SNN_ERR_INVALID_ARGUMENT);

    cfg = snn_bptt_default_config(layers, 3, 7);
    cfg.weight_init_gain = 0.0f;
    ASSERT_EQ_INT(snn_bptt_config_validate(&cfg), SNN_ERR_INVALID_ARGUMENT);
    cfg.weight_init_gain = NAN;
    ASSERT_EQ_INT(snn_bptt_config_validate(&cfg), SNN_ERR_INVALID_ARGUMENT);

    cfg = snn_bptt_default_config(zero_layer, 3, 7);
    ASSERT_EQ_INT(snn_bptt_config_validate(&cfg), SNN_ERR_INVALID_ARGUMENT);

    /* beta_from_lif bridges the simulator's tau/dt to the trainable decay. */
    lif.dt_ms = 1.0f;
    lif.membrane_tau_ms = 20.0f;
    ASSERT_NEAR(snn_bptt_beta_from_lif(&lif), expf(-0.05f), 1e-6f);
    ASSERT_NEAR(snn_bptt_beta_from_lif(NULL), 0.0f, 0.0f);
    lif.dt_ms = -1.0f;
    ASSERT_NEAR(snn_bptt_beta_from_lif(&lif), 0.0f, 0.0f);
}

static void test_network_lifecycle(void) {
    snn_size_t layers[] = {4, 3, 2};
    snn_bptt_config_t cfg = snn_bptt_default_config(layers, 3, 5);
    snn_bptt_network_t *net = NULL;
    float *params = NULL;
    float *copy = NULL;
    snn_size_t n = 0;
    snn_size_t i = 0;

    ASSERT_EQ_INT(snn_bptt_network_create(&cfg, NULL), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_network_create(NULL, &net), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_TRUE(net == NULL);
    ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &net), SNN_OK);
    ASSERT_TRUE(net != NULL);

    /* 3*4 + 3 + 2*3 + 2 */
    n = snn_bptt_parameter_count(net);
    ASSERT_EQ_U64(n, 23);
    ASSERT_EQ_U64(snn_bptt_layer_count(net), 3);
    ASSERT_EQ_U64(snn_bptt_layer_size(net, 0), 4);
    ASSERT_EQ_U64(snn_bptt_layer_size(net, 2), 2);
    ASSERT_EQ_U64(snn_bptt_layer_size(net, 3), 0); /* out of range */
    ASSERT_EQ_U64(snn_bptt_input_size(net), 4);
    ASSERT_EQ_U64(snn_bptt_output_size(net), 2);
    ASSERT_EQ_U64(snn_bptt_timesteps(net), 5);
    ASSERT_EQ_INT(snn_bptt_network_surrogate(net), SNN_SURROGATE_ATAN);
    ASSERT_NEAR(snn_bptt_network_alpha(net), 2.0f, 0.0f);
    ASSERT_EQ_INT(snn_bptt_network_detach_reset(net), 0);

    ASSERT_EQ_U64(snn_bptt_layer_count(NULL), 0);
    ASSERT_EQ_U64(snn_bptt_layer_size(NULL, 0), 0);
    ASSERT_EQ_U64(snn_bptt_input_size(NULL), 0);
    ASSERT_EQ_U64(snn_bptt_output_size(NULL), 0);
    ASSERT_EQ_U64(snn_bptt_timesteps(NULL), 0);
    ASSERT_EQ_U64(snn_bptt_parameter_count(NULL), 0);
    ASSERT_EQ_INT(snn_bptt_network_surrogate(NULL), SNN_SURROGATE_COUNT);
    ASSERT_NEAR(snn_bptt_network_alpha(NULL), 0.0f, 0.0f);
    ASSERT_EQ_INT(snn_bptt_network_detach_reset(NULL), 0);

    ASSERT_EQ_INT(snn_bptt_network_set_surrogate(net, SNN_SURROGATE_TRIANGLE, 3.0f), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_network_surrogate(net), SNN_SURROGATE_TRIANGLE);
    ASSERT_NEAR(snn_bptt_network_alpha(net), 3.0f, 0.0f);
    ASSERT_EQ_INT(snn_bptt_network_set_surrogate(NULL, SNN_SURROGATE_ATAN, 1.0f), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_network_set_surrogate(net, SNN_SURROGATE_COUNT, 1.0f), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_network_set_surrogate(net, SNN_SURROGATE_ATAN, 0.0f), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_network_set_surrogate(net, SNN_SURROGATE_ATAN, NAN), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_network_set_surrogate(net, SNN_SURROGATE_ATAN, 2.0f), SNN_OK);

    params = (float *)xcalloc((size_t)n, sizeof(float));
    copy = (float *)xcalloc((size_t)n, sizeof(float));
    ASSERT_EQ_INT(snn_bptt_get_parameters(net, params, n), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_get_parameters(NULL, params, n), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_get_parameters(net, NULL, n), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_get_parameters(net, params, n - 1u), SNN_ERR_INVALID_ARGUMENT);

    /* Kaiming-uniform on fan_in 4 and 3; biases (the tail of each block) zero. */
    for (i = 0; i < 12u; ++i) {
        ASSERT_TRUE(fabsf(params[i]) <= sqrtf(3.0f / 4.0f));
    }
    for (i = 12u; i < 15u; ++i) {
        ASSERT_NEAR(params[i], 0.0f, 0.0f);
    }
    for (i = 21u; i < 23u; ++i) {
        ASSERT_NEAR(params[i], 0.0f, 0.0f);
    }
    ASSERT_TRUE(params[0] != params[1]);

    for (i = 0; i < n; ++i) {
        copy[i] = (float)i * 0.01f;
    }
    ASSERT_EQ_INT(snn_bptt_set_parameters(net, copy, n), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_get_parameters(net, params, n), SNN_OK);
    ASSERT_NEAR(params[7], 0.07f, 0.0f);
    ASSERT_EQ_INT(snn_bptt_set_parameters(NULL, copy, n), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_set_parameters(net, NULL, n), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_set_parameters(net, copy, n - 1u), SNN_ERR_INVALID_ARGUMENT);
    copy[3] = NAN;
    ASSERT_EQ_INT(snn_bptt_set_parameters(net, copy, n), SNN_ERR_INVALID_ARGUMENT);
    /* rejected before any write */
    ASSERT_EQ_INT(snn_bptt_get_parameters(net, params, n), SNN_OK);
    ASSERT_NEAR(params[3], 0.03f, 0.0f);

    free(params);
    free(copy);
    snn_bptt_network_free(net);
    snn_bptt_network_free(NULL);

    /* Different seeds give different initializations; equal seeds match. */
    {
        snn_bptt_network_t *a = NULL;
        snn_bptt_network_t *b = NULL;
        float pa[23];
        float pb[23];
        cfg = snn_bptt_default_config(layers, 3, 5);
        cfg.seed = 11;
        ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &a), SNN_OK);
        ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &b), SNN_OK);
        ASSERT_EQ_INT(snn_bptt_get_parameters(a, pa, 23), SNN_OK);
        ASSERT_EQ_INT(snn_bptt_get_parameters(b, pb, 23), SNN_OK);
        ASSERT_TRUE(memcmp(pa, pb, sizeof(pa)) == 0);
        snn_bptt_network_free(b);
        cfg.seed = 12;
        ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &b), SNN_OK);
        ASSERT_EQ_INT(snn_bptt_get_parameters(b, pb, 23), SNN_OK);
        ASSERT_TRUE(memcmp(pa, pb, sizeof(pa)) != 0);
        snn_bptt_network_free(a);
        snn_bptt_network_free(b);
    }

    /* Overflow: the weight-count product, then the running total. */
    {
        snn_size_t huge_mul[] = {UINT64_MAX, 2};
        snn_size_t huge_add[] = {1, UINT64_MAX};
        cfg = snn_bptt_default_config(huge_mul, 2, 1);
        ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &net), SNN_ERR_OVERFLOW);
        cfg = snn_bptt_default_config(huge_add, 2, 1);
        ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &net), SNN_ERR_OVERFLOW);
        ASSERT_TRUE(net == NULL);
    }
}

static void test_allocation_failures(void) {
#ifdef SNN_ENABLE_TEST_HOOKS
    snn_size_t layers[] = {4, 3, 2};
    snn_size_t tiny[] = {1, 1};
    snn_bptt_config_t cfg = snn_bptt_default_config(layers, 3, 5);
    snn_bptt_network_t *net = NULL;
    snn_bptt_workspace_t *ws = NULL;
    snn_bptt_grads_t *grads = NULL;
    snn_bptt_optimizer_t *opt = NULL;
    int k = 0;

    /* struct, sizes, param_offsets, params */
    for (k = 0; k < 4; ++k) {
        snn_test_set_alloc_fail_after(k);
        ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &net), SNN_ERR_OUT_OF_MEMORY);
        snn_test_disable_alloc_failure();
        ASSERT_TRUE(net == NULL);
    }
    ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &net), SNN_OK);

    /* struct, offsets, arena */
    for (k = 0; k < 3; ++k) {
        snn_test_set_alloc_fail_after(k);
        ASSERT_EQ_INT(snn_bptt_workspace_create(net, &ws), SNN_ERR_OUT_OF_MEMORY);
        snn_test_disable_alloc_failure();
        ASSERT_TRUE(ws == NULL);
    }
    /* struct, gradient vector */
    for (k = 0; k < 2; ++k) {
        snn_test_set_alloc_fail_after(k);
        ASSERT_EQ_INT(snn_bptt_grads_create(net, &grads), SNN_ERR_OUT_OF_MEMORY);
        snn_test_disable_alloc_failure();
        ASSERT_TRUE(grads == NULL);
    }
    /* struct, first moment, second moment */
    for (k = 0; k < 3; ++k) {
        snn_test_set_alloc_fail_after(k);
        ASSERT_EQ_INT(snn_bptt_optimizer_create(net, 1e-3f, 0.9f, 0.999f, 1e-8f, &opt), SNN_ERR_OUT_OF_MEMORY);
        snn_test_disable_alloc_failure();
        ASSERT_TRUE(opt == NULL);
    }
    snn_bptt_network_free(net);
    net = NULL;

    /* The workspace arena's size is a product of caller-controlled 64-bit
     * values, so it can overflow even for a network that allocated fine. */
    cfg = snn_bptt_default_config(tiny, 2, UINT64_MAX);
    ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &net), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_workspace_create(net, &ws), SNN_ERR_OVERFLOW);
    ASSERT_TRUE(ws == NULL);
    snn_bptt_network_free(net);
#endif
}

static void test_workspace_grads_optimizer(void) {
    snn_size_t layers[] = {4, 3, 2};
    snn_size_t other_layers[] = {4, 3, 2};
    snn_bptt_config_t cfg = snn_bptt_default_config(layers, 3, 5);
    snn_bptt_network_t *net = NULL;
    snn_bptt_network_t *other = NULL;
    snn_bptt_workspace_t *ws = NULL;
    snn_bptt_grads_t *a = NULL;
    snn_bptt_grads_t *b = NULL;
    snn_bptt_grads_t *foreign = NULL;
    snn_bptt_optimizer_t *opt = NULL;
    float g[23];
    snn_size_t i = 0;

    ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &net), SNN_OK);
    cfg = snn_bptt_default_config(other_layers, 3, 5);
    ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &other), SNN_OK);

    ASSERT_EQ_INT(snn_bptt_workspace_create(net, NULL), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_workspace_create(NULL, &ws), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_TRUE(ws == NULL);
    ASSERT_EQ_INT(snn_bptt_workspace_create(net, &ws), SNN_OK);
    snn_bptt_workspace_free(NULL);

    ASSERT_EQ_INT(snn_bptt_grads_create(net, NULL), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_grads_create(NULL, &a), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_TRUE(a == NULL);
    ASSERT_EQ_INT(snn_bptt_grads_create(net, &a), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_grads_create(net, &b), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_grads_create(other, &foreign), SNN_OK);
    snn_bptt_grads_free(NULL);

    ASSERT_EQ_INT(snn_bptt_grads_zero(NULL), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_grads_zero(a), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_grads_copy_out(a, g, 23), SNN_OK);
    for (i = 0; i < 23u; ++i) {
        ASSERT_NEAR(g[i], 0.0f, 0.0f);
    }
    ASSERT_EQ_INT(snn_bptt_grads_copy_out(NULL, g, 23), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_grads_copy_out(a, NULL, 23), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_grads_copy_out(a, g, 22), SNN_ERR_INVALID_ARGUMENT);

    /* Reduction rejects accumulators belonging to different networks. */
    ASSERT_EQ_INT(snn_bptt_grads_add(a, b), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_grads_add(NULL, b), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_grads_add(a, NULL), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_grads_add(a, foreign), SNN_ERR_INVALID_ARGUMENT);

    ASSERT_EQ_INT(snn_bptt_optimizer_create(net, 1e-3f, 0.9f, 0.999f, 1e-8f, NULL), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_optimizer_create(NULL, 1e-3f, 0.9f, 0.999f, 1e-8f, &opt), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_optimizer_create(net, 0.0f, 0.9f, 0.999f, 1e-8f, &opt), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_optimizer_create(net, NAN, 0.9f, 0.999f, 1e-8f, &opt), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_optimizer_create(net, 1e-3f, 1.0f, 0.999f, 1e-8f, &opt), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_optimizer_create(net, 1e-3f, -0.1f, 0.999f, 1e-8f, &opt), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_optimizer_create(net, 1e-3f, NAN, 0.999f, 1e-8f, &opt), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_optimizer_create(net, 1e-3f, 0.9f, 1.0f, 1e-8f, &opt), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_optimizer_create(net, 1e-3f, 0.9f, -0.1f, 1e-8f, &opt), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_optimizer_create(net, 1e-3f, 0.9f, NAN, 1e-8f, &opt), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_optimizer_create(net, 1e-3f, 0.9f, 0.999f, 0.0f, &opt), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_optimizer_create(net, 1e-3f, 0.9f, 0.999f, NAN, &opt), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_TRUE(opt == NULL);
    ASSERT_EQ_INT(snn_bptt_optimizer_create(net, 1e-3f, 0.9f, 0.999f, 1e-8f, &opt), SNN_OK);
    snn_bptt_optimizer_free(NULL);

    ASSERT_EQ_INT(snn_bptt_optimizer_set_lr(NULL, 1e-3f), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_optimizer_set_lr(opt, 0.0f), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_optimizer_set_lr(opt, NAN), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_optimizer_set_lr(opt, 5e-4f), SNN_OK);

    ASSERT_EQ_INT(snn_bptt_optimizer_step(NULL, net, a, 1), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_optimizer_step(opt, NULL, a, 1), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_optimizer_step(opt, net, NULL, 1), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_optimizer_step(opt, net, a, 0), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_optimizer_step(opt, other, a, 1), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_optimizer_step(opt, net, foreign, 1), SNN_ERR_INVALID_ARGUMENT);

    /* A zero gradient moves nothing: m_hat == 0 and the eps floor keeps the
     * 0/0 out of the update. */
    {
        float before[23];
        float after[23];
        ASSERT_EQ_INT(snn_bptt_get_parameters(net, before, 23), SNN_OK);
        ASSERT_EQ_INT(snn_bptt_grads_zero(a), SNN_OK);
        ASSERT_EQ_INT(snn_bptt_optimizer_step(opt, net, a, 1), SNN_OK);
        ASSERT_EQ_INT(snn_bptt_get_parameters(net, after, 23), SNN_OK);
        for (i = 0; i < 23u; ++i) {
            ASSERT_NEAR(after[i], before[i], 0.0f);
        }
    }

    /*
     * Adam's very first step is scale-free: m_hat == g and v_hat == g^2, so
     * every parameter with a nonzero gradient moves by exactly -lr*sign(g),
     * whatever the gradient's magnitude. That pins the bias correction, both
     * moment updates and the eps floor in one assertion -- and it is the
     * reason a surrogate that is uniformly 10x larger trains identically.
     */
    {
        float before[23];
        float after[23];
        float in[4] = {0.8f, -0.5f, 1.2f, 0.3f};
        snn_bptt_optimizer_t *fresh = NULL;
        int moved = 0;
        ASSERT_EQ_INT(snn_bptt_optimizer_create(net, 0.01f, 0.9f, 0.999f, 1e-8f, &fresh), SNN_OK);
        ASSERT_EQ_INT(snn_bptt_grads_zero(a), SNN_OK);
        ASSERT_EQ_INT(snn_bptt_forward_backward(net, ws, in, 1, 0, a, NULL, NULL), SNN_OK);
        ASSERT_EQ_INT(snn_bptt_grads_copy_out(a, g, 23), SNN_OK);
        ASSERT_EQ_INT(snn_bptt_get_parameters(net, before, 23), SNN_OK);
        ASSERT_EQ_INT(snn_bptt_optimizer_step(fresh, net, a, 1), SNN_OK);
        ASSERT_EQ_INT(snn_bptt_get_parameters(net, after, 23), SNN_OK);
        for (i = 0; i < 23u; ++i) {
            if (fabsf(g[i]) > 1e-4f) {
                ASSERT_NEAR(after[i] - before[i], g[i] > 0.0f ? -0.01f : 0.01f, 1e-5f);
                ++moved;
            }
        }
        ASSERT_TRUE(moved > 0);
        snn_bptt_optimizer_free(fresh);
    }

    snn_bptt_optimizer_free(opt);
    snn_bptt_grads_free(a);
    snn_bptt_grads_free(b);
    snn_bptt_grads_free(foreign);
    snn_bptt_workspace_free(ws);
    snn_bptt_network_free(net);
    snn_bptt_network_free(other);
}

/* ------------------------------------------------------------------ */
/* forward                                                            */
/* ------------------------------------------------------------------ */

static void test_forward_basics(void) {
    snn_size_t layers[] = {3, 5, 2};
    snn_size_t flat[] = {3, 2};
    snn_bptt_config_t cfg = snn_bptt_default_config(layers, 3, 4);
    snn_bptt_network_t *net = NULL;
    snn_bptt_network_t *other = NULL;
    snn_bptt_workspace_t *ws = NULL;
    snn_bptt_workspace_t *foreign_ws = NULL;
    snn_bptt_grads_t *grads = NULL;
    float frame[3] = {0.9f, -0.4f, 1.3f};
    float tape[12];
    float logits_static[2];
    float logits_tape[2];
    float loss = 0.0f;
    snn_size_t t = 0;
    snn_size_t i = 0;

    cfg.threshold = 0.5f;
    cfg.beta = 0.85f;
    ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &net), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &other), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_workspace_create(net, &ws), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_workspace_create(other, &foreign_ws), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_grads_create(net, &grads), SNN_OK);

    ASSERT_EQ_INT(snn_bptt_forward(NULL, ws, frame, 1), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_forward(net, NULL, frame, 1), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_forward(net, ws, NULL, 1), SNN_ERR_INVALID_ARGUMENT);
    /* A workspace belongs to exactly one network. */
    ASSERT_EQ_INT(snn_bptt_forward(net, foreign_ws, frame, 1), SNN_ERR_INVALID_ARGUMENT);

    /* A static frame and a tape of that frame repeated must agree exactly:
     * the static path collapses the layer-0 drive but computes the same map. */
    for (t = 0; t < 4u; ++t) {
        for (i = 0; i < 3u; ++i) {
            tape[t * 3u + i] = frame[i];
        }
    }
    ASSERT_EQ_INT(snn_bptt_forward(net, ws, frame, 1), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_copy_logits(ws, logits_static, 2), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_forward(net, ws, tape, 0), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_copy_logits(ws, logits_tape, 2), SNN_OK);
    ASSERT_NEAR(logits_static[0], logits_tape[0], 1e-6f);
    ASSERT_NEAR(logits_static[1], logits_tape[1], 1e-6f);

    ASSERT_EQ_INT(snn_bptt_copy_logits(NULL, logits_static, 2), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_copy_logits(ws, NULL, 2), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_copy_logits(ws, logits_static, 1), SNN_ERR_INVALID_ARGUMENT);

    /* prediction is the argmax of the logits. */
    ASSERT_TRUE(snn_bptt_prediction(ws) == (logits_tape[1] > logits_tape[0] ? 1u : 0u));
    ASSERT_EQ_U64(snn_bptt_prediction(NULL), 0);
    ASSERT_EQ_U64(snn_bptt_spike_count(NULL), 0);

    ASSERT_EQ_INT(snn_bptt_cross_entropy(ws, 0, &loss), SNN_OK);
    ASSERT_TRUE(loss > 0.0f && isfinite(loss));
    ASSERT_EQ_INT(snn_bptt_cross_entropy(NULL, 0, &loss), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_cross_entropy(ws, 0, NULL), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_cross_entropy(ws, 2, &loss), SNN_ERR_INVALID_ARGUMENT);

    /* Non-finite input is rejected, on both the static and the tape path. */
    frame[1] = NAN;
    ASSERT_EQ_INT(snn_bptt_forward(net, ws, frame, 1), SNN_ERR_INVALID_ARGUMENT);
    frame[1] = -0.4f;
    tape[7] = INFINITY;
    ASSERT_EQ_INT(snn_bptt_forward(net, ws, tape, 0), SNN_ERR_INVALID_ARGUMENT);
    tape[7] = frame[1];

    ASSERT_EQ_INT(snn_bptt_forward_backward(NULL, ws, frame, 1, 0, grads, NULL, NULL), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_forward_backward(net, ws, frame, 1, 0, NULL, NULL, NULL), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_forward_backward(net, ws, frame, 1, 2, grads, NULL, NULL), SNN_ERR_INVALID_ARGUMENT);
    /* the forward inside forward_backward can still fail */
    ASSERT_EQ_INT(snn_bptt_forward_backward(net, ws, NULL, 1, 0, grads, NULL, NULL), SNN_ERR_INVALID_ARGUMENT);
    {
        snn_bptt_grads_t *foreign = NULL;
        ASSERT_EQ_INT(snn_bptt_grads_create(other, &foreign), SNN_OK);
        ASSERT_EQ_INT(snn_bptt_forward_backward(net, ws, frame, 1, 0, foreign, NULL, NULL), SNN_ERR_INVALID_ARGUMENT);
        snn_bptt_grads_free(foreign);
    }
    {
        int correct = -1;
        ASSERT_EQ_INT(snn_bptt_grads_zero(grads), SNN_OK);
        ASSERT_EQ_INT(snn_bptt_forward_backward(net, ws, frame, 1, 1, grads, &loss, &correct), SNN_OK);
        ASSERT_TRUE(correct == 0 || correct == 1);
        ASSERT_TRUE(isfinite(loss) && loss > 0.0f);
    }

    snn_bptt_grads_free(grads);
    snn_bptt_workspace_free(ws);
    snn_bptt_workspace_free(foreign_ws);
    snn_bptt_network_free(net);
    snn_bptt_network_free(other);

    /* layer_count == 2: no hidden layer, hence no spikes at all. */
    {
        snn_bptt_network_t *linear = NULL;
        snn_bptt_workspace_t *lws = NULL;
        float in2[3] = {0.5f, 1.0f, -0.25f};
        cfg = snn_bptt_default_config(flat, 2, 3);
        ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &linear), SNN_OK);
        ASSERT_EQ_INT(snn_bptt_workspace_create(linear, &lws), SNN_OK);
        ASSERT_EQ_INT(snn_bptt_forward(linear, lws, in2, 1), SNN_OK);
        ASSERT_EQ_U64(snn_bptt_spike_count(lws), 0);
        snn_bptt_workspace_free(lws);
        snn_bptt_network_free(linear);
    }
}

/*
 * Pins the two scalar outputs nothing else constrains. The gradient checks all
 * differentiate the loss, so they are blind to a constant factor on it, and
 * out_correct feeds no gradient at all -- both could be silently wrong while
 * every other test in this file passes.
 */
static void test_loss_and_correct_are_pinned(void) {
    static const snn_size_t sizes[] = {3, 5, 4};
    const snn_size_t classes = 4;
    snn_bptt_config_t cfg = snn_bptt_default_config(sizes, 3, 4);
    snn_bptt_network_t *net = NULL;
    snn_bptt_workspace_t *ws = NULL;
    snn_bptt_grads_t *grads = NULL;
    float params[64];
    float input[3];
    float logits[4];
    snn_size_t n = 0;
    snn_size_t k = 0;
    snn_size_t prediction = 0;
    uint64_t rng = 606u;
    int saw_correct = 0;
    int saw_wrong = 0;

    cfg.beta = 0.85f;
    cfg.threshold = 0.6f;
    ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &net), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_workspace_create(net, &ws), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_grads_create(net, &grads), SNN_OK);
    n = snn_bptt_parameter_count(net);
    ASSERT_TRUE(n <= 64u);
    fill_signed(params, n, &rng, 0.9f);
    fill_signed(input, 3, &rng, 1.5f);
    ASSERT_EQ_INT(snn_bptt_set_parameters(net, params, n), SNN_OK);

    ASSERT_EQ_INT(snn_bptt_forward(net, ws, input, 1), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_copy_logits(ws, logits, classes), SNN_OK);
    prediction = snn_bptt_prediction(ws);

    for (k = 0; k < classes; ++k) {
        /* Independent softmax cross-entropy straight from the logits. */
        float max_logit = logits[0];
        float sum = 0.0f;
        float reference = 0.0f;
        float loss = 0.0f;
        float bp_loss = 0.0f;
        int correct = -1;
        snn_size_t c = 0;
        for (c = 1; c < classes; ++c) {
            if (logits[c] > max_logit) {
                max_logit = logits[c];
            }
        }
        for (c = 0; c < classes; ++c) {
            sum += expf(logits[c] - max_logit);
        }
        reference = (max_logit + logf(sum)) - logits[k];

        ASSERT_EQ_INT(snn_bptt_cross_entropy(ws, k, &loss), SNN_OK);
        ASSERT_NEAR(loss, reference, 1e-5f);

        ASSERT_EQ_INT(snn_bptt_grads_zero(grads), SNN_OK);
        ASSERT_EQ_INT(snn_bptt_forward_backward(net, ws, input, 1, k, grads, &bp_loss, &correct), SNN_OK);
        /* The backward's loss is the same quantity, not merely proportional. */
        ASSERT_NEAR(bp_loss, reference, 1e-5f);
        /* out_correct is exactly the argmax test, not its negation. */
        ASSERT_EQ_INT(correct, prediction == k);
        saw_correct |= correct;
        saw_wrong |= correct == 0;
    }
    /* Both branches of out_correct were exercised, so the assertion above is
     * not vacuously true for one polarity. */
    ASSERT_TRUE(saw_correct && saw_wrong);

    /* All-zero parameters give equal logits, so the loss is exactly ln(K). */
    memset(params, 0, sizeof(params));
    ASSERT_EQ_INT(snn_bptt_set_parameters(net, params, n), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_forward(net, ws, input, 1), SNN_OK);
    ASSERT_EQ_U64(snn_bptt_spike_count(ws), 0);
    for (k = 0; k < classes; ++k) {
        float loss = 0.0f;
        ASSERT_EQ_INT(snn_bptt_cross_entropy(ws, k, &loss), SNN_OK);
        ASSERT_NEAR(loss, logf((float)classes), 1e-6f);
    }

    snn_bptt_grads_free(grads);
    snn_bptt_workspace_free(ws);
    snn_bptt_network_free(net);
}

/* ------------------------------------------------------------------ */
/* gradient check 1: the backward is the transpose of the forward      */
/* linearization                                                      */
/* ------------------------------------------------------------------ */

static void check_transpose(const snn_size_t *sizes,
                            size_t layer_count,
                            snn_size_t timesteps,
                            float beta,
                            float threshold,
                            snn_surrogate_t surrogate,
                            float alpha,
                            int detach,
                            int static_input,
                            uint64_t seed) {
    snn_bptt_config_t cfg = snn_bptt_default_config(sizes, layer_count, timesteps);
    snn_bptt_network_t *net = NULL;
    snn_bptt_workspace_t *ws = NULL;
    snn_bptt_grads_t *grads = NULL;
    ref_net_t ref;
    ref_tape_t tape;
    snn_size_t n = 0;
    snn_size_t in_size = sizes[0];
    snn_size_t out_size = sizes[layer_count - 1u];
    snn_size_t input_len = static_input ? in_size : timesteps * in_size;
    snn_size_t label = seed % out_size;
    float *params = NULL;
    float *grad = NULL;
    float *dir = NULL;
    float *input = NULL;
    float lib_logits[REF_MAX_OUT];
    float gz[REF_MAX_OUT];
    float dz[REF_MAX_OUT];
    uint64_t rng = seed;
    int trial = 0;
    snn_size_t k = 0;
    snn_size_t hidden = 0;
    size_t j = 0;

    cfg.beta = beta;
    cfg.threshold = threshold;
    cfg.surrogate = surrogate;
    cfg.surrogate_alpha = alpha;
    cfg.detach_reset = detach;
    cfg.seed = seed;
    ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &net), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_workspace_create(net, &ws), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_grads_create(net, &grads), SNN_OK);
    n = snn_bptt_parameter_count(net);

    params = (float *)xcalloc((size_t)n, sizeof(float));
    grad = (float *)xcalloc((size_t)n, sizeof(float));
    dir = (float *)xcalloc((size_t)n, sizeof(float));
    input = (float *)xcalloc((size_t)input_len, sizeof(float));

    /* A weight scale that drives the hidden layers across threshold in both
     * directions -- an all-silent or saturated net has degenerate gradients
     * and would let a wrong backward pass. */
    fill_signed(params, n, &rng, 0.9f);
    fill_signed(input, input_len, &rng, 1.5f);
    ASSERT_EQ_INT(snn_bptt_set_parameters(net, params, n), SNN_OK);

    ASSERT_EQ_INT(snn_bptt_grads_zero(grads), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_forward_backward(net, ws, input, static_input, label, grads, NULL, NULL), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_grads_copy_out(grads, grad, n), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_copy_logits(ws, lib_logits, out_size), SNN_OK);

    for (j = 0; j + 1u < layer_count - 1u; ++j) {
        hidden += sizes[j + 1u];
    }
    if (hidden > 0u) {
        const snn_size_t fired = snn_bptt_spike_count(ws);
        ASSERT_TRUE(fired > 0u);
        ASSERT_TRUE(fired < hidden * timesteps);
    }

    ref_init(&ref, net, beta, threshold, 0);
    ref_tape_alloc(&ref, &tape);
    ref_forward(&ref, params, input, static_input, &tape);
    /* The reference forward reproduces the library's logits. */
    for (k = 0; k < out_size; ++k) {
        ASSERT_NEAR(tape.logits[k], lib_logits[k], 1e-4f);
    }
    ref_softmax_grad(tape.logits, out_size, label, gz);

    /* <dLoss/dz, J*d> == <dLoss/dparams, d> for arbitrary directions d. */
    for (trial = 0; trial < 6; ++trial) {
        float lhs = 0.0f;
        float rhs = 0.0f;
        fill_signed(dir, n, &rng, 1.0f);
        ref_jvp(&ref, params, dir, input, static_input, &tape, dz);
        lhs = dot(gz, dz, out_size);
        rhs = dot(grad, dir, n);
        if (fabsf(lhs - rhs) > 2e-4f * (fabsf(lhs) + fabsf(rhs)) + 1e-6f) {
            fprintf(stderr,
                    "transpose test failed: surrogate=%s alpha=%g detach=%d static=%d T=%llu trial=%d "
                    "lhs=%.9g rhs=%.9g\n",
                    snn_surrogate_string(surrogate), (double)alpha, detach, static_input,
                    (unsigned long long)timesteps, trial, (double)lhs, (double)rhs);
            exit(1);
        }
        /* A nondegenerate check: the directional derivative is not ~0. */
        ASSERT_TRUE(fabsf(rhs) > 1e-5f);
    }

    ref_tape_free(&ref, &tape);
    free(params);
    free(grad);
    free(dir);
    free(input);
    snn_bptt_grads_free(grads);
    snn_bptt_workspace_free(ws);
    snn_bptt_network_free(net);
}

static void test_gradient_transpose(void) {
    static const snn_size_t deep[] = {3, 4, 3, 2};   /* two hidden layers */
    static const snn_size_t one_hidden[] = {4, 5, 3};
    static const snn_size_t linear[] = {4, 3}; /* no hidden layer, no surrogate */
    int s = 0;
    int detach = 0;
    int static_input = 0;

    for (s = 0; s < 6; ++s) {
        const snn_surrogate_t surrogate = (snn_surrogate_t)s;
        for (detach = 0; detach < 2; ++detach) {
            for (static_input = 0; static_input < 2; ++static_input) {
                check_transpose(deep, 4, 6, 0.85f, 0.6f, surrogate, 2.0f, detach, static_input,
                                1000u + (uint64_t)(s * 4 + detach * 2 + static_input));
            }
        }
        check_transpose(one_hidden, 3, 5, 0.9f, 1.0f, surrogate, 3.0f, 0, 1, 2000u + (uint64_t)s);
    }
    /* T == 1 isolates the same-timestep cross-layer edge with every temporal
     * term switched off; T > 1 above covers the recurrence and the reset. */
    check_transpose(deep, 4, 1, 0.85f, 0.6f, SNN_SURROGATE_ATAN, 2.0f, 0, 1, 31u);
    check_transpose(one_hidden, 3, 1, 0.5f, 0.8f, SNN_SURROGATE_FAST_SIGMOID, 2.0f, 0, 0, 32u);
    /* beta == 0: a memoryless membrane still resets. */
    check_transpose(deep, 4, 5, 0.0f, 0.6f, SNN_SURROGATE_GAUSSIAN, 2.0f, 0, 1, 33u);
    /* No hidden layer at all: the surrogate is never evaluated. */
    check_transpose(linear, 2, 5, 0.9f, 1.0f, SNN_SURROGATE_ATAN, 2.0f, 0, 1, 34u);
    check_transpose(linear, 2, 5, 0.9f, 1.0f, SNN_SURROGATE_ATAN, 2.0f, 0, 0, 35u);
}

/* ------------------------------------------------------------------ */
/* gradient check 2: finite differences of a genuinely smooth model    */
/* ------------------------------------------------------------------ */

static float loss_at(snn_bptt_network_t *net,
                     snn_bptt_workspace_t *ws,
                     const float *params,
                     snn_size_t n,
                     const float *input,
                     int static_input,
                     snn_size_t label) {
    float loss = 0.0f;
    ASSERT_EQ_INT(snn_bptt_set_parameters(net, params, n), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_forward(net, ws, input, static_input), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_cross_entropy(ws, label, &loss), SNN_OK);
    return loss;
}

/* Central differences of the cross-entropy w.r.t. every parameter. */
static void finite_difference_grad(snn_bptt_network_t *net,
                                   snn_bptt_workspace_t *ws,
                                   float *params,
                                   snn_size_t n,
                                   const float *input,
                                   int static_input,
                                   snn_size_t label,
                                   float *out_grad) {
    /* Near the cube root of the float epsilon, where the truncation error of a
     * central difference and its cancellation error are comparable. */
    const float h = 4e-3f;
    snn_size_t i = 0;
    for (i = 0; i < n; ++i) {
        const float saved = params[i];
        float plus = 0.0f;
        float minus = 0.0f;
        params[i] = saved + h;
        plus = loss_at(net, ws, params, n, input, static_input, label);
        params[i] = saved - h;
        minus = loss_at(net, ws, params, n, input, static_input, label);
        params[i] = saved;
        out_grad[i] = (plus - minus) / (2.0f * h);
    }
    ASSERT_EQ_INT(snn_bptt_set_parameters(net, params, n), SNN_OK);
}

/*
 * With soft spikes the network computes a C^1 function of its parameters whose
 * exact gradient is what the surrogate backward produces, so this compares the
 * analytic gradient with central differences of a real scalar loss.
 */
static void check_finite_difference(const snn_size_t *sizes,
                                    size_t layer_count,
                                    snn_size_t timesteps,
                                    float beta,
                                    float threshold,
                                    snn_surrogate_t surrogate,
                                    float alpha,
                                    int soft,
                                    uint64_t seed) {
    snn_bptt_config_t cfg = snn_bptt_default_config(sizes, layer_count, timesteps);
    snn_bptt_network_t *net = NULL;
    snn_bptt_workspace_t *ws = NULL;
    snn_bptt_grads_t *grads = NULL;
    snn_size_t n = 0;
    snn_size_t in_size = sizes[0];
    snn_size_t label = 1u % sizes[layer_count - 1u];
    float *params = NULL;
    float *analytic = NULL;
    float *numeric = NULL;
    float *input = NULL;
    uint64_t rng = seed;
    snn_size_t i = 0;

    cfg.beta = beta;
    cfg.threshold = threshold;
    cfg.surrogate = surrogate;
    cfg.surrogate_alpha = alpha;
    cfg.seed = seed;
    ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &net), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_workspace_create(net, &ws), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_grads_create(net, &grads), SNN_OK);
    n = snn_bptt_parameter_count(net);

    params = (float *)xcalloc((size_t)n, sizeof(float));
    analytic = (float *)xcalloc((size_t)n, sizeof(float));
    numeric = (float *)xcalloc((size_t)n, sizeof(float));
    input = (float *)xcalloc((size_t)in_size, sizeof(float));
    fill_signed(params, n, &rng, 0.9f);
    fill_signed(input, in_size, &rng, 1.5f);

#ifdef SNN_ENABLE_TEST_HOOKS
    snn_test_bptt_set_soft_spikes(net, soft);
#else
    (void)soft;
#endif
    ASSERT_EQ_INT(snn_bptt_set_parameters(net, params, n), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_grads_zero(grads), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_forward_backward(net, ws, input, 1, label, grads, NULL, NULL), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_grads_copy_out(grads, analytic, n), SNN_OK);

    finite_difference_grad(net, ws, params, n, input, 1, label, numeric);

    for (i = 0; i < n; ++i) {
        const float tol = 3e-4f + 3e-3f * fabsf(analytic[i]);
        if (fabsf(analytic[i] - numeric[i]) > tol) {
            fprintf(stderr,
                    "finite-difference test failed: surrogate=%s alpha=%g soft=%d param=%llu "
                    "analytic=%.9g numeric=%.9g\n",
                    snn_surrogate_string(surrogate), (double)alpha, soft, (unsigned long long)i, (double)analytic[i],
                    (double)numeric[i]);
            exit(1);
        }
    }
    /* Nondegenerate: at least one parameter has a gradient worth checking. */
    {
        float biggest = 0.0f;
        for (i = 0; i < n; ++i) {
            if (fabsf(analytic[i]) > biggest) {
                biggest = fabsf(analytic[i]);
            }
        }
        ASSERT_TRUE(biggest > 1e-3f);
    }

    free(params);
    free(analytic);
    free(numeric);
    free(input);
    snn_bptt_grads_free(grads);
    snn_bptt_workspace_free(ws);
    snn_bptt_network_free(net);
}

static void test_gradient_finite_difference(void) {
    static const snn_size_t deep[] = {3, 4, 3, 2};
    static const snn_size_t linear[] = {4, 3};
    int s = 0;

    /* layer_count == 2 is exactly differentiable with no hook: a leaky
     * integrator readout and a softmax. */
    check_finite_difference(linear, 2, 5, 0.85f, 0.6f, SNN_SURROGATE_ATAN, 2.0f, 0, 77u);

#ifdef SNN_ENABLE_TEST_HOOKS
    /* Every surrogate, on a two-hidden-layer net unrolled over 6 steps. */
    for (s = 0; s < 6; ++s) {
        check_finite_difference(deep, 4, 6, 0.85f, 0.6f, (snn_surrogate_t)s, 2.0f, 1, 400u + (uint64_t)s);
    }
    check_finite_difference(deep, 4, 6, 0.5f, 1.0f, SNN_SURROGATE_FAST_SIGMOID, 1.0f, 1, 410u);
#else
    (void)s;
    (void)deep;
#endif
}

/*
 * The reset term -threshold*s[j][t-1] carries real gradient: detaching it
 * changes the answer. Proving both halves of that -- that the non-detached
 * gradient matches the true gradient of the smooth model, and that the
 * detached one measurably does not -- pins down the one path most likely to
 * be silently missing or sign-flipped.
 */
static void test_detach_reset_is_the_reset_path(void) {
#ifdef SNN_ENABLE_TEST_HOOKS
    static const snn_size_t sizes[] = {3, 4, 3, 2};
    snn_bptt_config_t cfg = snn_bptt_default_config(sizes, 4, 6);
    snn_bptt_network_t *full = NULL;
    snn_bptt_network_t *detached = NULL;
    snn_bptt_workspace_t *ws_full = NULL;
    snn_bptt_workspace_t *ws_det = NULL;
    snn_bptt_grads_t *g_full = NULL;
    snn_bptt_grads_t *g_det = NULL;
    snn_size_t n = 0;
    snn_size_t i = 0;
    uint64_t rng = 909u;
    float params[64];
    float input[3];
    float grad_full[64];
    float grad_det[64];
    float numeric[64];
    float worst_full = 0.0f;
    float worst_det = 0.0f;

    cfg.beta = 0.85f;
    cfg.threshold = 0.6f;
    cfg.surrogate = SNN_SURROGATE_ATAN;
    cfg.surrogate_alpha = 2.0f;
    cfg.detach_reset = 0;
    ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &full), SNN_OK);
    cfg.detach_reset = 1;
    ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &detached), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_network_detach_reset(detached), 1);

    n = snn_bptt_parameter_count(full);
    ASSERT_TRUE(n <= 64u);
    ASSERT_EQ_INT(snn_bptt_workspace_create(full, &ws_full), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_workspace_create(detached, &ws_det), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_grads_create(full, &g_full), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_grads_create(detached, &g_det), SNN_OK);

    fill_signed(params, n, &rng, 0.9f);
    fill_signed(input, 3, &rng, 1.5f);

    /* Identical smooth forward on both; only the backward differs. */
    snn_test_bptt_set_soft_spikes(full, 1);
    snn_test_bptt_set_soft_spikes(detached, 1);
    snn_test_bptt_set_soft_spikes(NULL, 1); /* tolerated */
    ASSERT_EQ_INT(snn_bptt_set_parameters(full, params, n), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_set_parameters(detached, params, n), SNN_OK);

    ASSERT_EQ_INT(snn_bptt_grads_zero(g_full), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_forward_backward(full, ws_full, input, 1, 1, g_full, NULL, NULL), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_grads_copy_out(g_full, grad_full, n), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_grads_zero(g_det), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_forward_backward(detached, ws_det, input, 1, 1, g_det, NULL, NULL), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_grads_copy_out(g_det, grad_det, n), SNN_OK);

    finite_difference_grad(full, ws_full, params, n, input, 1, 1, numeric);

    for (i = 0; i < n; ++i) {
        const float ef = fabsf(grad_full[i] - numeric[i]);
        const float ed = fabsf(grad_det[i] - numeric[i]);
        if (ef > worst_full) {
            worst_full = ef;
        }
        if (ed > worst_det) {
            worst_det = ed;
        }
    }
    /* The full gradient is the true one... */
    ASSERT_TRUE(worst_full < 1e-3f);
    /* ...and dropping the reset path is a visibly different answer, so the
     * path is present, nonzero, and not cancelling itself out. */
    ASSERT_TRUE(worst_det > 1e-3f);
    ASSERT_TRUE(worst_det > 20.0f * worst_full);

    snn_bptt_grads_free(g_full);
    snn_bptt_grads_free(g_det);
    snn_bptt_workspace_free(ws_full);
    snn_bptt_workspace_free(ws_det);
    snn_bptt_network_free(full);
    snn_bptt_network_free(detached);
#endif
}

/*
 * snn_bptt_spike_count reports the hard threshold crossings of the trajectory
 * that actually ran. Under the soft-spike hook the trajectory itself differs
 * -- the reset feeds back S(x) rather than 1 -- so the census legitimately
 * changes; what must hold is that it still counts crossings of the membrane it
 * was given. The reference forward is the oracle for both.
 */
static void test_spike_count_matches_reference(void) {
    static const snn_size_t sizes[] = {3, 6, 2};
    snn_bptt_config_t cfg = snn_bptt_default_config(sizes, 3, 5);
    snn_bptt_network_t *net = NULL;
    snn_bptt_workspace_t *ws = NULL;
    ref_net_t ref;
    ref_tape_t tape;
    float input[3] = {1.1f, -0.7f, 0.9f};
    float params[44];
    snn_size_t n = 0;
    uint64_t rng = 5150u;
    float logits_hard[2];
    float logits_soft[2];
    snn_size_t hard = 0;

    cfg.beta = 0.85f;
    cfg.threshold = 0.6f;
    cfg.seed = 5150u;
    ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &net), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_workspace_create(net, &ws), SNN_OK);
    n = snn_bptt_parameter_count(net);
    ASSERT_TRUE(n <= 44u);
    fill_signed(params, n, &rng, 0.9f);
    ASSERT_EQ_INT(snn_bptt_set_parameters(net, params, n), SNN_OK);

    ASSERT_EQ_INT(snn_bptt_forward(net, ws, input, 1), SNN_OK);
    hard = snn_bptt_spike_count(ws);
    ASSERT_EQ_INT(snn_bptt_copy_logits(ws, logits_hard, 2), SNN_OK);
    ref_init(&ref, net, 0.85f, 0.6f, 0);
    ref_tape_alloc(&ref, &tape);
    ref_forward(&ref, params, input, 1, &tape);
    ASSERT_EQ_U64(hard, ref_count_crossings(&ref, &tape));
    /* A mix of firing and silent neuron-steps, not a degenerate extreme. */
    ASSERT_TRUE(hard > 0u && hard < 6u * 5u);
    ref_tape_free(&ref, &tape);

#ifdef SNN_ENABLE_TEST_HOOKS
    snn_test_bptt_set_soft_spikes(net, 1);
    ASSERT_EQ_INT(snn_bptt_forward(net, ws, input, 1), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_copy_logits(ws, logits_soft, 2), SNN_OK);
    ref_init(&ref, net, 0.85f, 0.6f, 1);
    ref_tape_alloc(&ref, &tape);
    ref_forward(&ref, params, input, 1, &tape);
    ASSERT_EQ_U64(snn_bptt_spike_count(ws), ref_count_crossings(&ref, &tape));
    ref_tape_free(&ref, &tape);
    /* S != H, so the soft model is a genuinely different forward map. */
    ASSERT_TRUE(fabsf(logits_hard[0] - logits_soft[0]) > 1e-5f);
    snn_test_bptt_set_soft_spikes(net, 0);
#else
    (void)logits_soft;
#endif

    snn_bptt_workspace_free(ws);
    snn_bptt_network_free(net);
}

/* ------------------------------------------------------------------ */
/* optimization                                                       */
/* ------------------------------------------------------------------ */

/* Summing per-sample gradients into one accumulator must equal reducing
 * per-thread accumulators with snn_bptt_grads_add -- the contract the MNIST
 * driver's OpenMP batch parallelism relies on. */
static void test_grads_add_matches_direct_accumulation(void) {
    static const snn_size_t sizes[] = {4, 5, 3};
    snn_bptt_config_t cfg = snn_bptt_default_config(sizes, 3, 4);
    snn_bptt_network_t *net = NULL;
    snn_bptt_workspace_t *ws = NULL;
    snn_bptt_grads_t *direct = NULL;
    snn_bptt_grads_t *part[3];
    snn_bptt_grads_t *reduced = NULL;
    float input[3][4];
    float a[64];
    float b[64];
    snn_size_t n = 0;
    snn_size_t i = 0;
    uint64_t rng = 4242u;
    int k = 0;

    cfg.beta = 0.9f;
    cfg.threshold = 0.7f;
    ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &net), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_workspace_create(net, &ws), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_grads_create(net, &direct), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_grads_create(net, &reduced), SNN_OK);
    for (k = 0; k < 3; ++k) {
        ASSERT_EQ_INT(snn_bptt_grads_create(net, &part[k]), SNN_OK);
        fill_signed(input[k], 4, &rng, 1.4f);
    }
    n = snn_bptt_parameter_count(net);
    ASSERT_TRUE(n <= 64u);

    ASSERT_EQ_INT(snn_bptt_grads_zero(direct), SNN_OK);
    for (k = 0; k < 3; ++k) {
        ASSERT_EQ_INT(snn_bptt_forward_backward(net, ws, input[k], 1, (snn_size_t)k, direct, NULL, NULL), SNN_OK);
    }
    ASSERT_EQ_INT(snn_bptt_grads_copy_out(direct, a, n), SNN_OK);

    for (k = 0; k < 3; ++k) {
        ASSERT_EQ_INT(snn_bptt_grads_zero(part[k]), SNN_OK);
        ASSERT_EQ_INT(snn_bptt_forward_backward(net, ws, input[k], 1, (snn_size_t)k, part[k], NULL, NULL), SNN_OK);
    }
    ASSERT_EQ_INT(snn_bptt_grads_zero(reduced), SNN_OK);
    for (k = 0; k < 3; ++k) {
        ASSERT_EQ_INT(snn_bptt_grads_add(reduced, part[k]), SNN_OK);
    }
    ASSERT_EQ_INT(snn_bptt_grads_copy_out(reduced, b, n), SNN_OK);

    for (i = 0; i < n; ++i) {
        ASSERT_NEAR(a[i], b[i], 1e-5f);
    }

    for (k = 0; k < 3; ++k) {
        snn_bptt_grads_free(part[k]);
    }
    snn_bptt_grads_free(direct);
    snn_bptt_grads_free(reduced);
    snn_bptt_workspace_free(ws);
    snn_bptt_network_free(net);
}

/*
 * A batch whose forward overflows produces a non-finite gradient. Adam's
 * moments are exponential moving averages, so admitting one NaN would poison
 * them forever; the step must refuse it and leave both objects untouched.
 */
static void test_optimizer_rejects_non_finite_gradient(void) {
    static const snn_size_t sizes[] = {2, 2, 2};
    snn_bptt_config_t cfg = snn_bptt_default_config(sizes, 3, 2);
    snn_bptt_network_t *net = NULL;
    snn_bptt_workspace_t *ws = NULL;
    snn_bptt_grads_t *grads = NULL;
    snn_bptt_optimizer_t *opt = NULL;
    float params[16];
    float before[16];
    float after[16];
    float grad[16];
    float input[2] = {1.0f, 1.0f};
    float loss = 0.0f;
    snn_size_t n = 0;
    snn_size_t i = 0;
    int any_non_finite = 0;

    ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &net), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_workspace_create(net, &ws), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_grads_create(net, &grads), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_optimizer_create(net, 1e-3f, 0.9f, 0.999f, 1e-8f, &opt), SNN_OK);
    n = snn_bptt_parameter_count(net);
    ASSERT_EQ_U64(n, 12); /* (2*2 + 2) + (2*2 + 2) */

    /* Finite parameters whose products overflow float: the membrane, and then
     * the logits, become inf, and softmax(inf, inf) is NaN. */
    for (i = 0; i < n; ++i) {
        params[i] = 3.0e38f;
    }
    ASSERT_EQ_INT(snn_bptt_set_parameters(net, params, n), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_get_parameters(net, before, n), SNN_OK);

    ASSERT_EQ_INT(snn_bptt_grads_zero(grads), SNN_OK);
    /* The forward and backward themselves still succeed: it is the caller's
     * loss that is NaN, exactly as in any framework. */
    ASSERT_EQ_INT(snn_bptt_forward_backward(net, ws, input, 1, 0, grads, &loss, NULL), SNN_OK);
    ASSERT_TRUE(!isfinite(loss));
    ASSERT_EQ_INT(snn_bptt_grads_copy_out(grads, grad, n), SNN_OK);
    for (i = 0; i < n; ++i) {
        any_non_finite |= !isfinite(grad[i]);
    }
    ASSERT_TRUE(any_non_finite);

    ASSERT_EQ_INT(snn_bptt_optimizer_step(opt, net, grads, 1), SNN_ERR_INVALID_ARGUMENT);
    ASSERT_EQ_INT(snn_bptt_get_parameters(net, after, n), SNN_OK);
    ASSERT_TRUE(memcmp(before, after, (size_t)n * sizeof(float)) == 0);

    /* And the optimizer is still usable afterwards: the rejected step did not
     * advance its bias-correction counter or touch its moments. */
    ASSERT_EQ_INT(snn_bptt_grads_zero(grads), SNN_OK);
    for (i = 0; i < n; ++i) {
        params[i] = 0.1f;
    }
    ASSERT_EQ_INT(snn_bptt_set_parameters(net, params, n), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_forward_backward(net, ws, input, 1, 0, grads, &loss, NULL), SNN_OK);
    ASSERT_TRUE(isfinite(loss));
    ASSERT_EQ_INT(snn_bptt_optimizer_step(opt, net, grads, 1), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_get_parameters(net, after, n), SNN_OK);
    ASSERT_TRUE(memcmp(params, after, (size_t)n * sizeof(float)) != 0);

    snn_bptt_optimizer_free(opt);
    snn_bptt_grads_free(grads);
    snn_bptt_workspace_free(ws);
    snn_bptt_network_free(net);
}

/* End to end: BPTT + Adam must actually drive a spiking network to memorize a
 * small classification problem. A gradient with the wrong sign anywhere -- or
 * a surrogate that never fires -- cannot do this. */
static void test_training_converges(void) {
    static const snn_size_t sizes[] = {6, 16, 3};
    const int samples = 9;
    snn_bptt_config_t cfg = snn_bptt_default_config(sizes, 3, 5);
    snn_bptt_network_t *net = NULL;
    snn_bptt_workspace_t *ws = NULL;
    snn_bptt_grads_t *grads = NULL;
    snn_bptt_optimizer_t *opt = NULL;
    float input[9][6];
    snn_size_t label[9];
    uint64_t rng = 20260709u;
    float first_loss = 0.0f;
    float last_loss = 0.0f;
    int epoch = 0;
    int k = 0;

    cfg.beta = 0.9f;
    cfg.threshold = 0.8f;
    cfg.surrogate = SNN_SURROGATE_FAST_SIGMOID;
    cfg.surrogate_alpha = 2.0f;
    cfg.seed = 987u;
    ASSERT_EQ_INT(snn_bptt_network_create(&cfg, &net), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_workspace_create(net, &ws), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_grads_create(net, &grads), SNN_OK);
    ASSERT_EQ_INT(snn_bptt_optimizer_create(net, 0.02f, 0.9f, 0.999f, 1e-8f, &opt), SNN_OK);

    for (k = 0; k < samples; ++k) {
        fill_signed(input[k], 6, &rng, 1.2f);
        label[k] = (snn_size_t)(k % 3);
    }

    for (epoch = 0; epoch < 250; ++epoch) {
        float total = 0.0f;
        ASSERT_EQ_INT(snn_bptt_grads_zero(grads), SNN_OK);
        for (k = 0; k < samples; ++k) {
            float loss = 0.0f;
            ASSERT_EQ_INT(snn_bptt_forward_backward(net, ws, input[k], 1, label[k], grads, &loss, NULL), SNN_OK);
            total += loss;
        }
        ASSERT_EQ_INT(snn_bptt_optimizer_step(opt, net, grads, (snn_size_t)samples), SNN_OK);
        if (epoch == 0) {
            first_loss = total / (float)samples;
        }
        last_loss = total / (float)samples;
    }

    ASSERT_TRUE(isfinite(last_loss));
    ASSERT_TRUE(last_loss < 0.25f * first_loss);

    {
        int correct = 0;
        for (k = 0; k < samples; ++k) {
            ASSERT_EQ_INT(snn_bptt_forward(net, ws, input[k], 1), SNN_OK);
            correct += snn_bptt_prediction(ws) == label[k];
        }
        ASSERT_EQ_INT(correct, samples);
    }

    snn_bptt_optimizer_free(opt);
    snn_bptt_grads_free(grads);
    snn_bptt_workspace_free(ws);
    snn_bptt_network_free(net);
}

void run_bptt_tests(void) {
    test_surrogate_functions();
    test_config_and_defaults();
    test_network_lifecycle();
    test_allocation_failures();
    test_workspace_grads_optimizer();
    test_forward_basics();
    test_loss_and_correct_are_pinned();
    test_gradient_transpose();
    test_gradient_finite_difference();
    test_detach_reset_is_the_reset_path();
    test_spike_count_matches_reference();
    test_grads_add_matches_direct_accumulation();
    test_optimizer_rejects_non_finite_gradient();
    test_training_converges();
}
