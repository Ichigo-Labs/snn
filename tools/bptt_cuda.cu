/*
 * CUDA/cuBLAS training backend for tools/mnist_bptt.c. See bptt_cuda.h for
 * the contract and mnist_bptt.c --mode gputest for the verification battery.
 *
 * The math is src/snn_bptt.c batched over a minibatch:
 *
 *   tapes    U[j], S[j], GS[j] are [T, B, N_j] row-major blocks; GU is one
 *            reused [T+1, B, maxN] tape whose row T is the zero seed, exactly
 *            like the CPU workspace's gu tape.
 *   forward  per (t, j): one GEMM I = S[j-1][t] * W[j]^T written straight
 *            into U[j]'s block, then a pointwise LIF kernel adds bias, decay
 *            and reset and emits spikes. Layer 0's drive is one GEMM per
 *            minibatch (constant-current input), the CPU path's same trick.
 *   backward per (t, j): a pointwise kernel forms dL/dU, then one GEMM
 *            scatters GS[j-1][t] = GU[t] * W[j]. The per-timestep rank-1
 *            weight updates of the CPU path collapse into ONE GEMM per layer
 *            over the flattened (T*B) dimension: gw = GU_flat^T * Pre_flat.
 *            That reordering (and cuBLAS's own accumulation order) is why
 *            this backend matches the CPU trainer to float tolerance, not
 *            bitwise.
 *   adam     one fused kernel; a sticky device flag rejects the epoch if any
 *            minibatch gradient is non-finite, mirroring the CPU
 *            optimizer_step's refusal to poison the moments.
 */

#include "bptt_cuda.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NEURON_LAYERS 9u /* MAX_HIDDEN_LAYERS + output */
#define THREADS 256

struct bptt_gpu {
    cublasHandle_t blas;
    cudaStream_t stream;

    size_t layers; /* neuron layers L = layer_count - 1 */
    snn_size_t sizes[MAX_NEURON_LAYERS + 1u];
    snn_size_t w_off[MAX_NEURON_LAYERS]; /* into the flat param vector */
    snn_size_t b_off[MAX_NEURON_LAYERS];
    snn_size_t param_count;
    snn_size_t timesteps;
    snn_size_t max_size; /* max over j of sizes[j+1] */
    snn_size_t in_size;
    snn_size_t out_size;
    float beta;
    float threshold;
    float alpha;
    int surrogate;
    int detach_reset;
    int soft_spikes;

    snn_size_t batch;      /* training minibatch rows */
    snn_size_t eval_batch; /* evaluation rows; tapes sized for the larger */
    snn_size_t train_count;
    snn_size_t test_count;

    float *d_params;
    float *d_grads;
    float *d_m;
    float *d_v;
    uint64_t adam_steps;
    float lr, adam_b1, adam_b2, adam_eps;

    float *d_train_x; /* [train_count, in_size], pixels / 255 */
    float *d_test_x;
    uint8_t *d_train_y;
    uint8_t *d_test_y;
    snn_size_t *d_order;

    float *d_u[MAX_NEURON_LAYERS];  /* [T, maxB, N_{j+1}] */
    float *d_s[MAX_NEURON_LAYERS];  /* hidden layers only */
    float *d_gs[MAX_NEURON_LAYERS]; /* hidden layers only, train batch */
    float *d_gu;                    /* [T+1, batch, max_size] */
    float *d_xb;                    /* [maxB, in_size] gathered minibatch */
    float *d_drive;                 /* [maxB, sizes[1]] */
    float *d_gsum;                  /* [batch, sizes[1]] */
    float *d_logits;                /* [maxB, out_size] */
    float *d_gz;                    /* [batch, out_size] */

    double *d_loss;                 /* epoch / eval accumulator */
    int *d_correct;
    unsigned long long *d_spikes;
    int *d_reject; /* sticky non-finite-gradient flag */

    char err[256];
};

static void set_err(char *dst, size_t len, const char *what, const char *detail) {
    if (dst != NULL && len > 0u) {
        snprintf(dst, len, "%s: %s", what, detail);
    }
}

#define CUDA_TRY(g, call)                                                          \
    do {                                                                           \
        const cudaError_t e_ = (call);                                             \
        if (e_ != cudaSuccess) {                                                   \
            set_err((g)->err, sizeof((g)->err), #call, cudaGetErrorString(e_));    \
            return -1;                                                             \
        }                                                                          \
    } while (0)

#define BLAS_TRY(g, call)                                                          \
    do {                                                                           \
        if ((call) != CUBLAS_STATUS_SUCCESS) {                                     \
            set_err((g)->err, sizeof((g)->err), #call, "cublas failure");          \
            return -1;                                                             \
        }                                                                          \
    } while (0)

/* ------------------------------------------------------------------ */
/* device math                                                         */
/* ------------------------------------------------------------------ */

/* Byte-for-byte the formulas of src/snn_bptt.c's surrogate_grad_at. */
static __device__ float d_surrogate_grad(int surrogate, float x, float alpha) {
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
        const float s = 1.0f / (1.0f + expf(-alpha * x));
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

/* snn_surrogate_primitive, for the soft-spike verification mode. */
static __device__ float d_surrogate_primitive(int surrogate, float x, float alpha) {
    switch (surrogate) {
    case SNN_SURROGATE_FAST_SIGMOID:
        return (float)(0.5 + (double)x / (1.0 + (double)alpha * fabs((double)x)));
    case SNN_SURROGATE_ATAN:
        return 0.5f + atanf(alpha * x) / alpha;
    case SNN_SURROGATE_SIGMOID:
        return 0.5f + (4.0f / alpha) * (1.0f / (1.0f + expf(-alpha * x)) - 0.5f);
    case SNN_SURROGATE_TRIANGLE: {
        const float ax = fabsf(x);
        const float half_area = 0.5f / alpha;
        return ax * alpha >= 1.0f ? (x >= 0.0f ? 0.5f + half_area : 0.5f - half_area)
                                  : 0.5f + x - 0.5f * alpha * x * ax;
    }
    case SNN_SURROGATE_GAUSSIAN:
        return 0.5f + (1.2533141373155003f / alpha) * erff(alpha * x * 0.70710678118654752f);
    case SNN_SURROGATE_RECTANGULAR: {
        const float lim = 1.0f / alpha;
        return 0.5f + (x < -lim ? -lim : (x > lim ? lim : x));
    }
    default:
        return 0.0f;
    }
}

/* ------------------------------------------------------------------ */
/* kernels                                                             */
/* ------------------------------------------------------------------ */

__global__ void k_u8_to_f32(const uint8_t *src, float *dst, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        dst[i] = (float)src[i] * (1.0f / 255.0f);
    }
}

/* xb[b, :] = x[order == NULL ? start + b : order[start + b], :] */
__global__ void k_gather_rows(const float *x,
                              const snn_size_t *order,
                              size_t start,
                              int b_count,
                              int width,
                              float *xb) {
    const size_t total = (size_t)b_count * width;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t b = i / width;
        const size_t src = order == NULL ? start + b : (size_t)order[start + b];
        xb[i] = x[src * width + i % width];
    }
}

/*
 * One LIF timestep for one layer, batched. `in` is the layer's synaptic
 * drive: the drive buffer for layer 0 (constant across t), or u_t itself,
 * where the preceding GEMM already left W*pre (in-place). Term order matches
 * the CPU forward: (I + b) + beta*u_prev - threshold*s_prev.
 */
__global__ void k_lif_step(float *u_t,
                           float *s_t, /* NULL for the output layer */
                           const float *in,
                           const float *u_prev, /* NULL at t == 0 */
                           const float *s_prev, /* NULL at t == 0 or for the output layer */
                           const float *bias,
                           int b_count,
                           int n,
                           float beta,
                           float threshold,
                           int soft,
                           int surrogate,
                           float alpha) {
    const size_t total = (size_t)b_count * n;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        float u = in[i] + bias[i % n];
        if (u_prev != NULL) {
            u += beta * u_prev[i];
        }
        if (s_prev != NULL) {
            u -= threshold * s_prev[i];
        }
        u_t[i] = u;
        if (s_t != NULL) {
            s_t[i] = soft ? d_surrogate_primitive(surrogate, u - threshold, alpha)
                          : (u - threshold >= 0.0f ? 1.0f : 0.0f);
        }
    }
}

/* logits[b, k] = mean over t of the output layer's membrane tape. */
__global__ void k_logits(const float *u_out, int timesteps, int b_count, int out_size, float *logits) {
    const size_t total = (size_t)b_count * out_size;
    const size_t step = (size_t)b_count * out_size;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < timesteps; ++t) {
            acc += u_out[(size_t)t * step + i];
        }
        logits[i] = acc / (float)timesteps;
    }
}

/*
 * Per-sample softmax cross-entropy. Writes gz = softmax - onehot when gz is
 * non-NULL (training) and counts first-strict-maximum predictions when
 * correct is non-NULL (evaluation), replicating the CPU argmax tie rule.
 */
__global__ void k_loss(const float *logits,
                       const uint8_t *labels,
                       const snn_size_t *order, /* NULL: samples start..start+b_count */
                       size_t start,
                       int b_count,
                       int out_size,
                       float *gz,
                       double *loss_accum,
                       int *correct) {
    for (int b = blockIdx.x * blockDim.x + threadIdx.x; b < b_count; b += gridDim.x * blockDim.x) {
        const float *l = logits + (size_t)b * out_size;
        const size_t sample = order == NULL ? start + b : (size_t)order[start + b];
        const int label = labels[sample];
        float max_logit = l[0];
        int best = 0;
        float sum = 0.0f;
        for (int k = 1; k < out_size; ++k) {
            if (l[k] > max_logit) {
                max_logit = l[k];
            }
            if (l[k] > l[best]) {
                best = k;
            }
        }
        for (int k = 0; k < out_size; ++k) {
            const float e = expf(l[k] - max_logit);
            sum += e;
            if (gz != NULL) {
                gz[(size_t)b * out_size + k] = e;
            }
        }
        if (gz != NULL) {
            for (int k = 0; k < out_size; ++k) {
                gz[(size_t)b * out_size + k] /= sum;
            }
            gz[(size_t)b * out_size + label] -= 1.0f;
        }
        atomicAdd(loss_accum, (double)((max_logit + logf(sum)) - l[label]));
        if (correct != NULL && best == label) {
            atomicAdd(correct, 1);
        }
    }
}

/*
 * dL/dU for one timestep of one layer. The output layer takes gz/T; a hidden
 * layer folds the same-timestep credit from the layer above (gs tape) with
 * the reset path -threshold * gu[t+1] through the surrogate, exactly as the
 * CPU backward does.
 */
__global__ void k_lif_backward(const float *u_t,   /* hidden only */
                               const float *gs_t,  /* hidden only */
                               const float *gz,    /* output only */
                               const float *gu_next,
                               float *gu_t,
                               int b_count,
                               int n,
                               int gu_stride, /* max_size */
                               float inv_t,
                               float beta,
                               float threshold,
                               int detach,
                               int surrogate,
                               float alpha) {
    const size_t total = (size_t)b_count * n;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t b = i / n;
        const size_t k = i % n;
        const size_t gu_idx = b * gu_stride + k;
        if (gz != NULL) {
            gu_t[gu_idx] = gz[i] * inv_t + beta * gu_next[gu_idx];
        } else {
            const float gs = detach ? gs_t[i] : gs_t[i] - threshold * gu_next[gu_idx];
            const float phi = d_surrogate_grad(surrogate, u_t[i] - threshold, alpha);
            gu_t[gu_idx] = gs * phi + beta * gu_next[gu_idx];
        }
    }
}

/* gsum[b, k] = sum over t of gu[t, b, k]: the static-input collapse. */
__global__ void k_gu_time_sum(const float *gu, int timesteps, int b_count, int n, int gu_stride, float *gsum) {
    const size_t total = (size_t)b_count * n;
    const size_t step = (size_t)b_count * gu_stride;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t idx = (i / n) * gu_stride + i % n;
        float acc = 0.0f;
        for (int t = 0; t < timesteps; ++t) {
            acc += gu[(size_t)t * step + idx];
        }
        gsum[i] = acc;
    }
}

/* out[c] += column sums of an [rows, cols] matrix with leading dim ld. */
__global__ void k_col_sum(const float *m, long rows, int cols, int ld, float *out) {
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < cols; c += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (long r = 0; r < rows; ++r) {
            acc += m[(size_t)r * ld + c];
        }
        out[c] += acc;
    }
}

__global__ void k_finite_check(const float *g, size_t n, int *reject) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        if (!isfinite(g[i])) {
            *reject = 1;
        }
    }
}

/* One Adam step over the whole parameter vector; a no-op once the sticky
 * reject flag is set, so a poisoned batch never touches the moments. */
__global__ void k_adam(float *params,
                       const float *g,
                       float *m,
                       float *v,
                       size_t n,
                       float scale,
                       float lr,
                       float b1,
                       float b2,
                       float bias1,
                       float bias2,
                       float eps,
                       const int *reject) {
    if (*reject) {
        return;
    }
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float gi = g[i] * scale;
        const float mi = b1 * m[i] + (1.0f - b1) * gi;
        const float vi = b2 * v[i] + (1.0f - b2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        params[i] -= lr * (mi / bias1) / (sqrtf(vi / bias2) + eps);
    }
}

/* Spike totals over a [count] slab of a spike tape (entries are exactly 0/1). */
__global__ void k_spike_sum(const float *s, size_t n, unsigned long long *acc) {
    __shared__ unsigned int block_sum;
    if (threadIdx.x == 0u) {
        block_sum = 0u;
    }
    __syncthreads();
    unsigned int mine = 0u;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        mine += s[i] != 0.0f;
    }
    atomicAdd(&block_sum, mine);
    __syncthreads();
    if (threadIdx.x == 0u) {
        atomicAdd(acc, (unsigned long long)block_sum);
    }
}

/* ------------------------------------------------------------------ */
/* row-major GEMM helper                                               */
/* ------------------------------------------------------------------ */

/*
 * C[M, N] = alpha * op(A) * op(B) + beta * C with row-major storage. A
 * row-major [R, C] matrix with row stride ld is a column-major [C, R] matrix
 * with the same ld, so cuBLAS computes C^T = op(B)^T * op(A)^T: swap the
 * operands, keep the op flags. ld arguments are the stored row widths.
 */
static int gemm_rm(bptt_gpu_t *g,
                   cublasOperation_t op_a,
                   cublasOperation_t op_b,
                   int m,
                   int n,
                   int k,
                   float alpha,
                   const float *a,
                   int lda,
                   const float *b,
                   int ldb,
                   float beta,
                   float *c,
                   int ldc) {
    BLAS_TRY(g, cublasSgemm(g->blas, op_b, op_a, n, m, k, &alpha, b, ldb, a, lda, &beta, c, ldc));
    return 0;
}

/* ------------------------------------------------------------------ */
/* context                                                             */
/* ------------------------------------------------------------------ */

int bptt_gpu_available(void) {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

const char *bptt_gpu_last_error(const bptt_gpu_t *gpu) {
    return gpu == NULL ? "no gpu context" : gpu->err;
}

void bptt_gpu_set_soft_spikes(bptt_gpu_t *gpu, int enable) {
    if (gpu != NULL) {
        gpu->soft_spikes = enable;
    }
}

static int grid_for(size_t n) {
    const size_t blocks = (n + THREADS - 1u) / THREADS;
    return blocks > 4096u ? 4096 : (int)blocks;
}

static int upload_set(bptt_gpu_t *g,
                      const uint8_t *px,
                      const uint8_t *lb,
                      snn_size_t count,
                      float **out_x,
                      uint8_t **out_y) {
    const size_t pixels = (size_t)count * g->in_size;
    uint8_t *staging = NULL;
    CUDA_TRY(g, cudaMalloc((void **)&staging, pixels));
    CUDA_TRY(g, cudaMalloc((void **)out_x, pixels * sizeof(float)));
    CUDA_TRY(g, cudaMalloc((void **)out_y, (size_t)count));
    CUDA_TRY(g, cudaMemcpy(staging, px, pixels, cudaMemcpyHostToDevice));
    CUDA_TRY(g, cudaMemcpy(*out_y, lb, (size_t)count, cudaMemcpyHostToDevice));
    k_u8_to_f32<<<grid_for(pixels), THREADS, 0, g->stream>>>(staging, *out_x, pixels);
    CUDA_TRY(g, cudaStreamSynchronize(g->stream));
    CUDA_TRY(g, cudaFree(staging));
    return 0;
}

void bptt_gpu_destroy(bptt_gpu_t *gpu) {
    if (gpu == NULL) {
        return;
    }
    for (size_t j = 0; j < gpu->layers; ++j) {
        cudaFree(gpu->d_u[j]);
        cudaFree(gpu->d_s[j]);
        cudaFree(gpu->d_gs[j]);
    }
    cudaFree(gpu->d_gu);
    cudaFree(gpu->d_xb);
    cudaFree(gpu->d_drive);
    cudaFree(gpu->d_gsum);
    cudaFree(gpu->d_logits);
    cudaFree(gpu->d_gz);
    cudaFree(gpu->d_loss);
    cudaFree(gpu->d_correct);
    cudaFree(gpu->d_spikes);
    cudaFree(gpu->d_reject);
    cudaFree(gpu->d_params);
    cudaFree(gpu->d_grads);
    cudaFree(gpu->d_m);
    cudaFree(gpu->d_v);
    cudaFree(gpu->d_train_x);
    cudaFree(gpu->d_test_x);
    cudaFree(gpu->d_train_y);
    cudaFree(gpu->d_test_y);
    cudaFree(gpu->d_order);
    if (gpu->blas != NULL) {
        cublasDestroy(gpu->blas);
    }
    if (gpu->stream != NULL) {
        cudaStreamDestroy(gpu->stream);
    }
    free(gpu);
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
    bptt_gpu_t *g = NULL;
    snn_size_t running = 0;
    snn_size_t max_b = 0;

    if (layer_sizes == NULL || layer_count < 2u || layer_count > MAX_NEURON_LAYERS + 1u ||
        params == NULL || batch == 0u || eval_batch == 0u || timesteps == 0u) {
        set_err(err, err_len, "bptt_gpu_create", "bad configuration");
        return NULL;
    }
    if (!bptt_gpu_available()) {
        set_err(err, err_len, "bptt_gpu_create", "no CUDA device");
        return NULL;
    }
    g = (bptt_gpu_t *)calloc(1u, sizeof(*g));
    if (g == NULL) {
        set_err(err, err_len, "bptt_gpu_create", "out of host memory");
        return NULL;
    }

    g->layers = layer_count - 1u;
    for (size_t j = 0; j <= g->layers; ++j) {
        g->sizes[j] = layer_sizes[j];
    }
    for (size_t j = 0; j < g->layers; ++j) {
        g->w_off[j] = running;
        running += g->sizes[j + 1u] * g->sizes[j];
        g->b_off[j] = running;
        running += g->sizes[j + 1u];
        if (g->sizes[j + 1u] > g->max_size) {
            g->max_size = g->sizes[j + 1u];
        }
    }
    if (running != param_count) {
        set_err(err, err_len, "bptt_gpu_create", "parameter layout mismatch");
        free(g);
        return NULL;
    }
    g->param_count = param_count;
    g->timesteps = timesteps;
    g->in_size = g->sizes[0];
    g->out_size = g->sizes[g->layers];
    g->beta = beta;
    g->threshold = threshold;
    g->alpha = alpha;
    g->surrogate = (int)surrogate;
    g->detach_reset = detach_reset;
    g->batch = batch;
    g->eval_batch = eval_batch;
    g->train_count = train_count;
    g->test_count = test_count;
    g->lr = lr;
    g->adam_b1 = adam_beta1;
    g->adam_b2 = adam_beta2;
    g->adam_eps = adam_eps;
    max_b = batch > eval_batch ? batch : eval_batch;

#define CREATE_TRY(call)                                                          \
    do {                                                                          \
        if ((call) != 0) {                                                        \
            set_err(err, err_len, "bptt_gpu_create", g->err);                     \
            bptt_gpu_destroy(g);                                                  \
            return NULL;                                                          \
        }                                                                         \
    } while (0)

    {
        int rc = cudaStreamCreate(&g->stream) != cudaSuccess || cublasCreate(&g->blas) != CUBLAS_STATUS_SUCCESS;
        if (rc == 0) {
            rc = cublasSetStream(g->blas, g->stream) != CUBLAS_STATUS_SUCCESS;
        }
        if (rc != 0) {
            set_err(err, err_len, "bptt_gpu_create", "stream/cublas init failed");
            bptt_gpu_destroy(g);
            return NULL;
        }
    }

    /* Parameters, gradients, Adam moments. */
    {
        const size_t pb = (size_t)param_count * sizeof(float);
        int rc = cudaMalloc((void **)&g->d_params, pb) != cudaSuccess ||
                 cudaMalloc((void **)&g->d_grads, pb) != cudaSuccess ||
                 cudaMalloc((void **)&g->d_m, pb) != cudaSuccess ||
                 cudaMalloc((void **)&g->d_v, pb) != cudaSuccess;
        if (rc == 0) {
            rc = cudaMemcpy(g->d_params, params, pb, cudaMemcpyHostToDevice) != cudaSuccess ||
                 cudaMemset(g->d_m, 0, pb) != cudaSuccess || cudaMemset(g->d_v, 0, pb) != cudaSuccess;
        }
        if (rc != 0) {
            set_err(err, err_len, "bptt_gpu_create", "parameter buffers failed");
            bptt_gpu_destroy(g);
            return NULL;
        }
    }

    CREATE_TRY(upload_set(g, train_px, train_lb, train_count, &g->d_train_x, &g->d_train_y));
    CREATE_TRY(upload_set(g, test_px, test_lb, test_count, &g->d_test_x, &g->d_test_y));

    /* Tapes and scratch. */
    {
        const size_t T = (size_t)timesteps;
        int rc = 0;
        for (size_t j = 0; j < g->layers && rc == 0; ++j) {
            const size_t n = g->sizes[j + 1u];
            rc |= cudaMalloc((void **)&g->d_u[j], T * max_b * n * sizeof(float)) != cudaSuccess;
            if (j + 1u < g->layers) {
                rc |= cudaMalloc((void **)&g->d_s[j], T * max_b * n * sizeof(float)) != cudaSuccess;
                rc |= cudaMalloc((void **)&g->d_gs[j], T * (size_t)batch * n * sizeof(float)) != cudaSuccess;
            }
        }
        rc |= cudaMalloc((void **)&g->d_gu, (T + 1u) * (size_t)batch * g->max_size * sizeof(float)) != cudaSuccess;
        rc |= cudaMalloc((void **)&g->d_xb, (size_t)max_b * g->in_size * sizeof(float)) != cudaSuccess;
        rc |= cudaMalloc((void **)&g->d_drive, (size_t)max_b * g->sizes[1] * sizeof(float)) != cudaSuccess;
        rc |= cudaMalloc((void **)&g->d_gsum, (size_t)batch * g->sizes[1] * sizeof(float)) != cudaSuccess;
        rc |= cudaMalloc((void **)&g->d_logits, (size_t)max_b * g->out_size * sizeof(float)) != cudaSuccess;
        rc |= cudaMalloc((void **)&g->d_gz, (size_t)batch * g->out_size * sizeof(float)) != cudaSuccess;
        rc |= cudaMalloc((void **)&g->d_loss, sizeof(double)) != cudaSuccess;
        rc |= cudaMalloc((void **)&g->d_correct, sizeof(int)) != cudaSuccess;
        rc |= cudaMalloc((void **)&g->d_spikes, sizeof(unsigned long long)) != cudaSuccess;
        rc |= cudaMalloc((void **)&g->d_reject, sizeof(int)) != cudaSuccess;
        rc |= cudaMalloc((void **)&g->d_order, (size_t)train_count * sizeof(snn_size_t)) != cudaSuccess;
        if (rc != 0) {
            set_err(err, err_len, "bptt_gpu_create", "tape allocation failed (out of device memory?)");
            bptt_gpu_destroy(g);
            return NULL;
        }
    }
#undef CREATE_TRY
    return g;
}

/* ------------------------------------------------------------------ */
/* forward / backward over one minibatch                               */
/* ------------------------------------------------------------------ */

/*
 * Runs the batched forward for b_count samples already gathered into d_xb,
 * leaving membrane/spike tapes and logits behind. Mirrors snn_bptt_forward
 * with static_input == 1.
 */
static int forward_batch(bptt_gpu_t *g, int b_count) {
    const int T = (int)g->timesteps;
    const int n1 = (int)g->sizes[1];

    /* Layer 0 drive: one GEMM per minibatch, bias folded in by k_lif_step. */
    if (gemm_rm(g, CUBLAS_OP_N, CUBLAS_OP_T, b_count, n1, (int)g->in_size, 1.0f, g->d_xb, (int)g->in_size,
                g->d_params + g->w_off[0], (int)g->in_size, 0.0f, g->d_drive, n1) != 0) {
        return -1;
    }

    for (int t = 0; t < T; ++t) {
        for (size_t j = 0; j < g->layers; ++j) {
            const int rows = (int)g->sizes[j + 1u];
            const int cols = (int)g->sizes[j];
            float *u_t = g->d_u[j] + (size_t)t * b_count * rows;
            const int hidden = j + 1u < g->layers;
            float *s_t = hidden ? g->d_s[j] + (size_t)t * b_count * rows : NULL;
            const float *in = g->d_drive;

            if (j > 0u) {
                /* u_t <- S[j-1][t] * W[j]^T, then the LIF kernel finishes in place. */
                const float *pre = g->d_s[j - 1u] + (size_t)t * b_count * cols;
                if (gemm_rm(g, CUBLAS_OP_N, CUBLAS_OP_T, b_count, rows, cols, 1.0f, pre, cols,
                            g->d_params + g->w_off[j], cols, 0.0f, u_t, rows) != 0) {
                    return -1;
                }
                in = u_t;
            }
            k_lif_step<<<grid_for((size_t)b_count * rows), THREADS, 0, g->stream>>>(
                u_t, s_t, in, t > 0 ? g->d_u[j] + (size_t)(t - 1) * b_count * rows : NULL,
                (t > 0 && hidden) ? g->d_s[j] + (size_t)(t - 1) * b_count * rows : NULL,
                g->d_params + g->b_off[j], b_count, rows, g->beta, g->threshold, g->soft_spikes, g->surrogate,
                g->alpha);
        }
    }
    k_logits<<<grid_for((size_t)b_count * g->out_size), THREADS, 0, g->stream>>>(
        g->d_u[g->layers - 1u], T, b_count, (int)g->out_size, g->d_logits);
    return 0;
}

/* BPTT for the minibatch: fills d_grads (accumulating; caller zeroes it). */
static int backward_batch(bptt_gpu_t *g, int b_count) {
    const int T = (int)g->timesteps;
    const int max_n = (int)g->max_size;
    const float inv_t = 1.0f / (float)T;
    const size_t gu_block = (size_t)b_count * max_n;

    for (size_t j = g->layers; j-- > 0u;) {
        const int rows = (int)g->sizes[j + 1u];
        const int cols = (int)g->sizes[j];
        const int is_output = j + 1u == g->layers;
        float *gw = g->d_grads + g->w_off[j];
        float *gb = g->d_grads + g->b_off[j];
        const float *w = g->d_params + g->w_off[j];

        /* Row T of the gu tape is the zero seed, as in the CPU workspace. */
        CUDA_TRY(g, cudaMemsetAsync(g->d_gu + (size_t)T * gu_block, 0, gu_block * sizeof(float), g->stream));

        for (int t = T - 1; t >= 0; --t) {
            float *gu_t = g->d_gu + (size_t)t * gu_block;
            const float *gu_next = g->d_gu + (size_t)(t + 1) * gu_block;
            k_lif_backward<<<grid_for((size_t)b_count * rows), THREADS, 0, g->stream>>>(
                is_output ? NULL : g->d_u[j] + (size_t)t * b_count * rows,
                is_output ? NULL : g->d_gs[j] + (size_t)t * b_count * rows, is_output ? g->d_gz : NULL, gu_next,
                gu_t, b_count, rows, max_n, inv_t, g->beta, g->threshold, g->detach_reset, g->surrogate, g->alpha);
            if (j > 0u) {
                /* GS[j-1][t] = GU[t] * W[j] */
                if (gemm_rm(g, CUBLAS_OP_N, CUBLAS_OP_N, b_count, cols, rows, 1.0f, gu_t, max_n, w, cols, 0.0f,
                            g->d_gs[j - 1u] + (size_t)t * b_count * cols, cols) != 0) {
                    return -1;
                }
            }
        }

        if (j == 0u) {
            /* Static input: T rank-1 updates collapse through the time sum. */
            k_gu_time_sum<<<grid_for((size_t)b_count * rows), THREADS, 0, g->stream>>>(g->d_gu, T, b_count, rows,
                                                                                       max_n, g->d_gsum);
            k_col_sum<<<grid_for((size_t)rows), THREADS, 0, g->stream>>>(g->d_gsum, b_count, rows, rows, gb);
            if (gemm_rm(g, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, b_count, 1.0f, g->d_gsum, rows, g->d_xb, cols,
                        1.0f, gw, cols) != 0) {
                return -1;
            }
        } else {
            /* gw += GU_flat^T * Pre_flat over the whole (T*B) extent at once. */
            const long flat = (long)T * b_count;
            k_col_sum<<<grid_for((size_t)rows), THREADS, 0, g->stream>>>(g->d_gu, flat, rows, max_n, gb);
            if (gemm_rm(g, CUBLAS_OP_T, CUBLAS_OP_N, rows, cols, (int)flat, 1.0f, g->d_gu, max_n,
                        g->d_s[j - 1u], cols, 1.0f, gw, cols) != 0) {
                return -1;
            }
        }
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/* public entry points                                                 */
/* ------------------------------------------------------------------ */

int bptt_gpu_train_epoch(bptt_gpu_t *g, const snn_size_t *order, snn_size_t count, float *out_mean_loss) {
    double loss = 0.0;
    int reject = 0;

    if (g == NULL || order == NULL || count == 0u || count > g->train_count) {
        return -1;
    }
    CUDA_TRY(g, cudaMemcpyAsync(g->d_order, order, (size_t)count * sizeof(snn_size_t), cudaMemcpyHostToDevice,
                                g->stream));
    CUDA_TRY(g, cudaMemsetAsync(g->d_loss, 0, sizeof(double), g->stream));
    CUDA_TRY(g, cudaMemsetAsync(g->d_reject, 0, sizeof(int), g->stream));

    for (snn_size_t start = 0; start < count; start += g->batch) {
        const int b_count = (int)(count - start < g->batch ? count - start : g->batch);
        const float scale = 1.0f / (float)b_count;
        float bias1 = 0.0f;
        float bias2 = 0.0f;

        CUDA_TRY(g, cudaMemsetAsync(g->d_grads, 0, (size_t)g->param_count * sizeof(float), g->stream));
        k_gather_rows<<<grid_for((size_t)b_count * g->in_size), THREADS, 0, g->stream>>>(
            g->d_train_x, g->d_order, start, b_count, (int)g->in_size, g->d_xb);
        if (forward_batch(g, b_count) != 0) {
            return -1;
        }
        k_loss<<<grid_for((size_t)b_count), THREADS, 0, g->stream>>>(g->d_logits, g->d_train_y, g->d_order, start,
                                                                     b_count, (int)g->out_size, g->d_gz, g->d_loss,
                                                                     NULL);
        if (backward_batch(g, b_count) != 0) {
            return -1;
        }
        k_finite_check<<<grid_for((size_t)g->param_count), THREADS, 0, g->stream>>>(g->d_grads, g->param_count,
                                                                                    g->d_reject);
        g->adam_steps += 1u;
        bias1 = 1.0f - powf(g->adam_b1, (float)g->adam_steps);
        bias2 = 1.0f - powf(g->adam_b2, (float)g->adam_steps);
        k_adam<<<grid_for((size_t)g->param_count), THREADS, 0, g->stream>>>(
            g->d_params, g->d_grads, g->d_m, g->d_v, g->param_count, scale, g->lr, g->adam_b1, g->adam_b2, bias1,
            bias2, g->adam_eps, g->d_reject);
    }

    CUDA_TRY(g, cudaMemcpyAsync(&loss, g->d_loss, sizeof(double), cudaMemcpyDeviceToHost, g->stream));
    CUDA_TRY(g, cudaMemcpyAsync(&reject, g->d_reject, sizeof(int), cudaMemcpyDeviceToHost, g->stream));
    CUDA_TRY(g, cudaStreamSynchronize(g->stream));
    CUDA_TRY(g, cudaGetLastError());
    if (reject != 0) {
        set_err(g->err, sizeof(g->err), "bptt_gpu_train_epoch", "non-finite gradient batch (as on CPU, fatal)");
        return -1;
    }
    if (out_mean_loss != NULL) {
        *out_mean_loss = (float)(loss / (double)count);
    }
    return 0;
}

int bptt_gpu_evaluate(bptt_gpu_t *g,
                      int use_test,
                      snn_size_t limit,
                      float *out_accuracy,
                      float *out_loss,
                      double *out_spikes) {
    const float *x = NULL;
    const uint8_t *y = NULL;
    snn_size_t count = 0;
    double loss = 0.0;
    int correct = 0;
    unsigned long long spikes = 0;

    if (g == NULL) {
        return -1;
    }
    x = use_test ? g->d_test_x : g->d_train_x;
    y = use_test ? g->d_test_y : g->d_train_y;
    count = use_test ? g->test_count : g->train_count;
    if (limit != 0u && limit < count) {
        count = limit;
    }
    CUDA_TRY(g, cudaMemsetAsync(g->d_loss, 0, sizeof(double), g->stream));
    CUDA_TRY(g, cudaMemsetAsync(g->d_correct, 0, sizeof(int), g->stream));
    CUDA_TRY(g, cudaMemsetAsync(g->d_spikes, 0, sizeof(unsigned long long), g->stream));

    for (snn_size_t start = 0; start < count; start += g->eval_batch) {
        const int b_count = (int)(count - start < g->eval_batch ? count - start : g->eval_batch);
        k_gather_rows<<<grid_for((size_t)b_count * g->in_size), THREADS, 0, g->stream>>>(x, NULL, start, b_count,
                                                                                         (int)g->in_size, g->d_xb);
        if (forward_batch(g, b_count) != 0) {
            return -1;
        }
        k_loss<<<grid_for((size_t)b_count), THREADS, 0, g->stream>>>(g->d_logits, y, NULL, start, b_count,
                                                                     (int)g->out_size, NULL, g->d_loss,
                                                                     g->d_correct);
        for (size_t j = 0; j + 1u < g->layers; ++j) {
            const size_t n = (size_t)g->timesteps * b_count * g->sizes[j + 1u];
            k_spike_sum<<<grid_for(n), THREADS, 0, g->stream>>>(g->d_s[j], n, g->d_spikes);
        }
    }

    CUDA_TRY(g, cudaMemcpyAsync(&loss, g->d_loss, sizeof(double), cudaMemcpyDeviceToHost, g->stream));
    CUDA_TRY(g, cudaMemcpyAsync(&correct, g->d_correct, sizeof(int), cudaMemcpyDeviceToHost, g->stream));
    CUDA_TRY(g, cudaMemcpyAsync(&spikes, g->d_spikes, sizeof(unsigned long long), cudaMemcpyDeviceToHost,
                                g->stream));
    CUDA_TRY(g, cudaStreamSynchronize(g->stream));
    CUDA_TRY(g, cudaGetLastError());
    if (out_accuracy != NULL) {
        *out_accuracy = (float)((double)correct / (double)count);
    }
    if (out_loss != NULL) {
        *out_loss = (float)(loss / (double)count);
    }
    if (out_spikes != NULL) {
        *out_spikes = (double)spikes / (double)count;
    }
    return 0;
}

int bptt_gpu_batch_grads(bptt_gpu_t *g,
                         const snn_size_t *indices,
                         snn_size_t count,
                         float *out_grads,
                         float *out_loss_sum) {
    double loss = 0.0;
    if (g == NULL || indices == NULL || count == 0u || count > g->batch || out_grads == NULL) {
        return -1;
    }
    CUDA_TRY(g, cudaMemcpyAsync(g->d_order, indices, (size_t)count * sizeof(snn_size_t), cudaMemcpyHostToDevice,
                                g->stream));
    CUDA_TRY(g, cudaMemsetAsync(g->d_loss, 0, sizeof(double), g->stream));
    CUDA_TRY(g, cudaMemsetAsync(g->d_grads, 0, (size_t)g->param_count * sizeof(float), g->stream));
    k_gather_rows<<<grid_for((size_t)count * g->in_size), THREADS, 0, g->stream>>>(
        g->d_train_x, g->d_order, 0, (int)count, (int)g->in_size, g->d_xb);
    if (forward_batch(g, (int)count) != 0) {
        return -1;
    }
    k_loss<<<grid_for((size_t)count), THREADS, 0, g->stream>>>(g->d_logits, g->d_train_y, g->d_order, 0, (int)count,
                                                               (int)g->out_size, g->d_gz, g->d_loss, NULL);
    if (backward_batch(g, (int)count) != 0) {
        return -1;
    }
    CUDA_TRY(g, cudaMemcpyAsync(out_grads, g->d_grads, (size_t)g->param_count * sizeof(float),
                                cudaMemcpyDeviceToHost, g->stream));
    CUDA_TRY(g, cudaMemcpyAsync(&loss, g->d_loss, sizeof(double), cudaMemcpyDeviceToHost, g->stream));
    CUDA_TRY(g, cudaStreamSynchronize(g->stream));
    CUDA_TRY(g, cudaGetLastError());
    if (out_loss_sum != NULL) {
        *out_loss_sum = (float)loss;
    }
    return 0;
}
