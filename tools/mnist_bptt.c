/*
 * Trains the BPTT / surrogate-gradient SNN on MNIST and compares surrogate
 * functions.
 *
 * Encoding is constant-current ("direct"): the normalized 784-pixel vector is
 * injected unchanged at every timestep, so the input layer's drive is computed
 * once per sample rather than once per timestep. Readout is the time-averaged
 * membrane potential of a non-spiking output layer, trained with softmax
 * cross-entropy.
 *
 * Batch parallelism is over samples: each thread owns a workspace and a
 * gradient accumulator, and the accumulators are reduced in a fixed thread
 * order, so the parameter update is reproducible for a fixed thread count.
 * (The reported loss/spike averages use OpenMP reductions, whose combination
 * order is unspecified; they can differ in the last bits without the training
 * trajectory differing at all.)
 *
 * Modes:
 *   single  one configuration, reporting per-epoch curves
 *   sweep   every surrogate x every alpha, to find each surrogate's own best
 *           gradient-window width before comparing shapes
 *   final   every surrogate at a given alpha, several seeds, full dataset
 */

#include <snn/snn.h>
#include <snn/snn_bptt.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <zlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "bptt_cuda.h"

#ifdef SNN_ENABLE_TEST_HOOKS
/* Library test hook (no public header): soft spikes make the model smooth so
 * the gputest gradient comparison cannot be tripped by a spike flipping on a
 * last-bit membrane difference between the CPU and GPU implementations. */
void snn_test_bptt_set_soft_spikes(snn_bptt_network_t *network, int enable);
#endif

#define MNIST_PIXELS 784u
#define MNIST_CLASSES 10u
#define MAX_HIDDEN_LAYERS 8u

typedef struct {
    uint8_t *pixels; /* count * 784 */
    uint8_t *labels; /* count */
    snn_size_t count;
} mnist_set_t;

typedef struct {
    const char *data_dir;
    const char *mode;
    const char *csv_path;
    const char *tag;
    snn_size_t hidden[MAX_HIDDEN_LAYERS];
    size_t hidden_count;
    snn_size_t timesteps;
    int epochs;
    snn_size_t train_limit;
    snn_size_t test_limit;
    snn_size_t train_eval; /* post-epoch train-set eval on this many samples, 0 = off */
    snn_size_t batch;
    float lr;
    float beta;
    float threshold;
    float gain;
    float alpha;
    snn_surrogate_t surrogate;
    int detach_reset;
    int seeds;
    int threads;
    int gpu;
    uint64_t seed0;
} options_t;

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static uint64_t splitmix64(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static void die(const char *what) {
    fprintf(stderr, "mnist_bptt: %s\n", what);
    exit(1);
}

/* ------------------------------------------------------------------ */
/* IDX loading (gzip, big-endian headers)                              */
/* ------------------------------------------------------------------ */

static int read_u32_be(gzFile f, uint32_t *out) {
    unsigned char b[4];
    if (gzread(f, b, 4) != 4) {
        return 0;
    }
    *out = ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | (uint32_t)b[3];
    return 1;
}

static uint8_t *read_idx(const char *path, uint32_t expect_magic, uint32_t *out_count, uint32_t *out_stride) {
    gzFile f = gzopen(path, "rb");
    uint32_t magic = 0;
    uint32_t count = 0;
    uint32_t rows = 1;
    uint32_t cols = 1;
    uint64_t bytes = 0;
    uint8_t *data = NULL;
    if (f == NULL) {
        fprintf(stderr, "mnist_bptt: cannot open %s\n", path);
        exit(1);
    }
    if (!read_u32_be(f, &magic) || magic != expect_magic || !read_u32_be(f, &count)) {
        die("bad idx header");
    }
    if (expect_magic == 0x00000803u) {
        if (!read_u32_be(f, &rows) || !read_u32_be(f, &cols)) {
            die("bad idx image header");
        }
        /* rows*cols is 32-bit: a crafted header (e.g. 0x10000031 x 0x10) wraps
         * to exactly 784 and would pass the stride check downstream while the
         * real extent is astronomically larger. MNIST geometry is fixed. */
        if (rows != 28u || cols != 28u) {
            die("idx images are not 28x28");
        }
    }
    bytes = (uint64_t)count * (uint64_t)rows * (uint64_t)cols;
    data = (uint8_t *)malloc((size_t)bytes);
    if (data == NULL) {
        die("out of memory loading idx");
    }
    {
        uint64_t got = 0;
        while (got < bytes) {
            const unsigned chunk = (unsigned)((bytes - got) > (1u << 30) ? (1u << 30) : (bytes - got));
            const int n = gzread(f, data + got, chunk);
            if (n <= 0) {
                die("short idx read");
            }
            got += (uint64_t)n;
        }
    }
    gzclose(f);
    *out_count = count;
    *out_stride = rows * cols;
    return data;
}

static void mnist_load(const char *dir, const char *images, const char *labels, snn_size_t limit, mnist_set_t *out) {
    char path[1024];
    uint32_t image_count = 0;
    uint32_t stride = 0;
    uint32_t label_count = 0;
    uint32_t label_stride = 0;

    snprintf(path, sizeof(path), "%s/%s", dir, images);
    out->pixels = read_idx(path, 0x00000803u, &image_count, &stride);
    snprintf(path, sizeof(path), "%s/%s", dir, labels);
    out->labels = read_idx(path, 0x00000801u, &label_count, &label_stride);
    if (stride != MNIST_PIXELS || image_count != label_count) {
        die("unexpected mnist geometry");
    }
    out->count = image_count;
    if (limit != 0u && limit < out->count) {
        out->count = limit;
    }
}

static void mnist_free(mnist_set_t *set) {
    free(set->pixels);
    free(set->labels);
}

/* Constant-current encoding: pixel/255 injected unchanged every timestep. */
static void encode(const mnist_set_t *set, snn_size_t index, float *frame) {
    const uint8_t *px = set->pixels + (size_t)index * MNIST_PIXELS;
    snn_size_t i = 0;
    for (i = 0; i < MNIST_PIXELS; ++i) {
        frame[i] = (float)px[i] * (1.0f / 255.0f);
    }
}

/* ------------------------------------------------------------------ */
/* training                                                            */
/* ------------------------------------------------------------------ */

typedef struct {
    snn_bptt_workspace_t **ws;
    snn_bptt_grads_t **grads;
    float **frames;
    int threads;
} pool_t;

static void pool_create(pool_t *pool, const snn_bptt_network_t *net, int threads) {
    int t = 0;
    pool->threads = threads;
    pool->ws = (snn_bptt_workspace_t **)calloc((size_t)threads, sizeof(*pool->ws));
    pool->grads = (snn_bptt_grads_t **)calloc((size_t)threads, sizeof(*pool->grads));
    pool->frames = (float **)calloc((size_t)threads, sizeof(*pool->frames));
    if (pool->ws == NULL || pool->grads == NULL || pool->frames == NULL) {
        die("out of memory allocating thread pool");
    }
    for (t = 0; t < threads; ++t) {
        if (snn_bptt_workspace_create(net, &pool->ws[t]) != SNN_OK ||
            snn_bptt_grads_create(net, &pool->grads[t]) != SNN_OK) {
            die("out of memory allocating workspaces");
        }
        pool->frames[t] = (float *)malloc(MNIST_PIXELS * sizeof(float));
        if (pool->frames[t] == NULL) {
            die("out of memory allocating frames");
        }
    }
}

static void pool_free(pool_t *pool) {
    int t = 0;
    for (t = 0; t < pool->threads; ++t) {
        snn_bptt_workspace_free(pool->ws[t]);
        snn_bptt_grads_free(pool->grads[t]);
        free(pool->frames[t]);
    }
    free(pool->ws);
    free(pool->grads);
    free(pool->frames);
}

static int thread_id(void) {
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

/* Leaving an OpenMP structured block by calling exit() is not allowed, so the
 * worker loops record a failure and the caller aborts once the region ends. */
static void check_failed(int failed, const char *what) {
    if (failed) {
        die(what);
    }
}

static void evaluate(const snn_bptt_network_t *net,
                     pool_t *pool,
                     const mnist_set_t *set,
                     float *out_accuracy,
                     float *out_loss,
                     double *out_spikes) {
    long long correct = 0;
    double loss_sum = 0.0;
    double spike_sum = 0.0;
    long long i = 0;
    int failed = 0;
    const long long n = (long long)set->count;

#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : correct, loss_sum, spike_sum) reduction(| : failed)
#endif
    for (i = 0; i < n; ++i) {
        const int tid = thread_id();
        float *frame = pool->frames[tid];
        float loss = 0.0f;
        encode(set, (snn_size_t)i, frame);
        failed |= snn_bptt_forward(net, pool->ws[tid], frame, 1) != SNN_OK;
        failed |= snn_bptt_cross_entropy(pool->ws[tid], set->labels[i], &loss) != SNN_OK;
        correct += snn_bptt_prediction(pool->ws[tid]) == (snn_size_t)set->labels[i];
        loss_sum += (double)loss;
        spike_sum += (double)snn_bptt_spike_count(pool->ws[tid]);
    }
    check_failed(failed, "forward or loss failed during evaluation");
    *out_accuracy = (float)((double)correct / (double)n);
    *out_loss = (float)(loss_sum / (double)n);
    *out_spikes = spike_sum / (double)n;
}

/* Runs one training epoch over a shuffled index list. */
static float train_epoch(snn_bptt_network_t *net,
                         snn_bptt_optimizer_t *opt,
                         pool_t *pool,
                         const mnist_set_t *set,
                         const snn_size_t *order,
                         snn_size_t batch) {
    double loss_sum = 0.0;
    snn_size_t start = 0;
    for (start = 0; start < set->count; start += batch) {
        const long long span = (long long)((set->count - start < batch) ? (set->count - start) : batch);
        double batch_loss = 0.0;
        long long k = 0;
        int t = 0;
        int failed = 0;
        for (t = 0; t < pool->threads; ++t) {
            if (snn_bptt_grads_zero(pool->grads[t]) != SNN_OK) {
                die("grads_zero failed");
            }
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) reduction(+ : batch_loss) reduction(| : failed)
#endif
        for (k = 0; k < span; ++k) {
            const int tid = thread_id();
            const snn_size_t index = order[start + (snn_size_t)k];
            float *frame = pool->frames[tid];
            float loss = 0.0f;
            encode(set, index, frame);
            failed |= snn_bptt_forward_backward(net, pool->ws[tid], frame, 1, set->labels[index], pool->grads[tid],
                                                &loss, NULL) != SNN_OK;
            batch_loss += (double)loss;
        }
        check_failed(failed, "forward_backward failed during training");
        /* Fixed thread order: the reduction is reproducible for a fixed
         * thread count, exactly as the library's OpenMP scatter is. */
        for (t = 1; t < pool->threads; ++t) {
            if (snn_bptt_grads_add(pool->grads[0], pool->grads[t]) != SNN_OK) {
                die("grads_add failed");
            }
        }
        if (snn_bptt_optimizer_step(opt, net, pool->grads[0], (snn_size_t)span) != SNN_OK) {
            die("optimizer_step failed");
        }
        loss_sum += batch_loss;
    }
    return (float)(loss_sum / (double)set->count);
}

static void shuffle(snn_size_t *order, snn_size_t count, uint64_t *rng) {
    snn_size_t i = 0;
    for (i = count; i > 1u; --i) {
        const snn_size_t j = (snn_size_t)(splitmix64(rng) % i);
        const snn_size_t tmp = order[i - 1u];
        order[i - 1u] = order[j];
        order[j] = tmp;
    }
}

typedef struct {
    float best_accuracy;
    float final_accuracy;
    float final_loss;
    float firing_rate;
    float epoch1_accuracy;
    double seconds;
} run_result_t;

static run_result_t train_one(const options_t *opt,
                              const mnist_set_t *train,
                              const mnist_set_t *test,
                              snn_surrogate_t surrogate,
                              float alpha,
                              uint64_t seed,
                              FILE *csv,
                              const char *tag,
                              int verbose) {
    snn_size_t layers[MAX_HIDDEN_LAYERS + 2u];
    snn_bptt_config_t cfg;
    snn_bptt_network_t *net = NULL;
    snn_bptt_optimizer_t *adam = NULL;
    bptt_gpu_t *gpu = NULL;
    pool_t pool;
    snn_size_t *order = NULL;
    snn_size_t i = 0;
    size_t h = 0;
    uint64_t rng = seed ^ UINT64_C(0x9e3779b97f4a7c15);
    run_result_t result;
    int epoch = 0;
    double hidden_steps = 0.0;
    /* The train-eval set is a fixed prefix of the (unshuffled) training set,
     * scored after the epoch's last update like the test set is, so the two
     * losses are the same measurement on the same frozen model. */
    mnist_set_t train_eval = *train;
    if (opt->train_eval != 0u && opt->train_eval < train_eval.count) {
        train_eval.count = opt->train_eval;
    }

    memset(&result, 0, sizeof(result));
    layers[0] = MNIST_PIXELS;
    for (h = 0; h < opt->hidden_count; ++h) {
        layers[h + 1u] = opt->hidden[h];
        hidden_steps += (double)opt->hidden[h] * (double)opt->timesteps;
    }
    layers[opt->hidden_count + 1u] = MNIST_CLASSES;
    cfg = snn_bptt_default_config(layers, opt->hidden_count + 2u, opt->timesteps);
    cfg.beta = opt->beta;
    cfg.threshold = opt->threshold;
    cfg.surrogate = surrogate;
    cfg.surrogate_alpha = alpha;
    cfg.detach_reset = opt->detach_reset;
    cfg.weight_init_gain = opt->gain;
    cfg.seed = seed;
    if (snn_bptt_network_create(&cfg, &net) != SNN_OK) {
        die("network create failed");
    }
    if (opt->gpu) {
        /* The library still creates the network so initialization is the
         * identical Kaiming-uniform draw; only the flat vector moves over. */
        const snn_size_t pc = snn_bptt_parameter_count(net);
        float *p0 = (float *)malloc((size_t)pc * sizeof(float));
        char errbuf[256];
        if (p0 == NULL || snn_bptt_get_parameters(net, p0, pc) != SNN_OK) {
            die("parameter export failed");
        }
        gpu = bptt_gpu_create(layers, opt->hidden_count + 2u, opt->timesteps, cfg.beta, cfg.threshold, surrogate,
                              alpha, opt->detach_reset, p0, pc, train->pixels, train->labels, train->count,
                              test->pixels, test->labels, test->count, opt->batch, 1000u, opt->lr, 0.9f, 0.999f,
                              1e-8f, errbuf, sizeof(errbuf));
        free(p0);
        if (gpu == NULL) {
            die(errbuf);
        }
    } else {
        if (snn_bptt_optimizer_create(net, opt->lr, 0.9f, 0.999f, 1e-8f, &adam) != SNN_OK) {
            die("optimizer create failed");
        }
        pool_create(&pool, net, opt->threads);
    }

    order = (snn_size_t *)malloc((size_t)train->count * sizeof(*order));
    if (order == NULL) {
        die("out of memory allocating shuffle order");
    }
    for (i = 0; i < train->count; ++i) {
        order[i] = i;
    }

    for (epoch = 1; epoch <= opt->epochs; ++epoch) {
        const double t0 = now_seconds();
        float train_loss = 0.0f;
        float acc = 0.0f;
        float loss = 0.0f;
        float train_eval_loss = (float)NAN;
        float train_eval_acc = (float)NAN;
        double spikes = 0.0;
        double elapsed = 0.0;

        shuffle(order, train->count, &rng);
        if (gpu != NULL) {
            if (bptt_gpu_train_epoch(gpu, order, train->count, &train_loss) != 0) {
                die(bptt_gpu_last_error(gpu));
            }
            elapsed = now_seconds() - t0;
            if (bptt_gpu_evaluate(gpu, 1, 0u, &acc, &loss, &spikes) != 0) {
                die(bptt_gpu_last_error(gpu));
            }
            if (opt->train_eval != 0u) {
                double eval_spikes = 0.0;
                if (bptt_gpu_evaluate(gpu, 0, opt->train_eval, &train_eval_acc, &train_eval_loss, &eval_spikes) !=
                    0) {
                    die(bptt_gpu_last_error(gpu));
                }
            }
        } else {
            train_loss = train_epoch(net, adam, &pool, train, order, opt->batch);
            elapsed = now_seconds() - t0;
            evaluate(net, &pool, test, &acc, &loss, &spikes);
            if (opt->train_eval != 0u) {
                double eval_spikes = 0.0;
                evaluate(net, &pool, &train_eval, &train_eval_acc, &train_eval_loss, &eval_spikes);
            }
        }

        result.final_accuracy = acc;
        result.final_loss = loss;
        result.firing_rate = (float)(spikes / hidden_steps);
        result.seconds += elapsed;
        if (acc > result.best_accuracy) {
            result.best_accuracy = acc;
        }
        if (epoch == 1) {
            result.epoch1_accuracy = acc;
        }
        if (verbose) {
            printf("  epoch %2d  train_loss %.4f  test_loss %.4f  test_acc %6.2f%%  firing %.3f  %.1fs", epoch,
                   (double)train_loss, (double)loss, 100.0 * (double)acc, spikes / hidden_steps, elapsed);
            if (opt->train_eval != 0u) {
                printf("  eval_loss %.4f  eval_acc %6.2f%%", (double)train_eval_loss, 100.0 * (double)train_eval_acc);
            }
            printf("\n");
            fflush(stdout);
        }
        if (csv != NULL) {
            fprintf(csv, "%s,%s,%g,%llu,%d,%.6f,%.6f,%.6f,%.6f,%.3f,%.6f,%.6f\n", tag, snn_surrogate_string(surrogate),
                    (double)alpha, (unsigned long long)seed, epoch, (double)train_loss, (double)loss, (double)acc,
                    spikes / hidden_steps, elapsed, (double)train_eval_loss, (double)train_eval_acc);
            fflush(csv);
        }
    }

    free(order);
    if (gpu != NULL) {
        bptt_gpu_destroy(gpu);
    } else {
        pool_free(&pool);
        snn_bptt_optimizer_free(adam);
    }
    snn_bptt_network_free(net);
    return result;
}

/* ------------------------------------------------------------------ */
/* GPU verification (--mode gputest)                                   */
/* ------------------------------------------------------------------ */

/*
 * Two-part battery for the CUDA backend, run on real dataset samples.
 *
 * Part 1 compares batch-summed gradients against the library under the
 * soft-spike test hook. With S(U - threshold) in place of the Heaviside the
 * unrolled model is smooth, so the two implementations can only differ by
 * float summation order -- a strict tolerance holds, and no tolerance has to
 * absorb a spike flipping on a last-bit membrane difference. A wrong index,
 * transpose, reset path or detach branch fails loudly.
 *
 * Part 2 trains with hard spikes for two epochs from an identical
 * initialization and shuffle order, then requires the CPU and GPU
 * trajectories to land together within loose bounds. This exercises the one
 * code path soft spikes cannot: the discrete spike itself.
 */

#ifdef SNN_ENABLE_TEST_HOOKS
typedef struct {
    const char *name;
    snn_size_t hidden[3];
    size_t hidden_count;
    snn_size_t timesteps;
    snn_surrogate_t surrogate;
    float alpha;
    int detach;
} gputest_case_t;

static double rel_l2(const float *a, const float *b, snn_size_t n) {
    double diff = 0.0;
    double ref = 0.0;
    snn_size_t i = 0;
    for (i = 0; i < n; ++i) {
        const double d = (double)a[i] - (double)b[i];
        diff += d * d;
        ref += (double)a[i] * (double)a[i];
    }
    return sqrt(diff) / (sqrt(ref) + 1e-30);
}
#endif /* SNN_ENABLE_TEST_HOOKS */

static int run_gputest(const options_t *opt, const mnist_set_t *train, const mnist_set_t *test) {
#ifndef SNN_ENABLE_TEST_HOOKS
    (void)opt;
    (void)train;
    (void)test;
    die("gputest needs the soft-spike hook: configure with -DSNN_BUILD_TESTS=ON");
    return 1;
#else
    static const gputest_case_t cases[] = {
        {"d1 atan T7", {40, 0, 0}, 1, 7, SNN_SURROGATE_ATAN, 1.0f, 0},
        {"d1 gaussian T1", {40, 0, 0}, 1, 1, SNN_SURROGATE_GAUSSIAN, 1.0f, 0},
        {"d2 atan T7", {32, 24, 0}, 2, 7, SNN_SURROGATE_ATAN, 1.0f, 0},
        {"d2 atan T7 detach", {32, 24, 0}, 2, 7, SNN_SURROGATE_ATAN, 1.0f, 1},
        {"d2 sigmoid T7", {32, 24, 0}, 2, 7, SNN_SURROGATE_SIGMOID, 1.0f, 0},
        {"d2 triangle T7", {32, 24, 0}, 2, 7, SNN_SURROGATE_TRIANGLE, 0.5f, 0},
        {"d2 rectangular T7", {32, 24, 0}, 2, 7, SNN_SURROGATE_RECTANGULAR, 0.5f, 0},
        {"d3 fast_sigmoid T5", {24, 16, 16}, 3, 5, SNN_SURROGATE_FAST_SIGMOID, 2.0f, 0},
    };
    const snn_size_t GRAD_BATCH = 64;
    const snn_size_t SUBSET = 256; /* uploaded slice of the training set */
    int failures = 0;
    size_t c = 0;

    if (!bptt_gpu_available()) {
        die("gputest: no CUDA device");
    }
    printf("gradient parity, CPU library vs GPU backend, soft spikes, batch of %llu real samples\n",
           (unsigned long long)GRAD_BATCH);
    printf("%-22s %14s %14s  %s\n", "case", "grad rel L2", "loss rel", "verdict");

    for (c = 0; c < sizeof(cases) / sizeof(cases[0]); ++c) {
        const gputest_case_t *tc = &cases[c];
        snn_size_t layers[5];
        snn_bptt_config_t cfg;
        snn_bptt_network_t *net = NULL;
        snn_bptt_workspace_t *ws = NULL;
        snn_bptt_grads_t *grads = NULL;
        bptt_gpu_t *gpu = NULL;
        char errbuf[256];
        float *cpu_g = NULL;
        float *gpu_g = NULL;
        float *p0 = NULL;
        float *frame = NULL;
        snn_size_t *indices = NULL;
        snn_size_t pc = 0;
        double cpu_loss = 0.0;
        float gpu_loss = 0.0f;
        double g_err = 0.0;
        double l_err = 0.0;
        int ok = 0;
        snn_size_t i = 0;
        size_t h = 0;

        layers[0] = MNIST_PIXELS;
        for (h = 0; h < tc->hidden_count; ++h) {
            layers[h + 1u] = tc->hidden[h];
        }
        layers[tc->hidden_count + 1u] = MNIST_CLASSES;
        cfg = snn_bptt_default_config(layers, tc->hidden_count + 2u, tc->timesteps);
        cfg.surrogate = tc->surrogate;
        cfg.surrogate_alpha = tc->alpha;
        cfg.detach_reset = tc->detach;
        cfg.weight_init_gain = 0.577f;
        cfg.seed = 1000u + (uint64_t)c;
        if (snn_bptt_network_create(&cfg, &net) != SNN_OK || snn_bptt_workspace_create(net, &ws) != SNN_OK ||
            snn_bptt_grads_create(net, &grads) != SNN_OK) {
            die("gputest: cpu setup failed");
        }
        snn_test_bptt_set_soft_spikes(net, 1);
        pc = snn_bptt_parameter_count(net);
        cpu_g = (float *)malloc((size_t)pc * sizeof(float));
        gpu_g = (float *)malloc((size_t)pc * sizeof(float));
        p0 = (float *)malloc((size_t)pc * sizeof(float));
        frame = (float *)malloc(MNIST_PIXELS * sizeof(float));
        indices = (snn_size_t *)malloc((size_t)GRAD_BATCH * sizeof(snn_size_t));
        if (cpu_g == NULL || gpu_g == NULL || p0 == NULL || frame == NULL || indices == NULL) {
            die("gputest: out of memory");
        }

        for (i = 0; i < GRAD_BATCH; ++i) {
            float loss = 0.0f;
            indices[i] = i;
            encode(train, i, frame);
            if (snn_bptt_forward_backward(net, ws, frame, 1, train->labels[i], grads, &loss, NULL) != SNN_OK) {
                die("gputest: cpu forward_backward failed");
            }
            cpu_loss += (double)loss;
        }
        if (snn_bptt_grads_copy_out(grads, cpu_g, pc) != SNN_OK ||
            snn_bptt_get_parameters(net, p0, pc) != SNN_OK) {
            die("gputest: cpu export failed");
        }

        gpu = bptt_gpu_create(layers, tc->hidden_count + 2u, tc->timesteps, cfg.beta, cfg.threshold, tc->surrogate,
                              tc->alpha, tc->detach, p0, pc, train->pixels, train->labels, SUBSET, test->pixels,
                              test->labels, SUBSET, GRAD_BATCH, GRAD_BATCH, 1e-3f, 0.9f, 0.999f, 1e-8f, errbuf,
                              sizeof(errbuf));
        if (gpu == NULL) {
            die(errbuf);
        }
        bptt_gpu_set_soft_spikes(gpu, 1);
        if (bptt_gpu_batch_grads(gpu, indices, GRAD_BATCH, gpu_g, &gpu_loss) != 0) {
            die(bptt_gpu_last_error(gpu));
        }

        g_err = rel_l2(cpu_g, gpu_g, pc);
        l_err = fabs(cpu_loss - (double)gpu_loss) / (fabs(cpu_loss) + 1e-30);
        ok = g_err < 5e-4 && l_err < 1e-4;
        failures += !ok;
        printf("%-22s %14.3e %14.3e  %s\n", tc->name, g_err, l_err, ok ? "pass" : "FAIL");

        bptt_gpu_destroy(gpu);
        free(indices);
        free(frame);
        free(p0);
        free(gpu_g);
        free(cpu_g);
        snn_bptt_grads_free(grads);
        snn_bptt_workspace_free(ws);
        snn_bptt_network_free(net);
    }

    /* Part 2: hard spikes, identical init and shuffle, two epochs. */
    {
        snn_size_t layers[4] = {MNIST_PIXELS, 32, 24, MNIST_CLASSES};
        snn_bptt_config_t cfg = snn_bptt_default_config(layers, 4, 7);
        snn_bptt_network_t *net = NULL;
        snn_bptt_optimizer_t *adam = NULL;
        bptt_gpu_t *gpu = NULL;
        pool_t pool;
        char errbuf[256];
        mnist_set_t train2 = *train;
        mnist_set_t test2 = *test;
        snn_size_t *order = NULL;
        uint64_t rng = 42u ^ UINT64_C(0x9e3779b97f4a7c15);
        uint64_t rng_gpu = rng;
        float cpu_train_loss = 0.0f;
        float gpu_train_loss = 0.0f;
        float cpu_acc = 0.0f;
        float gpu_acc = 0.0f;
        float l = 0.0f;
        double sp = 0.0;
        float *p0 = NULL;
        snn_size_t pc = 0;
        snn_size_t i = 0;
        int e = 0;
        int ok = 0;

        train2.count = train2.count < 2000u ? train2.count : 2000u;
        test2.count = test2.count < 1000u ? test2.count : 1000u;
        cfg.surrogate = SNN_SURROGATE_ATAN;
        cfg.surrogate_alpha = 1.0f;
        cfg.weight_init_gain = 0.577f;
        cfg.seed = 42u;
        if (snn_bptt_network_create(&cfg, &net) != SNN_OK ||
            snn_bptt_optimizer_create(net, 1e-3f, 0.9f, 0.999f, 1e-8f, &adam) != SNN_OK) {
            die("gputest: trajectory cpu setup failed");
        }
        pc = snn_bptt_parameter_count(net);
        p0 = (float *)malloc((size_t)pc * sizeof(float));
        order = (snn_size_t *)malloc((size_t)train2.count * sizeof(*order));
        if (p0 == NULL || order == NULL || snn_bptt_get_parameters(net, p0, pc) != SNN_OK) {
            die("gputest: trajectory export failed");
        }
        pool_create(&pool, net, opt->threads);
        for (i = 0; i < train2.count; ++i) {
            order[i] = i;
        }
        for (e = 0; e < 2; ++e) {
            shuffle(order, train2.count, &rng);
            cpu_train_loss = train_epoch(net, adam, &pool, &train2, order, 128u);
        }
        evaluate(net, &pool, &test2, &cpu_acc, &l, &sp);

        gpu = bptt_gpu_create(layers, 4, 7, cfg.beta, cfg.threshold, SNN_SURROGATE_ATAN, 1.0f, 0, p0, pc,
                              train->pixels, train->labels, train2.count, test->pixels, test->labels, test2.count,
                              128u, 500u, 1e-3f, 0.9f, 0.999f, 1e-8f, errbuf, sizeof(errbuf));
        if (gpu == NULL) {
            die(errbuf);
        }
        for (i = 0; i < train2.count; ++i) {
            order[i] = i;
        }
        for (e = 0; e < 2; ++e) {
            shuffle(order, train2.count, &rng_gpu);
            if (bptt_gpu_train_epoch(gpu, order, train2.count, &gpu_train_loss) != 0) {
                die(bptt_gpu_last_error(gpu));
            }
        }
        if (bptt_gpu_evaluate(gpu, 1, 0u, &gpu_acc, &l, &sp) != 0) {
            die(bptt_gpu_last_error(gpu));
        }

        ok = fabs((double)cpu_acc - (double)gpu_acc) <= 0.02 &&
             fabs((double)cpu_train_loss - (double)gpu_train_loss) / ((double)cpu_train_loss + 1e-30) <= 0.05;
        failures += !ok;
        printf("\nhard-spike trajectory, 784-32-24-10 T=7, 2 epochs on %llu samples, shared init and order\n",
               (unsigned long long)train2.count);
        printf("  train_loss  cpu %.4f  gpu %.4f\n", (double)cpu_train_loss, (double)gpu_train_loss);
        printf("  test_acc    cpu %.2f%%  gpu %.2f%%  %s\n", 100.0 * (double)cpu_acc, 100.0 * (double)gpu_acc,
               ok ? "pass" : "FAIL");

        bptt_gpu_destroy(gpu);
        free(order);
        free(p0);
        pool_free(&pool);
        snn_bptt_optimizer_free(adam);
        snn_bptt_network_free(net);
    }

    printf("\ngputest: %s\n", failures == 0 ? "all comparisons passed" : "FAILURES");
    return failures != 0;
#endif
}

/* ------------------------------------------------------------------ */
/* CLI                                                                 */
/* ------------------------------------------------------------------ */

/* "256" or "256,256,128": one width per hidden layer, at most MAX_HIDDEN_LAYERS. */
static void parse_hidden(const char *arg, options_t *opt) {
    const char *p = arg;
    opt->hidden_count = 0;
    for (;;) {
        char *end = NULL;
        const unsigned long long v = strtoull(p, &end, 10);
        if (end == p || v == 0ull) {
            die("bad --hidden list");
        }
        if (opt->hidden_count == MAX_HIDDEN_LAYERS) {
            die("too many hidden layers");
        }
        opt->hidden[opt->hidden_count++] = (snn_size_t)v;
        if (*end == '\0') {
            return;
        }
        if (*end != ',') {
            die("bad --hidden list");
        }
        p = end + 1;
    }
}

static snn_surrogate_t parse_surrogate(const char *name) {
    int i = 0;
    for (i = 0; i < (int)SNN_SURROGATE_COUNT; ++i) {
        if (strcmp(name, snn_surrogate_string((snn_surrogate_t)i)) == 0) {
            return (snn_surrogate_t)i;
        }
    }
    die("unknown surrogate name");
    return SNN_SURROGATE_COUNT;
}

static void usage(void) {
    printf("usage: mnist_bptt [options]\n"
           "  --data DIR         directory holding the four MNIST .gz files (default data/mnist)\n"
           "  --mode M           single | sweep | final | gputest    (default single)\n"
           "  --gpu              train on the CUDA backend (tool-level; see tools/bptt_cuda.h)\n"
           "  --hidden N[,N...]  hidden layer widths, comma-separated (default 256)\n"
           "  --timesteps T      unrolled steps (default 20)\n"
           "  --epochs E         epochs per run (default 3)\n"
           "  --train N          cap on training images, 0 = all (default 0)\n"
           "  --test N           cap on test images, 0 = all (default 0)\n"
           "  --train-eval N     also score the first N train images after each epoch (default 0 = off)\n"
           "  --batch B          minibatch size (default 128)\n"
           "  --lr LR            Adam learning rate (default 2e-3)\n"
           "  --beta B           membrane decay (default 0.95)\n"
           "  --threshold TH     spike threshold (default 1.0)\n"
           "  --gain G           weight init gain (default 0.577)\n"
           "  --surrogate NAME   fast_sigmoid|atan|sigmoid|triangle|gaussian|rectangular\n"
           "  --alpha A          surrogate width (default 2.0)\n"
           "  --detach           detach the reset path from the gradient\n"
           "  --seeds S          repeats per configuration (default 1)\n"
           "  --threads N        OpenMP threads (default: all)\n"
           "  --seed0 S          base RNG seed (default 1)\n"
           "  --csv PATH         append per-epoch records here\n"
           "  --tag T            label written into the csv's tag column (default: the mode)\n");
}

int main(int argc, char **argv) {
    options_t opt;
    mnist_set_t train;
    mnist_set_t test;
    FILE *csv = NULL;
    int i = 0;

    memset(&opt, 0, sizeof(opt));
    opt.data_dir = "data/mnist";
    opt.mode = "single";
    opt.tag = NULL;
    opt.hidden[0] = 256;
    opt.hidden_count = 1;
    opt.timesteps = 20;
    opt.epochs = 3;
    opt.batch = 128;
    opt.lr = 2e-3f;
    opt.beta = 0.95f;
    opt.threshold = 1.0f;
    opt.gain = 0.577f;
    opt.alpha = 2.0f;
    opt.surrogate = SNN_SURROGATE_ATAN;
    opt.seeds = 1;
    opt.seed0 = 1;
    opt.threads = 0;

    for (i = 1; i < argc; ++i) {
        const char *a = argv[i];
        const int has_next = i + 1 < argc;
#define NEXT() (has_next ? argv[++i] : (die("missing argument"), ""))
        if (strcmp(a, "--help") == 0) {
            usage();
            return 0;
        } else if (strcmp(a, "--data") == 0) {
            opt.data_dir = NEXT();
        } else if (strcmp(a, "--mode") == 0) {
            opt.mode = NEXT();
        } else if (strcmp(a, "--hidden") == 0) {
            parse_hidden(NEXT(), &opt);
        } else if (strcmp(a, "--timesteps") == 0) {
            opt.timesteps = strtoull(NEXT(), NULL, 10);
        } else if (strcmp(a, "--epochs") == 0) {
            opt.epochs = atoi(NEXT());
        } else if (strcmp(a, "--train") == 0) {
            opt.train_limit = strtoull(NEXT(), NULL, 10);
        } else if (strcmp(a, "--test") == 0) {
            opt.test_limit = strtoull(NEXT(), NULL, 10);
        } else if (strcmp(a, "--train-eval") == 0) {
            opt.train_eval = strtoull(NEXT(), NULL, 10);
        } else if (strcmp(a, "--batch") == 0) {
            opt.batch = strtoull(NEXT(), NULL, 10);
        } else if (strcmp(a, "--lr") == 0) {
            opt.lr = (float)atof(NEXT());
        } else if (strcmp(a, "--beta") == 0) {
            opt.beta = (float)atof(NEXT());
        } else if (strcmp(a, "--threshold") == 0) {
            opt.threshold = (float)atof(NEXT());
        } else if (strcmp(a, "--gain") == 0) {
            opt.gain = (float)atof(NEXT());
        } else if (strcmp(a, "--alpha") == 0) {
            opt.alpha = (float)atof(NEXT());
        } else if (strcmp(a, "--surrogate") == 0) {
            opt.surrogate = parse_surrogate(NEXT());
        } else if (strcmp(a, "--detach") == 0) {
            opt.detach_reset = 1;
        } else if (strcmp(a, "--gpu") == 0) {
            opt.gpu = 1;
        } else if (strcmp(a, "--seeds") == 0) {
            opt.seeds = atoi(NEXT());
        } else if (strcmp(a, "--threads") == 0) {
            opt.threads = atoi(NEXT());
        } else if (strcmp(a, "--seed0") == 0) {
            opt.seed0 = strtoull(NEXT(), NULL, 10);
        } else if (strcmp(a, "--csv") == 0) {
            opt.csv_path = NEXT();
        } else if (strcmp(a, "--tag") == 0) {
            opt.tag = NEXT();
        } else {
            fprintf(stderr, "unknown option %s\n", a);
            usage();
            return 2;
        }
#undef NEXT
    }

#ifdef _OPENMP
    /* Without this the runtime may hand a parallel region fewer threads than
     * omp_get_max_threads(), which changes which samples a thread sums and so
     * the last bits of the reported loss. The gradient reduction is already
     * order-fixed, but the run would not be bit-reproducible. */
    omp_set_dynamic(0);
    if (opt.threads > 0) {
        omp_set_num_threads(opt.threads);
    }
    opt.threads = omp_get_max_threads();
#else
    opt.threads = 1;
#endif

    if (opt.tag == NULL) {
        opt.tag = opt.mode;
    }

    mnist_load(opt.data_dir, "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", opt.train_limit, &train);
    mnist_load(opt.data_dir, "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", opt.test_limit, &test);

    if (opt.csv_path != NULL) {
        FILE *probe = fopen(opt.csv_path, "r");
        const int fresh = probe == NULL;
        if (probe != NULL) {
            fclose(probe);
        }
        csv = fopen(opt.csv_path, "a");
        if (csv == NULL) {
            die("cannot open csv");
        }
        if (fresh) {
            fprintf(csv, "tag,surrogate,alpha,seed,epoch,train_loss,test_loss,test_acc,firing_rate,seconds,"
                         "train_eval_loss,train_eval_acc\n");
        }
    }

    {
        char arch[128];
        size_t off = (size_t)snprintf(arch, sizeof(arch), "784");
        size_t h = 0;
        for (h = 0; h < opt.hidden_count && off < sizeof(arch); ++h) {
            off += (size_t)snprintf(arch + off, sizeof(arch) - off, "-%llu", (unsigned long long)opt.hidden[h]);
        }
        if (off < sizeof(arch)) {
            snprintf(arch + off, sizeof(arch) - off, "-10");
        }
        printf("mnist_bptt: %llu train, %llu test, %s, T=%llu, beta=%g, thr=%g, gain=%g, lr=%g, batch=%llu, "
               "threads=%d, detach_reset=%d\n",
               (unsigned long long)train.count, (unsigned long long)test.count, arch,
               (unsigned long long)opt.timesteps, (double)opt.beta, (double)opt.threshold, (double)opt.gain,
               (double)opt.lr, (unsigned long long)opt.batch, opt.threads, opt.detach_reset);
    }

    if (strcmp(opt.mode, "gputest") == 0) {
        const int rc = run_gputest(&opt, &train, &test);
        mnist_free(&train);
        mnist_free(&test);
        if (csv != NULL) {
            fclose(csv);
        }
        return rc;
    }
    if (strcmp(opt.mode, "single") == 0) {
        int s = 0;
        for (s = 0; s < opt.seeds; ++s) {
            run_result_t r;
            printf("surrogate=%s alpha=%g seed=%llu\n", snn_surrogate_string(opt.surrogate), (double)opt.alpha,
                   (unsigned long long)(opt.seed0 + (uint64_t)s));
            r = train_one(&opt, &train, &test, opt.surrogate, opt.alpha, opt.seed0 + (uint64_t)s, csv, opt.tag, 1);
            printf("  => best %.2f%%  final %.2f%%  firing %.3f  %.1fs\n", 100.0 * (double)r.best_accuracy,
                   100.0 * (double)r.final_accuracy, (double)r.firing_rate, r.seconds);
        }
    } else if (strcmp(opt.mode, "sweep") == 0) {
        static const float alphas[] = {0.5f, 1.0f, 2.0f, 5.0f, 10.0f, 25.0f};
        const int alpha_count = (int)(sizeof(alphas) / sizeof(alphas[0]));
        int su = 0;
        int ai = 0;
        int s = 0;
        printf("\n%-13s %7s %8s %8s %8s %8s\n", "surrogate", "alpha", "acc_mean", "acc_std", "acc_e1", "firing");
        for (su = 0; su < (int)SNN_SURROGATE_COUNT; ++su) {
            for (ai = 0; ai < alpha_count; ++ai) {
                double sum = 0.0;
                double sumsq = 0.0;
                double e1 = 0.0;
                double fire = 0.0;
                for (s = 0; s < opt.seeds; ++s) {
                    const run_result_t r = train_one(&opt, &train, &test, (snn_surrogate_t)su, alphas[ai],
                                                     opt.seed0 + (uint64_t)s, csv, opt.tag, 0);
                    sum += (double)r.final_accuracy;
                    sumsq += (double)r.final_accuracy * (double)r.final_accuracy;
                    e1 += (double)r.epoch1_accuracy;
                    fire += (double)r.firing_rate;
                }
                {
                    const double mean = sum / opt.seeds;
                    const double var = sumsq / opt.seeds - mean * mean;
                    printf("%-13s %7g %7.2f%% %7.2f%% %7.2f%% %8.3f\n", snn_surrogate_string((snn_surrogate_t)su),
                           (double)alphas[ai], 100.0 * mean, 100.0 * sqrt(var > 0.0 ? var : 0.0),
                           100.0 * e1 / opt.seeds, fire / opt.seeds);
                    fflush(stdout);
                }
            }
        }
    } else if (strcmp(opt.mode, "final") == 0) {
        int su = 0;
        int s = 0;
        printf("\n%-13s %8s %8s %8s %8s %8s\n", "surrogate", "acc_mean", "acc_std", "acc_best", "firing", "sec/ep");
        for (su = 0; su < (int)SNN_SURROGATE_COUNT; ++su) {
            double sum = 0.0;
            double sumsq = 0.0;
            double best = 0.0;
            double fire = 0.0;
            double secs = 0.0;
            for (s = 0; s < opt.seeds; ++s) {
                const run_result_t r = train_one(&opt, &train, &test, (snn_surrogate_t)su, opt.alpha,
                                                 opt.seed0 + (uint64_t)s, csv, opt.tag, 0);
                sum += (double)r.final_accuracy;
                sumsq += (double)r.final_accuracy * (double)r.final_accuracy;
                fire += (double)r.firing_rate;
                secs += r.seconds / opt.epochs;
                if ((double)r.best_accuracy > best) {
                    best = (double)r.best_accuracy;
                }
            }
            {
                const double mean = sum / opt.seeds;
                const double var = sumsq / opt.seeds - mean * mean;
                printf("%-13s %7.2f%% %7.2f%% %7.2f%% %8.3f %8.1f\n", snn_surrogate_string((snn_surrogate_t)su),
                       100.0 * mean, 100.0 * sqrt(var > 0.0 ? var : 0.0), 100.0 * best, fire / opt.seeds,
                       secs / opt.seeds);
                fflush(stdout);
            }
        }
    } else {
        die("unknown mode");
    }

    if (csv != NULL) {
        fclose(csv);
    }
    mnist_free(&train);
    mnist_free(&test);
    return 0;
}
