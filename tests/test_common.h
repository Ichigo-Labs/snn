#ifndef SNN_TEST_COMMON_H
#define SNN_TEST_COMMON_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

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

#endif /* SNN_TEST_COMMON_H */
