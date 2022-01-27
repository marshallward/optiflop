#include <immintrin.h>  /* __m256, _m256_* */
#include <pthread.h>    /* pthread_* */
#include <stdint.h>     /* uint64_t */

#include "roof.h"
#include "avx.h"
#include "stopwatch.h"

/* TODO: Make this dynamic */
#define VADDPS_LATENCY 3
#define VMULPS_LATENCY 3

typedef void (*loop_kernel)(int, __m256[], __m256[]);

static inline void avx_add_kernel(int, __m256[], __m256[])
    __attribute__((always_inline));
static inline void avx_mul_kernel(int, __m256[], __m256[])
    __attribute__((always_inline));
static inline void avx_mac_kernel(int, __m256[], __m256[])
    __attribute__((always_inline));

static float sum_avx(__m256);


void avx_test(struct roof_args *args, loop_kernel loop)
{
    const int n_avx = 32 / sizeof(float);   // Values per SIMD register
    const int n_reg = VADDPS_LATENCY;       // Number of loop-unrolled stages
    __m256 reg1[n_reg];
    __m256 reg2[n_reg];

    long r_max;
    double runtime;
    Stopwatch *t;

    /* Declare as volatile to prevent removal during optimisation */
    volatile float result __attribute__((unused));

    t = args->timer;

    for (int j = 0; j < n_reg; j++) {
        reg1[j] = _mm256_set1_ps((float) j);
        reg2[j] = _mm256_set1_ps((float) j);
    }

    r_max = 1;
    *(args->runtime_flag) = 0;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (long r = 0; r < r_max; r++) {
            /* Intel icc requires an explicit unroll */
            #ifdef __ICC
            #pragma unroll(n_reg)
            #endif
            for (int i = 0; i < n_reg; i++)
                loop(i, reg1, reg2);

            // Create an impossible branch to prevent loop interchange
            //if (reg[0] < 0.) dummy(a, b, x, y);
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > (args->min_runtime)) {
            pthread_mutex_lock(args->mutex);
            *(args->runtime_flag) = 1;
            pthread_mutex_unlock(args->mutex);
        }

        pthread_barrier_wait(args->barrier);
        if (! *(args->runtime_flag)) r_max *= 2;

    } while (! *(args->runtime_flag));

    for (int j = 0; j < n_reg; j++) {
        reg1[0] = _mm256_add_ps(reg1[0], reg1[j]);
        reg2[0] = _mm256_add_ps(reg2[0], reg2[j]);
    }
    result = sum_avx(reg1[0]);
    /* TODO: Validate the result */

    args->runtime = runtime;
    args->flops = args->kflops * r_max * n_avx * n_reg / runtime;
    args->bw_load = 0.;
    args->bw_store = 0.;
}


void avx_add_kernel(int i, __m256 r1[], __m256 r2[])
{
    const __m256 c = _mm256_set1_ps((float) 1e-6);
    r1[i] = _mm256_add_ps(r1[i], c);
}


void simd_avx_add(struct roof_args *args)
{
    args->kflops = 1;
    avx_test(args, avx_add_kernel);
}


void avx_mul_kernel(int i, __m256 r1[], __m256 r2[])
{
    const __m256 a = _mm256_set1_ps((float) (1. + 1e-6));
    r1[i] = _mm256_mul_ps(r1[i], a);
}


void simd_avx_mul(struct roof_args *args)
{
    args->kflops = 1;
    avx_test(args, avx_mul_kernel);
}


void avx_mac_kernel(int i, __m256 r1[], __m256 r2[])
{
    const __m256 a = _mm256_set1_ps((float) (1. + 1e-6));
    const __m256 c = _mm256_set1_ps((float) 1e-6);
    r1[i] = _mm256_add_ps(r1[i], c);
    r2[i] = _mm256_mul_ps(r2[i], a);
}


void simd_avx_mac(struct roof_args *args)
{
    args->kflops = 2;
    avx_test(args, avx_mac_kernel);
}


float sum_avx(__m256 x) {
    const int n_avx = 32 / sizeof(float);
    union vec {
        __m256 reg;
        float val[n_avx];
    } v;
    float result = 0;
    int i;

    v.reg = x;
    for (i = 0; i < n_avx; i++)
        result += v.val[i];

    return result;
}
