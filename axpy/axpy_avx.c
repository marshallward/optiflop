#include <immintrin.h>  /* __m256, _m256_* */
#include <stdio.h>
#include <stdlib.h>
#include <time.h> /* timespec, clock_gettime */

#define BYTEALIGN 32

float axpy(float, float *, float *, int, int);
void dummy(float, float *, float *);

int main(int argc, char **argv)
{
    float *x, *y;
    float a;

    float rt; // runtime

    int n;  // Vector length
    int r;  // Repeat counter
    int i;  // Loop counter

    // TODO: Safe argv parsing
    n = atoi(argv[1]);
    r = atoi(argv[2]);

    posix_memalign((void *) &x, BYTEALIGN, n * sizeof(float));
    posix_memalign((void *) &y, BYTEALIGN, n * sizeof(float));
    
    // x = _mm_malloc(n * sizeof(float), BYTEALIGN);
    // y = _mm_malloc(n * sizeof(float), BYTEALIGN);

    a = 2.;
    for (i = 0; i < n; i++) {
        x[i] = 1.;
        y[i] = 2.;
    }

    /* a x + y */

    // Warmup
    // rt = axpy(a, x, y, n, 1);

    rt = axpy(a, x, y, n, r);

    printf("mean axpy time: %.12f\n", rt / r);
    printf("GFLOP/sec estimate: %.12f\n", 2. * n * r / rt / 1e9);

    return 0;
}


float axpy(float a, float *x, float *y, int n, int r_max)
{
    /* Is GCC ignoring this? */
    __builtin_assume_aligned(x, BYTEALIGN);
    __builtin_assume_aligned(y, BYTEALIGN);

    int i, r;
    struct timespec ts_start, ts_end;
    float runtime;

    int midpt = n / 2;
    float sum;

    // AVX test
    __m256 rx1, rx2, ry1, ry2;

    ry1 = _mm256_load_ps(&y[0]);
    ry2 = _mm256_load_ps(&y[8]);
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (r = 0; r < r_max; r++) {
        for (i = 0; i < n; i = i + 16) {
            rx1 = _mm256_load_ps(&x[i]);
            rx2 = _mm256_load_ps(&x[i + 8]);
            // ry = _mm256_load_ps(&y[i]);

            ry1 = _mm256_add_ps(rx1, ry1);
            ry2 = _mm256_add_ps(rx2, ry2);

            //_mm256_store_ps(&x[i], rx);
            _mm256_store_ps(&y[i], ry1);
            _mm256_store_ps(&y[i + 8], ry2);
        }

        // To prevent outer loop removal during optimisation
        if (y[midpt] < 0.) dummy(a, x, y);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);

    runtime = (float) (ts_end.tv_sec - ts_start.tv_sec)
        + (float) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    return runtime;
}
