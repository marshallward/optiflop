#include <stdio.h>
#include <stdlib.h>
#include <time.h> /* timespec, clock_gettime */

#define BYTEALIGN 32

double axpy(float, float, float *, float *, int, int);
void dummy(float, float, float *, float *);

int main(int argc, char **argv)
{
    float *x, *y;
    float a, b;

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
    b = 3.;
    for (i = 0; i < n; i++) {
        x[i] = 1.;
        y[i] = 2.;
    }

    /* a x + y */

    // Warmup
    // rt = axpy(a, b, x, y, n, 1);

    rt = axpy(a, b, x, y, n, r);

    // TODO: Do the "vector triad" version:
    //  1. (x) Do iterations inside `axpy`, get absolute number
    //  2. Replace `rt`, don't append the time increases
    //  3. Step up `r` exponentially

    //rt = 0.;
    //do {
    //    rt += axpy(a, b, x, y, n, 1);
    //    r++;
    //    // rt = axpy(a, b, x, y, n, r);
    //    // r = 2 * r;
    //} while(rt < 0.2);

    printf("mean axpy time: %.12f\n", rt / r);
    printf("GFLOP/sec estimate: %.12f\n", (double) 2. * n * r / rt / 1e9);

    return 0;
}


double axpy(float a, float b, float *x, float *y, int n, int r_max)
{
    __builtin_assume_aligned(x, BYTEALIGN);
    __builtin_assume_aligned(y, BYTEALIGN);

    int i, r;
    struct timespec ts_start, ts_end;
    float runtime;

    int midpt = n / 2;
    float sum;

    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (r = 0; r < r_max; r++) {
        for (i = 0; i < n; i++)
            //y[i] = a * y[i];
            //y[i] = y[i] + y[i];
            //y[i] = x[i] + y[i];
            y[i] = a * x[i] + y[i];
            //y[i] = a * x[i] + b * y[i];
        // To prevent outer loop removal during optimisation
        if (y[midpt] < 0.) dummy(a, b, x, y);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);

    runtime = (double) (ts_end.tv_sec - ts_start.tv_sec)
        + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    return runtime;
}
