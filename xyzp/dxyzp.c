#include <stdio.h>
#include <stdlib.h>
#include <time.h> /* timespec, clock_gettime */

#define BYTEALIGN 32

double xyzp(double, double, double, double *, double *, double *, int, int);
void dummy(double, double *, double *, double *);

int main(int argc, char **argv)
{
    double *x, *y, *z;
    double a, b, c;

    double rt; // runtime

    int n;  // Vector length
    int r;  // Repeat counter
    int i;  // Loop counter

    // TODO: Safe argv parsing
    n = atoi(argv[1]);
    r = atoi(argv[2]);

    posix_memalign((void *) &x, BYTEALIGN, n * sizeof(double));
    posix_memalign((void *) &y, BYTEALIGN, n * sizeof(double));
    posix_memalign((void *) &z, BYTEALIGN, n * sizeof(double));

    // x = _mm_malloc(n * sizeof(double), BYTEALIGN);
    // y = _mm_malloc(n * sizeof(double), BYTEALIGN);

    a = 2.;
    b = 3.;
    c = 4.;
    for (i = 0; i < n; i++) {
        x[i] = 1.;
        y[i] = 2.;
        z[i] = 3.;
    }

    /* a x + y */

    // Warmup
    // rt = xyzp(a, x, y, n, 1);

    rt = xyzp(a, b, c, x, y, z, n, r);

    // TODO: Do the "vector triad" version:
    //  1. (x) Do iterations inside `xyzp`, get absolute number
    //  2. Replace `rt`, don't append the time increases
    //  3. Step up `r` exponentially

    //rt = 0.;
    //do {
    //    rt += xyzp(a, x, y, n, 1);
    //    r++;
    //    // rt = xyzp(a, x, y, n, r);
    //    // r = 2 * r;
    //} while(rt < 0.2);

    printf("mean xyzp time: %.12f\n", rt / r);
    printf("GFLOP/sec estimate: %.12f\n", (double) 2. * n * r / rt / 1e9);

    return 0;
}


double xyzp(double a, double b, double c, double *x, double *y, double *z,
            int n, int r_max)
{
    __builtin_assume_aligned(x, BYTEALIGN);
    __builtin_assume_aligned(y, BYTEALIGN);
    __builtin_assume_aligned(z, BYTEALIGN);

    int i, r;
    struct timespec ts_start, ts_end;
    double runtime;

    int midpt = n / 2;
    double sum;

    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_start);
    for (r = 0; r < r_max; r++) {
        //#pragma unroll(8)
        for (i = 0; i < n; i++)
            //z[i] = x[i] + y[i] + z[i];
            //z[i] = x[i] * y[i] + z[i];
            //z[i] = a * x[i] + y[i] + z[i];
            //z[i] = a * x[i] + b * y[i] + z[i];
            //z[i] = a * x[i] + b * y[i] + c * z[i];
            //z[i] = a * x[i] + b * y[i] + c;
            z[i] = 0.25 * x[i] * y[i];
            //z[i] = 0.25 + x[i] + y[i];
            //z[i] = a * x[i] * y[i];
            //z[i] = a + x[i] + y[i];
        // To prevent outer loop removal during optimisation
        if (y[midpt] < 0.) dummy(a, x, y, z);
    }
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts_end);

    runtime = (double) (ts_end.tv_sec - ts_start.tv_sec)
        + (double) (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;

    return runtime;
}
