#include <stdio.h>
#include <stdlib.h>
#include <time.h> /* timespec, clock_gettime */

#include "timer.h"

#define BYTEALIGN 32

double axpy(float, float, float *, float *, int, double *);
void dummy(float, float, float *, float *);

void axpy_main(double *runtime, double *flops)
{
    float *x, *y;
    float a, b;

    float rt; // runtime

    int n;  // Vector length
    int i;  // Loop counter

    n = 3200;

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

    rt = axpy(a, b, x, y, n, flops);

    *runtime = rt;
}


double axpy(float a, float b, float * x_in, float * y_in,
            int n, double *flops)
{
    float *x, *y;
    int midpt = n / 2;
    float sum;

    Timer *t;
    float runtime;
    int runtime_flag;

    int i, r;
    int r_max;

    // TODO: Create a macro somewhere else
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 6)
    x = __builtin_assume_aligned(x_in, BYTEALIGN);
    y = __builtin_assume_aligned(y_in, BYTEALIGN);
#else
    x = x_in; y = y_in;
#endif

    t = mtimer_create(TIMER_POSIX);

    r_max = 1000;
    runtime_flag = 0;
    do {
        r_max *= 2;
        t->start(t);
        for (r = 0; r < r_max; r++) {
            for (i = 0; i < n; i++)
                //y[i] = a * y[i];
                //y[i] = y[i] + y[i];
                //y[i] = x[i] + y[i];
                y[i] = a + x[i] + y[i];
                //y[i] = a * x[i] + y[i];
                //y[i] = a * x[i] * y[i];
                //y[i] = a * x[i] + b * y[i];
            // To prevent outer loop removal during optimisation
            if (y[midpt] < 0.) dummy(a, b, x, y);
        }
        t->stop(t);
        runtime = t->runtime(t);

        if (runtime > 0.5) runtime_flag = 1;
    } while (!runtime_flag);

    *flops = 2. * n * r_max / runtime;
    return runtime;
}
