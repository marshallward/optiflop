#include <stdlib.h>     /* malloc, posix_memalign, free */

#include "roof.h"      /* thread_args */
#include "stopwatch.h"  /* Stopwatch */

static inline void dgemm_kernel(int n, double *x, double *y)
    __attribute__((always_inline));


void * dgemm(void *args_in)
{
    /* Thread input */
    struct thread_args *args;
    args = (struct thread_args *) args_in;

    Stopwatch *timer;
    timer = stopwatch_create(args->timer_type);
    double runtime;

    /* Boo! get rid of these! */
    double a, b;

    /* Rev up the core */
    volatile int v = 0;
    unsigned long iter = 1;
    do {
        timer->start(timer);
        for (unsigned long i = 0; i < iter; i++)
            v++;
        timer->stop(timer);
        iter *= 2;
    } while (timer->runtime(timer) < 0.01);

    /* Benchmark inputs */
    int n;
    double *x, *y;

    /* Initialize to NULL to prevent initialization warnings */
    n = args->vlen;

    /* Allocate matrix as a 1d array */
    x = NULL;
    posix_memalign((void *) &x, BYTEALIGN, n * n * sizeof(double));
    y = NULL;
    posix_memalign((void *) &y, BYTEALIGN, n * n * sizeof(double));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            x[i*n+j] = (i == j ? 1. : 0.);
            y[i*n+j] = (i == j ? 1. : 0.);
        }
    }

    /* Normally the test would be called here, but we implement it here. */
    long r_max = 1;
    *(args->runtime_flag) = 0;
    do {
        pthread_barrier_wait(args->barrier);
        timer->start(timer);
        for (long r = 0; r < r_max; r++) {
            dgemm_kernel(n, x, y);
            if (y[0] < 0.) dummy(a, b, x, y);
        }
        timer->stop(timer);
        runtime = timer->runtime(timer);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > (args->min_runtime)) {
            pthread_mutex_lock(args->mutex);
            *(args->runtime_flag) = 1;
            pthread_mutex_unlock(args->mutex);
        } else {
            r_max *= 2;
        }
        pthread_barrier_wait(args->barrier);

    } while (!*(args->runtime_flag));

    args->runtime = runtime;
    args->flops = r_max * 2. * n * n * n / runtime;
    args->bw_load = r_max * 2. * n * n / runtime;
    args->bw_store = r_max * n * n / runtime;

    timer->destroy(timer);
    free(x);
    free(y);

    pthread_exit(NULL);
}


void dgemm_kernel(int n, double *x, double *y) {
    double sum;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            sum = 0.;
            for (int k = 0; k < n; k++) {
                sum += x[i*n+k] * y[k+j*n];
            }
            y[i*n+j] = sum;
        }
    }
}
