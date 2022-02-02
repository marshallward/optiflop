#include <stdlib.h>     /* malloc, posix_memalign, free */
#include <cblas.h>      /* Cblas*, cblas_dgemm */

#include "roof.h"      /* thread_args */
#include "stopwatch.h"  /* Stopwatch */

static inline void dgemm_kernel(int n, double *x, double *y)
    __attribute__((always_inline));


void dgemm(int n, double a, double b,
           double * restrict x_in, double * restrict y_in,
           struct roof_args *args)
{
    /* Timer config */
    Stopwatch *timer = args->timer;
    double runtime;

    /* dgemm inputs */
    double *x, *y;

    /* We don't use any of the input arguments here!  Make new ones */
    /* TODO: Yes, this indicates bad design... */

    /* Allocate matrix as a 1d array */
    /* NOTE: Initialize to NULL to prevent warnings */
    x = NULL;
    y = NULL;

    posix_memalign((void *) &x, BYTEALIGN, n * n * sizeof(double));
    posix_memalign((void *) &y, BYTEALIGN, n * n * sizeof(double));

    /* Initialize to the identity matrix */
    /* (... and pray it's not optimized out) */
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            x[i*n+j] = (i == j ? 1. : 0.);
            y[i*n+j] = (i == j ? 1. : 0.);
        }
    }

    long r_max = 1;
    *(args->runtime_flag) = 0;

    do {
        pthread_barrier_wait(args->barrier);
        timer->start(timer);
        for (long r = 0; r < r_max; r++) {
            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                n, n, n, 1., x, n, y, n, 0., x, n);
            /* Prevent loop interchange */
            //if (y[0] < 0.) dummy(a, b, x, y);
        }
        timer->stop(timer);
        runtime = timer->runtime(timer);

        if (runtime > (args->min_runtime)) {
            /* Set runtime flag if any thread exceeds runtime limit */
            pthread_mutex_lock(args->mutex);
            *(args->runtime_flag) = 1;
            pthread_mutex_unlock(args->mutex);
        } else {
            /* Otherwise, extend the loop iteration */
            r_max *= 2;
        }
        pthread_barrier_wait(args->barrier);

    } while (!*(args->runtime_flag));

    free(x);
    free(y);

    /* Thread output */
    args->runtime = runtime;
    args->flops = r_max * 2. * n * n * n / runtime;
    args->bw_load = r_max * 2. * n * n / runtime;
    args->bw_store = r_max * n * n / runtime;
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
