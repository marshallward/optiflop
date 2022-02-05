#include <pthread.h>    /* pthread_* */
#include <mkl.h>

#include "roof.h"       /* roof_args */
#include "stopwatch.h"  /* Stopwatch */


void roof_daxpy_blas(int n, double a, double b,
                     double * restrict x_in, double * restrict y_in,
                     struct roof_args *args)
{
    double *x, *y;

    Stopwatch *t;
    long r, r_max;
    double runtime;

    /* If possible, assert alignment of x_in and y_in */
    x = ASSUME_ALIGNED(x_in);
    y = ASSUME_ALIGNED(y_in);

    t = args->timer;

    r_max = 1;
    *(args->runtime_flag) = 0;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (r = 0; r < r_max; r++) {
            cblas_daxpy(n, a, x, 1, y, 1);
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

    /* Total number of kernel calls */
    args->runtime = runtime;
    args->flops = 2 * n * r_max / runtime;
    args->bw_load = n * sizeof(double) * r_max / runtime;
    args->bw_store = n * sizeof(double) * r_max / runtime;
}
