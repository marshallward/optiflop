#include <stdlib.h>     /* malloc, posix_memalign, free */

#include "roof.h"       /* roof_args */
#include "bench.h"      /* thread_args */
#include "stopwatch.h"  /* Stopwatch */


void * roof_thread(void *args_in)
{
    /* Thread input */
    struct thread_args *args;
    args = (struct thread_args *) args_in;

    Stopwatch *timer;
    timer = stopwatch_create(args->timer_type);

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
    float a, b;
    float *x, *y;

    /* Compilers cannot infer that posix_memalign() initializes x and y and may
     * raise a warning, so we explicitly initalize them here. */
    x = NULL;
    y = NULL;

    n = args->vlen;
    posix_memalign((void *) &x, BYTEALIGN, n * sizeof(float));
    posix_memalign((void *) &y, BYTEALIGN, n * sizeof(float));

    a = 2.;
    b = 3.;
    for (int i = 0; i < n; i++) {
        x[i] = 1.;
        y[i] = 2.;
    }

    struct roof_args *rargs;
    rargs = malloc(sizeof(struct roof_args));

    rargs->timer = timer;
    rargs->min_runtime = args->min_runtime;

    rargs->mutex = args->mutex;
    rargs->barrier = args->barrier;
    rargs->runtime_flag = args->runtime_flag;

    (*(args->benchmark).roof)(n, a, b, x, y, rargs);

    args->runtime = rargs->runtime;
    args->flops = rargs->flops;
    args->bw_load = rargs->bw_load;
    args->bw_store = rargs->bw_store;

    timer->destroy(timer);
    free(x);
    free(y);
    free(rargs);

    pthread_exit(NULL);
}
