#include <stdlib.h>     /* malloc, posix_memalign, free */

#include "bench.h"      /* thread_args */
#include "stopwatch.h"  /* Stopwatch */


void * simd_thread(void *args_in)
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
    } while (timer->runtime(timer) < 0.02);

    /* Currently identical to roof_args!  Perhaps not always though */
    struct roof_args *kargs;
    kargs = malloc(sizeof(struct roof_args));

    kargs->timer = timer;
    kargs->min_runtime = args->min_runtime;

    kargs->mutex = args->mutex;
    kargs->barrier = args->barrier;
    kargs->runtime_flag = args->runtime_flag;

    (*(args->benchmark).simd)(kargs);

    args->runtime = kargs->runtime;
    args->flops = kargs->flops;
    args->bw_load = kargs->bw_load;
    args->bw_store = kargs->bw_store;

    timer->destroy(timer);
    free(kargs);

    pthread_exit(NULL);
}
