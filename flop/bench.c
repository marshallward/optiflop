#include <pthread.h>    /* pthread_* */
#include <stdlib.h>     /* malloc, free */

#include "bench.h"

/* Thread control */
pthread_barrier_t timer_barrier;
pthread_mutex_t runtime_mutex;
volatile int runtime_flag;

void * bench_thread(void *arg)
{
    thread_arg_t *tinfo;
    bench_arg_t *bench_args;

    tinfo = (thread_arg_t *) arg;
    bench_args = malloc(sizeof(bench_arg_t));

    /* Set inputs */
    bench_args->min_runtime = tinfo->min_runtime;

    (*(tinfo->bench))(bench_args);

    /* Save output */
    tinfo->runtime = bench_args->runtime;
    tinfo->flops = bench_args->flops;

    free(bench_args);

    pthread_exit(NULL);
}
