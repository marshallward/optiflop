#include <pthread.h>    /* pthread_* */

#include "bench.h"


void * avx512_add(void *args_in)
{
    struct thread_args *args;

    args = (struct thread_args *) args_in;

    args->runtime = 0.;
    args->flops = 0.;
    args->bw_load = 0.;
    args->bw_store = 0.;

    pthread_exit(NULL);
}


void * avx512_fma(void *args_in)
{
    struct thread_args *args;

    args = (struct thread_args *) args_in;

    args->runtime = 0.;
    args->flops = 0.;
    args->bw_load = 0.;
    args->bw_store = 0.;

    pthread_exit(NULL);
}
