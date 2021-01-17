#ifndef FLOP_BENCH_H_
#define FLOP_BENCH_H_

#include <pthread.h>
#include "roof.h"
#include "stopwatch.h"


typedef void * (*bench_ptr_t) (void *);


struct thread_args {
    /* Input */
    int tid;
    int vlen;
    roof_ptr_t roof;    /* TODO: Phase this out! */

    double min_runtime;
    enum stopwatch_backend timer_type;
    pthread_mutex_t *mutex;
    pthread_barrier_t *barrier;
    volatile int *runtime_flag;

    /* Output */
    double runtime;
    double flops;
    double bw_load;
    double bw_store;
};


/* Types */
struct task {
    /* Simple descriptive name of test */
    char *name;

    /* Thread subroutine */
    union {
        bench_ptr_t simd;
        roof_ptr_t roof;
    } thread;

    /* Thread arguments */
    union {
        struct thread_args *simd;
        struct roof_args   *roof;
    } args;
};

#endif  // FLOP_BENCH_H_
