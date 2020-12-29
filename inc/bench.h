#ifndef FLOP_BENCH_H_
#define FLOP_BENCH_H_

#include <pthread.h>
#include "roof.h"
#include "stopwatch.h"

/* Types */
struct microbench {
    char *name;
    void * (*thread) (void *);
};


/* TODO: Create a type here, or reduce the number of arguments */
typedef void * (*bench_ptr_t) (void *);


struct thread_args {
    /* Input */
    int tid;
    int vlen;
    roof_ptr_t roof;

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

#endif  // FLOP_BENCH_H_
