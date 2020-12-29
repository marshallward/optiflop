#ifndef FLOP_BENCH_H_
#define FLOP_BENCH_H_

#include <pthread.h>
#include "stopwatch.h"

/* Types */

struct microbench {
    char *name;
    void * (*thread) (void *);
};

/* TODO: Create a type here, or reduce the number of arguments */
typedef void * (*bench_ptr_t) (void *);

struct roof_args {
    /* Config */
    float min_runtime;
    enum stopwatch_backend timer_type;
    pthread_mutex_t *mutex;
    pthread_barrier_t *barrier;

    /* Fields */
    int n;
    float a;
    float b;
    float *x;
    float *y;

    /* Output */
    double runtime;
    double flops;
    double bw_load;
    double bw_store;
};

typedef void (*roof_ptr_t) (int, float, float, float *, float *,
                            struct roof_args *);

typedef void (*kernel_ptr_t) (int, float, float, float *, float *);

struct thread_args {
    /* Input */
    int tid;
    int vlen;
    roof_ptr_t roof;

    double min_runtime;
    enum stopwatch_backend timer_type;
    pthread_mutex_t *mutex;
    pthread_barrier_t *barrier;

    /* Output */
    double runtime;
    double flops;
    double bw_load;
    double bw_store;
};

/* Declarations */
extern volatile int runtime_flag;
#endif  // FLOP_BENCH_H_
