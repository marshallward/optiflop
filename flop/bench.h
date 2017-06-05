#ifndef FLOP_H_
#define FLOP_H_

#include <pthread.h>

/* Types */

/* TODO: Create a type here, or reduce the number of arguments */
typedef void * (*bench_ptr_t) (void *);

typedef double (*roof_ptr_t) (float, float, float *, float *,
                              int, double *, double);

struct thread_args {
    /* Input */
    int tid;
    double min_runtime;
    roof_ptr_t roof;

    /* Output */
    double runtime;
    double flops;
    double bw_load;
    double bw_store;
};

/* Declarations */
extern pthread_barrier_t timer_barrier;
extern pthread_mutex_t runtime_mutex;
extern volatile int runtime_flag;
#endif
