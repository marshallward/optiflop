#ifndef FLOP_H_
#define FLOP_H_

#include <pthread.h>

/* Types */

/* TODO: Create a type here, or reduce the number of arguments */
typedef double (*roof_ptr_t) (float, float, float *, float *,
                              int, double *, double);

typedef struct _thread_arg_t {
    /* Input */
    int tid;
    double min_runtime;
    roof_ptr_t roof;

    /* Output */
    double runtime;
    double flops;
    double bw_load;
    double bw_store;
} thread_arg_t;

typedef void * (*bench_ptr_t) (void *);

/* Declarations */
extern pthread_barrier_t timer_barrier;
extern pthread_mutex_t runtime_mutex;
extern volatile int runtime_flag;

void * bench_thread(void *);
#endif
