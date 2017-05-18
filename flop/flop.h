#ifndef FLOP_H_
#define FLOP_H_

typedef void (*bench_ptr_t) (double *, double *);

typedef struct _thread_arg_t {

    /* Global input */
    int tid;
    double min_runtime;
    bench_ptr_t bench;

    /* Thread input */

    /* Output */
    double runtime;
    double flops;
} thread_arg_t;

#endif
