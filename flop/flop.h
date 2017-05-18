#ifndef FLOP_H_
#define FLOP_H_

typedef struct _bench_arg_t {
    /* Output */
    double runtime;
    double flops;
} bench_arg_t;

//typedef void (*bench_ptr_t) (double *, double *);
typedef void (*bench_ptr_t) (bench_arg_t *);

typedef struct _thread_arg_t {
    /* Input */
    int tid;
    double min_runtime;
    bench_ptr_t bench;

    /* Output */
    double runtime;
    double flops;
} thread_arg_t;

#endif
