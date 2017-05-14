/* FLOP test (based heavily on Alex Yee source) */

#include <immintrin.h>  /* __m256, _m256_* */
#include <stdint.h>     /* uint64_t */
#include <stdio.h>      /* printf */
#include <string.h>     /* strcpy */
#include <unistd.h>     /* getopt */

// pthread testing
#define __USE_GNU   /* (Optional) pthread_attr_setaffinity_np declaration */
#include <pthread.h>
#include <omp.h>

#include "timer.h"
#include "avx.h"


typedef struct _thread_arg_t {
    int tid;
    void (*bench)(double *, double *);
    double runtime;
    double flops;
} thread_arg_t;


void * bench_thread(void *arg)
{
    thread_arg_t *tinfo;
    int tid;
    double runtime;
    double flops;

    tinfo = (thread_arg_t *) arg;
    tid = tinfo->tid;
    (*((thread_arg_t *) tinfo)->bench)(&runtime, &flops);

    /* Save output */
    tinfo->runtime = runtime;
    tinfo->flops = flops;

    pthread_exit(NULL);
}


int main(int argc, char *argv[])
{
    pthread_t *threads;
    pthread_attr_t attr;
    cpu_set_t cpus;
    int nthreads, nprocs;
    thread_arg_t *t_args;
    void *status;

    int t;
    int optflag;
    double *runtimes, *flops;

    /* getopt */

    nthreads = 1;
    while ((optflag = getopt(argc, argv, "p:")) != -1) {
        switch(optflag) {
            case 'p':
                nthreads = (int) strtol(optarg, (char **) NULL, 10);
                break;
            default:
                abort();
        }
    }

    nprocs = omp_get_num_procs();
    if (nthreads > nprocs) {
        printf("flop: Number of threads (%i) exceeds maximum "
               "core count (%i).\n", nthreads, nprocs);
        return -1;
    }

    /* Thread setup */

    threads = malloc(nthreads * sizeof(pthread_t));
    t_args = malloc(nthreads * sizeof(thread_arg_t));
    runtimes = malloc(nthreads * sizeof(double));
    flops = malloc(nthreads * sizeof(double));

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    pthread_barrier_init(&timer_barrier, NULL, nthreads);

    /* avx_add */

    for (t = 0; t < nthreads; t++) {
        /* TODO: Better way to keep processes off the busy threads */
        if (nthreads > 1) {
            CPU_ZERO(&cpus);
            CPU_SET(t, &cpus);
            pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
        }

        t_args[t].tid = t;
        t_args[t].bench = &avx_add;

        pthread_create(&threads[t], &attr, bench_thread, (void *) &t_args[t]);
    }

    for (t = 0; t < nthreads; t++) {
        pthread_join(threads[t], &status);
        runtimes[t] = t_args[t].runtime;
        flops[t] = t_args[t].flops;

        printf("Thread %i avx_add runtime: %.12f\n", t, runtimes[t]);
        /* (iterations) * (8 flops / register) * (8 registers / iteration) */
        printf("Thread %i avx_add gflops: %.12f\n", t, flops[t] /  1e9);
    }

    /* avx_mac */

    for (t = 0; t < nthreads; t++) {
        if (nthreads > 1) {
            CPU_ZERO(&cpus);
            CPU_SET(t, &cpus);
            pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
        }

        t_args[t].tid = t;
        t_args[t].bench = &avx_mac;
        pthread_create(&threads[t], &attr, bench_thread, (void *) &t_args[t]);
    }

    for (t = 0; t < nthreads; t++) {
        pthread_join(threads[t], &status);
        runtimes[t] = t_args[t].runtime;
        flops[t] = t_args[t].flops;

        printf("Thread %i avx_mac runtime: %.12f\n", t, runtimes[t]);
        /* (iterations) * (8 flops / register) * (24 registers / iteration) */
        printf("Thread %i avx_mac gflops: %.12f\n", t, flops[t] /  1e9);
    }

    pthread_attr_destroy(&attr);
    free(threads);

    pthread_exit(NULL);
}
