/* FLOP test (based heavily on Alex Yee source) */
#define _GNU_SOURCE     /* CPU_*, pthread_attr_setaffinity_np declaration */

#include <omp.h>        /* omp_get_num_procs */
#include <pthread.h>    /* pthread_*, CPU_* */
#include <sched.h>      /* CPU_* */
#include <stdio.h>      /* printf */
#include <stdlib.h>     /* strtol, malloc */
#include <unistd.h>     /* getopt */

#include "bench.h"
#include "avx.h"
#include "axpy.h"


int main(int argc, char *argv[])
{
    pthread_t *threads;
    pthread_attr_t attr;
    cpu_set_t cpus;
    int nthreads, nprocs;
    double min_runtime;
    thread_arg_t *t_args;
    void *status;

    int b, t;
    int optflag;

    /* Default values */
    nthreads = 1;
    min_runtime = 1e-2;

    /* Command line parser */

    while ((optflag = getopt(argc, argv, "p:r:")) != -1) {
        switch(optflag) {
            case 'p':
                nthreads = (int) strtol(optarg, (char **) NULL, 10);
                break;
            case 'r':
                min_runtime = strtod(optarg, NULL);
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

    pthread_mutex_init(&runtime_mutex, NULL);
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    pthread_barrier_init(&timer_barrier, NULL, nthreads);

    /* General benchmark loop */
    /* TODO: Combine name and bench into a struct, or add to t_args? */

    const bench_ptr_t benchmarks[] = {&avx_add, &avx_mac, &axpy_main, 0};
    const char * benchnames[] = {"avx_add", "avx_mac", "axpy", 0};

    for (b = 0; benchmarks[b]; b++) {
        for (t = 0; t < nthreads; t++) {
            /* TODO: Better way to keep processes off the busy threads */
            if (nthreads > 1) {
                CPU_ZERO(&cpus);
                CPU_SET(t, &cpus);
                pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
            }

            t_args[t].tid = t;
            t_args[t].min_runtime = min_runtime;
            t_args[t].bench = benchmarks[b];

            pthread_create(&threads[t], &attr, bench_thread, (void *) &t_args[t]);
        }

        for (t = 0; t < nthreads; t++) {
            pthread_join(threads[t], &status);

            printf("Thread %i %s runtime: %.12f\n",
                   t, benchnames[b], t_args[t].runtime);
            printf("Thread %i %s gflops: %.12f\n",
                   t, benchnames[b], t_args[t].flops /  1e9);
        }
    }

    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&runtime_mutex);
    free(threads);

    pthread_exit(NULL);
}
