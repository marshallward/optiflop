/* FLOP test (based heavily on Alex Yee source) */

#include <immintrin.h>  /* __m256, _m256_* */
#include <stdint.h>     /* uint64_t */
#include <stdio.h>      /* printf */
#include <string.h>     /* strcpy */
#include <time.h>       /* timespec, clock_gettime */

// pthread testing
#define __USE_GNU   /* (Optional) pthread_attr_setaffinity_np declaration */
#include <pthread.h>

#include "timer.h"
#include "avx.h"


typedef struct _thread_arg_t {
    int tid;
    char name[7];   /* TODO: Dynamic length (or just ditch it) */
    double (*bench)();
    long flops;
} thread_arg_t;


void * bench_thread(void *tinfo)
{
    int tid;
    char name[7];
    double runtime;
    long flops;

    tid = (int) ((thread_arg_t *) tinfo)->tid;
    strcpy(name, ((thread_arg_t *) tinfo)->name);
    runtime = (*((thread_arg_t *) tinfo)->bench)();
    flops = ((thread_arg_t *) tinfo)->flops;

    printf("Thread %i %s runtime: %.12f\n", tid, name, runtime);
    /* (iterations) * (8 flops / register) * (8 registers / iteration) */
    printf("Thread %i %s gflops: %.12f\n", tid, name, flops / (runtime * 1e9));

    pthread_exit(NULL);
}


int main(int argc, char *argv[])
{
    /* pthread implementation */
    pthread_t *threads;
    pthread_attr_t attr;
    cpu_set_t cpus;
    int nthreads;
    thread_arg_t *t_args;

    void *status;
    int t;

    /* TODO: proper getopt */
    if (argc == 2)
        nthreads = (int) strtol(argv[1], (char **) NULL, 10);
    else
        nthreads = 1;

    threads = malloc(nthreads * sizeof(pthread_t));
    t_args = malloc(nthreads * sizeof(thread_arg_t));

    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    /* avx_add */

    for (t = 0; t < nthreads; t++) {
        /* TODO: Better way to keep processes off the busy threads */
        if (nthreads > 1) {
            CPU_ZERO(&cpus);
            CPU_SET(t, &cpus);
            pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
        }

        t_args[t].tid = t;
        strcpy(t_args[t].name, "avx_add");
        t_args[t].bench = &avx_add;
        t_args[t].flops = N * 8 * 8;

        pthread_create(&threads[t], &attr, bench_thread, (void *) &t_args[t]);
    }

    for (t = 0; t < nthreads; t++)
        pthread_join(threads[t], &status);

    /* avx_mac */

    for (t = 0; t < nthreads; t++) {
        if (nthreads > 1) {
            CPU_ZERO(&cpus);
            CPU_SET(t, &cpus);
            pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t), &cpus);
        }
        t_args[t].tid = t;
        strcpy(t_args[t].name, "avx_mac");
        t_args[t].bench = &avx_mac;
        t_args[t].flops = N * 8 * 24;
        pthread_create(&threads[t], &attr, bench_thread, (void *) &t_args[t]);
    }

    for (t = 0; t < nthreads; t++)
        pthread_join(threads[t], &status);

    pthread_attr_destroy(&attr);
    free(threads);

    pthread_exit(NULL);
}
