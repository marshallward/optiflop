/* FLOP test (based heavily on Alex Yee source) */

#include <immintrin.h>  /* __m256, _m256_* */
#include <stdint.h>     /* uint64_t */
#include <stdio.h>      /* printf */
#include <time.h>       /* timespec, clock_gettime */

// pthread testing
#define __USE_GNU   /* (Optional) pthread_attr_setaffinity_np declaration */
#include <pthread.h>

#include "timer.h"
#include "avx.h"


int main(int argc, char *argv[])
{
    double runtime;
    int nthreads;

    /* TODO: proper getopt */
    if (argc == 2)
        nthreads = (int) strtol(argv[1], (char **) NULL, 10);
    else
        nthreads = 1;

    /* pthread implementation */
    pthread_t threads[nthreads];
    pthread_attr_t attr;
    cpu_set_t cpus;

    void *status;
    long t;

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
        pthread_create(&threads[t], &attr, avx_add_thread, (void *) t);
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
        pthread_create(&threads[t], &attr, avx_mac_thread, (void *) t);
    }

    for (t = 0; t < nthreads; t++)
        pthread_join(threads[t], &status);

    pthread_attr_destroy(&attr);
    pthread_exit(NULL);
}
