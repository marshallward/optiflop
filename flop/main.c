#define _GNU_SOURCE     /* CPU_*, pthread_attr_setaffinity_np declaration */
#include <features.h>   /* Manually set __USE_GNU (some platforms need this) */

#include <math.h>
#include <omp.h>        /* omp_get_num_procs */
#include <pthread.h>    /* pthread_*, CPU_* */
#include <sched.h>      /* CPU_* */
#include <stdio.h>      /* printf */
#include <stdlib.h>     /* strtol, malloc */

#include "avx.h"
#include "axpy.h"
#include "bench.h"
#include "input.h"

#define ENSEMBLE_COUNT 10

int main(int argc, char *argv[])
{
    pthread_t *threads;
    pthread_attr_t attr;
    cpu_set_t cpus;
    int nthreads, nprocs;
    struct thread_args *t_args;
    void *status;

    struct input_config *cfg;

    int b, t;
    int optflag;
    int print_help;
    int verbose;
    int vlen, vlen_start, vlen_end;
    double vlen_scale;
    double min_runtime;

    int nbench;
    int save_output;
    FILE *output;
    double **results;

    double total_flops, total_bw_load, total_bw_store;

    /* Ensemble handler */
    int ens;
    double max_total_flops, max_total_bw_load, max_total_bw_store;

    cfg = malloc(sizeof(struct input_config));
    parse_input(argc, argv, cfg);

    print_help = cfg->print_help;
    verbose = cfg->verbose;
    vlen_start = cfg->vlen_start;
    vlen_end = cfg->vlen_end;
    vlen_scale = cfg->vlen_scale;
    nthreads = cfg->nthreads;
    min_runtime = cfg->min_runtime;

    /* Display help if requested */
    if (print_help) {
        printf("Help info here.\n");
        return 0;
    }

    if (vlen_end < 0) vlen_end = vlen_start + 1;

    nprocs = omp_get_num_procs();
    if (nthreads > nprocs) {
        printf("flop: Number of threads (%i) exceeds maximum "
               "core count (%i).\n", nthreads, nprocs);
        return -1;
    }

    /* Thread setup */

    threads = malloc(nthreads * sizeof(pthread_t));
    t_args = malloc(nthreads * sizeof(struct thread_args));

    pthread_mutex_init(&runtime_mutex, NULL);
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    pthread_barrier_init(&timer_barrier, NULL, nthreads);

    /* General benchmark loop */
    /* TODO: Combine name and bench into a struct, or add to t_args? */

    const bench_ptr_t benchmarks[] = {
        &avx_add,
        &avx_mac,
        //&avx_fma,
        &axpy_main,
        &axpy_main,
        &axpy_main,
        &axpy_main,
        &axpy_main,
    0};

    const char * benchnames[] = {
        "avx_add",
        "avx_mac",
        //"avx_fma",
        "y[:] = x[:]",
        "y[:] = a x[:]",
        "y[:] = x[:] + y[:]",
        "y[:] = a x[:] + y[:]",
        "y[:] = a x[:] + b y[:]",
    0};

    const roof_ptr_t roof_tests[] = {
        NULL,
        NULL,
        //NULL,
        &roof_copy,
        &roof_ax,
        &roof_xpy,
        &roof_axpy,
        &roof_axpby,
    0};

    /* IO setup */
    if (save_output) {
        for (nbench = 0; benchmarks[nbench]; nbench++) {}

        results = malloc(2 * sizeof(double *));
        results[0] = malloc(nbench * sizeof(double));
        results[1] = malloc(nbench * sizeof(double));

        output = fopen("results.txt", "w");
    }

    /* NOTE: the avx_* tests don't depend on vector length */
    for (vlen = vlen_start; vlen < vlen_end; vlen = ceil(vlen * vlen_scale)) {
        for (b = 0; benchmarks[b]; b++) {
            max_total_flops = 0.;
            max_total_bw_load = 0.;
            max_total_bw_store = 0.;

            for (ens = 0; ens < ENSEMBLE_COUNT; ens++) {
                for (t = 0; t < nthreads; t++) {
                    /* TODO: Better way to keep processes off busy threads */
                    if (nthreads > 1) {
                        CPU_ZERO(&cpus);
                        CPU_SET(t, &cpus);
                        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t),
                                                    &cpus);
                    }

                    /* Thread inputs */
                    t_args[t].tid = t;
                    t_args[t].vlen = vlen;
                    t_args[t].min_runtime = min_runtime;
                    t_args[t].roof = roof_tests[b];

                    pthread_create(&threads[t], &attr, benchmarks[b],
                                   (void *) &t_args[t]);
                }

                for (t = 0; t < nthreads; t++)
                    pthread_join(threads[t], &status);

                total_flops = 0.0;
                total_bw_load = 0.0;
                total_bw_store = 0.0;

                for (t = 0; t < nthreads; t++) {
                    total_flops += t_args[t].flops;
                    total_bw_load += t_args[t].bw_load;
                    total_bw_store += t_args[t].bw_store;
                }

                /* Ensemble maximum */
                if (total_flops > max_total_flops)
                    max_total_flops = total_flops;

                if (total_bw_load > max_total_bw_load)
                    max_total_bw_load = total_bw_load;

                if (total_bw_store > max_total_bw_store)
                    max_total_bw_store = total_bw_store;
            }

            total_flops = max_total_flops;
            total_bw_load = max_total_bw_load;
            total_bw_store = max_total_bw_store;

            if (total_flops > 0.)
                printf("%s GFLOP/s: %.12f (%.12f / thread)\n",
                        benchnames[b], total_flops / 1e9,
                        total_flops / nthreads / 1e9);

            if (total_bw_load > 0. && total_bw_store > 0.)
                printf("%s GB/s: %.12f (%.12f / thread)\n",
                        benchnames[b], (total_bw_load + total_bw_store) / 1e9,
                        (total_bw_load + total_bw_store) / nthreads / 1e9);

            if (verbose) {
                for (t = 0; t < nthreads; t++) {
                    printf("    - Thread %i %s runtime: %.12f\n",
                           t, benchnames[b], t_args[t].runtime);
                    printf("    - Thread %i %s gflops: %.12f\n",
                           t, benchnames[b], t_args[t].flops /  1e9);
                    printf("    - Thread %i %s load BW: %.12f\n",
                           t, benchnames[b], t_args[t].bw_load /  1e9);
                    printf("    - Thread %i %s store BW: %.12f\n",
                           t, benchnames[b], t_args[t].bw_store /  1e9);
                }
            }

            /* Store results for model output */
            if (save_output) {
                results[0][b] = total_flops;
                results[1][b] = total_bw_load + total_bw_store;
            }
        }

        if (save_output)
            fprintf(output, "%i,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n",
                    vlen,
                    results[0][3], results[0][4], results[0][5], results[0][6],
                    results[1][2], results[1][3], results[1][4], results[1][5],
                    results[1][6]);
    }

    /* IO cleanup */
    if (save_output) {
        free(results);
        fclose(output);
    }

    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&runtime_mutex);
    free(t_args);
    free(threads);
    free(cfg);

    return 0;
}
