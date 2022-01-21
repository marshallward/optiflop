#define _GNU_SOURCE     /* CPU_*, pthread_attr_setaffinity_np declaration */
#include <features.h>   /* Manually set __USE_GNU (some platforms need this) */

#include <math.h>
#include <pthread.h>    /* pthread_*, CPU_* */
#include <sched.h>      /* CPU_* */
#include <stdio.h>      /* printf */
#include <stdlib.h>     /* strtol, malloc */

#include "sse.h"
#include "sse_fma.h"
#include "avx.h"
#include "avx_fma.h"
#include "avx512.h"
#include "roof.h"
#include "bench.h"
#include "input.h"

#include "stopwatch.h"
#include "gpu.h"

int main(int argc, char *argv[])
{
    /* Input variables */
    struct input_config *cfg;

    cfg = malloc(sizeof(struct input_config));
    parse_input(argc, argv, cfg);

    /* CPU set variables */
    cpu_set_t cpuset;
    int ncpus;
    int *cpus;
    int id, c;

    int b, t;
    long vlen, vlen_start, vlen_end;
    double vlen_scale;
    int nthreads;

    /* Thread control variables */
    pthread_mutex_t mutex;
    pthread_attr_t attr;
    pthread_barrier_t barrier;
    pthread_t *threads;
    struct thread_args *t_args;
    void *status;

    volatile int runtime_flag;

    /* Output variables */
    FILE *output = NULL;
    double **results = NULL;

    double total_flops, total_bw_load, total_bw_store;

    /* Ensemble handler */
    int ens;
    double max_total_flops, max_total_bw_load, max_total_bw_store;

    vlen_start = cfg->vlen_start;
    vlen_end = cfg->vlen_end;
    vlen_scale = cfg->vlen_scale;
    nthreads = cfg->nthreads;

    /* Thread setup */

    threads = malloc(nthreads * sizeof(pthread_t));
    t_args = malloc(nthreads * sizeof(struct thread_args));

    pthread_mutex_init(&mutex, NULL);
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    pthread_barrier_init(&barrier, NULL, nthreads);

    /* Generate the CPU set */
    sched_getaffinity(0, sizeof(cpuset), &cpuset);
    ncpus = CPU_COUNT(&cpuset);
    cpus = malloc(ncpus * sizeof(int));

    c = 0;
    for (id = 0; c < ncpus; id++) {
        if (CPU_ISSET(id, &cpuset)) {
            cpus[c] = id;
            c++;
        }
    }

    const struct task simd_tasks[] = {
        {.name = "sse_add",     .thread = {.simd = &sse_add}},
        {.name = "sse_fma",     .thread = {.simd = &sse_fma}},
        {.name = "sse_fmac",    .thread = {.simd = &sse_fmac}},
        {.name = "avx_add",     .thread = {.simd = &avx_add}},
        {.name = "avx_mac",     .thread = {.simd = &avx_mac}},
        {.name = "avx_fma",     .thread = {.simd = &avx_fma}},
        {.name = "avx_fmac",    .thread = {.simd = &avx_fmac}},
        {.name = "avx512_add",  .thread = {.simd = &avx512_add}},
        {.name = "avx512_fma",  .thread = {.simd = &avx512_fma}},
        {.name = "gpu_add",  .thread = {.simd = &gpu_add}},
    };
    int nsimd = sizeof(simd_tasks) / sizeof(struct task);

    const struct task roof_tasks[] = {
        {.name = "y[:] = x[:]",                 .thread = {.roof = &roof_copy}},
        {.name = "y[:] = a x[:]",               .thread = {.roof = &roof_ax}},
        {.name = "y[:] = x[:] + x[:]",          .thread = {.roof = &roof_xpx}},
        {.name = "y[:] = x[:] + y[:]",          .thread = {.roof = &roof_xpy}},
        {.name = "y[:] = a x[:] + y[:]",        .thread = {.roof = &roof_axpy}},
        {.name = "y[:] = a x[:] + b y[:]",      .thread = {.roof = &roof_axpby}},
        {.name = "y[1:] = x[1:] + x[:-1]",      .thread = {.roof = &roof_diff}},
        {.name = "y[8:] = x[8:] + x[:-8]",      .thread = {.roof = &roof_diff8}},
        {.name = "GPU: y[:] = a * x[:] + y[:]", .thread = {.roof = &gpu_axpy}},
    };
    int nroof = sizeof(roof_tasks) / sizeof(struct task);

    /* IO setup */
    if (cfg->save_output) {
        results = malloc(2 * sizeof(double *));
        results[0] = malloc(nroof * sizeof(double));
        results[1] = malloc(nroof * sizeof(double));

        output = fopen("results.txt", "w");
    }

    /* Timer setup */
    /* TODO: Evaluate in separate file so function can be declared as static */
    /* TODO: Don't do this, Make a "calibrate" func for each Stopwatch type */
    if (cfg->timer_type == TIMER_TSC)
        stopwatch_set_tsc_freq();

    /* SIMD tests */
    for (b = 0; b < nsimd; b++) {
        max_total_flops = 0.;
        max_total_bw_load = 0.;
        max_total_bw_store = 0.;

        for (ens = 0; ens < cfg->ensembles; ens++) {
            for (t = 0; t < nthreads; t++) {
                /* TODO: Better way to keep processes off busy threads */
                if (nthreads > 1) {
                    CPU_ZERO(&cpuset);
                    CPU_SET(cpus[t], &cpuset);
                    pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t),
                                                &cpuset);
                }

                /* Thread inputs */
                t_args[t].tid = t;
                t_args[t].vlen = 1;
                t_args[t].min_runtime = cfg->min_runtime;
                t_args[t].benchmark.simd = simd_tasks[b].thread.simd;
                t_args[t].timer_type = cfg->timer_type;
                t_args[t].mutex = &mutex;
                t_args[t].barrier = &barrier;
                t_args[t].runtime_flag = &runtime_flag;

                pthread_create(&threads[t], &attr, &simd_thread,
                    (void *) &t_args[t]
                );
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
            printf("%s GFLOP/s: %.3f (%.3f / thread)\n",
                    simd_tasks[b].name, total_flops / 1e9,
                    total_flops / nthreads / 1e9);

        if (total_bw_load > 0. && total_bw_store > 0.)
            printf("%s GB/s: %.3f (%.3f / thread)\n",
                    simd_tasks[b].name, (total_bw_load + total_bw_store) / 1e9,
                    (total_bw_load + total_bw_store) / nthreads / 1e9);

        if (cfg->verbose) {
            for (t = 0; t < nthreads; t++) {
                printf("    - Thread %i %s runtime: %.12f\n",
                       t, simd_tasks[b].name, t_args[t].runtime);
                printf("    - Thread %i %s gflops: %.12f\n",
                       t, simd_tasks[b].name, t_args[t].flops /  1e9);
                printf("    - Thread %i %s load BW: %.12f\n",
                       t, simd_tasks[b].name, t_args[t].bw_load /  1e9);
                printf("    - Thread %i %s store BW: %.12f\n",
                       t, simd_tasks[b].name, t_args[t].bw_store /  1e9);
            }
        }

        /* Store results for model output */
        if (cfg->save_output) {
            results[0][b] = total_flops;
            results[1][b] = total_bw_load + total_bw_store;
        }
    }

    /* Roofline tests */
    /* TODO: Merge with the SIMD loops */
    for (vlen = vlen_start; vlen < vlen_end; vlen = ceil(vlen * vlen_scale)) {
        for (b = 0; b < nroof; b++) {
            max_total_flops = 0.;
            max_total_bw_load = 0.;
            max_total_bw_store = 0.;

            for (ens = 0; ens < cfg->ensembles; ens++) {
                for (t = 0; t < nthreads; t++) {
                    /* TODO: Better way to keep processes off busy threads */
                    if (nthreads > 1) {
                        CPU_ZERO(&cpuset);
                        CPU_SET(cpus[t], &cpuset);
                        pthread_attr_setaffinity_np(&attr, sizeof(cpu_set_t),
                                                    &cpuset);
                    }

                    /* Thread inputs */
                    t_args[t].tid = t;
                    t_args[t].vlen = vlen;
                    t_args[t].min_runtime = cfg->min_runtime;
                    t_args[t].benchmark.roof = roof_tasks[b].thread.roof;
                    t_args[t].timer_type = cfg->timer_type;
                    t_args[t].mutex = &mutex;
                    t_args[t].barrier = &barrier;
                    t_args[t].runtime_flag = &runtime_flag;

                    pthread_create(&threads[t], &attr, &roof_thread,
                        (void *) &t_args[t]
                    );
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
                printf("%s GFLOP/s: %.3f (%.3f / thread)\n",
                        roof_tasks[b].name, total_flops / 1e9,
                        total_flops / nthreads / 1e9);

            if (total_bw_load > 0. && total_bw_store > 0.)
                printf("%s GB/s: %.3f (%.3f / thread)\n",
                        roof_tasks[b].name, (total_bw_load + total_bw_store) / 1e9,
                        (total_bw_load + total_bw_store) / nthreads / 1e9);

            if (cfg->verbose) {
                for (t = 0; t < nthreads; t++) {
                    printf("    - Thread %i %s runtime: %.12f\n",
                           t, roof_tasks[b].name, t_args[t].runtime);
                    printf("    - Thread %i %s gflops: %.12f\n",
                           t, roof_tasks[b].name, t_args[t].flops /  1e9);
                    printf("    - Thread %i %s load BW: %.12f\n",
                           t, roof_tasks[b].name, t_args[t].bw_load /  1e9);
                    printf("    - Thread %i %s store BW: %.12f\n",
                           t, roof_tasks[b].name, t_args[t].bw_store /  1e9);
                }
            }

            /* Store results for model output */
            if (cfg->save_output) {
                results[0][b] = total_flops;
                results[1][b] = total_bw_load + total_bw_store;
            }
        }

        if (cfg->save_output)
            fprintf(output, "%li,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f\n",
                    vlen,
                    results[0][1], results[0][2], results[0][3], results[0][4],
                    results[0][5], results[0][6], results[0][7], results[0][8],
                    results[1][0],
                    results[1][1], results[1][2], results[1][3], results[1][4],
                    results[1][5], results[1][6], results[1][7], results[1][8]
            );
    }

    /* IO cleanup */
    if (cfg->save_output) {
        free(results);
        fclose(output);
    }

    pthread_barrier_destroy(&barrier);
    pthread_attr_destroy(&attr);
    pthread_mutex_destroy(&mutex);
    free(cpus);
    free(t_args);
    free(threads);
    free(cfg);

    return 0;
}
