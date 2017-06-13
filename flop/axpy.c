#include <stdlib.h>
#include <time.h> /* timespec, clock_gettime */

#include "axpy.h"
#include "bench.h"
#include "stopwatch.h"

/* If unset, assume AVX alignment */
#ifndef BYTEALIGN
#define BYTEALIGN 32
#endif

void * axpy_main(void *args_in)
{
    /* Thread input */
    struct thread_args *args;

    float *x, *y;
    float a, b;

    int n;  // Vector length
    int i;  // Loop counter

    struct roof_args *rargs;

    /* Read inputs */
    args = (struct thread_args *) args_in;

    n = args->vlen;

    posix_memalign((void *) &x, BYTEALIGN, n * sizeof(float));
    posix_memalign((void *) &y, BYTEALIGN, n * sizeof(float));

    a = 2.;
    b = 3.;
    for (i = 0; i < n; i++) {
        x[i] = 1.;
        y[i] = 2.;
    }

    rargs = malloc(sizeof(struct roof_args));
    rargs->min_runtime = args->min_runtime;

    (*(args->roof))(n, a, b, x, y, rargs);

    args->runtime = rargs->runtime;
    args->flops = rargs->flops;
    args->bw_load = rargs->bw_load;
    args->bw_store = rargs->bw_store;

    free(x);
    free(y);
    free(rargs);
    pthread_exit(NULL);
}


/* Many compilers (gcc, icc) will ignore this loop and use its builtin memcpy
 * function, which can perform worse than vectorised loops.
 *
 * To avoid this issue, make sure to disable builtins (usually `-fno-builtin`).
 */
void roof_copy(int n, float a, float b,
               float * restrict x_in, float * restrict y_in,
               struct roof_args *args)
{
    float *x;
    float *y;
    Stopwatch *t;

    int r, r_max;
    int i;
    int midpt = args->n / 2;
    double runtime;

    x = ASSUME_ALIGNED(x_in, BYTEALIGN);
    y = ASSUME_ALIGNED(y_in, BYTEALIGN);

    t = stopwatch_create(TIMER_POSIX);

    r_max = 1;
    runtime_flag = 0;
    do {
        pthread_barrier_wait(&timer_barrier);
        t->start(t);
        for (r = 0; r < r_max; r++) {
            for (i = 0; i < n; i++)
                y[i] = x[i];
            // Create an impossible branch to prevent loop interchange
            if (y[midpt] < 0.) dummy(a, b, x, y);
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > (args->min_runtime)) {
            pthread_mutex_lock(&runtime_mutex);
            runtime_flag = 1;
            pthread_mutex_unlock(&runtime_mutex);
        }

        pthread_barrier_wait(&timer_barrier);
        if (!runtime_flag) r_max *= 2;

    } while (!runtime_flag);

    args->runtime = runtime;
    args->flops = 0.;
    args->bw_load = n * sizeof(float) * r_max / runtime;
    args->bw_store = n * sizeof(float) * r_max / runtime;

    /* Cleanup */
    t->destroy(t);
}


void roof_ax(int n, float a, float b,
             float * restrict x_in, float * restrict y_in,
             struct roof_args *args)
{
    float *x, *y;

    Stopwatch *t;

    int r, r_max;
    int i;
    int midpt = args->n / 2;
    double runtime;

    x = ASSUME_ALIGNED(x_in, BYTEALIGN);
    y = ASSUME_ALIGNED(y_in, BYTEALIGN);

    t = stopwatch_create(TIMER_POSIX);

    r_max = 1;
    runtime_flag = 0;
    do {
        pthread_barrier_wait(&timer_barrier);
        t->start(t);
        for (r = 0; r < r_max; r++) {
            for (i = 0; i < n; i++)
                y[i] = a * x[i];
            // Create an impossible branch to prevent loop interchange
            if (y[midpt] < 0.) dummy(a, b, x, y);
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > (args->min_runtime)) {
            pthread_mutex_lock(&runtime_mutex);
            runtime_flag = 1;
            pthread_mutex_unlock(&runtime_mutex);
        }

        pthread_barrier_wait(&timer_barrier);
        if (!runtime_flag) r_max *= 2;

    } while (!runtime_flag);

    args->runtime = runtime;
    args->flops = n * r_max / runtime;
    args->bw_load = n * sizeof(float) * r_max / runtime;
    args->bw_store = n * sizeof(float) * r_max / runtime;

    /* Cleanup */
    t->destroy(t);
}


void roof_xpy(int n, float a, float b,
              float * restrict x_in, float * restrict y_in,
              struct roof_args *args)
{
    float *x, *y;

    Stopwatch *t;

    int r, r_max;
    int i;
    int midpt = args->n / 2;
    double runtime;

    x = ASSUME_ALIGNED(x_in, BYTEALIGN);
    y = ASSUME_ALIGNED(y_in, BYTEALIGN);

    t = stopwatch_create(TIMER_POSIX);

    r_max = 1;
    runtime_flag = 0;
    do {
        pthread_barrier_wait(&timer_barrier);
        t->start(t);
        for (r = 0; r < r_max; r++) {
            for (i = 0; i < n; i++)
                y[i] = x[i] + y[i];
            // Create an impossible branch to prevent loop interchange
            if (y[midpt] < 0.) dummy(a, b, x, y);
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > (args->min_runtime)) {
            pthread_mutex_lock(&runtime_mutex);
            runtime_flag = 1;
            pthread_mutex_unlock(&runtime_mutex);
        }

        pthread_barrier_wait(&timer_barrier);
        if (!runtime_flag) r_max *= 2;

    } while (!runtime_flag);

    args->runtime = runtime;
    args->flops = n * r_max / runtime;
    args->bw_load = 2. * n * sizeof(float) * r_max / runtime;
    args->bw_store = n * sizeof(float) * r_max / runtime;

    /* Cleanup */
    t->destroy(t);
}


void roof_axpy(int n, float a, float b,
               float * restrict x_in, float * restrict y_in,
               struct roof_args *args)
{
    float *x, *y;

    Stopwatch *t;

    int r, r_max;
    int i;
    int midpt = args->n / 2;
    double runtime;

    x = ASSUME_ALIGNED(x_in, BYTEALIGN);
    y = ASSUME_ALIGNED(y_in, BYTEALIGN);

    t = stopwatch_create(TIMER_POSIX);

    r_max = 1;
    runtime_flag = 0;
    do {
        pthread_barrier_wait(&timer_barrier);
        t->start(t);
        for (r = 0; r < r_max; r++) {
            for (i = 0; i < n; i++)
                y[i] = a * x[i] + y[i];
            // Create an impossible branch to prevent loop interchange
            if (y[midpt] < 0.) dummy(a, b, x, y);
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > (args->min_runtime)) {
            pthread_mutex_lock(&runtime_mutex);
            runtime_flag = 1;
            pthread_mutex_unlock(&runtime_mutex);
        }

        pthread_barrier_wait(&timer_barrier);
        if (!runtime_flag) r_max *= 2;

    } while (!runtime_flag);

    args->runtime = runtime;
    args->flops = 2. * n * r_max / runtime;
    args->bw_load = 2. * n * sizeof(float) * r_max / runtime;
    args->bw_store = n * sizeof(float) * r_max / runtime;

    /* Cleanup */
    t->destroy(t);
}


void roof_axpby(int n, float a, float b,
                float * restrict x_in, float * restrict y_in,
                struct roof_args *args)
{
    float *x, *y;

    Stopwatch *t;

    int r, r_max;
    int i;
    int midpt = args->n / 2;
    double runtime;

    x = ASSUME_ALIGNED(x_in, BYTEALIGN);
    y = ASSUME_ALIGNED(y_in, BYTEALIGN);

    t = stopwatch_create(TIMER_POSIX);

    r_max = 1;
    runtime_flag = 0;
    do {
        pthread_barrier_wait(&timer_barrier);
        t->start(t);
        for (r = 0; r < r_max; r++) {
            for (i = 0; i < n; i++)
                y[i] = a * x[i] + b * y[i];
            // Create an impossible branch to prevent loop interchange
            if (y[midpt] < 0.) dummy(a, b, x, y);
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > (args->min_runtime)) {
            pthread_mutex_lock(&runtime_mutex);
            runtime_flag = 1;
            pthread_mutex_unlock(&runtime_mutex);
        }

        pthread_barrier_wait(&timer_barrier);
        if (!runtime_flag) r_max *= 2;

    } while (!runtime_flag);

    args->runtime = runtime;
    args->flops = 3. * n * r_max / runtime;
    args->bw_load = 2. * n * sizeof(float) * r_max / runtime;
    args->bw_store = n * sizeof(float) * r_max / runtime;

    /* Cleanup */
    t->destroy(t);
}
