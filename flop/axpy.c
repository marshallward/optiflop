#include <stdlib.h>
#include <time.h> /* timespec, clock_gettime */

#include "axpy.h"
#include "bench.h"
#include "stopwatch.h"

#define BYTEALIGN 32

void * axpy_main(void *args_in)
{
    /* Thread input */
    struct thread_args *args;

    float *x, *y;
    float a, b;

    double runtime, flops;

    int n;  // Vector length
    int i;  // Loop counter

    /* Read inputs */
    args = (struct thread_args *) args_in;

    /* TODO: Determine dynamically with L1 size */
    n = args->vlen;

    posix_memalign((void *) &x, BYTEALIGN, n * sizeof(float));
    posix_memalign((void *) &y, BYTEALIGN, n * sizeof(float));

    a = 2.;
    b = 3.;
    for (i = 0; i < n; i++) {
        x[i] = 1.;
        y[i] = 2.;
    }

    /* a x + y */
    runtime = (*(args->roof))(a, b, x, y, n, &flops, args->min_runtime);

    args->runtime = runtime;
    args->flops = flops;
    args->bw_load = 0.;
    args->bw_store = 0.;

    pthread_exit(NULL);
}


double roof_axpy(float a, float b,
                 float * restrict x_in, float * restrict y_in,
                 int n, double *flops, double min_runtime)
{
    float *x, *y;

    Stopwatch *t;
    float runtime;

    int i, r;
    int r_max;

    // TODO: Create a macro somewhere else
    #if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 6)
    x = __builtin_assume_aligned(x_in, BYTEALIGN);
    y = __builtin_assume_aligned(y_in, BYTEALIGN);
    #else
    x = x_in; y = y_in;
    #endif

    t = stopwatch_create(TIMER_POSIX);

    r_max = 1;
    runtime_flag = 0;
    do {
        pthread_barrier_wait(&timer_barrier);
        t->start(t);
        for (r = 0; r < r_max; r++) {
            for (i = 0; i < n; i++)
                //y[i] = y[i] + y[i];
                //y[i] = a + x[i] + y[i];
                y[i] = a * x[i] + y[i];
                //y[i] = a * x[i] * y[i];
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > min_runtime) {
            pthread_mutex_lock(&runtime_mutex);
            runtime_flag = 1;
            pthread_mutex_unlock(&runtime_mutex);
        }

        pthread_barrier_wait(&timer_barrier);
        if (!runtime_flag) r_max *= 2;

    } while (!runtime_flag);

    *flops = 2. * n * r_max / runtime;

    /* Cleanup */
    t->destroy(t);

    return runtime;
}


double roof_copy(float a, float b,
                 float * restrict x_in, float * restrict y_in,
                 int n, double *flops, double min_runtime)
{
    float *x, *y;

    Stopwatch *t;
    float runtime;

    int i, r;
    int r_max;

    // TODO: Create a macro somewhere else
    #if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 6)
    x = __builtin_assume_aligned(x_in, BYTEALIGN);
    y = __builtin_assume_aligned(y_in, BYTEALIGN);
    #else
    x = x_in; y = y_in;
    #endif

    t = stopwatch_create(TIMER_POSIX);

    r_max = 1;
    runtime_flag = 0;
    do {
        pthread_barrier_wait(&timer_barrier);
        t->start(t);
        for (r = 0; r < r_max; r++) {
            for (i = 0; i < n; i++)
                y[i] = x[i];
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > min_runtime) {
            pthread_mutex_lock(&runtime_mutex);
            runtime_flag = 1;
            pthread_mutex_unlock(&runtime_mutex);
        }

        pthread_barrier_wait(&timer_barrier);
        if (!runtime_flag) r_max *= 2;

    } while (!runtime_flag);

    /* No actual flops... but should add STREAM bandwidth */
    *flops = 2. * n * r_max / runtime;

    /* Cleanup */
    t->destroy(t);

    return runtime;
}


double roof_xpy(float a, float b,
                float * restrict x_in, float * restrict y_in,
                int n, double *flops, double min_runtime)
{
    float *x, *y;

    Stopwatch *t;
    float runtime;

    int i, r;
    int r_max;

    // TODO: Create a macro somewhere else
    #if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 6)
    x = __builtin_assume_aligned(x_in, BYTEALIGN);
    y = __builtin_assume_aligned(y_in, BYTEALIGN);
    #else
    x = x_in; y = y_in;
    #endif

    t = stopwatch_create(TIMER_POSIX);

    r_max = 1;
    runtime_flag = 0;
    do {
        pthread_barrier_wait(&timer_barrier);
        t->start(t);
        for (r = 0; r < r_max; r++) {
            for (i = 0; i < n; i++)
                y[i] = x[i] + y[i];
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > min_runtime) {
            pthread_mutex_lock(&runtime_mutex);
            runtime_flag = 1;
            pthread_mutex_unlock(&runtime_mutex);
        }

        pthread_barrier_wait(&timer_barrier);
        if (!runtime_flag) r_max *= 2;

    } while (!runtime_flag);

    /* No actual flops... but should add STREAM bandwidth */
    *flops = n * r_max / runtime;

    /* Cleanup */
    t->destroy(t);

    return runtime;
}


double roof_ax(float a, float b,
               float * restrict x_in, float * restrict y_in,
               int n, double *flops, double min_runtime)
{
    float *x, *y;

    Stopwatch *t;
    float runtime;

    int i, r;
    int r_max;

    // TODO: Create a macro somewhere else
    #if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 6)
    x = __builtin_assume_aligned(x_in, BYTEALIGN);
    y = __builtin_assume_aligned(y_in, BYTEALIGN);
    #else
    x = x_in; y = y_in;
    #endif

    t = stopwatch_create(TIMER_POSIX);

    r_max = 1;
    runtime_flag = 0;
    do {
        pthread_barrier_wait(&timer_barrier);
        t->start(t);
        for (r = 0; r < r_max; r++) {
            for (i = 0; i < n; i++)
                y[i] = a * x[i];
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > min_runtime) {
            pthread_mutex_lock(&runtime_mutex);
            runtime_flag = 1;
            pthread_mutex_unlock(&runtime_mutex);
        }

        pthread_barrier_wait(&timer_barrier);
        if (!runtime_flag) r_max *= 2;

    } while (!runtime_flag);

    /* No actual flops... but should add STREAM bandwidth */
    *flops = n * r_max / runtime;

    /* Cleanup */
    t->destroy(t);

    return runtime;
}


double roof_axpby(float a, float b,
                  float * restrict x_in, float * restrict y_in,
                  int n, double *flops, double min_runtime)
{
    float *x, *y;

    Stopwatch *t;
    float runtime;

    int i, r;
    int r_max;

    // TODO: Create a macro somewhere else
    #if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 6)
    x = __builtin_assume_aligned(x_in, BYTEALIGN);
    y = __builtin_assume_aligned(y_in, BYTEALIGN);
    #else
    x = x_in; y = y_in;
    #endif

    t = stopwatch_create(TIMER_POSIX);

    r_max = 1;
    runtime_flag = 0;
    do {
        pthread_barrier_wait(&timer_barrier);
        t->start(t);
        for (r = 0; r < r_max; r++) {
            for (i = 0; i < n; i++)
                y[i] = a * x[i] + b * y[i];
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > min_runtime) {
            pthread_mutex_lock(&runtime_mutex);
            runtime_flag = 1;
            pthread_mutex_unlock(&runtime_mutex);
        }

        pthread_barrier_wait(&timer_barrier);
        if (!runtime_flag) r_max *= 2;

    } while (!runtime_flag);

    *flops = 3. * n * r_max / runtime;

    /* Cleanup */
    t->destroy(t);

    return runtime;
}
