#include <stdlib.h>
#include <time.h> /* timespec, clock_gettime */

#include "roof.h"
#include "bench.h"
#include "stopwatch.h"

inline void copy_kernel(int i, float a, float b, float *x, float *y)
    __attribute__((always_inline));
inline void ax_kernel(int i, float a, float b, float *x, float *y)
    __attribute__((always_inline));
inline void xpy_kernel(int i, float a, float b, float *x, float *y)
    __attribute__((always_inline));
inline void axpy_kernel(int i, float a, float b, float *x, float *y)
    __attribute__((always_inline));
inline void axpby_kernel(int i, float a, float b, float *x, float *y)
    __attribute__((always_inline));
inline void diff_kernel(int i, float a, float b, float *x, float *y)
    __attribute__((always_inline));

void * roof_thread(void *args_in)
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


void roof_kernel(int n, float a, float b,
                 float * restrict x_in, float * restrict y_in,
                 struct roof_args *args, kernel_ptr_t kernel,
                 int flops, int load_bytes, int store_bytes, int offset)
{
    float *x, *y;

    Stopwatch *t;

    long r, r_max;
    int i;
    double runtime;

    x = ASSUME_ALIGNED(x_in);
    y = ASSUME_ALIGNED(y_in);

    t = stopwatch_create(TIMER_POSIX);

    r_max = 1;
    runtime_flag = 0;
    do {
        pthread_barrier_wait(&timer_barrier);
        t->start(t);
        for (r = 0; r < r_max; r++) {
            for (i = 0; i < (n - offset); i++)
                kernel(i, a, b, x, y);
            // Create an impossible branch to prevent loop interchange
            if (y[0] < 0.) dummy(a, b, x, y);
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
    args->flops = flops * (n - offset) * r_max / runtime;
    args->bw_load = load_bytes * (n - offset) * sizeof(float) * r_max
                        / runtime;
    args->bw_store = store_bytes * (n - offset) * sizeof(float) * r_max
                        / runtime;

    /* Cleanup */
    t->destroy(t);
}


/* roof_copy */

/* NOTE: Many compilers (gcc, icc) will ignore the loop and use its builtin
 * memcpy function, which can perform worse or better than vectorised
 * loops.
 *
 * If you want to use SIMD vectorization, then disable builtins
 * (usually with `-fno-builtin`).
 */

void copy_kernel(int i, float a, float b, float *x, float *y) {
    y[i] = x[i];
}

void roof_copy(int n, float a, float b,
               float * restrict x_in, float * restrict y_in,
               struct roof_args *args)
{
    roof_kernel(n, a, b, x_in, y_in, args, copy_kernel, 0, 1, 1, 0);
}


/* roof_ax */

void ax_kernel(int i, float a, float b, float *x, float *y) {
    y[i] = a * x[i];
}

void roof_ax(int n, float a, float b,
             float * restrict x_in, float * restrict y_in,
             struct roof_args *args)
{
    roof_kernel(n, a, b, x_in, y_in, args, ax_kernel, 1, 1, 1, 0);
}


/* roof_xpy */

void xpy_kernel(int i, float a, float b, float *x, float *y)
{
    y[i] = x[i] + y[i];
}

void roof_xpy(int n, float a, float b,
              float * restrict x_in, float * restrict y_in,
              struct roof_args *args)
{
    roof_kernel(n, a, b, x_in, y_in, args, xpy_kernel, 1, 2, 1, 0);
}


/* roof_axpy */

void axpy_kernel(int i, float a, float b, float *x, float *y)
{
    y[i] = a * x[i] + y[i];
}

void roof_axpy(int n, float a, float b,
               float * restrict x_in, float * restrict y_in,
               struct roof_args *args)
{
    roof_kernel(n, a, b, x_in, y_in, args, axpy_kernel, 2, 2, 1, 0);
}


/* roof_axpby */

void axpby_kernel(int i, float a, float b, float *x, float *y)
{
    y[i] = a * x[i] + b * y[i];
}


void roof_axpby(int n, float a, float b,
                float * restrict x_in, float * restrict y_in,
                struct roof_args *args)
{
    roof_kernel(n, a, b, x_in, y_in, args, axpby_kernel, 3, 2, 1, 0);
}


void diff_kernel(int i, float a, float b, float *x, float *y)
{
    y[i] = x[i + 1] - x[i];
}


void roof_diff(int n, float a, float b,
               float * restrict x_in, float * restrict y_in,
               struct roof_args *args)
{
    roof_kernel(n, a, b, x_in, y_in, args, diff_kernel, 1, 1, 1, 1);
}
