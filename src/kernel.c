#include <pthread.h>    /* pthread_* */
#include <math.h>       /* sqrt */

#include "roof.h"       /* roof_args */
#include "stopwatch.h"  /* Stopwatch */


/* Kernel pointer definition */
typedef void (*compute_kernel) (int, double, double, double *, double *);


/* Kernel definitions */
static inline void copy_kernel(int i, double a, double b, double *x, double *y)
    __attribute__((always_inline));
static inline void ax_kernel(int i, double a, double b, double *x, double *y)
    __attribute__((always_inline));
static inline void xpx_kernel(int i, double a, double b, double *x, double *y)
    __attribute__((always_inline));
static inline void xpy_kernel(int i, double a, double b, double *x, double *y)
    __attribute__((always_inline));
static inline void axpy_kernel(int i, double a, double b, double *x, double *y)
    __attribute__((always_inline));
static inline void axpby_kernel(int i, double a, double b, double *x, double *y)
    __attribute__((always_inline));
static inline void diff_kernel(int i, double a, double b, double *x, double *y)
    __attribute__((always_inline));
static inline void diff_simd_kernel(int i, double a, double b, double *x, double *y)
    __attribute__((always_inline));
static inline void mean_kernel(int i, double a, double b, double *x, double *y)
    __attribute__((always_inline));
static inline void mean_simd_kernel(int i, double a, double b, double *x, double *y)
    __attribute__((always_inline));
static inline void sqrt_kernel(int i, double a, double b, double *x, double *y)
    __attribute__((always_inline));
static inline void demo_kernel(int i, double a, double b, double *x, double *y)
    __attribute__((always_inline));


void roof_kernel(int n, double a, double b,
                 double * restrict x_in, double * restrict y_in,
                 struct roof_args *args, compute_kernel kernel)
{
    double *x, *y;

    Stopwatch *t;
    long r, r_max;
    int nk;
    double runtime;

    /* If possible, assert alignment of x_in and y_in */
    x = ASSUME_ALIGNED(x_in);
    y = ASSUME_ALIGNED(y_in);

    t = args->timer;

    nk = n > args->offset ? n - args->offset : 0;

    r_max = 1;
    *(args->runtime_flag) = 0;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (r = 0; r < r_max; r++) {
            for (int i = 0; i < nk; i++)
                kernel(i, a, b, x, y);
            // Create an impossible branch to prevent loop interchange
            if (y[0] < 0.) dummy(a, b, x, y);
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > (args->min_runtime)) {
            pthread_mutex_lock(args->mutex);
            *(args->runtime_flag) = 1;
            pthread_mutex_unlock(args->mutex);
        }

        pthread_barrier_wait(args->barrier);
        if (! *(args->runtime_flag)) r_max *= 2;

    } while (! *(args->runtime_flag));

    /* Total number of kernel calls */
    args->runtime = runtime;
    args->flops = args->kflops * nk * r_max / runtime;
    args->bw_load = args->kloads * sizeof(double) * nk * r_max / runtime;
    args->bw_store = args->kstores * sizeof(double) * nk * r_max / runtime;
}


void roof_kernel_set(int n, double a, double b,
                     double * restrict x_in, double * restrict y_in,
                     struct roof_args *args)
{
    double *x, *y;

    Stopwatch *t;
    long r, r_max;
    int nk;
    double runtime;

    /* If possible, assert alignment of x_in and y_in */
    x = ASSUME_ALIGNED(x_in);
    y = ASSUME_ALIGNED(y_in);

    t = args->timer;

    nk = n > args->offset ? n - args->offset : 0;

    r_max = 1;
    *(args->runtime_flag) = 0;
    do {
        pthread_barrier_wait(args->barrier);
        t->start(t);
        for (r = 0; r < r_max; r++) {
            //for (int k = 0; k < nker; k++)  {
            //    for (int i = 0; i < nk; i++)
            //        kernels[k](i, a, b, x, y);
            //}
            for (int i = 0; i < nk; i++)
                demo_kernel(i, a, b, x, y);
            for (int i = 0; i < nk; i++)
                copy_kernel(i, a, b, y, x);
            // Create an impossible branch to prevent loop interchange
            if (y[0] < 0.) dummy(a, b, x, y);
        }
        t->stop(t);
        runtime = t->runtime(t);

        /* Set runtime flag if any thread exceeds runtime limit */
        if (runtime > (args->min_runtime)) {
            pthread_mutex_lock(args->mutex);
            *(args->runtime_flag) = 1;
            pthread_mutex_unlock(args->mutex);
        }

        pthread_barrier_wait(args->barrier);
        if (! *(args->runtime_flag)) r_max *= 2;

    } while (! *(args->runtime_flag));

    /* Total number of kernel calls */
    args->runtime = runtime;
    args->flops = args->kflops * nk * r_max / runtime;
    args->bw_load = args->kloads * sizeof(double) * nk * r_max / runtime;
    args->bw_store = args->kstores * sizeof(double) * nk * r_max / runtime;
}


/* roof_copy */

/* NOTE: Many compilers (gcc, icc) will ignore the loop and use its builtin
 * memcpy function, which can perform worse or better than vectorised
 * loops.
 *
 * If you want to use SIMD vectorization, then disable builtins
 * (usually with `-fno-builtin`).
 */

void copy_kernel(int i, double a, double b, double *x, double *y) {
    y[i] = x[i];
}

void roof_copy(int n, double a, double b,
               double * restrict x_in, double * restrict y_in,
               struct roof_args *args)
{
    args->kflops = 0;
    args->kloads = 1;
    args->kstores = 1;
    args->offset = 0;

    roof_kernel(n, a, b, x_in, y_in, args, copy_kernel);
}


/* roof_ax */

void ax_kernel(int i, double a, double b, double *x, double *y) {
    y[i] = a * x[i];
}

void roof_ax(int n, double a, double b,
             double * restrict x_in, double * restrict y_in,
             struct roof_args *args)
{
    args->kflops = 1;
    args->kloads = 1;
    args->kstores = 1;
    args->offset = 0;

    roof_kernel(n, a, b, x_in, y_in, args, ax_kernel);
}


/* roof_xpx */

void xpx_kernel(int i, double a, double b, double *x, double *y)
{
    y[i] = x[i] + x[i];
}

void roof_xpx(int n, double a, double b,
              double * restrict x_in, double * restrict y_in,
              struct roof_args *args)
{
    args->kflops = 1;
    args->kloads = 1;
    args->kstores = 1;
    args->offset = 0;

    roof_kernel(n, a, b, x_in, y_in, args, xpx_kernel);
}


/* roof_xpy */

void xpy_kernel(int i, double a, double b, double *x, double *y)
{
    y[i] = x[i] + y[i];
}

void roof_xpy(int n, double a, double b,
              double * restrict x_in, double * restrict y_in,
              struct roof_args *args)
{
    args->kflops = 1;
    args->kloads = 2;
    args->kstores = 1;
    args->offset = 0;

    roof_kernel(n, a, b, x_in, y_in, args, xpy_kernel);
}


/* roof_axpy */

void axpy_kernel(int i, double a, double b, double *x, double *y)
{
    y[i] = a * x[i] + y[i];
}

void roof_axpy(int n, double a, double b,
               double * restrict x_in, double * restrict y_in,
               struct roof_args *args)
{
    args->kflops = 2;
    args->kloads = 2;
    args->kstores = 1;
    args->offset = 0;

    roof_kernel(n, a, b, x_in, y_in, args, axpy_kernel);
}


/* roof_axpby */

void axpby_kernel(int i, double a, double b, double *x, double *y)
{
    y[i] = a * x[i] + b * y[i];
}


void roof_axpby(int n, double a, double b,
                double * restrict x_in, double * restrict y_in,
                struct roof_args *args)
{
    args->kflops = 3;
    args->kloads = 2;
    args->kstores = 1;
    args->offset = 0;

    roof_kernel(n, a, b, x_in, y_in, args, axpby_kernel);
}


/* diff */


void diff_kernel(int i, double a, double b, double *x, double *y)
{
    y[i] = x[i + 1] - x[i];
}


void roof_diff(int n, double a, double b,
               double * restrict x_in, double * restrict y_in,
               struct roof_args *args)
{
    args->kflops = 1;
    args->kloads = 1;
    args->kstores = 1;
    args->offset = 1;

    roof_kernel(n, a, b, x_in, y_in, args, diff_kernel);
}


/* diff_simd */
/* NOTE: We actually do not need to do 2x SIMD width here!
 * I don't yet know why, something unique to X[1]-X[0] vs X[1]+X[0] *
 */

void diff_simd_kernel(int i, double a, double b, double *x, double *y)
{
    const int d = 2 * (32 / sizeof(double));
    y[i] = x[i + d] - x[i];
}


void roof_diff_simd(int n, double a, double b,
                    double * restrict x_in, double * restrict y_in,
                    struct roof_args *args)
{
    const int d = 2 * (32 / sizeof(double));

    args->kflops = 1;
    args->kloads = 1;
    args->kstores = 1;
    args->offset = d;

    roof_kernel(n, a, b, x_in, y_in, args, diff_simd_kernel);
}


/* mean */


void mean_kernel(int i, double a, double b, double *x, double *y)
{
    y[i] = 0.5 * (x[i + 1] + x[i]);
}


void roof_mean(int n, double a, double b,
               double * restrict x_in, double * restrict y_in,
               struct roof_args *args)
{
    args->kflops = 2;
    args->kloads = 1;
    args->kstores = 1;
    args->offset = 1;

    roof_kernel(n, a, b, x_in, y_in, args, mean_kernel);
}


/* mean_simd */
/* NOTE: Optimal will allow summation of two concurrent pairs of two registers.
 * So we need a 2x SIMD offset to do these concurrently.
 */

void mean_simd_kernel(int i, double a, double b, double *x, double *y)
{
    const int d = 2 * (32 / sizeof(double));
    y[i] = 0.5 * (x[i + d] + x[i]);
}


void roof_mean_simd(int n, double a, double b,
                    double * restrict x_in, double * restrict y_in,
                    struct roof_args *args)
{
    const int d = 2 * (32 / sizeof(double));

    args->kflops = 2;
    args->kloads = 1;
    args->kstores = 1;
    args->offset = d;

    roof_kernel(n, a, b, x_in, y_in, args, mean_simd_kernel);
}


void sqrt_kernel(int i, double a, double b, double *x, double *y)
{
    y[i] = sqrt(x[i]);
}


void roof_sqrt(int n, double a, double b,
                double * restrict x_in, double * restrict y_in,
                struct roof_args *args)
{
    args->kflops = 1;
    args->kloads = 1;
    args->kstores = 1;
    args->offset = 0;

    roof_kernel(n, a, b, x_in, y_in, args, sqrt_kernel);
}


void demo_kernel(int i, double a, double b, double *x, double *y)
{
    y[i] = x[i] + a * x[i-1] + b * x[i] + a * x[i+1];
}


void roof_demo(int n, double a, double b,
               double * restrict x_in, double * restrict y_in,
               struct roof_args *args)
{
    compute_kernel kernels[2];
    kernels[0] = demo_kernel;
    kernels[1] = copy_kernel;

    args->kflops = 5;
    args->kloads = 2;
    args->kstores = 1;
    args->offset = 1;

    roof_kernel_set(n, a, b, x_in, y_in, args);
}
