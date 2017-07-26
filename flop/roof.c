#include <stdlib.h>
#include <time.h> /* timespec, clock_gettime */

#include "roof.h"
#include "bench.h"
#include "stopwatch.h"

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


/* Note for roof_copy:
 * Many compilers (gcc, icc) will ignore this loop and use its builtin memcpy
 * function, which can perform worse than vectorised loops.
 *
 * To avoid this issue, make sure to disable builtins (usually `-fno-builtin`).
 */

#define ROOF_TEST roof_copy
#define ROOF_KERNEL y[i] = x[i]
#define ROOF_FLOPS 0
#define ROOF_BW_LOAD 1
#define ROOF_BW_STORE 1
#include "roof.inc"


#define ROOF_TEST roof_ax
#define ROOF_KERNEL y[i] = a * x[i]
#define ROOF_FLOPS 1
#define ROOF_BW_LOAD 1
#define ROOF_BW_STORE 1
#include "roof.inc"


#define ROOF_TEST roof_xpy
#define ROOF_KERNEL y[i] = x[i] + y[i]
#define ROOF_FLOPS 1
#define ROOF_BW_LOAD 2
#define ROOF_BW_STORE 1
#include "roof.inc"


#define ROOF_TEST roof_axpy
#define ROOF_KERNEL y[i] = a * x[i] + y[i]
#define ROOF_FLOPS 2
#define ROOF_BW_LOAD 2
#define ROOF_BW_STORE 1
#include "roof.inc"


#define ROOF_TEST roof_axpby
#define ROOF_KERNEL y[i] = a * x[i] + b * y[i]
#define ROOF_FLOPS 3
#define ROOF_BW_LOAD 2
#define ROOF_BW_STORE 1
#include "roof.inc"

#define ROOF_TEST roof_diff
#define ROOF_KERNEL y[i] = x[i] + x[i-1]
#define ROOF_FLOPS 1
#define ROOF_BW_LOAD 2
#define ROOF_BW_STORE 1
#define ROOF_OFFSET 1
#include "roof.inc"
