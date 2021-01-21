#ifndef OPTIFLOP_ROOF_H_
#define OPTIFLOP_ROOF_H_

#include <pthread.h>
#include "stopwatch.h"

/* If unset, assume AVX alignment */
#ifndef BYTEALIGN
#define BYTEALIGN 32
#endif

#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 6)
#define ASSUME_ALIGNED(x) __builtin_assume_aligned(x, BYTEALIGN)
#else
#define ASSUME_ALIGNED(x) x
#endif


struct roof_args {
    /* Timimg */
    Stopwatch *timer;
    double min_runtime;

    /* Thread control */
    pthread_mutex_t *mutex;
    pthread_barrier_t *barrier;
    volatile int *runtime_flag;

    /* Work per kernel operation */
    int kflops;
    int kloads;
    int kstores;
    int offset;

    /* Output */
    double runtime;
    double flops;
    double bw_load;
    double bw_store;
};


/* A dummy function in the SIMD benchmark list indicatind a roofline test.
 * This can be phased out once those tests have been separated.
 */
void * roof_thread(void *);


/* Roofline test function pointer */
typedef void (*roof_ptr_t) (int, float, float, float *, float *,
                            struct roof_args *);

/* Roofline tests */
void roof_copy(int, float, float, float *, float *, struct roof_args *);
void roof_ax(int, float, float, float *, float *, struct roof_args *);
void roof_xpx(int, float, float, float *, float *, struct roof_args *);
void roof_xpy(int, float, float, float *, float *, struct roof_args *);
void roof_axpy(int, float, float, float *, float *, struct roof_args *);
void roof_axpby(int, float, float, float *, float *, struct roof_args *);
void roof_diff(int, float, float, float *, float *, struct roof_args *);
void roof_diff8(int, float, float, float *, float *, struct roof_args *);

void dummy(float, float, float *, float *);

#endif  // OPTIFLOP_ROOF_H_
