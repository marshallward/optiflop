#ifndef FLOP_AXPY_H_
#define FLOP_AXPY_H_

#include "bench.h"

/* If unset, assume AVX alignment */
#ifndef BYTEALIGN
#define BYTEALIGN 32
#endif

#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 6)
#define ASSUME_ALIGNED(x) __builtin_assume_aligned(x, BYTEALIGN)
#else
#define ASSUME_ALIGNED(x) x
#endif

void * axpy_main(void *);

void roof_copy(int, float, float, float *, float *, struct roof_args *);
void roof_ax(int, float, float, float *, float *, struct roof_args *);
void roof_xpy(int, float, float, float *, float *, struct roof_args *);
void roof_axpy(int, float, float, float *, float *, struct roof_args *);
void roof_axpby(int, float, float, float *, float *, struct roof_args *);

void dummy(float, float, float *, float *);

#endif  // FLOP_AXPY_H_
