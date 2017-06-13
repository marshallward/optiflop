#ifndef AXPY_H_
#define AXPY_H_

#include "bench.h"

#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 6)
#define ASSUME_ALIGNED(x, n) __builtin_assume_aligned(x, n)
#else
#define ASSUME_ALIGNED(x, n) x
#endif

void * axpy_main(void *);

void roof_copy(int, float, float, float *, float *, struct roof_args *);
void roof_ax(int, float, float, float *, float *, struct roof_args *);
void roof_xpy(int, float, float, float *, float *, struct roof_args *);
void roof_axpy(int, float, float, float *, float *, struct roof_args *);
void roof_axpby(int, float, float, float *, float *, struct roof_args *);

void dummy(float, float, float *, float *);

#endif
