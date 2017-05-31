#ifndef AXPY_H_
#define AXPY_H_

#include "bench.h"

void axpy_main(bench_arg_t *);
double roof_copy(float, float, float *, float *, int, double *, double);
double roof_ax(float, float, float *, float *, int, double *, double);
double roof_xpy(float, float, float *, float *, int, double *, double);
double roof_axpy(float, float, float *, float *, int, double *, double);
double roof_axpby(float, float, float *, float *, int, double *, double);

#endif
