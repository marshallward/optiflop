#ifndef AXPY_H_
#define AXPY_H_

#include "bench.h"

void * axpy_main(void *);

void roof_copy(int, float, float, float *, float *, struct roof_args *);
void roof_ax(int, float, float, float *, float *, struct roof_args *);
void roof_xpy(int, float, float, float *, float *, struct roof_args *);
void roof_axpy(int, float, float, float *, float *, struct roof_args *);
void roof_axpby(int, float, float, float *, float *, struct roof_args *);

void dummy(float, float, float *, float *);

#endif
