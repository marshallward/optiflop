#ifndef AXPY_H_
#define AXPY_H_

#include "bench.h"

void * axpy_main(void *);

void roof_copy(struct roof_args *);
void roof_ax(struct roof_args *);
void roof_xpy(struct roof_args *);
void roof_axpy(struct roof_args *);
void roof_axpby(struct roof_args *);

void dummy(float, float, float *, float *);

#endif
